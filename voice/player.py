"""播放层：jitter buffer + flush + gen_id 竞态防护（AGENTS.md §5.2 铁律）。

- Player：sounddevice 输出。回调线程从字节缓冲消费；欠载出静音。
  首帧真实进 DAC 的时刻（回调线程观测）即 t_playback_start。
- NullPlayer：无声卡/CI 环境用。按真实速率模拟播放，音频落盘 wav。

gen_id 规则：feed_chunk 发现 chunk.gen_id != 当前 gen 立即丢弃；
flush() 清空缓冲并作废当前 gen（barge-in 用，M3）。
"""

from __future__ import annotations

import asyncio
import threading
import time
import wave
from pathlib import Path

import numpy as np

from voice.tts.base import AudioChunk

SAMPLE_RATE = 16000
FRAME_SAMPLES = 512
BYTES_PER_MS = SAMPLE_RATE * 2 // 1000


class Player:
    def __init__(self, prebuffer_ms: int = 32, device=None):
        self.prebuffer_bytes = prebuffer_ms * BYTES_PER_MS
        self._buf = bytearray()
        self._lock = threading.Lock()
        self._gen = -1
        self._eos = False              # 当前 gen 的 chunk 流已结束
        self._first_audio_t: float | None = None
        self._pending_first = False
        self._done: asyncio.Event | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        # M3 泄露检测：stale chunk 一律丢弃并计数；flush 清除量累计
        self.dropped_stale_chunks = 0
        self.dropped_stale_bytes = 0
        self.flushed_bytes = 0

        import sounddevice as sd

        def callback(outdata, frames, time_info, status):
            n_bytes = frames * 2
            with self._lock:
                if len(self._buf) >= self.prebuffer_bytes or (
                        self._eos and self._buf):
                    take = min(n_bytes, len(self._buf))
                    data = bytes(self._buf[:take])
                    del self._buf[:take]
                    empty_and_eos = not self._buf and self._eos
                else:
                    data, empty_and_eos = b"", False
            if data:
                if self._pending_first:
                    self._pending_first = False
                    self._first_audio_t = time.monotonic()
                pcm = np.frombuffer(data, dtype=np.int16)
                out = np.zeros(frames, dtype=np.int16)
                out[: len(pcm)] = pcm
                outdata[:, 0] = out
            else:
                outdata[:] = 0
            if empty_and_eos and self._done is not None and self._loop is not None:
                self._loop.call_soon_threadsafe(self._done.set)

        self._stream = sd.OutputStream(samplerate=SAMPLE_RATE, channels=1,
                                       dtype="int16", blocksize=FRAME_SAMPLES,
                                       device=device, callback=callback)
        self._stream.start()

    # ---- 管线侧接口（async） ----

    def start_gen(self, gen_id: int) -> None:
        self._loop = asyncio.get_running_loop()
        self._done = asyncio.Event()
        with self._lock:
            self._gen = gen_id
            self._eos = False
            self._pending_first = True
            self._first_audio_t = None

    async def feed_chunk(self, chunk: AudioChunk) -> bool:
        """返回 False 表示 gen 已作废，chunk 被丢弃。"""
        if chunk.gen_id != self._gen:
            self.dropped_stale_chunks += 1
            self.dropped_stale_bytes += len(chunk.pcm)
            return False
        with self._lock:
            self._buf.extend(chunk.pcm)
        return True

    def end_gen(self) -> None:
        with self._lock:
            self._eos = True
            if not self._buf and self._done is not None and self._loop is not None:
                self._loop.call_soon_threadsafe(self._done.set)

    def flush(self) -> None:
        """barge-in：立即静音，作废当前 gen。"""
        with self._lock:
            self.flushed_bytes += len(self._buf)
            self._buf.clear()
            self._gen = -1
            self._eos = False
            self._pending_first = False
        if self._done is not None and self._loop is not None:
            self._loop.call_soon_threadsafe(self._done.set)

    async def wait_done(self, gen_id: int) -> None:
        if gen_id != self._gen or self._done is None:
            return
        await self._done.wait()

    def first_audio_time(self, gen_id: int) -> float | None:
        return self._first_audio_t

    def stats(self) -> dict:
        return {"dropped_stale_chunks": self.dropped_stale_chunks,
                "dropped_stale_bytes": self.dropped_stale_bytes,
                "flushed_bytes": self.flushed_bytes}

    def close(self) -> None:
        self._stream.stop()
        self._stream.close()


class NullPlayer:
    """无声卡环境：按真实速率模拟播放，音频落盘 wav。接口与 Player 相同。

    保真模型（M3 起）：feed 进来的 chunk 进内存缓冲，只有"播放时刻已到"的
    部分才写入 wav（= 真实 DAC 已出声的部分）；flush 丢弃未播缓冲并计数——
    这样 tts_out.wav 精确等于用户实际听到的音频，barge-in 泄露可验证。
    """

    def __init__(self, out_path: str | Path = "reports/recordings/tts_out.wav",
                 prebuffer_ms: int = 32):
        self.out_path = Path(out_path)
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self.prebuffer_s = prebuffer_ms / 1000.0
        self._gen = -1
        self._first_audio_t: float | None = None
        self._gen_start_t = 0.0
        self._played_bytes = 0        # 当前 gen 已"播出"（已写 wav）的字节数
        self._buf = bytearray()       # 已 feed 未到播放时刻的部分
        self._wf: wave.Wave_write | None = None
        self.dropped_stale_chunks = 0
        self.dropped_stale_bytes = 0
        self.flushed_bytes = 0

    def start_gen(self, gen_id: int) -> None:
        self._gen = gen_id
        self._gen_start_t = time.monotonic()
        self._played_bytes = 0
        self._buf.clear()
        self._first_audio_t = None
        if self._wf is None:
            self._wf = wave.open(str(self.out_path), "wb")
            self._wf.setnchannels(1)
            self._wf.setsampwidth(2)
            self._wf.setframerate(SAMPLE_RATE)

    def _drain_played(self) -> None:
        """把缓冲中播放时刻已到的部分写入 wav。"""
        playable = int((time.monotonic() - self._gen_start_t - self.prebuffer_s)
                       * SAMPLE_RATE * 2) - self._played_bytes
        take = max(0, min(playable, len(self._buf)))
        if take:
            self._wf.writeframes(bytes(self._buf[:take]))
            del self._buf[:take]
            self._played_bytes += take

    async def feed_chunk(self, chunk: AudioChunk) -> bool:
        if chunk.gen_id != self._gen:
            self.dropped_stale_chunks += 1
            self.dropped_stale_bytes += len(chunk.pcm)
            return False
        if self._first_audio_t is None:
            self._first_audio_t = time.monotonic() + self.prebuffer_s
        self._buf.extend(chunk.pcm)
        self._drain_played()
        return True

    def end_gen(self) -> None:
        pass

    def flush(self) -> None:
        self.flushed_bytes += len(self._buf)
        self._buf.clear()
        self._gen = -1

    async def wait_done(self, gen_id: int) -> None:
        if gen_id != self._gen:
            return
        total_s = (self._played_bytes + len(self._buf)) / (SAMPLE_RATE * 2)
        remaining = self._gen_start_t + self.prebuffer_s + total_s - time.monotonic()
        if remaining > 0:
            await asyncio.sleep(remaining)
        if gen_id != self._gen:  # sleep 期间被 flush（防御；正常路径是 task 被取消）
            return
        self._drain_played()

    def first_audio_time(self, gen_id: int) -> float | None:
        return self._first_audio_t

    def stats(self) -> dict:
        return {"dropped_stale_chunks": self.dropped_stale_chunks,
                "dropped_stale_bytes": self.dropped_stale_bytes,
                "flushed_bytes": self.flushed_bytes}

    def close(self) -> None:
        if self._wf is not None:
            self._wf.close()
            self._wf = None
