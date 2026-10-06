"""WebSocket 双向音频传输（M5w Web 应用）。

- ws_frames：浏览器发来的 16kHz PCM16 mono 帧（1024B = 32ms）→ TurnManager；
  客户端断开/发 hangup → 迭代结束（等价于本地链路的 Ctrl+C）。
- WsPlayer：实现与 player.Player 相同的接口（§5.2 铁律不变）——
  TTS chunk 即时下行（浏览器侧做 jitter buffer），flush 发控制帧让客户端
  立即清播放缓冲（barge-in 静音的关键路径在客户端，服务端只负责第一时间通知）。

时序模型与 NullPlayer 一致：客户端播放起点 ≈ 首 chunk 到达 + prebuffer 估计值
（默认 96ms，含网络 + 浏览器缓冲；Web 链路延迟报告以此为准，正文注明口径）。
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import AsyncIterator

from aiohttp import WSMsgType

from voice.tts.base import AudioChunk

SAMPLE_RATE = 16000
FRAME_BYTES = 1024  # 512 样本 = 32ms


async def ws_frames(ws) -> AsyncIterator[bytes]:
    """WebSocket 二进制帧 → 32ms PCM 帧流。非 1024 整数倍的尾部碎片丢弃。"""
    async for msg in ws:
        if msg.type == WSMsgType.BINARY:
            data = msg.data
            for off in range(0, len(data) - FRAME_BYTES + 1, FRAME_BYTES):
                yield data[off:off + FRAME_BYTES]
        elif msg.type == WSMsgType.TEXT:
            try:
                if json.loads(msg.data).get("type") == "hangup":
                    return
            except (ValueError, AttributeError):
                pass
        elif msg.type in (WSMsgType.CLOSE, WSMsgType.CLOSED, WSMsgType.ERROR):
            return


class WsPlayer:
    """接口与 voice.player.Player 相同，输出走 WebSocket 二进制帧。"""

    def __init__(self, ws, prebuffer_ms: int = 96):
        self._ws = ws
        self.prebuffer_s = prebuffer_ms / 1000.0
        self._gen = -1
        self._gen_start_t = 0.0
        self._fed_bytes = 0
        self._first_audio_t: float | None = None
        self._closed = False
        self.dropped_stale_chunks = 0
        self.dropped_stale_bytes = 0
        self.flushed_bytes = 0

    async def _send_json(self, obj: dict) -> None:
        try:
            await self._ws.send_str(json.dumps(obj))
        except Exception:
            self._closed = True

    def start_gen(self, gen_id: int) -> None:
        self._gen = gen_id
        self._gen_start_t = time.monotonic()
        self._fed_bytes = 0
        self._first_audio_t = None

    async def feed_chunk(self, chunk: AudioChunk) -> bool:
        if chunk.gen_id != self._gen or self._closed:
            self.dropped_stale_chunks += 1
            self.dropped_stale_bytes += len(chunk.pcm)
            return False
        try:
            await self._ws.send_bytes(chunk.pcm)
        except Exception:
            self._closed = True
            return False
        if self._first_audio_t is None:
            self._first_audio_t = time.monotonic() + self.prebuffer_s
        self._fed_bytes += len(chunk.pcm)
        return True

    def end_gen(self) -> None:
        asyncio.get_running_loop().create_task(self._send_json({"type": "eos"}))

    def flush(self) -> None:
        """barge-in：通知客户端立即清缓冲静音；作废当前 gen。"""
        played_s = max(0.0, time.monotonic() - self._gen_start_t - self.prebuffer_s)
        played_bytes = int(played_s * SAMPLE_RATE * 2)
        self.flushed_bytes += max(0, self._fed_bytes - played_bytes)
        self._gen = -1
        self._fed_bytes = 0
        asyncio.get_running_loop().create_task(self._send_json({"type": "flush"}))

    async def wait_done(self, gen_id: int) -> None:
        if gen_id != self._gen or self._closed:
            return
        total_s = self._fed_bytes / (SAMPLE_RATE * 2)
        remaining = (self._gen_start_t + self.prebuffer_s + total_s
                     - time.monotonic())
        if remaining > 0:
            await asyncio.sleep(remaining)

    def first_audio_time(self, gen_id: int) -> float | None:
        return self._first_audio_t

    def stats(self) -> dict:
        return {"dropped_stale_chunks": self.dropped_stale_chunks,
                "dropped_stale_bytes": self.dropped_stale_bytes,
                "flushed_bytes": self.flushed_bytes}

    def close(self) -> None:
        self._closed = True
