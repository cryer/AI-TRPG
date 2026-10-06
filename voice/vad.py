"""Silero VAD（ONNX）封装：16kHz PCM16 mono，512 样本/窗（32ms），输出 speech_start / speech_end 事件。

事件判定规则：
- prob >= threshold → 进入语音（需持续 min_speech_ms 才发 speech_start，防瞬态噪音）
- 语音中 prob < neg_threshold 累计 >= endpoint_ms → speech_end
- ONNX 推理耗时毫秒级，但按 AGENTS.md 约定经 run_in_executor 调用，不阻塞 event loop
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import AsyncIterator

import numpy as np
import onnxruntime as ort

FRAME_SAMPLES = 512  # 32ms @ 16kHz
FRAME_BYTES = FRAME_SAMPLES * 2


@dataclass
class VADEvent:
    kind: str  # "speech_start" | "speech_end"
    t_ms: int  # 自 VAD 启动以来的毫秒（按样本数计）
    prob: float


class SileroVAD:
    def __init__(self, model_path: str, threshold: float = 0.5,
                 neg_threshold: float = 0.35, endpoint_ms: int = 400,
                 min_speech_ms: int = 96):
        opts = ort.SessionOptions()
        opts.log_severity_level = 3  # 关掉 initializer 清理警告
        self._sess = ort.InferenceSession(model_path, sess_options=opts,
                                          providers=["CPUExecutionProvider"])
        self.threshold = threshold
        self.neg_threshold = neg_threshold if neg_threshold is not None else threshold - 0.15
        self.endpoint_ms = endpoint_ms
        self.min_speech_ms = min_speech_ms
        self.reset()

    def reset(self) -> None:
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._sr = np.array(16000, dtype=np.int64)
        # v5 模型必须在每窗前拼接上一窗末尾 64 样本的 context，否则输出恒为 ~0
        self._context = np.zeros((1, 64), dtype=np.float32)
        self._samples_seen = 0
        self._in_speech = False
        self._speech_run_ms = 0      # 未确认语音的累计时长
        self._silence_run_ms = 0     # 语音中的静音累计
        self._speech_start_ms = 0    # 语音段起点（含未确认阶段回推）
        self.last_prob = 0.0         # 最近一窗的语音概率（barge-in 逐帧确认用）

    @property
    def now_ms(self) -> int:
        """已处理音频的毫秒数（VAD 时间轴上的"现在"）。"""
        return self._samples_seen * 1000 // 16000

    def _infer(self, frame: bytes) -> float:
        audio = np.frombuffer(frame, dtype=np.int16).astype(np.float32) / 32768.0
        audio = audio.reshape(1, -1)
        x = np.concatenate([self._context, audio], axis=1)
        out, self._state = self._sess.run(
            ["output", "stateN"],
            {"input": x, "state": self._state, "sr": self._sr})
        self._context = x[..., -64:]
        return float(out[0][0])

    def process(self, frame: bytes) -> VADEvent | None:
        """喂一窗 512 样本（1024 字节）PCM16，返回事件或 None。CPU 推理，请勿在 event loop 直接调用。"""
        if len(frame) != FRAME_BYTES:
            raise ValueError(f"frame must be {FRAME_BYTES} bytes, got {len(frame)}")
        prob = self._infer(frame)
        self.last_prob = prob
        frame_ms = FRAME_SAMPLES * 1000 // 16000
        now_ms = self._samples_seen * 1000 // 16000
        self._samples_seen += FRAME_SAMPLES
        event = None

        if not self._in_speech:
            if prob >= self.threshold:
                if self._speech_run_ms == 0:
                    self._speech_start_ms = now_ms
                self._speech_run_ms += frame_ms
                if self._speech_run_ms >= self.min_speech_ms:
                    self._in_speech = True
                    self._silence_run_ms = 0
                    event = VADEvent("speech_start", self._speech_start_ms, prob)
            else:
                self._speech_run_ms = 0
        else:
            if prob < self.neg_threshold:
                self._silence_run_ms += frame_ms
                if self._silence_run_ms >= self.endpoint_ms:
                    self._in_speech = False
                    self._speech_run_ms = 0
                    # 语音结束点 = 静音累计开始处
                    end_ms = now_ms + frame_ms - self._silence_run_ms
                    event = VADEvent("speech_end", end_ms, prob)
                    self._silence_run_ms = 0
            else:
                self._silence_run_ms = 0
        return event

    def flush(self) -> VADEvent | None:
        """输入流结束时，若仍在语音中，补发 speech_end。"""
        if self._in_speech:
            self._in_speech = False
            end_ms = self._samples_seen * 1000 // 16000 - self._silence_run_ms
            self._silence_run_ms = 0
            return VADEvent("speech_end", end_ms, 0.0)
        return None


async def vad_events(vad: SileroVAD, frames: AsyncIterator[bytes],
                     executor=None) -> AsyncIterator[VADEvent]:
    """把帧流变成事件流；推理放 executor，event loop 零阻塞。"""
    loop = asyncio.get_running_loop()
    async for frame in frames:
        ev = await loop.run_in_executor(executor, vad.process, frame)
        if ev is not None:
            yield ev
    ev = vad.flush()
    if ev is not None:
        yield ev
