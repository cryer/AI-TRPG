"""sherpa-onnx 本地流式 ASR（streaming zipformer，bilingual zh-en）。

- OnlineRecognizer 进程级常驻（懒加载），每次 stream() 新建 OnlineStream：
  一回合一个识别会话，与云端 provider 的会话模型一致。
- 输入 16kHz PCM16 mono（32ms 帧），解码推理放 executor，event loop 零阻塞。
- 文本变化 → partial；recognizer 内部 endpoint → 已识别段并入 final 前缀；
  音频迭代结束（VAD endpoint 关流）→ input_finished + 收尾 final。
- provider 可选 cuda/cpu：3050/4090 用 cuda（需 sherpa-onnx GPU 构建，
  Windows PyPI 只有 CPU wheel，GPU 构建在 4090 Linux 服务器上启用）。
"""

from __future__ import annotations

import asyncio
import os
import time
from typing import AsyncIterator

import numpy as np

from voice.asr.base import TranscriptEvent


class SherpaOnnxASR:
    def __init__(self, model_dir: str = "models/sherpa-asr-bilingual-zh-en",
                 encoder: str = "encoder-epoch-99-avg-1.onnx",
                 decoder: str = "decoder-epoch-99-avg-1.onnx",
                 joiner: str = "joiner-epoch-99-avg-1.onnx",
                 tokens: str = "tokens.txt",
                 provider: str = "cpu", num_threads: int = 2,
                 sample_rate: int = 16000,
                 decoding_method: str = "greedy_search"):
        self.model_dir = model_dir
        self.files = dict(encoder=encoder, decoder=decoder, joiner=joiner, tokens=tokens)
        self.provider = provider
        self.num_threads = num_threads
        self.sample_rate = sample_rate
        self.decoding_method = decoding_method
        self._recognizer = None

    def _get_recognizer(self):
        if self._recognizer is None:
            import sherpa_onnx
            f = {k: os.path.join(self.model_dir, v) for k, v in self.files.items()}
            for p in f.values():
                if not os.path.isfile(p):
                    raise FileNotFoundError(f"sherpa ASR model file missing: {p}")
            self._recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
                tokens=f["tokens"], encoder=f["encoder"], decoder=f["decoder"],
                joiner=f["joiner"], num_threads=self.num_threads,
                sample_rate=self.sample_rate, feature_dim=80,
                decoding_method=self.decoding_method, provider=self.provider,
            )
        return self._recognizer

    async def warmup(self) -> None:
        """开机加载模型并跑一次空解码，避免首回合冷启动。"""
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._warmup_sync)

    def _warmup_sync(self) -> None:
        rec = self._get_recognizer()
        s = rec.create_stream()
        s.accept_waveform(self.sample_rate, np.zeros(self.sample_rate // 2, dtype=np.float32))
        while rec.is_ready(s):
            rec.decode_stream(s)

    async def stream(self, audio: AsyncIterator[bytes]) -> AsyncIterator[TranscriptEvent]:
        loop = asyncio.get_running_loop()
        rec = await loop.run_in_executor(None, self._get_recognizer)
        s = rec.create_stream()
        prefix = ""          # 已被内部 endpoint 定稿的前缀文本
        last_text = ""       # 上次对外发出的全量文本（去重 partial）
        pending_final: str | None = None

        def feed(pcm: bytes) -> tuple[str, bool]:
            samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
            s.accept_waveform(self.sample_rate, samples)
            while rec.is_ready(s):
                rec.decode_stream(s)
            text = rec.get_result(s)
            return text, rec.is_endpoint(s)

        def finish() -> str:
            s.input_finished()
            while rec.is_ready(s):
                rec.decode_stream(s)
            return rec.get_result(s)

        try:
            async for pcm in audio:
                text, is_ep = await loop.run_in_executor(None, feed, pcm)
                now_ms = int(time.monotonic() * 1000)
                full = prefix + text
                if is_ep and text:
                    prefix = full
                    rec.reset(s)
                    yield TranscriptEvent(kind="final", text=full, t_ms=now_ms)
                    pending_final = None
                elif full != last_text:
                    yield TranscriptEvent(kind="partial", text=full, t_ms=now_ms)
                last_text = full
            tail = await loop.run_in_executor(None, finish)
            full = prefix + tail
            if full:
                yield TranscriptEvent(kind="final", text=full,
                                      t_ms=int(time.monotonic() * 1000))
        finally:
            # 取消（barge-in 打断段捕获）或异常：流对象随之释放
            del s
