"""Mock TTS：为每句生成可听的提示音调（时长随句长），模拟流式首 chunk 延迟。"""

from __future__ import annotations

import asyncio
from typing import AsyncIterator

import numpy as np

from voice.tts.base import AudioChunk

SR = 16000
CHUNK_MS = 32


class MockTTS:
    def __init__(self, first_chunk_ms: int = 120, ms_per_char: int = 120,
                 max_ms_per_sentence: int = 4000):
        self.first_chunk_ms = first_chunk_ms
        self.ms_per_char = ms_per_char
        self.max_ms_per_sentence = max_ms_per_sentence

    async def synth(self, sentences, gen_id: int) -> AsyncIterator[AudioChunk]:
        seq = 0
        sent_idx = 0
        try:
            async for sentence in sentences:
                if sent_idx == 0:
                    await asyncio.sleep(self.first_chunk_ms / 1000.0)
                dur_ms = min(len(sentence) * self.ms_per_char, self.max_ms_per_sentence)
                freq = 440.0 + sent_idx * 60.0
                pcm = _tone(freq, dur_ms)
                # 按 32ms 一切片流式吐出
                step = SR * CHUNK_MS // 1000 * 2
                for off in range(0, len(pcm), step):
                    yield AudioChunk(pcm=pcm[off:off + step], gen_id=gen_id, seq=seq)
                    seq += 1
                    await asyncio.sleep(0)  # 让出循环，模拟网络流
                sent_idx += 1
        finally:
            pass  # 真实实现在此关连接


def _tone(freq: float, dur_ms: int) -> bytes:
    n = SR * dur_ms // 1000
    t = np.arange(n) / SR
    sig = 0.3 * np.sin(2 * np.pi * freq * t)
    fade = int(SR * 0.02)
    if len(sig) > 2 * fade:
        sig[:fade] *= np.linspace(0, 1, fade)
        sig[-fade:] *= np.linspace(1, 0, fade)
    return (sig * 32767).astype(np.int16).tobytes()
