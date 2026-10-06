"""Mock ASR：无 key 时跑通管线用。收到音频后按可配延迟吐出 partial / final。"""

from __future__ import annotations

import asyncio
import time
from typing import AsyncIterator

from voice.asr.base import TranscriptEvent


class MockASR:
    def __init__(self, text: str = "你好，给我讲个短笑话。", final_ms: int = 120):
        self.text = text
        self.final_ms = final_ms

    async def stream(self, audio: AsyncIterator[bytes]) -> AsyncIterator[TranscriptEvent]:
        n_frames = 0
        partial_sent = False
        async for _frame in audio:
            n_frames += 1
            if not partial_sent and n_frames >= 5:
                partial_sent = True
                yield TranscriptEvent("partial", self.text[: len(self.text) // 2],
                                      _now_ms())
        await asyncio.sleep(self.final_ms / 1000.0)
        yield TranscriptEvent("final", self.text, _now_ms())


def _now_ms() -> int:
    return int(time.monotonic() * 1000)
