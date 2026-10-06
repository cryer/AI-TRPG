"""ASR 接口契约（AGENTS.md §5.1）。

实现方负责：流式消费 16kHz PCM16 mono 帧，产出 partial / final transcript 事件。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import AsyncIterator, Protocol


@dataclass
class TranscriptEvent:
    """partial = 未定稿的实时文本；final = 截至目前的完整定稿文本（增量覆盖，非分句片段）。
    消费方取最后一个 final 即全量文本。"""

    kind: str  # "partial" | "final"
    text: str
    t_ms: int  # time.monotonic() 毫秒


class ASR(Protocol):
    async def stream(self, audio: AsyncIterator[bytes]) -> AsyncIterator[TranscriptEvent]:
        ...
