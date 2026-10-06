"""TTS 接口契约（AGENTS.md §5.1）。

输入：分句器产出的句/从句流；输出：16kHz PCM16 mono 音频 chunk 流。
取消语义：外层 task cancel；迟到 chunk 由 gen_id 在消费侧丢弃。

可选方法（非 Protocol 必需，调用方用 getattr 探测）：
    async def warmup(text: str) -> bytes | None
        开机预热连接/音色（M4 playbook ⑤），返回合成的 PCM（可丢弃）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import AsyncIterator, Protocol


@dataclass
class AudioChunk:
    pcm: bytes
    gen_id: int
    seq: int


class TTS(Protocol):
    async def synth(self, sentences: AsyncIterator[str], gen_id: int) -> AsyncIterator[AudioChunk]:
        ...
