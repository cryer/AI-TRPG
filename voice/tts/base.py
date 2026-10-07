"""TTS 接口契约（AGENTS.md §5.1）。

输入：分句器产出的句/从句流；输出：16kHz PCM16 mono 音频 chunk 流。
取消语义：外层 task cancel；迟到 chunk 由 gen_id 在消费侧丢弃。

可选方法（非 Protocol 必需，调用方用 getattr 探测）：
    async def warmup(text: str) -> bytes | None
        开机预热连接/音色（M4 playbook ⑤），返回合成的 PCM（可丢弃）。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import AsyncIterator, Protocol


@dataclass
class AudioChunk:
    pcm: bytes
    gen_id: int
    seq: int


# 多音色内联标记：⟦v:voice_id e:情绪⟧，作用于其后直到下一个标记的文本。
# 无标记文本声线 id 为 ""（= provider 默认声线）。情绪词表见 game 层 DM 规则。
# game 层会把 LLM 写的 ⟦s:NPC名 e:情绪⟧ 改写成 v: 声线 id；正则对所有
# ⟦...⟧ 形式兜底匹配，确保任何残缺/未知标记都被剥离、绝不会念出来。
_TAG_RE = re.compile(r"⟦([^⟧]*)⟧")
_TAG_V = re.compile(r"\bv:([\w一-鿿·'-]+)")
_TAG_E = re.compile(r"\be:([^⟧]+)")


def _parse_tag(body: str) -> tuple[str, str | None]:
    vm = _TAG_V.search(body)
    em = _TAG_E.search(body)
    voice = vm.group(1) if vm else ""
    emotion = (em.group(1).strip() or None) if em else None
    return voice, emotion


def parse_tagged(sentence: str) -> list[tuple[str, str, str | None]]:
    """把带标记的句子拆成 (text, voice_id, emotion) 片段序列，标记本身被剥离。

    解析失败（残缺/嵌套标记）时退化为整句默认声线，保证标记永不被念出。
    """
    out: list[tuple[str, str, str | None]] = []
    voice, emotion = "", None
    pos = 0
    try:
        for m in _TAG_RE.finditer(sentence):
            seg = sentence[pos:m.start()]
            if seg.strip():
                out.append((seg, voice, emotion))
            voice, emotion = _parse_tag(m.group(1))
            pos = m.end()
        tail = sentence[pos:]
        if tail.strip():
            out.append((tail, voice, emotion))
    except Exception:
        return [(_TAG_RE.sub("", sentence), "", None)]
    if not out:
        cleaned = _TAG_RE.sub("", sentence)
        if cleaned.strip():
            out.append((cleaned, "", None))
    return out


def strip_tags(sentence: str) -> str:
    """给不支持多音色的 provider 用：剥掉全部标记，只留可合成文本。"""
    return _TAG_RE.sub("", sentence)


class TTS(Protocol):
    async def synth(self, sentences: AsyncIterator[str], gen_id: int) -> AsyncIterator[AudioChunk]:
        ...
