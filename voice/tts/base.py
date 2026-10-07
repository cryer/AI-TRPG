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
# 兜底剥离：完整标记 | 未闭合标记（流被截断）| 落单的 ⟧
_TAG_STRIP_RE = re.compile(r"⟦[^⟧]*⟧|⟦[^⟧]*$|⟧")


def _parse_tag(body: str) -> tuple[str, str | None]:
    vm = _TAG_V.search(body)
    em = _TAG_E.search(body)
    voice = vm.group(1) if vm else ""
    emotion = (em.group(1).strip() or None) if em else None
    return voice, emotion


def _emit(out: list, text: str, voice: str, emotion: str | None) -> None:
    """片段入列前剥离残余标记（未闭合的 ⟦…、落单 ⟧），保证绝不念出。"""
    text = _TAG_STRIP_RE.sub("", text)
    if text.strip():
        out.append((text, voice, emotion))


def parse_tagged(sentence: str) -> list[tuple[str, str, str | None]]:
    """把带标记的句子拆成 (text, voice_id, emotion) 片段序列，标记本身被剥离。

    解析失败（残缺/嵌套标记）时退化为整句默认声线，保证标记永不被念出。
    """
    out: list[tuple[str, str, str | None]] = []
    voice, emotion = "", None
    pos = 0
    try:
        for m in _TAG_RE.finditer(sentence):
            _emit(out, sentence[pos:m.start()], voice, emotion)
            voice, emotion = _parse_tag(m.group(1))
            pos = m.end()
        _emit(out, sentence[pos:], voice, emotion)
    except Exception:
        return [(_TAG_STRIP_RE.sub("", sentence), "", None)]
    return out


def parse_tagged_stateful(
        sentence: str, voice: str = "", emotion: str | None = None
) -> tuple[list[tuple[str, str, str | None]], str, str | None]:
    """parse_tagged 的有状态版：标记状态从 (voice, emotion) 开始，返回结束状态。

    供跨句保持声线：NPC 多句台词只在开头标一次，后续无标记句继承该声线
    （由调用方决定何时继承，如引号未闭合）。句内先文本后标记的残缺形式
    不影响状态提取；整条标记解析异常时该句按起始状态原样返回。
    """
    out: list[tuple[str, str, str | None]] = []
    pos = 0
    try:
        for m in _TAG_RE.finditer(sentence):
            _emit(out, sentence[pos:m.start()], voice, emotion)
            voice, emotion = _parse_tag(m.group(1))
            pos = m.end()
        _emit(out, sentence[pos:], voice, emotion)
    except Exception:
        return [(_TAG_STRIP_RE.sub("", sentence), voice, emotion)], voice, emotion
    return out, voice, emotion


def strip_tags(sentence: str) -> str:
    """给不支持多音色的 provider 用：剥掉全部标记，只留可合成文本。"""
    return _TAG_STRIP_RE.sub("", sentence)


class TTS(Protocol):
    async def synth(self, sentences: AsyncIterator[str], gen_id: int) -> AsyncIterator[AudioChunk]:
        ...
