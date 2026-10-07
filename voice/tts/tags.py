"""多音色内联标记 ⟦v:声线id e:情绪⟧ 的分段与声线解析（与具体 TTS 后端无关）。

供支持多音色的 provider（cosyvoice / volcengine）共用：

- segment_stream：句流 → (text, voice_id, emotion) 合成批次流。
  引号（「」/『』/“”）感知的声线状态机，规则详见函数 docstring。
- resolve_voice_id：声线 id 容错解析（LLM 会模仿 history 里的标记自创 id，
  如 female_young 漏 npc_ 前缀），精确 → 补/去 npc_ 前缀 → 唯一子串 → 默认。

标记语法本身（_parse_tag / _TAG_STRIP_RE / strip_tags）定义在 base.py。
"""

from __future__ import annotations

import asyncio
import re
from typing import AsyncIterator

from voice.tts.base import _parse_tag, _TAG_STRIP_RE

# NPC 台词的引号形式：LLM 实际会混用「」和“”（真人局实锤），
# 跨句声线继承按所有形式处理
_QUOTE_OPEN = "「『“"
_QUOTE_CLOSE = "」』”"
# 句流切分：完整标记 / 引号开 / 引号合
_TOKEN_RE = re.compile(r"(⟦[^⟧]*⟧|[「『“」』”])")


def resolve_voice_id(voice_id: str, registered, default: str) -> str:
    """容错解析标记/剧本给的声线 id，返回注册表内的真实 id。

    registered 是可迭代的已注册 id 集合。静默回退默认声线会让女 NPC 变
    男声（真人局实锤），所以依次尝试：精确 → 补/去 npc_ 前缀 →
    唯一子串匹配 → 默认声线（告警）。
    """
    registered = list(registered)
    if not voice_id:
        return default
    if voice_id in registered:
        return voice_id
    for cand in (f"npc_{voice_id}", voice_id.removeprefix("npc_")):
        if cand in registered:
            return cand
    matches = [v for v in registered if voice_id in v or v in voice_id]
    if len(matches) == 1:
        return matches[0]
    print(f"[warn] 声线 {voice_id!r} 未注册，回退默认声线")
    return default


async def segment_stream(
        sentences,
        merge_max_chars: int = 150,
        merge_wait_s: float = 0.05,
) -> AsyncIterator[tuple[str, str, str | None]]:
    """句流 → 合并后的 (text, voice, emotion) 合成批次流。

    引号感知的声线状态机（逐 token 走查标记 ⟦…⟧ 与引号开合）：
    - 引号（「」/『』/“”）未闭合时，无标记句继承当前声线/情绪
      （NPC 台词被分句器切成多句，DM 只在开头标一次）。
    - 引号闭合后，无标记文本回到默认声线（旁白）。
    - 无标记的**新引号**继承本回合上一个 NPC 声线：NPC 两段台词中间
      插叙述/动作时 DM 不会重新标记（「他顿了顿，“还有一件事。”」），
      不继承的话第二段就掉成旁白（真人局实锤）。
    - 同 (voice, emotion) 的相邻片段合并（≤merge_max_chars）：一次合成
      调用内 prosody 连续规划，句间语气/语速不跳变。下一句非阻塞拉取
      （merge_wait_s 宽限），上游没就绪就先合成，不拖慢首音。
    """
    it = sentences.__aiter__()
    explicit = ("", None)      # 最近一次标记设置的声线（引号开/合后失效）
    last_voice = ("", None)    # 本回合上一个 NPC 声线（供无标记新引号继承）
    quote_voice = ("", None)   # 当前引号内使用的声线
    in_quote = False
    buf: list[str] = []
    buf_key = ("", None)
    upstream_end = False
    pending: asyncio.Task | None = None

    async def pull(block: bool) -> str | None:
        # 不用 wait_for(anext)：超时取消会毁掉异步生成器（丢句）。
        # 改为挂起 task，超时不取消、下次接着等同一个。
        nonlocal upstream_end, pending
        if upstream_end:
            return None
        if pending is None:
            pending = asyncio.ensure_future(it.__anext__())
        if not block:
            done, _ = await asyncio.wait({pending}, timeout=merge_wait_s)
            if not done:
                return None
        try:
            raw = await pending
        except StopAsyncIteration:
            upstream_end = True
            raw = None
        pending = None
        return raw

    while True:
        raw = await pull(block=not buf)
        if raw is None:
            if buf:
                yield "".join(buf), buf_key[0], buf_key[1]
                buf = []
            if upstream_end:
                return
            continue
        for tok in _TOKEN_RE.split(raw):
            if not tok:
                continue
            if tok[0] == "⟦" and tok.endswith("⟧"):
                v, e = _parse_tag(tok[1:-1])
                if v:
                    explicit = (v, e)
                    last_voice = explicit
                    if in_quote:
                        quote_voice = explicit
                continue
            if tok in _QUOTE_OPEN:
                in_quote = True
                quote_voice = explicit if explicit[0] else last_voice
                explicit = ("", None)
                continue
            if tok in _QUOTE_CLOSE:
                in_quote = False
                quote_voice = ("", None)
                explicit = ("", None)
                continue
            text = _TAG_STRIP_RE.sub("", tok)
            if not text.strip():
                continue
            key = quote_voice if in_quote else explicit
            if buf and key != buf_key:
                yield "".join(buf), buf_key[0], buf_key[1]
                buf = []
            if not buf:
                buf_key = key
            buf.append(text)
            if sum(len(t) for t in buf) >= merge_max_chars:
                yield "".join(buf), buf_key[0], buf_key[1]
                buf = []
