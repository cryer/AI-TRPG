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

# NPC 台词的引号形式：LLM 实际会混用「」、“”和直引号 "…"（真人局实锤），
# 跨句声线继承按所有形式处理
_QUOTE_OPEN = "「『“"
_QUOTE_CLOSE = "」』”"
# 直引号（ASCII/全角）：同一字符既开又合，按当前状态切换。漏掉它会让
# 带标记的台词不触发引号开合，声线/情绪经 explicit 泄漏进后续旁白
_QUOTE_TOGGLE = "\"＂"
# 句流切分：完整标记 / 引号开 / 引号合 / 直引号（开合切换）
_TOKEN_RE = re.compile(r"(⟦[^⟧]*⟧|[「『“」』”\"＂])")


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


# 无标记新引号的声线继承先缓冲试探：引号内 ≤_INHERIT_MIN_CHARS 字且没有
# 句末标点（。？！…）的几乎都是叙述里的引用/强调（那声「边军」）而非真
# 台词——继承 NPC 声线会把这两三个字抠出来单独合成（超短文本 + 情绪
# instruct → 拖长音 artifacts），还把旁白批次从中间劈断（真人局实锤）。
# 带句末标点（「还有谁在里面？」）或超长（>_INHERIT_PROBE_CHARS 还没合上
# 引号）认定为真台词，按继承声线流出。
_INHERIT_PROBE_CHARS = 12
_INHERIT_MIN_CHARS = 8


async def segment_stream(
        sentences,
        merge_max_chars: int = 150,
        merge_wait_s: float = 0.4,
        narrator_voice: str = "",
        state: dict | None = None,
        merge_head_chars: int = 0,
) -> AsyncIterator[tuple[str, str, str | None]]:
    """句流 → 合并后的 (text, voice, emotion) 合成批次流。

    引号感知的声线状态机（逐 token 走查标记 ⟦…⟧ 与引号开合）：
    - 引号（「」/『』/“”/直引号 "…"）未闭合时，无标记句继承当前声线/情绪
      （NPC 台词被分句器切成多句，DM 只在开头标一次）。
    - 引号闭合后，无标记文本回到默认声线（旁白）。
    - 无标记的**新引号**继承本回合上一个 NPC 声线：NPC 两段台词中间
      插叙述/动作时 DM 不会重新标记（「他顿了顿，“还有一件事。”」），
      不继承的话第二段就掉成旁白（真人局实锤）。
    - 声线标记只对本句生效：引号未开时 explicit 在句（raw）边界作废——
      「⟦v:x⟧ 台词。」没带引号时，下一句旁白不再继承 NPC 声线
      （直引号台词 + 标记的组合曾让整段旁白变 NPC 音色，真人局实锤）。
    - 旁白情绪记忆：⟦v:旁白声线 e:情绪⟧ 标记只更新旁白氛围状态
      （state["narrator_emotion"]），不占 explicit、不污染 NPC 引号继承；
      之后所有无标记旁白沿用该情绪，直到下一个旁白标记切换——
      氛围不突变，也不每段乱跳。state 由 provider 持有可跨回合延续。
    - 同 (voice, emotion) 的相邻片段合并（≤merge_max_chars）：一次合成
      调用内 prosody 连续规划，句间语气/语速不跳变。下一句非阻塞拉取
      （merge_wait_s 宽限），上游没就绪就先合成，不拖慢首音。
    - 首批次上限 merge_head_chars（0=不单独限制，同 merge_max_chars）：
      回合第一批只攒到这个长度就先发合成，保住首音延迟；之后的批次
      （TTS 已领先播放）才放心攒到 merge_max_chars 换 prosody 连贯。
    """
    if state is None:
        state = {}
    it = sentences.__aiter__()
    explicit = ("", None)      # 最近一次标记设置的声线（引号开/合后失效）
    last_voice = ("", None)    # 本回合上一个 NPC 声线（供无标记新引号继承）
    quote_voice = ("", None)   # 当前引号内使用的声线
    in_quote = False
    buf: list[str] = []
    buf_key = ("", None)
    head_done = False          # 回合第一批是否已发出（首批用 merge_head_chars 封顶）
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

    head_cap = merge_head_chars or merge_max_chars

    def current_cap() -> int:
        return merge_max_chars if head_done else min(head_cap, merge_max_chars)

    def emit_text(text: str, key) -> list[tuple[str, str, str | None]]:
        """把一段文本按 key 并入合成缓冲，返回凑满/换声线而完成的批次。"""
        nonlocal buf_key, head_done
        done = []
        if buf and key != buf_key:
            done.append(("".join(buf), buf_key[0], buf_key[1]))
            buf.clear()
            head_done = True
        if not buf:
            buf_key = key
        buf.append(text)
        if sum(len(t) for t in buf) >= current_cap():
            done.append(("".join(buf), buf_key[0], buf_key[1]))
            buf.clear()
            head_done = True
        return done

    inherit_buf: list[str] | None = None   # 无标记引号的试探缓冲（见上方常量注释）

    while True:
        raw = await pull(block=not buf)
        if raw is None:
            if inherit_buf:
                # 引号没合上就到句尾/流尾：按继承声线放行，别吞掉文本
                for b in emit_text("".join(inherit_buf), quote_voice):
                    yield b
                inherit_buf = None
            if buf:
                yield "".join(buf), buf_key[0], buf_key[1]
                buf = []
                head_done = True
            if upstream_end:
                return
            continue
        for tok in _TOKEN_RE.split(raw):
            if not tok:
                continue
            if tok[0] == "⟦" and tok.endswith("⟧"):
                v, e = _parse_tag(tok[1:-1])
                if v:
                    if narrator_voice and v == narrator_voice:
                        # 旁白氛围标记：只更新旁白情绪状态，不占 explicit、
                        # 不更新 last_voice（避免污染 NPC 引号继承）
                        state["narrator_emotion"] = e
                        explicit = ("", None)
                    else:
                        explicit = (v, e)
                        last_voice = explicit
                        if in_quote:
                            quote_voice = explicit
                            # 引号内来了显式标记：试探缓冲立即按新声线放行
                            if inherit_buf:
                                for b in emit_text("".join(inherit_buf),
                                                   quote_voice):
                                    yield b
                                inherit_buf = None
                continue
            if tok in _QUOTE_TOGGLE:
                # 直引号：同一字符按当前状态切换开/合
                tok = "“" if not in_quote else "”"
            if tok in _QUOTE_OPEN:
                in_quote = True
                if explicit[0]:
                    quote_voice = explicit
                    explicit = ("", None)
                    inherit_buf = None
                else:
                    # 无标记新引号：先缓冲试探，短引用不继承 NPC 声线
                    quote_voice = last_voice
                    inherit_buf = []
                continue
            if tok in _QUOTE_CLOSE:
                in_quote = False
                if inherit_buf is not None:
                    inner = "".join(inherit_buf)
                    inherit_buf = None
                    if inner.strip():
                        # 短引用/强调（那声「边军」）：短且无句末标点 →
                        # 回旁白氛围，不继承 NPC 声线、不单独成批
                        is_quote = len(inner) <= _INHERIT_MIN_CHARS \
                            and not re.search(r"[。？！…?!]", inner)
                        key = (("", state.get("narrator_emotion"))
                               if is_quote else quote_voice)
                        for b in emit_text(inner, key):
                            yield b
                quote_voice = ("", None)
                explicit = ("", None)
                continue
            text = _TAG_STRIP_RE.sub("", tok)
            if not text.strip():
                continue
            if in_quote and inherit_buf is not None:
                inherit_buf.append(text)
                if sum(len(t) for t in inherit_buf) > _INHERIT_PROBE_CHARS:
                    for b in emit_text("".join(inherit_buf), quote_voice):
                        yield b
                    inherit_buf = None
                continue
            if in_quote:
                key = quote_voice
            elif explicit[0]:
                key = explicit
            else:
                # 无标记旁白：沿用旁白氛围情绪（默认 None = 声线自带 instruct）
                key = ("", state.get("narrator_emotion"))
            for b in emit_text(text, key):
                yield b
        # 句边界：引号未开时声线标记作废——「⟦v:x⟧ 无引号台词。」只允许
        # 影响本句，防止 NPC 声线/情绪泄漏进后续旁白
        if not in_quote:
            explicit = ("", None)
