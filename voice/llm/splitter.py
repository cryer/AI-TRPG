"""分句器：token 流 → 句/从句流（AGENTS.md §7.3：按从句切比整句更早出声）。

规则：
- 硬断点（。！？!? 换行）：立即切出；
- 软断点（，,、；;：:）：缓冲已达到 min_chars 才切（避免切出半截词）；
- 缓冲超过 max_chars：强制在最近一个软/硬断点切，都没有就硬切；
- token 流结束时 flush 剩余。
"""

from __future__ import annotations

from typing import AsyncIterator

HARD_END = set("。！？!?\n")
SOFT_END = set("，,、；;：:")


async def split_sentences(tokens: AsyncIterator[str],
                          min_chars: int = 6,
                          max_chars: int = 48) -> AsyncIterator[str]:
    buf = ""
    async for tok in tokens:
        buf += tok
        while True:
            cut = _find_cut(buf, min_chars, max_chars)
            if cut is None:
                break
            piece, buf = buf[:cut], buf[cut:]
            piece = piece.strip()
            if piece:
                yield piece
    buf = buf.strip()
    if buf:
        yield buf


def _find_cut(buf: str, min_chars: int, max_chars: int) -> int | None:
    # 硬断点：第一个出现就切
    for i, ch in enumerate(buf):
        if ch in HARD_END:
            return i + 1
    # 超长：优先切在 max_chars 内最后一个软断点，否则硬切
    if len(buf) >= max_chars:
        for i in range(min(len(buf), max_chars) - 1, 0, -1):
            if buf[i] in SOFT_END:
                return i + 1
        return max_chars
    # 软断点：够长才切
    if len(buf) >= min_chars:
        for i, ch in enumerate(buf):
            if ch in SOFT_END and i + 1 >= min_chars:
                return i + 1
    return None
