"""LLM 接口契约（AGENTS.md §5.1）。

取消语义：外层 asyncio task cancel，实现方在 finally 里关连接。
"""

from __future__ import annotations

from typing import AsyncIterator, Protocol


class LLM(Protocol):
    async def stream(self, messages: list[dict], gen_id: int) -> AsyncIterator[str]:
        """token/片段流。messages 为 OpenAI 风格 [{role, content}, ...]。"""
        ...
