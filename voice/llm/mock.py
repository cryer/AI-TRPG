"""Mock LLM：模拟低 TTFT 流式生成，按字/词片吐出 canned 回复。"""

from __future__ import annotations

import asyncio
from typing import AsyncIterator


class MockLLM:
    def __init__(self,
                 reply: str = "好的，我查一下。为什么程序员分不清万圣节和圣诞节？因为 Oct 31 等于 Dec 25。",
                 first_token_ms: int = 180,
                 token_interval_ms: int = 25):
        self.reply = reply
        self.first_token_ms = first_token_ms
        self.token_interval_ms = token_interval_ms

    async def stream(self, messages: list[dict], gen_id: int) -> AsyncIterator[str]:
        try:
            await asyncio.sleep(self.first_token_ms / 1000.0)
            # 按 2~4 字一片模拟 token 粒度
            i = 0
            while i < len(self.reply):
                step = min(3, len(self.reply) - i)
                yield self.reply[i:i + step]
                i += step
                await asyncio.sleep(self.token_interval_ms / 1000.0)
        finally:
            pass  # 真实实现在此关连接
