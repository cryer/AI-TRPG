"""Kimi LLM（OpenAI 兼容流式，coding 订阅 key 走 api.kimi.com/coding）。

注意：kimi-for-coding 系列强制 thinking，流里先出 reasoning_content 再出 content；
只转发 content，reasoning_effort=low 压 thinking 长度。
"""

from __future__ import annotations

import json
import os
from typing import AsyncIterator

import aiohttp

DEFAULT_BASE_URL = "https://api.kimi.com/coding/v1"
DEFAULT_MODEL = "kimi-for-coding-highspeed"


class KimiLLM:
    def __init__(self, api_key: str | None = None, model: str = DEFAULT_MODEL,
                 base_url: str = DEFAULT_BASE_URL, reasoning_effort: str = "low",
                 max_tokens: int = 300, temperature: float | None = None):
        # kimi-for-coding 系列只允许 temperature=1，默认 None 即不传该参数
        self.api_key = api_key or os.environ["KIMI_API_KEY"]
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.reasoning_effort = reasoning_effort
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._session: aiohttp.ClientSession | None = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=aiohttp.ClientTimeout(total=60))
        return self._session

    async def stream(self, messages: list[dict], gen_id: int) -> AsyncIterator[str]:
        body = {"model": self.model, "messages": messages, "stream": True,
                "max_tokens": self.max_tokens,
                "reasoning_effort": self.reasoning_effort}
        if self.temperature is not None:
            body["temperature"] = self.temperature
        session = await self._get_session()
        try:
            async with session.post(f"{self.base_url}/chat/completions", json=body) as resp:
                if resp.status != 200:
                    detail = (await resp.text())[:300]
                    raise RuntimeError(f"kimi HTTP {resp.status}: {detail}")
                async for line in resp.content:
                    line = line.strip()
                    if not line.startswith(b"data: "):
                        continue
                    payload = line[6:]
                    if payload == b"[DONE]":
                        break
                    delta = json.loads(payload)["choices"][0].get("delta", {})
                    content = delta.get("content")
                    if content:
                        yield content
        finally:
            pass  # session 常驻复用（TLS/TCP 预热），连接由 cancel 时 aiohttp 自动释放

    async def stream_events(self, messages: list[dict], gen_id: int,
                            tools: list[dict] | None = None) -> AsyncIterator:
        """带 function calling 的事件流。

        content token 照常以 str yield；流结束时若收到 tool_calls，最后 yield
        一个 dict：{"tool_calls": [{"id", "name", "arguments"}], "content": str}，
        其中 arguments 为 json.loads 后的 dict（解析失败则保留原始字符串）。
        """
        body = {"model": self.model, "messages": messages, "stream": True,
                "max_tokens": self.max_tokens,
                "reasoning_effort": self.reasoning_effort}
        if self.temperature is not None:
            body["temperature"] = self.temperature
        if tools is not None:
            body["tools"] = tools
        session = await self._get_session()
        content_parts: list[str] = []
        pending: dict[int, dict] = {}  # index → 累积中的 tool_call
        try:
            async with session.post(f"{self.base_url}/chat/completions", json=body) as resp:
                if resp.status != 200:
                    detail = (await resp.text())[:300]
                    raise RuntimeError(f"kimi HTTP {resp.status}: {detail}")
                async for line in resp.content:
                    line = line.strip()
                    if not line.startswith(b"data: "):
                        continue
                    payload = line[6:]
                    if payload == b"[DONE]":
                        break
                    delta = json.loads(payload)["choices"][0].get("delta", {})
                    content = delta.get("content")
                    if content:
                        content_parts.append(content)
                        yield content
                    for tc in delta.get("tool_calls") or []:
                        acc = pending.setdefault(tc.get("index", 0),
                                                 {"id": "", "name": "", "arguments": ""})
                        if tc.get("id"):
                            acc["id"] += tc["id"]
                        fn = tc.get("function") or {}
                        if fn.get("name"):
                            acc["name"] += fn["name"]
                        if fn.get("arguments"):
                            acc["arguments"] += fn["arguments"]
        finally:
            pass  # session 常驻复用（TLS/TCP 预热），连接由 cancel 时 aiohttp 自动释放
        if pending:
            tool_calls = []
            for idx in sorted(pending):
                acc = pending[idx]
                try:
                    args = json.loads(acc["arguments"]) if acc["arguments"] else {}
                except json.JSONDecodeError:
                    args = acc["arguments"]
                tool_calls.append({"id": acc["id"], "name": acc["name"],
                                   "arguments": args})
            yield {"tool_calls": tool_calls, "content": "".join(content_parts)}

    async def close(self) -> None:
        if self._session is not None and not self._session.closed:
            await self._session.close()
