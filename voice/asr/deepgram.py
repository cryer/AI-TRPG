"""Deepgram 流式 ASR（WebSocket）。

- 输入 16kHz PCM16 mono 帧（与内部格式一致，无需转换）；
- interim_results=true 出 partial；音频流结束后发 Finalize 强制 finalize，
  再发 CloseStream 收尾；
- 中文用 model=nova-2 + language=zh-CN。
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from typing import AsyncIterator

import websockets

from voice.asr.base import TranscriptEvent

DEFAULT_URL = "wss://api.deepgram.com/v1/listen"


class DeepgramASR:
    def __init__(self, api_key: str | None = None, model: str = "nova-2",
                 language: str = "zh-CN", base_url: str = DEFAULT_URL):
        self.api_key = api_key or os.environ["DEEPGRAM_API_KEY"]
        self.model = model
        self.language = language
        self.base_url = base_url

    def _url(self) -> str:
        params = (f"model={self.model}&language={self.language}"
                  "&encoding=linear16&sample_rate=16000&channels=1"
                  "&interim_results=true&punctuate=true&smart_format=true")
        return f"{self.base_url}?{params}"

    async def stream(self, audio: AsyncIterator[bytes]) -> AsyncIterator[TranscriptEvent]:
        ws = await websockets.connect(
            self._url(),
            additional_headers={"Authorization": f"Token {self.api_key}"},
            ping_interval=10, ping_timeout=20)

        async def sender():
            try:
                async for frame in audio:
                    await ws.send(frame)
                await ws.send(json.dumps({"type": "Finalize"}))
                await ws.send(json.dumps({"type": "CloseStream"}))
            except (websockets.ConnectionClosed, asyncio.CancelledError):
                pass

        send_task = asyncio.create_task(sender())
        acc: list[str] = []  # 已 final 的分句；final 事件统一为"截至目前的完整文本"语义
        sep = "" if self.language.startswith("zh") else " "
        try:
            async for raw in ws:
                msg = json.loads(raw)
                if msg.get("type") != "Results":
                    continue
                alt = msg["channel"]["alternatives"][0]
                text = (alt.get("transcript") or "").strip()
                if not text:
                    continue
                t_ms = int(time.monotonic() * 1000)
                if msg.get("is_final"):
                    acc.append(text)
                    yield TranscriptEvent("final", sep.join(acc), t_ms)
                else:
                    yield TranscriptEvent("partial", sep.join(acc + [text]), t_ms)
        finally:
            send_task.cancel()
            try:
                await send_task
            except asyncio.CancelledError:
                pass
            await ws.close()
