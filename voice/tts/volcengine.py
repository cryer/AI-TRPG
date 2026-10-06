"""火山引擎豆包语音合成（单向流式 HTTP chunked）。

- POST https://openspeech.bytedance.com/api/v3/tts/unidirectional
- 新版控制台鉴权只需 X-Api-Key；X-Api-Resource-Id 选模型版本
- 每句一个请求，流式返回 base64 PCM chunk；aiohttp session 常驻复用 TLS
- 输出直接要 16kHz PCM，与内部格式一致
"""

from __future__ import annotations

import base64
import json
import os
import uuid
from typing import AsyncIterator

import aiohttp

from voice.tts.base import AudioChunk

URL = "https://openspeech.bytedance.com/api/v3/tts/unidirectional"
DEFAULT_RESOURCE_ID = "seed-tts-2.0"
DEFAULT_SPEAKER = "zh_female_vv_uranus_bigtts"


class VolcengineTTS:
    def __init__(self, api_key: str | None = None, speaker: str = DEFAULT_SPEAKER,
                 resource_id: str = DEFAULT_RESOURCE_ID, sample_rate: int = 16000,
                 speech_rate: int = 0):
        self.api_key = api_key or os.environ["VOLCENGINE_API_KEY"]
        self.speaker = speaker
        self.resource_id = resource_id
        self.sample_rate = sample_rate
        self.speech_rate = speech_rate
        self._session: aiohttp.ClientSession | None = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=60))
        return self._session

    async def synth(self, sentences, gen_id: int) -> AsyncIterator[AudioChunk]:
        session = await self._get_session()
        seq = 0
        async for sentence in sentences:
            body = {"req_params": {
                "text": sentence,
                "speaker": self.speaker,
                "audio_params": {"format": "pcm", "sample_rate": self.sample_rate,
                                 "speech_rate": self.speech_rate},
            }}
            headers = {"X-Api-Key": self.api_key,
                       "X-Api-Resource-Id": self.resource_id,
                       "X-Api-Request-Id": str(uuid.uuid4())}
            try:
                async with session.post(URL, json=body, headers=headers) as resp:
                    if resp.status != 200:
                        detail = (await resp.text())[:300]
                        raise RuntimeError(f"volcengine tts HTTP {resp.status}: {detail}")
                    async for raw_line in resp.content:
                        line = raw_line.strip()
                        if not line:
                            continue
                        msg = json.loads(line)
                        code = msg.get("code", 0)
                        if code not in (0, 20000000):  # 0=数据包, 20000000=成功结束
                            raise RuntimeError(f"volcengine tts code={code}: {msg.get('message')}")
                        data = msg.get("data")
                        if data:
                            yield AudioChunk(pcm=base64.b64decode(data),
                                             gen_id=gen_id, seq=seq)
                            seq += 1
            except Exception:
                # 取消（barge-in）或单句失败：向上抛由外层处理（M6 做降级）
                raise

    async def warmup(self, text: str) -> bytes | None:
        """开机预热（playbook ⑤）：合成一句短文本，暖 TLS 连接与音色/模型，
        返回拼接好的 PCM（调用方可丢弃）。"""
        async def one():
            yield text
        pcm = bytearray()
        async for chunk in self.synth(one(), -1):
            pcm.extend(chunk.pcm)
        return bytes(pcm) if pcm else None

    async def close(self) -> None:
        if self._session is not None and not self._session.closed:
            await self._session.close()
