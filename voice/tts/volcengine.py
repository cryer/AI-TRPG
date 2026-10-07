"""火山引擎豆包语音合成（单向流式 HTTP chunked），多音色 + 情绪版。

- POST https://openspeech.bytedance.com/api/v3/tts/unidirectional
- 新版控制台鉴权只需 X-Api-Key；X-Api-Resource-Id 选模型版本
- 每段一个请求，流式返回 base64 PCM chunk；aiohttp session 常驻复用 TLS
- 输出直接要 16kHz PCM，与内部格式一致

多音色/情绪（与 cosyvoice provider 同一套 ⟦v:声线id e:情绪⟧ 标记体系）：
- 句流经 tags.segment_stream 切成 (text, voice_id, emotion) 段（引号感知
  状态机，规则与本地 cosyvoice 完全一致）；
- voices 配置声线注册表 {id: {"speaker": "火山 voice_type"}}，标记里的
  声线 id 经 tags.resolve_voice_id 容错解析（LLM 会自创 id，如漏 npc_
  前缀），查不到一律回退 speaker 参数兜底，标记绝不会被念出；
- emotions 配置情绪词映射 {中文情绪词: 火山 emotion code}，命中时写入
  audio_params.emotion（可选 emotion_scale，1~5，不填由服务端默认 4）。

emotion 参数依据（2026-10 调研，详见 docs/cloud-tts.md）：
- 官方 WebSocket 双向流式-V3 文档明确 audio_params.emotion +
  emotion_scale（1~5，默认 4）；HTTP 单向 V3 文档的 audio_params 未列
  emotion，但豆包语音合成 2.0 插件文档与多家第三方接入（ZEGO、
  xiaozhi-esp32 等）均在 audio_params 里传 emotion/emotion_scale，
  两条流式链路的 audio_params 同构，故单向流同样放 audio_params；
- 官方确定支持 emotion code 的是 1.0 多情感音色（*_emo_*_mars_bigtts，
  emotion code 表见官方音色列表）。2.0 uranus 大模型音色的官方情感机制
  是 context_texts 语音指令（additions），emotion code 在部分 2.0 音色
  上可用但官方未逐一承诺——效果以实测为准，不支持的音色会静默忽略，
  不会报错。
"""

from __future__ import annotations

import base64
import json
import os
import uuid
from typing import AsyncIterator

import aiohttp

from voice.tts.base import AudioChunk
from voice.tts.tags import resolve_voice_id, segment_stream

URL = "https://openspeech.bytedance.com/api/v3/tts/unidirectional"
DEFAULT_RESOURCE_ID = "seed-tts-2.0"
DEFAULT_SPEAKER = "zh_female_vv_uranus_bigtts"


class VolcengineTTS:
    def __init__(self, api_key: str | None = None, speaker: str = DEFAULT_SPEAKER,
                 resource_id: str = DEFAULT_RESOURCE_ID, sample_rate: int = 16000,
                 speech_rate: int = 0,
                 voices: dict | None = None, emotions: dict | None = None,
                 emotion_scale: int | None = None,
                 merge_max_chars: int = 150, merge_wait_s: float = 0.05):
        self.api_key = api_key or os.environ["VOLCENGINE_API_KEY"]
        # 兜底 speaker：标记缺失或声线 id 查不到映射时使用
        self.speaker = speaker
        self.resource_id = resource_id
        self.sample_rate = sample_rate
        self.speech_rate = speech_rate
        # 声线注册表：{voice_id: {"speaker": "火山 voice_type"}}（可含
        # 剧本 npcs[].voice 引用的 5 个标准声线 id，见 docs/cloud-tts.md）
        self.voices = voices or {}
        # 情绪词映射：{中文情绪词: 火山 emotion code}；未配置的词不带 emotion
        self.emotions = emotions or {}
        self.emotion_scale = emotion_scale
        self.merge_max_chars = merge_max_chars
        self.merge_wait_s = merge_wait_s
        self._session: aiohttp.ClientSession | None = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=60))
        return self._session

    def _speaker_for(self, voice_id: str) -> str:
        """标记声线 id → 火山 voice_type；空 id / 未注册 / 未配置都回退默认。"""
        if not voice_id or not self.voices:
            return self.speaker
        vid = resolve_voice_id(voice_id, self.voices, "")
        prof = self.voices.get(vid) if vid else None
        return (prof or {}).get("speaker") or self.speaker

    def _emotion_code(self, emotion: str | None) -> str | None:
        """中文情绪词 → 火山 emotion code；未配置映射则不带 emotion 参数。"""
        if not emotion:
            return None
        code = self.emotions.get(emotion)
        if code is None:
            print(f"[warn] 情绪 {emotion!r} 未配置火山映射，本段不带 emotion")
        return code

    def _build_body(self, text: str, voice_id: str,
                    emotion: str | None) -> dict:
        audio_params = {"format": "pcm", "sample_rate": self.sample_rate,
                        "speech_rate": self.speech_rate}
        code = self._emotion_code(emotion)
        if code:
            audio_params["emotion"] = code
            if self.emotion_scale is not None:
                audio_params["emotion_scale"] = self.emotion_scale
        return {"req_params": {
            "text": text,
            "speaker": self._speaker_for(voice_id),
            "audio_params": audio_params,
        }}

    async def synth(self, sentences, gen_id: int) -> AsyncIterator[AudioChunk]:
        session = await self._get_session()
        seq = 0
        async for text, voice_id, emotion in segment_stream(
                sentences, self.merge_max_chars, self.merge_wait_s):
            if not text.strip():
                continue
            body = self._build_body(text, voice_id, emotion)
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
                # 取消（barge-in）或单段失败：向上抛由外层处理（M6 做降级）
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
