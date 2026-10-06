"""火山引擎豆包流式语音识别（sauc 大模型 ASR，WebSocket 自定义二进制协议）。

协议要点（docs 6561/1354869）：
- 4 字节头：version(4bit)=1 | header_size(4bit)=1；msg_type(4bit) | flags(4bit)；
  serialization(4bit)=1(JSON) | compression(4bit)=1(gzip)；保留 1 字节
- full client request(type=1)：gzip(JSON 参数)；audio only request(type=2)：gzip(PCM)，
  最后一包 flags=0b0010
- full server response(type=9)：flags 含 sequence 时跟 4 字节序号，再 4 字节长度 + gzip JSON；
  flags=0b0011 为最终结果包
- error(type=15)：4 字节错误码 + 4 字节长度 + 消息
- 建议 200ms/包（内部 32ms 帧在此聚合成 200ms 包）

注意：需要新版控制台 X-Api-Key 且已开通豆包流式语音识别资源（否则握手 403）。
等包超时（45000081）：长连接空闲会断，每 turn 重连或后续做保活（M2/M6）。
"""

from __future__ import annotations

import asyncio
import gzip
import json
import os
import struct
import time
import uuid
from typing import AsyncIterator

import websockets

from voice.asr.base import TranscriptEvent

DEFAULT_URL = "wss://openspeech.bytedance.com/api/v3/sauc/bigmodel_async"
DEFAULT_RESOURCE_ID = "volc.seedasr.sauc.duration"

# message type
FULL_CLIENT_REQUEST = 0b0001
AUDIO_ONLY_REQUEST = 0b0010
FULL_SERVER_RESPONSE = 0b1001
SERVER_ERROR = 0b1111
# flags
FLAG_NONE = 0b0000
FLAG_POS_SEQ = 0b0001
FLAG_LAST_NO_SEQ = 0b0010
FLAG_LAST_WITH_SEQ = 0b0011


def _header(msg_type: int, flags: int, serialization: int, compression: int) -> bytes:
    return bytes([0x11, (msg_type << 4) | flags,
                  (serialization << 4) | compression, 0x00])


def _full_request(payload: dict) -> bytes:
    data = gzip.compress(json.dumps(payload).encode("utf-8"))
    return _header(FULL_CLIENT_REQUEST, FLAG_NONE, 1, 1) + struct.pack(">I", len(data)) + data


def _audio_request(pcm: bytes, last: bool) -> bytes:
    data = gzip.compress(pcm)
    flags = FLAG_LAST_NO_SEQ if last else FLAG_NONE
    return _header(AUDIO_ONLY_REQUEST, flags, 0, 1) + struct.pack(">I", len(data)) + data


def _parse_response(msg: bytes) -> tuple[str, int, object]:
    b0, b1, b2 = msg[0], msg[1], msg[2]
    header_size = (b0 & 0x0F) * 4
    msg_type = b1 >> 4
    flags = b1 & 0x0F
    compression = b2 & 0x0F
    off = header_size
    if msg_type == FULL_SERVER_RESPONSE:
        if flags in (FLAG_POS_SEQ, FLAG_LAST_WITH_SEQ):
            off += 4  # sequence，暂不消费
        (size,) = struct.unpack(">I", msg[off:off + 4])
        off += 4
        payload = msg[off:off + size]
        if compression == 1:
            payload = gzip.decompress(payload)
        return ("result", flags, json.loads(payload))
    if msg_type == SERVER_ERROR:
        (code,) = struct.unpack(">I", msg[off:off + 4])
        off += 4
        (size,) = struct.unpack(">I", msg[off:off + 4])
        off += 4
        return ("error", code, msg[off:off + size].decode("utf-8", "replace"))
    return ("unknown", msg_type, None)


class VolcengineASR:
    def __init__(self, api_key: str | None = None,
                 resource_id: str = DEFAULT_RESOURCE_ID,
                 base_url: str = DEFAULT_URL, packet_ms: int = 200,
                 end_window_size: int = 800, force_to_speech_time: int = 1000,
                 enable_nonstream: bool = True):
        self.api_key = api_key or os.environ["VOLCENGINE_API_KEY"]
        self.resource_id = resource_id
        self.base_url = base_url
        self.packet_bytes = 16000 * 2 * packet_ms // 1000
        # 实测（2026-10-03）：bigmodel_async + 二遍识别 + end_window_size=800 时，
        # 我方 VAD 判停后负包 → final 仅 47~78ms；不加这些参数需等 ~5s 语义分句
        self.end_window_size = end_window_size
        self.force_to_speech_time = force_to_speech_time
        self.enable_nonstream = enable_nonstream

    async def stream(self, audio: AsyncIterator[bytes]) -> AsyncIterator[TranscriptEvent]:
        headers = {
            "X-Api-Key": self.api_key,
            "X-Api-Resource-Id": self.resource_id,
            "X-Api-Request-Id": str(uuid.uuid4()),
            "X-Api-Sequence": "-1",
            "X-Api-Connect-Id": str(uuid.uuid4()),
        }
        ws = await websockets.connect(self.base_url, additional_headers=headers,
                                      ping_interval=10, ping_timeout=20)
        req = {
            "user": {"uid": "voice-agent"},
            "audio": {"format": "pcm", "rate": 16000, "bits": 16, "channel": 1},
            "request": {"model_name": "bigmodel", "enable_punc": True,
                        "enable_itn": True, "show_utterances": True,
                        "enable_nonstream": self.enable_nonstream,
                        "end_window_size": self.end_window_size,
                        "force_to_speech_time": self.force_to_speech_time},
        }
        await ws.send(_full_request(req))

        async def sender():
            try:
                buf = b""
                async for frame in audio:
                    buf += frame
                    while len(buf) >= self.packet_bytes:
                        await ws.send(_audio_request(buf[:self.packet_bytes], last=False))
                        buf = buf[self.packet_bytes:]
                await ws.send(_audio_request(buf, last=True))
            except (websockets.ConnectionClosed, asyncio.CancelledError):
                pass

        send_task = asyncio.create_task(sender())
        last_partial = ""
        last_final = ""
        try:
            async for raw in ws:
                kind, code, data = _parse_response(raw)
                if kind == "error":
                    raise RuntimeError(f"volcengine asr error {code}: {data}")
                if kind != "result":
                    continue
                t_ms = int(time.monotonic() * 1000)
                text = ((data.get("result") or {}).get("text") or "").strip()
                definite = any(u.get("definite") for u in
                               (data.get("result") or {}).get("utterances") or [])
                if code == FLAG_LAST_WITH_SEQ:
                    # 流收尾：有未落定文本则补一个 final
                    if text and text != last_final:
                        yield TranscriptEvent("final", text, t_ms)
                    break
                if text and definite and text != last_final:
                    last_final = text
                    yield TranscriptEvent("final", text, t_ms)
                elif text and text != last_partial:
                    yield TranscriptEvent("partial", text, t_ms)
                if text:
                    last_partial = text
        finally:
            send_task.cancel()
            try:
                await send_task
            except asyncio.CancelledError:
                pass
            await ws.close()
