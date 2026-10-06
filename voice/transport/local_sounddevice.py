"""本地麦克风输入：sounddevice 回调线程 → asyncio.Queue 桥接，event loop 零阻塞。

产出 16kHz PCM16 mono 帧（512 样本 = 1024 字节 = 32ms）。队列满时丢最旧帧
（实时音频宁可丢帧也不积压延迟）。
"""

from __future__ import annotations

import asyncio
from typing import AsyncIterator

import sounddevice as sd

FRAME_SAMPLES = 512
SAMPLE_RATE = 16000


async def mic_frames(device=None, queue_max: int = 100) -> AsyncIterator[bytes]:
    loop = asyncio.get_running_loop()
    q: asyncio.Queue[bytes] = asyncio.Queue(maxsize=queue_max)

    def _enqueue(data: bytes) -> None:
        if q.full():
            try:
                q.get_nowait()
            except asyncio.QueueEmpty:
                pass
        q.put_nowait(data)

    def callback(indata, frames, time_info, status):
        loop.call_soon_threadsafe(_enqueue, indata.tobytes())

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16",
                        blocksize=FRAME_SAMPLES, device=device, callback=callback):
        while True:
            yield await q.get()
