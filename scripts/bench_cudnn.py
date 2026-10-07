"""cudnn.benchmark 开关 × 变长句对比（ad-hoc）。

python scripts/bench_cudnn.py 0|1
合成 6 句长度各异的句子，输出每句耗时与总 RTF。
"""

import asyncio
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from voice.tts.cosyvoice import CosyVoiceTTS


def sprint(text: str):
    sys.stdout.buffer.write((text + "\n").encode("gbk", errors="replace"))


SENTENCES = [
    "门开了。",
    "你闻到一股陈旧的墨水味。",
    "书房的门在你身后无声地合上，烛火忽然矮了半截。",
    "他抬起头。",
    "你闻到一股陈旧的墨水和潮湿的木头混在一起的味道，还有一种说不清的、像是铁锈的气息。",
    "他的声音压得极低，眼睛却不停地瞟向那扇锁着的侧门，手指在桌下攥成了拳头。",
]


async def main(bench: bool):
    tts = CosyVoiceTTS(
        repo_path=r"vendor/CosyVoice",
        model_dir=str(ROOT / "models" / "CosyVoice2-0.5B"),
        voices={"narrator_m": {"prompt_wav": str(ROOT / "models" / "voices" / "narrator_m.wav"),
                               "instruct": "用低沉平稳、富有叙事感的语气讲述"}},
        default_voice="narrator_m",
        nfe=5,
        cudnn_benchmark=bench,
    )
    await tts.warmup("预热。")

    total_t, total_a = 0.0, 0.0
    for text in SENTENCES:
        async def one(t=text):
            yield t
        n = 0
        t0 = time.monotonic()
        async for chunk in tts.synth(one(), 0):
            n += len(chunk.pcm)
        dt = time.monotonic() - t0
        dur = n / 16000 / 2
        total_t += dt
        total_a += dur
        sprint(f"  {dt:5.2f}s audio={dur:5.2f}s rtf={dt / dur:4.2f}  {text[:16]}")

    sprint(f"cudnn_benchmark={bench}: total={total_t:.2f}s audio={total_a:.2f}s "
           f"rtf={total_t / total_a:.2f}")


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1] == "1"))
