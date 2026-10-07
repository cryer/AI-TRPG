"""CosyVoice2 provider 冒烟+延迟测试（ad-hoc，直接用 asyncio 跑 synth）。

用法：python scripts/bench_cosyvoice.py
"""

import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from voice.tts.cosyvoice import CosyVoiceTTS


def sprint(text: str):
    sys.stdout.buffer.write((text + "\n").encode("gbk", errors="replace"))

SENTENCES = [
    "雪是从傍晚开始下的，到这个时候，整条进山的路已经只剩下你身后这一串脚印。",
    "⟦v:npc_male_old e:紧张⟧先生……您可算来了。",
    "⟦v:npc_female e:神秘⟧有些事情，我得在警察到之前告诉您。",
]


async def main():
    tts = CosyVoiceTTS(
        repo_path=r"vendor/CosyVoice",
        model_dir=r".\models\CosyVoice2-0.5B",
        voices={
            "narrator_m": {"prompt_wav": r".\models\voices\narrator_m.wav",
                           "instruct": "用低沉平稳、富有叙事感的语气讲述"},
            "npc_male_old": {"prompt_wav": r".\models\voices\npc_male_old.wav",
                             "instruct": "用恭敬而闪躲的语气说话"},
            "npc_female": {"prompt_wav": r".\models\voices\npc_female.wav",
                           "instruct": "用冷静优雅的语气说话"},
        },
        default_voice="narrator_m",
    )

    t0 = time.monotonic()
    await tts.warmup("预热。")
    print(f"[load+warmup] {time.monotonic() - t0:.1f}s")

    out = Path(__file__).resolve().parent.parent / "reports" / "cosy_bench.wav"
    all_pcm = bytearray()
    for text in SENTENCES:
        async def one(t=text):
            yield t
        t0 = time.monotonic()
        first = None
        n = 0
        async for chunk in tts.synth(one(), 0):
            if first is None:
                first = time.monotonic() - t0
            all_pcm.extend(chunk.pcm)
            n += len(chunk.pcm)
        total = time.monotonic() - t0
        dur = n / 16000 / 2
        sprint(f"[synth] first_chunk={first * 1000:.0f}ms total={total:.2f}s "
               f"audio={dur:.2f}s rtf={total / dur:.2f}  {text[:24]}")

    import wave
    with wave.open(str(out), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(bytes(all_pcm))
    print(f"written {out}")


if __name__ == "__main__":
    asyncio.run(main())
