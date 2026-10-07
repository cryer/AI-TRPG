"""播放卡顿仿真（ad-hoc）：按 chunk 到达时间模拟 web 客户端调度，统计欠载停顿。

用法：python scripts/bench_playback_stall.py [nfe]
对比不同 nfe 下：首音、总合成时间、以及客户端 offset=0.06/0.3 两种
重缓冲策略各自的停顿次数与停顿时长。
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


# 模拟一段 DM 长叙述（多句，句间模拟 LLM 流式出句间隔）
SENTENCES = [
    "书房的门在你身后无声地合上，烛火忽然矮了半截。",
    "你闻到一股陈旧的墨水和潮湿的木头混在一起的味道，还有一种说不清的、像是铁锈的气息。",
    "⟦v:npc_male_old e:低语⟧先生，有些事情……我得在警察到之前告诉您。",
    "他的声音压得极低，眼睛却不停地瞟向那扇锁着的侧门。",
]


def simulate(arrivals, offset):
    """arrivals: [(t, dur)]。返回 (停顿次数, 总停顿秒)。"""
    play_time = 0.0
    stalls, stalled = 0, 0.0
    for t, d in arrivals:
        if play_time < t:
            if play_time > 0:
                stalls += 1
                stalled += t - play_time
            play_time = t + offset
        play_time += d
    return stalls, stalled


async def main(nfe: int):
    tts = CosyVoiceTTS(
        repo_path=r"vendor/CosyVoice",
        model_dir=str(ROOT / "models" / "CosyVoice2-0.5B"),
        voices={
            "narrator_m": {"prompt_wav": str(ROOT / "models" / "voices" / "narrator_m.wav"),
                           "instruct": "用低沉平稳、富有叙事感的语气讲述"},
            "npc_male_old": {"prompt_wav": str(ROOT / "models" / "voices" / "npc_male_old.wav"),
                             "instruct": "用苍老缓慢的语气说话"},
        },
        default_voice="narrator_m",
        nfe=nfe,
    )
    await tts.warmup("预热。")

    async def sentence_stream():
        for s in SENTENCES:
            yield s
            await asyncio.sleep(0.25)      # 模拟 LLM 分句间隔

    arrivals = []
    t0 = time.monotonic()
    first = None
    total_pcm = 0
    async for chunk in tts.synth(sentence_stream(), 0):
        now = time.monotonic() - t0
        if first is None:
            first = now
        dur = len(chunk.pcm) / 16000 / 2
        arrivals.append((now, dur))
        total_pcm += len(chunk.pcm)
    total = time.monotonic() - t0
    audio = total_pcm / 16000 / 2

    s_old = simulate(arrivals, 0.06)
    s_new = simulate(arrivals, 0.3)
    sprint(f"nfe={nfe}: first_chunk={first * 1000:.0f}ms total={total:.2f}s "
           f"audio={audio:.2f}s rtf={total / audio:.2f}")
    sprint(f"  offset=0.06: stalls={s_old[0]} stalled={s_old[1]:.2f}s")
    sprint(f"  offset=0.30: stalls={s_new[0]} stalled={s_new[1]:.2f}s")


if __name__ == "__main__":
    asyncio.run(main(int(sys.argv[1]) if len(sys.argv) > 1 else 6))
