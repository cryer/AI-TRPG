"""分组件耗时剖析（ad-hoc）：llm token 流 / flow.inference / hift.inference 分别计时。

python scripts/prof_cosyvoice.py [nfe]
"""

import asyncio
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from voice.tts.cosyvoice import CosyVoiceTTS

TEXT = "你闻到一股陈旧的墨水和潮湿的木头混在一起的味道，还有一种说不清的、像是铁锈的气息。"


def sprint(text: str):
    sys.stdout.buffer.write((text + "\n").encode("gbk", errors="replace"))


async def main(nfe: int):
    tts = CosyVoiceTTS(
        repo_path="vendor/CosyVoice",
        model_dir=str(ROOT / "models" / "CosyVoice2-0.5B"),
        voices={"narrator_m": {"prompt_wav": str(ROOT / "models" / "voices" / "narrator_m.wav"),
                               "instruct": "用低沉平稳、富有叙事感的语气讲述"}},
        default_voice="narrator_m",
        nfe=nfe,
    )
    await tts.warmup("预热。")
    model = tts._model.model

    stats = {"flow": [0.0, 0], "hift": [0.0, 0]}
    orig_flow = model.flow.inference
    orig_hift = model.hift.inference

    def timed_flow(**kw):
        t0 = time.monotonic()
        r = orig_flow(**kw)
        stats["flow"][0] += time.monotonic() - t0
        stats["flow"][1] += 1
        return r

    def timed_hift(**kw):
        t0 = time.monotonic()
        r = orig_hift(**kw)
        stats["hift"][0] += time.monotonic() - t0
        stats["hift"][1] += 1
        return r

    model.flow.inference = timed_flow
    model.hift.inference = timed_hift

    frontend = tts._model.frontend
    tpl = tts._get_template("narrator_m", None)
    norm = frontend.text_normalize(TEXT, split=False)
    text_token, text_token_len = frontend._extract_text_token(norm)
    model_input = {**tpl, "text": text_token, "text_len": text_token_len}

    # llm 在独立线程与 flow/hift 并行：total - flow - hift ≈ LLM 未藏住的部分
    t0 = time.monotonic()
    with tts._lock:
        model.token_hop_len = 25
        for out in model.tts(**model_input, stream=True, speed=1.0):
            _ = out["tts_speech"].shape
    total = time.monotonic() - t0
    sprint(f"tts total={total:.2f}s  flow={stats['flow'][0]:.2f}s/{stats['flow'][1]}次  "
           f"hift={stats['hift'][0]:.2f}s/{stats['hift'][1]}次  "
           f"other={total - stats['flow'][0] - stats['hift'][0]:.2f}s")


if __name__ == "__main__":
    asyncio.run(main(int(sys.argv[1]) if len(sys.argv) > 1 else 6))
