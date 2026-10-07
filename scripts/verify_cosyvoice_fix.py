"""验证 cosyvoice endofprompt 修复：逐句合成 -> sherpa ASR 转写核对内容。

用法：python scripts/verify_cosyvoice_fix.py
期望：转写结果与输入文本一致（不再念 instruct），并给出每句延迟。
"""

import asyncio
import sys
import time
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from voice.tts.cosyvoice import CosyVoiceTTS


def sprint(text: str):
    sys.stdout.buffer.write((text + "\n").encode("gbk", errors="replace"))


CASES = [
    ("narrator", "你好呀，欢迎来玩单人跑团！"),
    ("narrator", "雪是从傍晚开始下的，整条进山的路只剩下你身后这一串脚印。"),
    ("tagged", "⟦v:npc_male_old e:紧张⟧先生……您可算来了。"),
]


async def synth_all(tts):
    paths = []
    for name, text in CASES:
        async def one(t=text):
            yield t
        pcm = bytearray()
        t0 = time.monotonic()
        first = None
        async for chunk in tts.synth(one(), 0):
            if first is None:
                first = time.monotonic() - t0
            pcm.extend(chunk.pcm)
        total = time.monotonic() - t0
        dur = len(pcm) / 16000 / 2
        sprint(f"[synth:{name}] first_chunk={first * 1000:.0f}ms total={total:.2f}s "
               f"audio={dur:.2f}s rtf={(total / dur if dur else 0):.2f}")
        p = ROOT / "reports" / f"verify_{name}.wav"
        with wave.open(str(p), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(bytes(pcm))
        paths.append((name, p))
    return paths


def transcribe(paths):
    import sherpa_onnx
    import soundfile as sf

    d = r"../realtime-voice-agent\models\sherpa-asr-bilingual-zh-en"
    rec = sherpa_onnx.OnlineRecognizer.from_transducer(
        tokens=f"{d}/tokens.txt",
        encoder=f"{d}/encoder-epoch-99-avg-1.int8.onnx",
        decoder=f"{d}/decoder-epoch-99-avg-1.int8.onnx",
        joiner=f"{d}/joiner-epoch-99-avg-1.int8.onnx",
        num_threads=2, decoding_method="greedy_search",
    )
    for name, p in paths:
        samples, sr = sf.read(str(p), dtype="float32")
        s = rec.create_stream()
        s.accept_waveform(sr, samples)
        tail = [0.0] * int(sr * 0.5)
        s.accept_waveform(sr, tail)
        s.input_finished()
        while rec.is_ready(s):
            rec.decode_stream(s)
        sprint(f"[asr:{name}] {rec.get_result(s)}")


async def main():
    tts = CosyVoiceTTS(
        repo_path=r"vendor/CosyVoice",
        model_dir=str(ROOT / "models" / "CosyVoice2-0.5B"),
        voices={
            "narrator_m": {"prompt_wav": str(ROOT / "models" / "voices" / "narrator_m.wav"),
                           "instruct": "用低沉平稳、富有叙事感的语气讲述"},
            "npc_male_old": {"prompt_wav": str(ROOT / "models" / "voices" / "npc_male_old.wav"),
                             "instruct": "用恭敬而闪躲的语气说话"},
        },
        default_voice="narrator_m",
    )
    t0 = time.monotonic()
    await tts.warmup("预热。")
    sprint(f"[load+warmup] {time.monotonic() - t0:.1f}s")
    paths = await synth_all(tts)
    transcribe(paths)


if __name__ == "__main__":
    asyncio.run(main())
