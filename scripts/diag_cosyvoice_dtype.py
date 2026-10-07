"""LLM dtype (fp32/bf16/fp16) 对合成质量与延迟的影响诊断（ad-hoc）。

同一句子非流式合成 -> wav + sherpa ASR -> reports/diag_dtype.txt
"""

import asyncio
import io
import sys
import time
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from voice.tts.cosyvoice import CosyVoiceTTS

TEXT = "雪是从傍晚开始下的，整条进山的路只剩下你身后这一串脚印。"


async def main():
    import numpy as np
    import torch
    import audioop

    tts = CosyVoiceTTS(
        repo_path="vendor/CosyVoice",
        model_dir=str(ROOT / "models" / "CosyVoice2-0.5B"),
        voices={"narrator_m": {"prompt_wav": str(ROOT / "models" / "voices" / "narrator_m.wav"),
                               "instruct": "用低沉平稳、富有叙事感的语气讲述"}},
        default_voice="narrator_m",
    )
    await tts.warmup("预热。")
    model = tts._model

    results = {}
    for name, dtype in [("fp32", torch.float32), ("bf16", torch.bfloat16), ("fp16", torch.float16)]:
        model.model.llm.to(dtype)
        frontend = model.frontend
        tpl = tts._get_template("narrator_m", None)
        norm = frontend.text_normalize(TEXT, split=False)
        text_token, text_token_len = frontend._extract_text_token(norm)
        model_input = {**tpl, "text": text_token, "text_len": text_token_len}
        chunks = []
        t0 = time.monotonic()
        with tts._lock:
            for out in model.model.tts(**model_input, stream=False, speed=1.0):
                speech = np.clip(out["tts_speech"].detach().cpu().numpy().flatten(), -1.0, 1.0)
                chunks.append((speech * 32767.0).astype("int16").tobytes())
        total = (time.monotonic() - t0) * 1000
        pcm = audioop.ratecv(b"".join(chunks), 2, 1, 24000, 16000, None)[0]
        p = ROOT / "reports" / f"diag_dtype_{name}.wav"
        with wave.open(str(p), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(pcm)
        dur = len(pcm) / 16000 / 2
        results[name] = f"total={total / 1000:.2f}s audio={dur:.2f}s"

    model.model.llm.to(torch.float16)  # 收尾复原

    import sherpa_onnx
    import soundfile as sf
    d = "models/sherpa-asr-bilingual-zh-en"
    rec = sherpa_onnx.OnlineRecognizer.from_transducer(
        tokens=f"{d}/tokens.txt",
        encoder=f"{d}/encoder-epoch-99-avg-1.int8.onnx",
        decoder=f"{d}/decoder-epoch-99-avg-1.int8.onnx",
        joiner=f"{d}/joiner-epoch-99-avg-1.int8.onnx",
        num_threads=2, decoding_method="greedy_search")
    out = io.open(str(ROOT / "reports" / "diag_dtype.txt"), "w", encoding="utf-8")
    out.write(f"expect => {TEXT}\n")
    for name in ("fp32", "bf16", "fp16"):
        samples, sr = sf.read(str(ROOT / "reports" / f"diag_dtype_{name}.wav"), dtype="float32")
        s = rec.create_stream()
        s.accept_waveform(sr, samples)
        s.accept_waveform(sr, [0.0] * int(sr * 0.5))
        s.input_finished()
        while rec.is_ready(s):
            rec.decode_stream(s)
        out.write(f"{name} [{results[name]}] => {rec.get_result(s)}\n")
    out.close()


if __name__ == "__main__":
    asyncio.run(main())
