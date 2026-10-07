"""官方 API 对照：CosyVoice2.inference_instruct2 原生路径（fp32/torch estimator/非流式）。

两个 prompt：官方 asset/zero_shot_prompt.wav vs 我们的 models/voices/narrator_m.wav
输出 wav + sherpa ASR -> reports/diag_official.txt
"""

import io
import sys
import time
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, r"vendor/CosyVoice")
sys.path.insert(0, r"vendor/CosyVoice\third_party\Matcha-TTS")

TEXT = "雪是从傍晚开始下的，整条进山的路只剩下你身后这一串脚印。"
INSTRUCT = "用低沉平稳、富有叙事感的语气讲述<|endofprompt|>"


def main():
    import numpy as np
    import torch
    import audioop
    from cosyvoice.cli.cosyvoice import CosyVoice2

    cv = CosyVoice2(str(ROOT / "models" / "CosyVoice2-0.5B"),
                    load_jit=False, load_trt=False, fp16=False)

    cases = [
        ("official_prompt", r"vendor/CosyVoice\asset\zero_shot_prompt.wav"),
        ("our_prompt", str(ROOT / "models" / "voices" / "narrator_m.wav")),
    ]
    for name, wav_path in cases:
        chunks = []
        t0 = time.monotonic()
        for out in cv.inference_instruct2(TEXT, INSTRUCT, wav_path, stream=False):
            speech = np.clip(out["tts_speech"].detach().cpu().numpy().flatten(), -1.0, 1.0)
            chunks.append((speech * 32767.0).astype("int16").tobytes())
        total = time.monotonic() - t0
        pcm = audioop.ratecv(b"".join(chunks), 2, 1, 24000, 16000, None)[0]
        with wave.open(str(ROOT / "reports" / f"diag_official_{name}.wav"), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(pcm)
        print(f"{name}: total={total:.2f}s audio={len(pcm) / 32000:.2f}s", flush=True)

    import sherpa_onnx
    import soundfile as sf
    d = r"../realtime-voice-agent\models\sherpa-asr-bilingual-zh-en"
    rec = sherpa_onnx.OnlineRecognizer.from_transducer(
        tokens=f"{d}/tokens.txt",
        encoder=f"{d}/encoder-epoch-99-avg-1.int8.onnx",
        decoder=f"{d}/decoder-epoch-99-avg-1.int8.onnx",
        joiner=f"{d}/joiner-epoch-99-avg-1.int8.onnx",
        num_threads=2, decoding_method="greedy_search")
    out = io.open(str(ROOT / "reports" / "diag_official.txt"), "w", encoding="utf-8")
    out.write(f"expect => {TEXT}\n")
    for name, _ in cases:
        samples, sr = sf.read(str(ROOT / "reports" / f"diag_official_{name}.wav"), dtype="float32")
        s = rec.create_stream()
        s.accept_waveform(sr, samples)
        s.accept_waveform(sr, [0.0] * int(sr * 0.5))
        s.input_finished()
        while rec.is_ready(s):
            rec.decode_stream(s)
        out.write(f"{name} => {rec.get_result(s)}\n")
    out.close()


if __name__ == "__main__":
    main()
