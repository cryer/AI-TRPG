"""流式/非流式 × ORT/torch estimator 三模式对照诊断（ad-hoc）。

合成同一句子 -> 保存 wav -> sherpa ASR 转写，输出 UTF-8 到 reports/diag_modes.txt
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


def collect_sync(tts, stream: bool):
    """直接走 provider 内部同步路径，返回 (pcm, first_ms, total_ms)。"""
    model = tts._model
    frontend = model.frontend
    tpl = tts._get_template("narrator_m", None)
    norm = frontend.text_normalize(TEXT, split=False)
    text_token, text_token_len = frontend._extract_text_token(norm)
    model_input = {**tpl, "text": text_token, "text_len": text_token_len}
    chunks = []
    if stream:
        model.model.token_hop_len = 25
    t0 = time.monotonic()
    first = None
    with tts._lock:
        for out in model.model.tts(**model_input, stream=stream, speed=1.0):
            if first is None:
                first = (time.monotonic() - t0) * 1000
            speech = out["tts_speech"].detach().cpu().numpy().flatten()
            import numpy as np
            speech = np.clip(speech, -1.0, 1.0)
            chunks.append((speech * 32767.0).astype("int16").tobytes())
    total = (time.monotonic() - t0) * 1000
    import audioop
    pcm = audioop.ratecv(b"".join(chunks), 2, 1, 24000, 16000, None)[0]
    return pcm, first, total


async def main():
    tts = CosyVoiceTTS(
        repo_path=r"vendor/CosyVoice",
        model_dir=str(ROOT / "models" / "CosyVoice2-0.5B"),
        voices={"narrator_m": {"prompt_wav": str(ROOT / "models" / "voices" / "narrator_m.wav"),
                               "instruct": "用低沉平稳、富有叙事感的语气讲述"}},
        default_voice="narrator_m",
    )
    await tts.warmup("预热。")

    cfm = tts._model.model.flow.decoder
    # 实例属性 ort_forward_estimator 遮蔽了类方法；del 即恢复 torch 类方法
    ort_estimator = cfm.__dict__["forward_estimator"]
    import types
    orig_estimator = types.MethodType(type(cfm).forward_estimator, cfm)

    results = {}
    modes = [
        ("nonstream_ort", False, True),
        ("stream_ort", True, True),
        ("stream_torch", True, False),
    ]
    for name, stream, use_ort in modes:
        cfm.forward_estimator = ort_estimator if use_ort else orig_estimator
        pcm, first, total = collect_sync(tts, stream)
        p = ROOT / "reports" / f"diag_{name}.wav"
        with wave.open(str(p), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(pcm)
        dur = len(pcm) / 16000 / 2
        results[name] = f"first={first:.0f}ms total={total / 1000:.2f}s audio={dur:.2f}s"
        with open(str(p) + ".done", "w") as f:
            f.write("ok")

    import sherpa_onnx
    import soundfile as sf
    d = r"../realtime-voice-agent\models\sherpa-asr-bilingual-zh-en"
    rec = sherpa_onnx.OnlineRecognizer.from_transducer(
        tokens=f"{d}/tokens.txt",
        encoder=f"{d}/encoder-epoch-99-avg-1.int8.onnx",
        decoder=f"{d}/decoder-epoch-99-avg-1.int8.onnx",
        joiner=f"{d}/joiner-epoch-99-avg-1.int8.onnx",
        num_threads=2, decoding_method="greedy_search")
    out = io.open(str(ROOT / "reports" / "diag_modes.txt"), "w", encoding="utf-8")
    out.write(f"expect => {TEXT}\n")
    for name, _, _ in modes:
        samples, sr = sf.read(str(ROOT / "reports" / f"diag_{name}.wav"), dtype="float32")
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
