"""M3 验收用：生成《庄园魅影》完整对局 wav（SAPI + ffmpeg）。

路线设计：书房（挂钟/壁炉）→ 花房（地面/地窖门）→ 冰窖（冰面/木箱）→ 阁楼（遗物箱）
→ 对质沈曼丽（怀表链+日记+遗嘱草稿 ≥2 铁证，期望 truth 结局）→ 回 lobby。
第 6 段间隔仅 8 秒，刻意在 DM 叙述中插话，验证打断后自然衔接。
依赖 Windows SAPI 与 ffmpeg；输出默认写到 reports/（已 gitignore）。
"""
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

import numpy as np

SR = 16000

SEGMENTS = [
    ("我就玩庄园魅影吧。", 70.0),
    ("我先去书房看看现场。", 25.0),
    ("我仔细检查那座挂钟，看看钟摆里面有什么。", 25.0),
    ("我翻检壁炉里的灰烬。", 25.0),
    ("我去花房看看，检查泥地地面和角落的矮木门。", 25.0),
    ("我推开矮木门，下地窖进冰窖。", 8.0),          # 刻意打断花房叙述
    ("我仔细检查储冰槽和冰面。", 25.0),
    ("我翻找那些旧木箱。", 25.0),
    ("我上楼去阁楼。", 25.0),
    ("我仔细翻检夫人的遗物箱，包括夹层。", 25.0),
    ("我要当面质问沈曼丽：老爷是十月十四号下午死的，你伪造了死亡时间，冰窖里的银怀表链就是证据。", 40.0),
    ("好的，今天就到这儿，带我回店里吧。", 45.0),   # 期望 back_to_lobby
]


def tts(text: str, tmpdir: str, idx: int) -> np.ndarray:
    raw = Path(tmpdir) / f"sapi_{idx}.wav"
    conv = Path(tmpdir) / f"sapi_{idx}_16k.wav"
    ps = (
        "Add-Type -AssemblyName System.Speech; "
        "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
        f"$s.SetOutputToWaveFile('{raw}'); "
        f"$s.Speak('{text}'); $s.Dispose()"
    )
    subprocess.run(["powershell", "-NoProfile", "-Command", ps], check=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(raw),
                    "-ar", str(SR), "-ac", "1", "-c:a", "pcm_s16le", str(conv)],
                   check=True)
    with wave.open(str(conv), "rb") as wf:
        return np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)


def silence(sec: float) -> np.ndarray:
    return np.zeros(int(SR * sec), dtype=np.int16)


def main() -> None:
    out = sys.argv[1] if len(sys.argv) > 1 else "reports/test_fullgame.wav"
    parts = [silence(0.6)]
    with tempfile.TemporaryDirectory() as td:
        for i, (text, gap) in enumerate(SEGMENTS):
            parts.append(tts(text, td, i))
            parts.append(silence(gap))
    audio = np.concatenate(parts)
    with wave.open(out, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes(audio.tobytes())
    print(f"written {out}: {len(audio) / SR:.1f}s")


if __name__ == "__main__":
    main()
