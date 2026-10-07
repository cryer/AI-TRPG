"""M2 验收用：生成 DM 对局测试 wav（SAPI 中文语音 + ffmpeg 转 16kHz）。
依赖 Windows SAPI 与 ffmpeg；输出默认写到 reports/（已 gitignore）。
"""
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

import numpy as np

SR = 16000

# (文本, 说完后跟随的静音秒数)
SEGMENTS = [
    ("我就玩庄园魅影吧。", 70.0),                    # 等开场白播完
    ("我先去书房看看现场。", 25.0),                  # 期望 advance_scene(study)
    ("我仔细检查那座挂钟，看看钟摆里面有什么。", 25.0),  # 期望挂钟蜂蜡线索
    ("我要验看尸体，把老爷的身体翻动检查一下。", 25.0),  # 期望尸斑线索/判定
    ("我试试用细铁丝拨开书房侧窗的插销。", 25.0),    # 期望 roll_dice 巧手判定
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
    out = sys.argv[1] if len(sys.argv) > 1 else "reports/test_dm.wav"
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
