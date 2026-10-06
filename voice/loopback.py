"""M0 音频环回 + 打点：mic/wav → VAD → 语音段录音落盘；事件打印 + JSONL 落盘。

用法：
    python -m voice.loopback --source mic                 # 真实麦克风（戴耳机！）
    python -m voice.loopback --source wav --file test.wav # 固定录音回归测试
    python -m voice.loopback --source wav --file test.wav --realtime
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import wave
from datetime import datetime
from pathlib import Path

from voice.metrics import MetricsSink
from voice.vad import SileroVAD, FRAME_BYTES

SAMPLE_RATE = 16000


class SegmentRecorder:
    """speech_start → speech_end 之间的帧写成 wav，落盘到 reports/recordings/。"""

    def __init__(self, out_dir: Path):
        out_dir.mkdir(parents=True, exist_ok=True)
        self.out_dir = out_dir
        self._wf: wave.Wave_write | None = None
        self._idx = 0
        self.path: Path | None = None

    def start(self) -> None:
        self._idx += 1
        self.path = self.out_dir / f"segment_{self._idx:03d}.wav"
        self._wf = wave.open(str(self.path), "wb")
        self._wf.setnchannels(1)
        self._wf.setsampwidth(2)
        self._wf.setframerate(SAMPLE_RATE)

    def write(self, frame: bytes) -> None:
        if self._wf is not None:
            self._wf.writeframes(frame)

    def stop(self) -> Path | None:
        if self._wf is not None:
            self._wf.close()
            self._wf = None
        return self.path

    @property
    def recording(self) -> bool:
        return self._wf is not None


async def run(args) -> None:
    with open(args.config, encoding="utf-8") as fh:
        cfg = json.load(fh)
    vcfg = cfg["vad"]
    vad = SileroVAD(vcfg["model_path"], threshold=vcfg["threshold"],
                    neg_threshold=vcfg["neg_threshold"],
                    endpoint_ms=cfg["endpoint_ms"],
                    min_speech_ms=vcfg["min_speech_ms"])

    if args.source == "mic":
        from voice.transport.local_sounddevice import mic_frames
        frames = mic_frames()
        src_desc = "mic"
    else:
        from voice.transport.wav_file import wav_frames
        frames = wav_frames(args.file, realtime=args.realtime)
        src_desc = f"wav:{args.file}"

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    jsonl_path = Path(args.out) / f"m0_{stamp}.jsonl"
    recorder = SegmentRecorder(Path(args.out) / "recordings")
    raw_wf = None
    if args.record_raw:
        raw_path = Path(args.out) / "recordings" / f"raw_{stamp}.wav"
        raw_wf = wave.open(str(raw_path), "wb")
        raw_wf.setnchannels(1)
        raw_wf.setsampwidth(2)
        raw_wf.setframerate(SAMPLE_RATE)

    print(f"[M0] source={src_desc} endpoint_ms={cfg['endpoint_ms']} jsonl={jsonl_path}")
    print("[M0] 等待语音…（Ctrl+C 停止）")

    loop = asyncio.get_running_loop()
    n_frames = 0
    try:
        with MetricsSink(jsonl_path) as sink:
            async for frame in frames:
                n_frames += 1
                if raw_wf is not None:
                    raw_wf.writeframes(frame)
                ev = await loop.run_in_executor(None, vad.process, frame)
                if recorder.recording:
                    recorder.write(frame)
                if ev is None:
                    continue
                sink.vad_event(ev.kind, ev.t_ms, ev.prob)
                if ev.kind == "speech_start":
                    recorder.start()
                    print(f"[{ev.t_ms:>8}ms] speech_start (prob={ev.prob:.2f}) -> 录音 {recorder.path}")
                else:
                    path = recorder.stop()
                    dur_ms = 0
                    if path:
                        import contextlib
                        with contextlib.suppress(Exception):
                            with wave.open(str(path)) as wf:
                                dur_ms = wf.getnframes() * 1000 // wf.getframerate()
                    print(f"[{ev.t_ms:>8}ms] speech_end   (prob={ev.prob:.2f}) -> 段落时长 {dur_ms}ms, 落盘 {path}")
            ev = vad.flush()
            if ev is not None:
                sink.vad_event(ev.kind, ev.t_ms, ev.prob)
                recorder.stop()
                print(f"[{ev.t_ms:>8}ms] speech_end   (flush)")
    except KeyboardInterrupt:
        pass
    finally:
        recorder.stop()
        if raw_wf is not None:
            raw_wf.close()
    sec = n_frames * 32 / 1000.0
    print(f"[M0] 结束：{n_frames} 帧（{sec:.1f}s 音频），事件已写入 {jsonl_path}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m voice.loopback")
    ap.add_argument("--source", choices=["mic", "wav"], default="wav")
    ap.add_argument("--file", help="source=wav 时的 16kHz PCM16 mono wav 文件")
    ap.add_argument("--realtime", action="store_true", help="wav 按真实速率喂入")
    ap.add_argument("--config", default="configs/default.json")
    ap.add_argument("--out", default="reports")
    ap.add_argument("--record-raw", action="store_true", help="同时录全量原始音频")
    args = ap.parse_args(argv)
    if args.source == "wav" and not args.file:
        ap.error("--source wav 需要 --file")
    try:
        asyncio.run(run(args))
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
