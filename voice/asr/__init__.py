"""ASR 工厂：按 config 选择实现。加新 provider = 实现 ASR Protocol + 在此注册。"""

from __future__ import annotations

from voice.asr.base import ASR


def create_asr(cfg: dict) -> ASR:
    name = (cfg.get("providers") or {}).get("asr")
    if name in (None, "mock"):
        from voice.asr.mock import MockASR
        return MockASR(**(cfg.get("mock", {}).get("asr", {})))
    if name == "deepgram":
        from voice.asr.deepgram import DeepgramASR
        return DeepgramASR(**cfg.get("deepgram", {}))
    if name == "volcengine":
        from voice.asr.volcengine import VolcengineASR
        return VolcengineASR(**cfg.get("volcengine_asr", {}))
    if name == "sherpa":
        from voice.asr.sherpa import SherpaOnnxASR
        return SherpaOnnxASR(**cfg.get("sherpa_asr", {}))
    raise ValueError(f"unknown ASR provider: {name!r}")
