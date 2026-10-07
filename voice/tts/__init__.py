"""TTS 工厂：按 config 选择实现。"""

from __future__ import annotations

from voice.tts.base import TTS


def create_tts(cfg: dict) -> TTS:
    name = (cfg.get("providers") or {}).get("tts")
    if name in (None, "mock"):
        from voice.tts.mock import MockTTS
        return MockTTS(**(cfg.get("mock", {}).get("tts", {})))
    if name == "volcengine":
        from voice.tts.volcengine import VolcengineTTS
        return VolcengineTTS(**cfg.get("volcengine", {}))
    if name == "sherpa":
        from voice.tts.sherpa import SherpaOnnxTTS
        return SherpaOnnxTTS(**cfg.get("sherpa_tts", {}))
    if name == "cosyvoice":
        from voice.tts.cosyvoice import CosyVoiceTTS
        return CosyVoiceTTS(**cfg.get("cosyvoice", {}))
    raise ValueError(f"unknown TTS provider: {name!r}")
