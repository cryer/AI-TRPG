"""TTS 工厂：按 config 选择实现。"""

from __future__ import annotations

from voice.tts.base import TTS


def _kwargs(cfg: dict, section: str) -> dict:
    """取配置段，剥掉 "_" 开头的注释键（如 _readme）。"""
    return {k: v for k, v in cfg.get(section, {}).items()
            if not k.startswith("_")}


def create_tts(cfg: dict) -> TTS:
    name = (cfg.get("providers") or {}).get("tts")
    if name in (None, "mock"):
        from voice.tts.mock import MockTTS
        return MockTTS(**_kwargs(cfg.get("mock", {}), "tts"))
    if name == "volcengine":
        from voice.tts.volcengine import VolcengineTTS
        return VolcengineTTS(**_kwargs(cfg, "volcengine"))
    if name == "sherpa":
        from voice.tts.sherpa import SherpaOnnxTTS
        return SherpaOnnxTTS(**_kwargs(cfg, "sherpa_tts"))
    if name == "cosyvoice":
        from voice.tts.cosyvoice import CosyVoiceTTS
        return CosyVoiceTTS(**_kwargs(cfg, "cosyvoice"))
    raise ValueError(f"unknown TTS provider: {name!r}")
