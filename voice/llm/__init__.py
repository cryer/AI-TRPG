"""LLM 工厂：按 config 选择实现。"""

from __future__ import annotations

from voice.llm.base import LLM


def create_llm(cfg: dict) -> LLM:
    name = (cfg.get("providers") or {}).get("llm")
    if name in (None, "mock"):
        from voice.llm.mock import MockLLM
        return MockLLM(**(cfg.get("mock", {}).get("llm", {})))
    if name == "kimi":
        from voice.llm.kimi import KimiLLM
        return KimiLLM(**cfg.get("kimi", {}))
    raise ValueError(f"unknown LLM provider: {name!r}")
