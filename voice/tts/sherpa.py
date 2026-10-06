"""sherpa-onnx 本地中文 TTS（VITS vits-zh-hf-fanchen-C，22050Hz 女声）。

- OfflineTts 进程级常驻（懒加载）；逐句合成，推理放 executor，event loop 零阻塞。
- 合成输出 22050Hz float32 → audioop.ratecv 重采样到内部 16kHz PCM16 mono，
  每句切成 ~100ms chunk 顺流产出，首 chunk 即首帧（barge-in flush 粒度也够）。
- VITS 是整句离线合成，无真正的增量流式；句级首音频延迟≈单句合成时长
  （CPU ~0.3-0.6s，GPU 更低）。要真流式可后续换 CosyVoice2（4090 服务器）。
"""

from __future__ import annotations

import asyncio
import audioop
import os
from typing import AsyncIterator

import numpy as np

from voice.tts.base import AudioChunk

CHUNK_MS = 100


class SherpaOnnxTTS:
    def __init__(self, model_dir: str = "models/vits-zh-hf-fanchen-C",
                 model: str = "vits-zh-hf-fanchen-C.onnx",
                 lexicon: str = "lexicon.txt", tokens: str = "tokens.txt",
                 dict_dir: str = "dict",
                 rule_fsts: str = "date.fst,new_heteronym.fst,number.fst,phone.fst",
                 provider: str = "cpu", num_threads: int = 2,
                 speaker_id: int = 0, speed: float = 1.1,
                 sample_rate: int = 16000):
        self.model_dir = model_dir
        self.model = model
        self.lexicon = lexicon
        self.tokens = tokens
        self.dict_dir = dict_dir
        self.rule_fsts = rule_fsts
        self.provider = provider
        self.num_threads = num_threads
        self.speaker_id = speaker_id
        self.speed = speed
        self.sample_rate = sample_rate  # 内部目标采样率
        self._tts = None
        self._native_sr: int | None = None

    def _get_tts(self):
        if self._tts is None:
            import sherpa_onnx
            d = self.model_dir
            for rel in [self.model, self.lexicon, self.tokens]:
                if not os.path.isfile(os.path.join(d, rel)):
                    raise FileNotFoundError(f"sherpa TTS model file missing: {os.path.join(d, rel)}")
            vits = sherpa_onnx.OfflineTtsVitsModelConfig(
                model=os.path.join(d, self.model),
                lexicon=os.path.join(d, self.lexicon),
                tokens=os.path.join(d, self.tokens),
                dict_dir=os.path.join(d, self.dict_dir),
            )
            model_cfg = sherpa_onnx.OfflineTtsModelConfig(
                vits=vits, provider=self.provider, num_threads=self.num_threads)
            rule_fsts = ",".join(os.path.join(d, f) for f in self.rule_fsts.split(",") if f)
            cfg = sherpa_onnx.OfflineTtsConfig(
                model=model_cfg, rule_fsts=rule_fsts, max_num_sentences=2)
            if not cfg.validate():
                raise RuntimeError("sherpa TTS config invalid, check model files")
            self._tts = sherpa_onnx.OfflineTts(cfg)
            self._native_sr = self._tts.sample_rate
        return self._tts

    def _synth_sync(self, text: str) -> bytes:
        tts = self._get_tts()
        audio = tts.generate(text, sid=self.speaker_id, speed=self.speed)
        samples = np.asarray(audio.samples, dtype=np.float32)
        # 整句都是 OOV 字符（如纯标点「」）时 sherpa 转换失败，返回空音频且
        # sample_rate=0——直接当空句处理，不能进 ratecv（会抛 sampling rate）
        if audio.sample_rate <= 0 or samples.size == 0:
            return b""
        samples = np.clip(samples, -1.0, 1.0)
        pcm_native = (samples * 32767.0).astype(np.int16).tobytes()
        if audio.sample_rate != self.sample_rate:
            pcm_native, _ = audioop.ratecv(pcm_native, 2, 1, audio.sample_rate,
                                           self.sample_rate, None)
        return pcm_native

    async def synth(self, sentences, gen_id: int) -> AsyncIterator[AudioChunk]:
        loop = asyncio.get_running_loop()
        seq = 0
        chunk_bytes = self.sample_rate * 2 * CHUNK_MS // 1000
        async for sentence in sentences:
            try:
                pcm = await loop.run_in_executor(None, self._synth_sync, sentence)
            except Exception as e:
                # 单句合成失败不能炸掉整个回合（DM 长叙述里一句坏句就静默
                # 中断是体验灾难）——记日志跳过，后续句子照常合成
                print(f"[warn] TTS 合成失败，跳过该句 {sentence[:20]!r}: {e!r}")
                continue
            if not pcm:
                continue
            for off in range(0, len(pcm), chunk_bytes):
                yield AudioChunk(pcm=pcm[off:off + chunk_bytes], gen_id=gen_id, seq=seq)
                seq += 1

    async def warmup(self, text: str) -> bytes | None:
        """开机预热：加载模型 + 合成一句，返回 PCM（调用方丢弃）。"""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._synth_sync, text)
