"""CosyVoice2 本地 GPU TTS（阿里 FunAudioLLM，24kHz 原生，重采样到内部 16kHz）。

与 sherpa provider 同契约（voice/tts/base.py），可经 configs 直接互换。

- 官方仓库代码经 sys.path 注入加载（`repo_path` 配置），模型懒加载。
- 逐句合成，LLM 流式出 token + flow/hift 逐 hop 出音频（stream=True）：
  首音延迟 ≈ 首个 hop（25 token），而不是整句合成完。worker 线程 →
  asyncio 队列桥接，event loop 零阻塞；单 GPU 串行由 threading.Lock 保证。
- flow matching 解码器走 onnxruntime CUDA（`use_ort_estimator`，默认开）：
  模型自带的 flow.decoder.estimator.fp32.onnx 在 WDDM/Windows 下比 torch eager
  快约 10 倍（实测 67M UNet 单次 500ms→49ms），是 6GB 级显卡实时的关键。
  CFM 的 cache/因果掩码在 estimator 之外处理（solve_euler 固定 z/mu 前缀，
  流式 hop 内用全量注意力），因此该 ONNX 同样可用于 stream=True 的逐 hop
  合成——首音只需等第一个 25 token 的 hop，而非整句。
- fp16 默认开：ORT estimator 不经过 torch autocast，fp16 只作用于 LLM/hift；
  开启时 LLM 权重同步转半精度（实测 55ms→19ms/token，显存占用也减半）。
  早期「autocast 拖慢 estimator」的结论只适用于 torch eager 解码路径。
- 多音色零样本克隆：`voices` 配置声线注册表 {id: {prompt_wav, instruct}}，
  句子可带内联标记 `⟦v:voice_id e:情绪⟧` 切换声线/语气（base.parse_tagged
  解析）；标记非法或未注册 id 一律回退 default_voice，不会念出标记。
- (voice_id, instruct) 维度缓存前端特征（speech token/embedding/feat 提取
  只需一次），同声线换句零额外开销；情绪词动态拼 instruct 也走缓存。
- barge-in 取消：外层 cancel 后停止入队；当前句合成在后台收尾（GPU 调用
  不可中断），下一句合成经锁等待，不会并发抢卡。
"""

from __future__ import annotations

import asyncio
import audioop
import os
import sys
import threading
from typing import AsyncIterator

import numpy as np

from voice.tts.base import AudioChunk, parse_tagged

CHUNK_MS = 100
NATIVE_SR = 24000  # CosyVoice2 输出采样率


class CosyVoiceTTS:
    def __init__(self, repo_path: str = r"vendor/CosyVoice",
                 model_dir: str = "models/CosyVoice2-0.5B",
                 voices: dict | None = None,
                 default_voice: str = "narrator",
                 speed: float = 1.0, fp16: bool = True,
                 use_ort_estimator: bool = True, nfe: int = 10,
                 load_jit: bool = False,
                 cudnn_benchmark: bool = False,
                 sample_rate: int = 16000):
        self.repo_path = repo_path
        self.model_dir = model_dir
        # 声线注册表：{id: {"prompt_wav": ..., "instruct": ...}}
        self.voices = voices or {}
        self.default_voice = default_voice
        self.speed = speed
        self.fp16 = fp16
        self.use_ort_estimator = use_ort_estimator
        # flow matching 欧拉步数（官方默认 10）。解码器是唯一瓶颈，
        # 步数线性省时间；RTF>1 的中低端卡可降到 5~6 换流畅度
        self.nfe = nfe
        # flow encoder 是否用官方 JIT 版（flow.encoder.fp16.zip）。
        # 实测 RTX 3050/WDDM 上 JIT 版反而慢一倍多（RTF 1.1→2.1），默认关
        self.load_jit = load_jit
        # benchmark 模式会按新形状重新调优：真实对局每句长度都不同，
        # 实测 RTX 3050 上关闭更快（RTF 1.05→0.95），长句尤为明显
        self.cudnn_benchmark = cudnn_benchmark
        self.sample_rate = sample_rate
        self._model = None
        self._lock = threading.Lock()       # GPU 串行
        self._frontend_cache: dict = {}      # (voice_id, instruct) -> model_input 模板
        self._load_err: Exception | None = None

    # ---------- 模型加载 ----------

    def _load_sync(self):
        if self._model is not None or self._load_err is not None:
            return
        try:
            import torch
            torch.backends.cudnn.benchmark = self.cudnn_benchmark
            for p in (self.repo_path,
                      os.path.join(self.repo_path, "third_party", "Matcha-TTS")):
                if p not in sys.path:
                    sys.path.insert(0, p)
            if not os.path.isdir(self.model_dir):
                raise FileNotFoundError(
                    f"CosyVoice2 模型目录不存在: {self.model_dir}（用 modelscope 下载："
                    f"modelscope download --model iic/CosyVoice2-0.5B --local_dir {self.model_dir}）")
            from cosyvoice.cli.cosyvoice import CosyVoice2
            self._model = CosyVoice2(self.model_dir, load_jit=self.load_jit,
                                     load_trt=False, fp16=self.fp16)
            if self.use_ort_estimator:
                self._patch_ort_estimator()
            if self.nfe != 10:
                self._patch_nfe()
            if self.fp16:
                # autocast 只转计算不转存储；权重真转 fp16 后 LLM 每 token
                # 显存带宽减半（实测 55ms→19ms，RTX 3050）
                self._model.model.llm.half()
        except Exception as e:  # 加载失败只报一次，后续合成按句跳过
            self._load_err = e

    def _patch_ort_estimator(self):
        """flow decoder 的 estimator 换成 onnxruntime CUDA 版（非流式）。

        见模块 docstring：torch eager 在 WDDM 下慢 ~10 倍。CFM 的
        forward_estimator 是实例方法，直接替换即可；签名与 torch 版一致，
        输入已在正确的 device/dtype 上（solve_euler 预分配的 batch=2 CFG 张量）。
        """
        import onnxruntime as ort
        import torch

        onnx_path = os.path.join(self.model_dir, "flow.decoder.estimator.fp32.onnx")
        if not os.path.isfile(onnx_path):
            print(f"[warn] {onnx_path} 不存在，estimator 保持 torch eager")
            return
        sess = ort.InferenceSession(
            onnx_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        if sess.get_providers()[0] != "CUDAExecutionProvider":
            print("[warn] onnxruntime CUDA 不可用，estimator 回退 CPU（较慢）")
        cfm = self._model.model.flow.decoder

        def ort_forward_estimator(x, mask, mu, t, spks, cond, streaming=False):
            feeds = {
                "x": x.detach().cpu().numpy().astype(np.float32),
                "mask": mask.detach().cpu().numpy().astype(np.float32),
                "mu": mu.detach().cpu().numpy().astype(np.float32),
                "t": t.detach().cpu().numpy().astype(np.float32),
                "spks": spks.detach().cpu().numpy().astype(np.float32),
                "cond": cond.detach().cpu().numpy().astype(np.float32),
            }
            out = sess.run(None, feeds)[0]
            return torch.from_numpy(out).to(x.device)

        cfm.forward_estimator = ort_forward_estimator

    def _patch_nfe(self):
        """把 flow matching 的欧拉步数钳到 self.nfe（flow.py 里硬编码 10）。

        CFM.forward 是普通实例方法，遮蔽即可；n_timesteps 全以关键字传入。
        """
        cfm = self._model.model.flow.decoder
        orig = type(cfm).forward
        nfe = self.nfe

        def clamped_forward(mu, mask, n_timesteps, **kw):
            return orig(cfm, mu, mask, min(n_timesteps, nfe), **kw)

        cfm.forward = clamped_forward

    # ---------- 前端特征缓存（声线+语气 → 模型输入模板） ----------

    def _voice_profile(self, voice_id: str) -> dict:
        prof = self.voices.get(voice_id) or self.voices.get(self.default_voice)
        if prof is None:
            raise RuntimeError(f"cosyvoice 未配置声线 {self.default_voice!r}")
        return prof

    def _get_template(self, voice_id: str, emotion: str | None):
        prof = self._voice_profile(voice_id)
        instruct = (f"请用{emotion}的语气说话。" if emotion
                    else prof.get("instruct", ""))
        # CosyVoice2 instruct2 约定：instruct 必须以 <|endofprompt|> 结尾，
        # 否则 LM 会把 instruct 当正文念出来（llm.inference 把 prompt_text
        # 拼在 text 前，靠该 special token 区分指令与正文）
        if "<|endofprompt|>" not in instruct:
            instruct = instruct + "<|endofprompt|>"
        key = (voice_id if voice_id in self.voices else self.default_voice, instruct)
        tpl = self._frontend_cache.get(key)
        if tpl is None:
            frontend = self._model.frontend
            tpl = frontend.frontend_instruct2(
                "", instruct, prof["prompt_wav"], NATIVE_SR, "")
            self._frontend_cache[key] = tpl
        return tpl

    # ---------- 同步合成（worker 线程内） ----------

    def _synth_segment_sync(self, text: str, voice_id: str, emotion: str | None,
                            put) -> None:
        """合成一个带声线标记的片段，PCM16 16kHz 字节经 put() 顺流产出。"""
        model = self._model
        frontend = model.frontend
        norm = frontend.text_normalize(text, split=False)
        if isinstance(norm, list):      # split=False 也应返回 str，防御
            norm = "".join(norm)
        norm = norm.strip()
        if not norm:
            return
        with self._lock:
            tpl = self._get_template(voice_id, emotion)
            text_token, text_token_len = frontend._extract_text_token(norm)
            model_input = {**tpl, "text": text_token, "text_len": text_token_len}
            # speed!=1.0 只在非流式 finalize 路径支持
            stream = self.speed == 1.0
            if stream:
                # token_hop_len 在流式过程中会 ×2 递增且不复位（官方行为），
                # 每句开始前重置，否则后续短句首音要等 100 token
                model.model.token_hop_len = 25
            rs_state = None               # ratecv 状态跨 chunk 保持，避免边界咔哒声
            for out in model.model.tts(**model_input, stream=stream,
                                       speed=self.speed):
                speech = out["tts_speech"].detach().cpu().numpy().flatten()
                speech = np.clip(speech, -1.0, 1.0)
                pcm = (speech * 32767.0).astype(np.int16).tobytes()
                pcm, rs_state = audioop.ratecv(pcm, 2, 1, NATIVE_SR,
                                               self.sample_rate, rs_state)
                if pcm:
                    put(pcm)

    # ---------- TTS 契约 ----------

    async def synth(self, sentences, gen_id: int) -> AsyncIterator[AudioChunk]:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._load_sync)
        if self._load_err is not None:
            print(f"[warn] CosyVoice2 加载失败，本回合静音: {self._load_err!r}")
            return
        seq = 0
        chunk_bytes = self.sample_rate * 2 * CHUNK_MS // 1000
        async for sentence in sentences:
            q: asyncio.Queue = asyncio.Queue()
            cancelled = False

            def put(pcm: bytes):
                if not cancelled:
                    loop.call_soon_threadsafe(q.put_nowait, pcm)

            def run():
                try:
                    for text, voice_id, emotion in parse_tagged(sentence):
                        if cancelled:      # barge-in：丢弃本句剩余片段
                            break
                        self._synth_segment_sync(text, voice_id, emotion, put)
                except Exception as e:
                    loop.call_soon_threadsafe(q.put_nowait, e)
                finally:
                    loop.call_soon_threadsafe(q.put_nowait, None)

            threading.Thread(target=run, daemon=True).start()
            buf = b""
            try:
                while True:
                    item = await q.get()
                    if item is None:
                        break
                    if isinstance(item, Exception):
                        # 单句失败不能炸掉整个回合（同 sherpa 策略）
                        print(f"[warn] CosyVoice2 合成失败，跳过该句 "
                              f"{sentence[:20]!r}: {item!r}")
                        break
                    buf += item
                    while len(buf) >= chunk_bytes:
                        yield AudioChunk(pcm=buf[:chunk_bytes], gen_id=gen_id, seq=seq)
                        seq += 1
                        buf = buf[chunk_bytes:]
                if buf:
                    yield AudioChunk(pcm=buf, gen_id=gen_id, seq=seq)
                    seq += 1
            except asyncio.CancelledError:
                cancelled = True
                raise

    async def warmup(self, text: str) -> bytes | None:
        """开机预热：加载模型 + 构建全部声线模板 + 合成一句（结果丢弃）。"""
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._load_sync)
        if self._load_err is not None:
            print(f"[warn] CosyVoice2 预热失败: {self._load_err!r}")
            return None

        def _warm():
            try:
                for voice_id in self.voices:
                    self._get_template(voice_id, None)
                self._synth_segment_sync(text, self.default_voice, None,
                                         lambda pcm: None)
            except Exception as e:
                print(f"[warn] CosyVoice2 预热合成失败: {e!r}")

        await loop.run_in_executor(None, _warm)
        return None
