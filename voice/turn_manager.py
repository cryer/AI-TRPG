"""TurnManager 状态机（AGENTS.md §5.2）：全双工轮次核心。

状态：IDLE / USER_SPEAKING / PROCESSING / AGENT_SPEAKING

- 连续音频帧逐帧过 VAD（executor 推理，event loop 零阻塞），事件驱动状态迁移；
- 每个用户回合在 speech_start 时新建 ASR 会话：建连耗时藏在用户说话时间里，
  并带 ~320ms preroll 防首音节被切；VAD speech_end → 关音频流（负包触发 finalize）
  → final_grace_ms 内等 ASR final，超时用最新 partial；
- PROCESSING：LLM token 流 → 分句器 → 逐从句送 TTS → player，首个 TTS chunk 进
  AGENT_SPEAKING；播放结束回 IDLE；
- barge-in：AGENT_SPEAKING 中 VAD speech_start 后逐帧确认语音持续 > bargein_confirm_ms
  （不能等 speech_end——它本身要 endpoint_ms 静音才触发，确认会太晚）→ 取消管线 task、
  player.flush()、gen_id 作废；已说出的半句带"被打断"标记留在对话上下文；
- endpoint 切早（PROCESSING 中用户继续说话）：中止在途管线，已识别文本 stash，
  拼接到下一回合；
- M4 endpoint 自适应（playbook ④）：VAD 用短 endpoint（如 300ms），speech_end
  时 ASR 文本"看着没说完"（过短或结尾是延续标点）则延展等待，用户继续说话
  就接着听（capture 不关，无缝），超时才真正切轮；
- 竞态铁律：一切异步产出携带 gen_id，消费侧发现过期即丢弃；取消用
  asyncio.CancelledError，资源清理在 except/finally。
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from typing import AsyncIterator

from voice.llm.splitter import split_sentences
from voice.metrics import MetricsSink, render_turn_waterfall
from voice.vad import SileroVAD, VADEvent

IDLE = "IDLE"
USER_SPEAKING = "USER_SPEAKING"
PROCESSING = "PROCESSING"
AGENT_SPEAKING = "AGENT_SPEAKING"

FRAME_MS = 32


class _Capture:
    """一个用户回合的 ASR 会话状态（speech_start 建连，speech_end 关流负包）。"""

    def __init__(self) -> None:
        self.q: asyncio.Queue[bytes | None] = asyncio.Queue()
        self.task: asyncio.Task | None = None
        self.latest_partial: str | None = None
        self.latest_final: str | None = None
        self.last_kind: str | None = None      # 最近一个事件是 partial 还是 final
        self.last_ev_t: float | None = None    # 最近一个事件的 monotonic 时刻
        self.t_asr_final: float | None = None
        self.final_event = asyncio.Event()
        self.new_event = asyncio.Event()       # 每个事件都置位（追尾等待用）


class TurnManager:
    def __init__(self, *, vad: SileroVAD, asr, llm, tts, player,
                 cfg: dict, system_prompt: str):
        self.vad = vad
        self.asr = asr
        self.llm = llm
        self.tts = tts
        self.player = player
        self.system_prompt = system_prompt
        self.endpoint_ms = cfg.get("endpoint_ms", 400)
        self.final_grace_ms = cfg.get("final_grace_ms", 150)
        self.asr_catchup_ms = cfg.get("asr_catchup_ms", 400)
        self.bargein_confirm_ms = cfg.get("bargein_confirm_ms", 200)
        self.providers = cfg.get("providers")
        adaptive = cfg.get("adaptive_endpoint", {})
        self.adaptive_enabled = adaptive.get("enabled", False)
        self.endpoint_extend_ms = adaptive.get("extend_ms", 300)
        self.min_complete_chars = adaptive.get("min_complete_chars", 5)

        self.state = IDLE
        self.history: list[dict] = []
        self.turn_id = 0
        self.sink: MetricsSink | None = None

        self._gen = 0
        self._preroll: deque[bytes] = deque(maxlen=10)  # ~320ms
        self._capture: _Capture | None = None
        self._turn_capture: _Capture | None = None
        self._ts: dict = {}
        self._candidate = False            # AGENT_SPEAKING 中的 barge-in 候选
        self._candidate_run_ms = 0
        self._candidate_t0: float | None = None  # 候选语音的真实开口时刻（monotonic）
        self._pipeline_task: asyncio.Task | None = None
        self._pipeline_interrupted = False  # cancel 原因：True=barge-in，False=用户重说
        self._stashed: tuple[str, float] | None = None  # 被中止回合的文本（拼下一句）
        self._ext_task: asyncio.Task | None = None      # 自适应 endpoint 延展计时
        self._ext_speech_end_t = 0.0                    # 延展对应的 speech_end 时刻

    # ---------- 主循环 ----------

    async def run(self, frames: AsyncIterator[bytes], sink: MetricsSink) -> None:
        self.sink = sink
        loop = asyncio.get_running_loop()
        async for frame in frames:
            ev = await loop.run_in_executor(None, self.vad.process, frame)
            await self._on_frame(frame, ev)
        ev = self.vad.flush()
        if ev is not None:
            await self._on_event(ev)
        # 输入耗尽（文件模式）：先让未决的 endpoint 延展落定，再让在途回合播完
        if self._ext_task is not None and not self._ext_task.done():
            try:
                await self._ext_task
            except asyncio.CancelledError:
                pass
        if self._pipeline_task is not None and not self._pipeline_task.done():
            try:
                await self._pipeline_task
            except asyncio.CancelledError:
                pass
        if self._capture is not None:
            self._cancel_capture()

    async def _on_frame(self, frame: bytes, ev: VADEvent | None) -> None:
        self._preroll.append(frame)
        if self._capture is not None:
            self._capture.q.put_nowait(frame)
        if self._candidate:
            prob = self.vad.last_prob
            if prob >= self.vad.threshold:
                self._candidate_run_ms += FRAME_MS
            elif prob < self.vad.neg_threshold:
                self._candidate_run_ms = 0
            if self._candidate_run_ms >= self.bargein_confirm_ms:
                await self._do_barge_in()
        if ev is not None:
            await self._on_event(ev)

    # ---------- VAD 事件 → 状态迁移 ----------

    async def _on_event(self, ev: VADEvent) -> None:
        if self.sink is not None:
            self.sink.vad_event(ev.kind, ev.t_ms, ev.prob)
        if ev.kind == "speech_start":
            await self._on_speech_start(ev)
        else:
            await self._on_speech_end()

    async def _on_speech_start(self, ev: VADEvent) -> None:
        now = time.monotonic()
        if self.state == IDLE:
            self._start_capture()
            self._begin_user_turn(now)
        elif self.state == AGENT_SPEAKING:
            # barge-in 候选：先开 ASR 捕获（确认窗内不丢音频），持续够才确认
            self._start_capture()
            self._candidate = True
            self._candidate_run_ms = 0
            # ev.t_ms 已回推到语音真实起点；换算成 monotonic 时刻（含 VAD 确认延迟）
            self._candidate_t0 = now - max(0, self.vad.now_ms - ev.t_ms) / 1000.0
        elif self.state == PROCESSING:
            # endpoint 切早了 / 用户补充：中止在途管线，文本 stash 拼到下一回合
            self._abort_pipeline()
            self._start_capture()
            self._begin_user_turn(now)
        elif self.state == USER_SPEAKING and self._ext_task is not None:
            # 自适应 endpoint 延展窗内用户继续说话：取消延展，capture 没关，接着听
            self._ext_task.cancel()
            self._ext_task = None

    async def _on_speech_end(self) -> None:
        now = time.monotonic()
        if self._candidate:
            # 确认窗前语音结束 → 咳嗽/噪音虚警，agent 继续说完
            self._candidate = False
            self._candidate_t0 = None
            self._cancel_capture()
            return
        if self.state != USER_SPEAKING:
            return
        if self.adaptive_enabled:
            cap = self._turn_capture
            text = ((cap.latest_partial or cap.latest_final or "")
                    if cap is not None else "")
            if not self._looks_complete(text):
                # 文本看着没说完：延展等待，用户继续说则无缝接续（capture 不关）
                self._ext_speech_end_t = now
                self._ext_task = asyncio.create_task(self._endpoint_extension())
                return
        self._finish_endpoint(now)

    def _looks_complete(self, text: str) -> bool:
        """标点/长度启发式：完整句（结尾标点或足够长）→ 短 endpoint 直接切轮。"""
        text = (text or "").strip()
        if not text:
            return False
        if text[-1] in "。！？!?…":
            return True
        if text[-1] in "，、；：,;:":
            return False
        return len(text) >= self.min_complete_chars

    async def _endpoint_extension(self) -> None:
        try:
            await asyncio.sleep(self.endpoint_extend_ms / 1000.0)
        except asyncio.CancelledError:
            raise
        finally:
            self._ext_task = None
        if self.state == USER_SPEAKING:
            # 延展超时用户没再说：以原 speech_end 时刻真正切轮
            self._finish_endpoint(self._ext_speech_end_t)

    def _finish_endpoint(self, speech_end_t: float) -> None:
        ts = self._ts
        ts["t_endpoint"] = time.monotonic()
        ts["t_last_user_audio"] = speech_end_t - self.endpoint_ms / 1000.0
        cap = self._turn_capture
        self._close_capture()
        self._set_state(PROCESSING)
        self._gen += 1
        self._pipeline_task = asyncio.create_task(
            self._pipeline(cap, self.turn_id, self._gen, ts))

    def _begin_user_turn(self, now: float) -> None:
        self.turn_id += 1
        self._turn_capture = self._capture
        self._ts = {"t_speech_start": now}
        self._set_state(USER_SPEAKING)

    def _set_state(self, new: str) -> None:
        if new != self.state:
            print(f"[state] {self.state} → {new}")
            self.state = new

    # ---------- barge-in / 中止 ----------

    async def _do_barge_in(self) -> None:
        self._candidate = False
        now = time.monotonic()
        interrupted_gen = self._gen
        self._gen += 1  # 铁律：作废旧 gen，迟到 token/chunk 全部丢弃
        self.player.flush()
        mute_ms = None
        if self._candidate_t0 is not None:
            mute_ms = round((now - self._candidate_t0) * 1000)
        self._candidate_t0 = None
        if self.sink is not None:
            self.sink.barge_in(self.turn_id, interrupted_gen, int(now * 1000),
                               cancelled={"llm": True, "tts": True,
                                          "playback_flushed": True},
                               mute_ms=mute_ms)
        print(f"[barge-in] turn {self.turn_id} gen {interrupted_gen} 作废，开始聆听"
              f"（开口→静音 {mute_ms}ms）")
        if self._pipeline_task is not None and not self._pipeline_task.done():
            self._pipeline_interrupted = True
            self._pipeline_task.cancel()
        self._begin_user_turn(now)  # 复用候选期已开始的 capture，不丢音频

    def _abort_pipeline(self) -> None:
        self._gen += 1
        self._pipeline_interrupted = False
        if self._pipeline_task is not None and not self._pipeline_task.done():
            self._pipeline_task.cancel()

    # ---------- ASR 捕获 ----------

    def _start_capture(self) -> None:
        cap = _Capture()
        for f in self._preroll:
            cap.q.put_nowait(f)
        cap.task = asyncio.create_task(self._asr_pump(cap))
        self._capture = cap

    def _close_capture(self) -> None:
        if self._capture is not None:
            self._capture.q.put_nowait(None)  # 负包 → ASR finalize
            self._capture = None

    def _cancel_capture(self) -> None:
        if self._capture is not None:
            if self._capture.task is not None:
                self._capture.task.cancel()
            self._capture = None

    async def _asr_pump(self, cap: _Capture) -> None:
        async def frames():
            while True:
                item = await cap.q.get()
                if item is None:
                    return
                yield item

        try:
            async for ev in self.asr.stream(frames()):
                cap.last_ev_t = time.monotonic()
                if ev.kind == "partial":
                    cap.latest_partial = ev.text
                    cap.last_kind = "partial"
                elif ev.kind == "final":
                    cap.latest_final = ev.text
                    cap.last_kind = "final"
                    cap.t_asr_final = cap.last_ev_t
                    cap.final_event.set()
                cap.new_event.set()
        except asyncio.CancelledError:
            raise
        except Exception as e:
            print(f"[warn] ASR 流出错：{e}")
            cap.final_event.set()  # 别让 pipeline 白等 grace

    # ---------- 应答管线（PROCESSING → AGENT_SPEAKING → IDLE） ----------

    async def _pipeline(self, cap: _Capture, turn_id: int, gen_id: int, ts: dict) -> None:
        text = ""
        reply_parts: list[str] = []
        user_idx = -1
        speculative = False
        try:
            # --- ASR 文本采用（实测依据见 AGENTS.md M2/M4 条目） ---
            # endpoint 时 ASR 往往还差最后 1~2 字（尾部 partial 滞后 ~300ms），
            # 而负包 finalize 要 ~470ms、服务端语义定稿 ~1.5s+，都等不起。
            # 策略：最近事件很近（ASR 还在追音频尾）→ 等一个更新补足尾字；
            # 否则立即采用现有 partial（投机派发 §7.2）；final 晚到后修补 history。
            # （M4 曾用 filler 占位音换取"等真 final"的宽限；filler 按用户决定
            #   移除后恢复快速采用。）
            if cap.last_kind != "final":
                if cap.last_ev_t is None:
                    # 极短语音无任何事件：给 final 一个宽限
                    try:
                        await asyncio.wait_for(cap.final_event.wait(),
                                               self.final_grace_ms / 1000.0)
                    except asyncio.TimeoutError:
                        pass
                elif time.monotonic() - cap.last_ev_t < 0.20:
                    cap.new_event.clear()
                    try:
                        await asyncio.wait_for(cap.new_event.wait(),
                                               self.asr_catchup_ms / 1000.0)
                    except asyncio.TimeoutError:
                        pass
            if cap.last_kind == "final" and cap.latest_final:
                text = cap.latest_final.strip()
                if cap.t_asr_final is not None:
                    ts["t_asr_final"] = cap.t_asr_final
            else:
                text = (cap.latest_partial or cap.latest_final or "").strip()
                speculative = bool(text)
                ts["t_asr_final"] = time.monotonic()
            if self._stashed is not None:
                prefix, t0 = self._stashed
                self._stashed = None
                if time.monotonic() - t0 < 10.0:
                    text = prefix + text
            print(f"[turn {turn_id}] ASR{'(投机)' if speculative else ''}: {text!r}")
            if not text:
                if self._gen == gen_id:
                    self._set_state(IDLE)
                return
            self.history.append({"role": "user", "content": text})
            user_idx = len(self.history) - 1
            messages = [{"role": "system", "content": self.system_prompt}] + self.history

            self.player.start_gen(gen_id)
            async for chunk in self.tts.synth(
                    self._sentence_stream(messages, gen_id, ts, reply_parts, turn_id),
                    gen_id):
                if self._gen != gen_id:  # 已作废（防御；正常路径是 task 被取消）
                    break
                if "t_tts_first_chunk" not in ts:
                    ts["t_tts_first_chunk"] = time.monotonic()
                if self.state == PROCESSING:
                    self._set_state(AGENT_SPEAKING)
                if not await self.player.feed_chunk(chunk):
                    break
            self.player.end_gen()
            await self.player.wait_done(gen_id)
            if self._gen != gen_id:
                return
            fa = self.player.first_audio_time(gen_id)
            if fa is not None:
                ts["t_playback_start"] = fa
            reply = "".join(reply_parts)
            if reply:
                self.history.append({"role": "assistant", "content": reply})
            else:
                # LLM 空回复（如 kimi-for-coding 对域外请求只出 reasoning）：
                # 不进 history——空 assistant 消息会让后续请求被 400 拒绝
                print(f"[warn] turn {turn_id} LLM 空回复，不计入对话上下文")
            print(f"[turn {turn_id}] agent: {reply!r}")
            extra = {"user_text": text, "reply": reply, "speculative": speculative}
            if not reply:
                extra["llm_empty"] = True
            patched = self._patch_user_text(user_idx, cap, text)
            if patched is not None:
                extra["user_text_final"] = patched
            if speculative and cap.t_asr_final is not None and "t_endpoint" in ts:
                extra["asr_true_final_ms"] = round(
                    (cap.t_asr_final - ts["t_endpoint"]) * 1000)
            self._emit_turn(turn_id, gen_id, ts, interrupted=False, extra=extra)
            wf = render_turn_waterfall(ts)
            if wf:
                print(f"  waterfall: {wf}")
            if self._gen == gen_id and self.state in (PROCESSING, AGENT_SPEAKING):
                if self._candidate:
                    # 播放收尾期间起的候选直接转正为新回合
                    self._candidate = False
                    self._candidate_t0 = None
                    self._begin_user_turn(time.monotonic())
                else:
                    self._set_state(IDLE)
        except asyncio.CancelledError:
            reply = "".join(reply_parts)
            if self._pipeline_interrupted:
                # barge-in：已说出的半句带标记留在上下文（agent 知道自己被打断）
                self._pipeline_interrupted = False
                if reply:
                    self.history.append(
                        {"role": "assistant", "content": reply + " …（被打断）"})
                self._emit_turn(turn_id, gen_id, ts, interrupted=True,
                                extra={"user_text": text, "reply": reply,
                                       "barge_in": True})
            else:
                # PROCESSING 中被用户重说中止：弹出悬空 user 消息，文本 stash
                text = text or (cap.latest_final or cap.latest_partial or "").strip()
                if 0 <= user_idx < len(self.history) and \
                        self.history[user_idx].get("role") == "user":
                    self.history.pop(user_idx)
                if text:
                    self._stashed = (text, time.monotonic())
            raise
        except Exception as e:
            print(f"[warn] turn {turn_id} 管线出错：{e!r}")
            if self._gen == gen_id and self.state in (PROCESSING, AGENT_SPEAKING):
                self._set_state(IDLE)

    async def _sentence_stream(self, messages: list[dict], gen_id: int, ts: dict,
                               reply_parts: list[str], turn_id: int) -> AsyncIterator[str]:
        async def tokens():
            async for tok in self.llm.stream(messages, gen_id):
                if "t_llm_first_token" not in ts:
                    ts["t_llm_first_token"] = time.monotonic()
                reply_parts.append(tok)
                yield tok

        async for s in split_sentences(tokens()):
            if "t_first_sentence_out" not in ts:
                ts["t_first_sentence_out"] = time.monotonic()
            print(f"[turn {turn_id}] sentence: {s!r}")
            yield s

    def _patch_user_text(self, user_idx: int, cap: _Capture, used_text: str) -> str | None:
        """投机派发后 final 晚到：若与采用文本不同，修补 history 里的 user 消息。"""
        final = (cap.latest_final or "").strip()
        if not final or final == used_text:
            return None
        if 0 <= user_idx < len(self.history) and \
                self.history[user_idx].get("role") == "user" and \
                self.history[user_idx].get("content") == used_text:
            self.history[user_idx]["content"] = final
            return final
        return None

    def _emit_turn(self, turn_id: int, gen_id: int, ts: dict, *,
                   interrupted: bool, extra: dict | None = None) -> None:
        if self.sink is None:
            return
        meta = {"endpoint_ms": self.endpoint_ms, "providers": self.providers}
        if extra:
            meta.update(extra)
        self.sink.turn(turn_id, gen_id, ts, interrupted=interrupted, meta=meta)
