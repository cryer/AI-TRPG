"""M1 半双工全链路：按键说话（push-to-talk）：ASR → LLM → TTS → 播放 + 延迟分解。

用法：
    python -m voice.ptt                          # 麦克风交互：Enter 开始说话，再 Enter 结束
    python -m voice.ptt --file reports/test_speech.wav   # 用固定录音跑一轮（回归）
    python -m voice.ptt --file x.wav --turns 5 --no-play # 无声卡跑多轮，出报告

M1 无 VAD endpointing（M2 才接），t_endpoint = t_last_user_audio，meta 里注明 ptt。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import AsyncIterator

from dotenv import load_dotenv

from voice.asr import create_asr
from voice.llm import create_llm
from voice.llm.splitter import split_sentences
from voice.metrics import MetricsSink, load_records, render_report
from voice.tts import create_tts

SAMPLE_RATE = 16000
SYSTEM_PROMPT = "你是一个语音助手，回答简短口语化，每次不超过三句话。"


async def run_turn(frames: AsyncIterator[bytes], *, sink: MetricsSink,
                   turn_id: int, gen_id: int, asr, llm, tts, player,
                   history: list[dict], cfg: dict, ts: dict) -> dict:
    """一轮完整交互，时间戳写入 ts（调用方需预置 t_last_user_audio / t_endpoint）。
    取消语义：外层 cancel，gen_id 作废由消费侧保证。"""

    # --- ASR ---
    # final 语义为"截至目前的完整定稿文本"（见 asr/base.py），取最后一个即可；
    # t_asr_final 取最后一个 final 的时刻（完整文本就绪）
    latest_partial = None
    latest_final = None
    async for ev in asr.stream(frames):
        if ev.kind == "partial":
            latest_partial = ev.text
        elif ev.kind == "final":
            latest_final = ev.text
            ts["t_asr_final"] = time.monotonic()
    text = latest_final or latest_partial or ""
    print(f"[turn {turn_id}] ASR: {text!r}")
    if not text:
        return ts

    history.append({"role": "user", "content": text})
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history

    # --- LLM → 分句 → TTS → 播放 ---
    player.start_gen(gen_id)
    reply_parts: list[str] = []

    async def tapped_tokens():
        async for tok in llm.stream(messages, gen_id):
            if "t_llm_first_token" not in ts:
                ts["t_llm_first_token"] = time.monotonic()
            reply_parts.append(tok)
            yield tok

    async def tapped_sentences():
        async for s in split_sentences(tapped_tokens()):
            if "t_first_sentence_out" not in ts:
                ts["t_first_sentence_out"] = time.monotonic()
            print(f"[turn {turn_id}] sentence: {s!r}")
            yield s

    async for chunk in tts.synth(tapped_sentences(), gen_id):
        if "t_tts_first_chunk" not in ts:
            ts["t_tts_first_chunk"] = time.monotonic()
        if not await player.feed_chunk(chunk):  # gen 已作废（M3 barge-in）
            break
    player.end_gen()
    await player.wait_done(gen_id)

    fa = player.first_audio_time(gen_id)
    if fa is not None:
        ts["t_playback_start"] = fa

    reply = "".join(reply_parts)
    history.append({"role": "assistant", "content": reply})
    print(f"[turn {turn_id}] agent: {reply!r}")

    sink.turn(turn_id, gen_id, ts, meta={
        "ptt": True, "endpoint_ms": 0,
        "providers": cfg.get("providers"),
        "user_text": text, "reply": reply,
    })
    return ts


async def _frames_from_list(frames: list[bytes]) -> AsyncIterator[bytes]:
    for f in frames:
        yield f


def _print_turn_waterfall(ts: dict) -> None:
    from voice.metrics import render_turn_waterfall
    wf = render_turn_waterfall(ts)
    if wf:
        print(f"  waterfall: {wf}")


async def run_file_mode(args, cfg, sink) -> None:
    from voice.transport.wav_file import wav_frames

    asr, llm, tts = create_asr(cfg), create_llm(cfg), create_tts(cfg)
    player = _make_player(args)
    history: list[dict] = []
    try:
        for turn_id in range(1, args.turns + 1):
            frames = [f async for f in wav_frames(args.file)]
            ts = {"t_last_user_audio": time.monotonic()}
            ts["t_endpoint"] = ts["t_last_user_audio"]
            await run_turn(_frames_from_list(frames), sink=sink,
                           turn_id=turn_id, gen_id=turn_id,
                           asr=asr, llm=llm, tts=tts, player=player,
                           history=history, cfg=cfg, ts=ts)
            _print_turn_waterfall(ts)
    finally:
        player.close()
        await _close_providers(asr, llm, tts)


async def run_mic_mode(args, cfg, sink) -> None:
    from voice.transport.local_sounddevice import mic_frames

    asr, llm, tts = create_asr(cfg), create_llm(cfg), create_tts(cfg)
    player = _make_player(args)
    history: list[dict] = []
    frames_q: asyncio.Queue[bytes] = asyncio.Queue(maxsize=200)
    collecting = asyncio.Event()

    async def collector():
        async for f in mic_frames():
            if collecting.is_set():
                if frames_q.full():
                    try:
                        frames_q.get_nowait()
                    except asyncio.QueueEmpty:
                        pass
                frames_q.put_nowait(f)

    task = asyncio.create_task(collector())
    turn_id = 0
    print("[M1] push-to-talk：Enter 开始说话，再 Enter 结束并发送；Ctrl+C 退出")
    try:
        while True:
            await asyncio.get_running_loop().run_in_executor(None, input, ">>> 按 Enter 开始说话…")
            while not frames_q.empty():
                frames_q.get_nowait()
            collecting.set()
            print("    录音中…（说完按 Enter）")
            await asyncio.get_running_loop().run_in_executor(None, input)
            collecting.clear()
            frames = []
            while not frames_q.empty():
                frames.append(frames_q.get_nowait())
            if len(frames) < 5:
                print("    太短，忽略")
                continue
            turn_id += 1
            ts = {"t_last_user_audio": time.monotonic()}
            ts["t_endpoint"] = ts["t_last_user_audio"]
            await run_turn(_frames_from_list(frames), sink=sink,
                           turn_id=turn_id, gen_id=turn_id,
                           asr=asr, llm=llm, tts=tts, player=player,
                           history=history, cfg=cfg, ts=ts)
            _print_turn_waterfall(ts)
    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        task.cancel()
        player.close()
        await _close_providers(asr, llm, tts)


async def _close_providers(*providers) -> None:
    for p in providers:
        close = getattr(p, "close", None)
        if close is not None:
            try:
                await close()
            except Exception:
                pass


def _make_player(args):
    if args.no_play:
        from voice.player import NullPlayer
        return NullPlayer()
    try:
        from voice.player import Player
        return Player()
    except Exception as e:
        print(f"[warn] 声卡不可用（{e}），改用 NullPlayer 落盘 reports/recordings/tts_out.wav")
        from voice.player import NullPlayer
        return NullPlayer()


async def amain(args) -> None:
    with open(args.config, encoding="utf-8") as fh:
        cfg = json.load(fh)
    for slot in ("asr", "llm", "tts"):
        override = getattr(args, slot)
        if override:
            cfg.setdefault("providers", {})[slot] = override
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    jsonl_path = Path(args.out) / f"m1_{stamp}.jsonl"
    with MetricsSink(jsonl_path) as sink:
        if args.file:
            await run_file_mode(args, cfg, sink)
        else:
            await run_mic_mode(args, cfg, sink)
    print(f"[M1] 会话落盘 {jsonl_path}")
    turns, _, barge_ins = load_records(jsonl_path)
    if turns:
        print(render_report(turns, barge_ins, cfg.get("budget_ms", {})))


def main(argv=None) -> int:
    load_dotenv()
    ap = argparse.ArgumentParser(prog="python -m voice.ptt")
    ap.add_argument("--file", help="用 wav 文件模拟一轮按键说话")
    ap.add_argument("--turns", type=int, default=1)
    ap.add_argument("--no-play", action="store_true", help="不出声，TTS 音频落盘")
    ap.add_argument("--config", default="configs/default.json")
    ap.add_argument("--out", default="reports")
    ap.add_argument("--asr", help="覆盖 config 的 ASR provider（如 mock / deepgram / volcengine）")
    ap.add_argument("--llm", help="覆盖 config 的 LLM provider（如 mock / kimi）")
    ap.add_argument("--tts", help="覆盖 config 的 TTS provider（如 mock / volcengine）")
    args = ap.parse_args(argv)
    try:
        asyncio.run(amain(args))
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
