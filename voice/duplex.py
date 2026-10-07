"""M2 全双工轮次：VAD endpointing 自动切轮，免按键连续对话。

用法：
    python -m voice.duplex                # 麦克风全双工（务必戴耳机，见 AGENTS.md §6）
    python -m voice.duplex --file reports/test_speech.wav --no-play   # 回归：realtime 喂帧

文件模式固定 realtime 吐帧（burst 下 VAD/延迟数字无意义）。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from voice.asr import create_asr
from voice.llm import create_llm
from voice.metrics import MetricsSink, load_records, render_report
from voice.tts import create_tts
from voice.turn_manager import TurnManager
from voice.vad import SileroVAD

SYSTEM_PROMPT = (
    "你是一个语音助手，正在和用户进行实时的语音通话。"
    "你的所有回复都会被语音合成朗读出来，用户用耳朵听，看不到任何文字。"
    "所以：像真正说话一样回答，简短口语化，每次不超过三句话；"
    "永远不要说自己无法发声、无法播放音频、没有声音或'只是文字助手'——你的回复本身就是声音；"
    "被要求唱歌、讲故事、讲笑话、模仿声音时，直接用口语演绎出来"
    "（唱歌就直接把歌词唱出来，可以带语气词、重复和延长音）；"
    "不要使用 Markdown、列表、编号、表情符号等任何视觉排版。"
)


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

    vad_cfg = cfg.get("vad", {})
    vad = SileroVAD(vad_cfg.get("model_path", "models/silero_vad.onnx"),
                    threshold=vad_cfg.get("threshold", 0.5),
                    neg_threshold=vad_cfg.get("neg_threshold", 0.35),
                    endpoint_ms=cfg.get("endpoint_ms", 400),
                    min_speech_ms=vad_cfg.get("min_speech_ms", 96))
    asr, llm, tts = create_asr(cfg), create_llm(cfg), create_tts(cfg)
    player = _make_player(args)

    # M4 playbook ⑤：TTS 开机预热（TLS 连接 + 音色/模型），预热文本合成后丢弃。
    # （filler 占位快车道曾在 M4 实现，真人验收后按用户决定移除——每轮播
    #   占位音太机械；保留预热因为它无侵入且改善首轮真实 TTS 延迟。）
    warmup = getattr(tts, "warmup", None)
    if warmup is not None:
        try:
            await warmup("嗯。")
            print("[M4] TTS 连接/音色已预热")
        except Exception as e:
            print(f"[warn] TTS 预热失败：{e}")

    # 游戏模式（AGENTS.md §2.2）：cfg 带 "game" 段时由 GameAgent 接管
    # system prompt（可调用，每轮解析）与 token 产出（responder，含工具调用循环）
    agent = None
    if "game" in cfg:
        from game.agent import GameAgent
        agent = GameAgent(cfg, llm)
        print(f"[game] 已加载 {len(agent.adventures)} 个剧本，进入推荐员模式")
    tm = TurnManager(vad=vad, asr=asr, llm=llm, tts=tts, player=player,
                     cfg=cfg,
                     system_prompt=(agent.system_prompt if agent is not None
                                    else cfg.get("system_prompt", SYSTEM_PROMPT)))
    if agent is not None:
        agent.bind_turn_manager(tm)   # 共享 history、记忆、打断模式回调

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    jsonl_path = Path(args.out) / f"m2_{stamp}.jsonl"
    try:
        with MetricsSink(jsonl_path) as sink:
            if args.file:
                from voice.transport.wav_file import wav_frames
                print(f"[M2] 文件回归：{args.file}（realtime）")
                await tm.run(wav_frames(args.file, realtime=True), sink)
            else:
                from voice.transport.local_sounddevice import mic_frames
                print("[M2] 全双工：直接说话，停顿自动切轮；agent 说话时可打断"
                      "（务必戴耳机）；Ctrl+C 退出")
                await tm.run(mic_frames(), sink)
    finally:
        player.close()
        await _close_providers(asr, llm, tts)

    print(f"[M2] 会话落盘 {jsonl_path}")
    stats_fn = getattr(player, "stats", None)
    if stats_fn is not None:
        st = stats_fn()
        print(f"[泄露检查] 迟到 stale chunk 丢弃 {st['dropped_stale_chunks']} 个"
              f"（{st['dropped_stale_bytes']}B，全部未播）；"
              f"flush 清除未播 {st['flushed_bytes']}B")
    turns, _, barge_ins = load_records(jsonl_path)
    if turns:
        print(render_report(turns, barge_ins, cfg.get("budget_ms", {})))


def main(argv=None) -> int:
    load_dotenv()
    ap = argparse.ArgumentParser(prog="python -m voice.duplex")
    ap.add_argument("--file", help="用 wav 文件模拟用户语音（realtime 喂帧）")
    ap.add_argument("--no-play", action="store_true", help="不出声，TTS 音频落盘")
    ap.add_argument("--config", default="configs/default.json")
    ap.add_argument("--out", default="reports")
    ap.add_argument("--asr", help="覆盖 config 的 ASR provider")
    ap.add_argument("--llm", help="覆盖 config 的 LLM provider")
    ap.add_argument("--tts", help="覆盖 config 的 TTS provider")
    args = ap.parse_args(argv)
    try:
        asyncio.run(amain(args))
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
