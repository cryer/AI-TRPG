"""Instrumentation: per-turn latency timestamps, JSONL sink, latency report.

JSONL 格式（定稿，每行一个 JSON 对象）：

1. turn 记录 —— 一轮完整交互（用户说完 → agent 出声）::

    {"type": "turn", "turn_id": 1, "gen_id": 1, "interrupted": false,
     "timestamps": {"t_last_user_audio": 12.300, "t_endpoint": 12.700,
                    "t_asr_final": 12.810, "t_llm_first_token": 12.950,
                    "t_first_sentence_out": 13.020, "t_tts_first_chunk": 13.150,
                    "t_playback_start": 13.180},
     "meta": {"endpoint_ms": 400, "providers": {...}, "notes": "..."}}

   时间戳为 time.monotonic() 秒；未发生的环节缺省该键（报告按缺失处理）。

2. VAD 事件::

    {"type": "vad_event", "kind": "speech_start"|"speech_end", "t_ms": 1234, "prob": 0.87}

3. barge-in 事件（`mute_ms` = 用户真实开口（VAD 回推起点）→ player.flush()）::

    {"type": "barge_in", "turn_id": 1, "gen_id": 1, "t_ms": 15000,
     "cancelled": {"llm": true, "tts": true, "playback_flushed": true},
     "mute_ms": 320}

用法：
    python -m voice.metrics report <session.jsonl> [--budget configs/default.json]
    python -m voice.metrics demo [--turns 20] [-o reports/fake_session.jsonl]
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

# 环节定义：(报告名, 起始时间戳键, 结束时间戳键, 预算键)
SEGMENTS = [
    ("vad_endpoint", "t_last_user_audio", "t_endpoint", "vad_endpoint"),
    ("asr_final", "t_endpoint", "t_asr_final", "asr_final"),
    ("llm_first_sentence", "t_asr_final", "t_first_sentence_out", "llm_first_sentence"),
    ("tts_first_chunk", "t_first_sentence_out", "t_tts_first_chunk", "tts_first_chunk"),
    ("transport_buffer", "t_tts_first_chunk", "t_playback_start", "transport_buffer"),
]

DEFAULT_BUDGET = {
    "vad_endpoint": 400,
    "asr_final": 150,
    "llm_first_sentence": 400,
    "tts_first_chunk": 200,
    "transport_buffer": 50,
    "total": 1200,
    "telephony_total": 1600,
}


class MetricsSink:
    """追加写 JSONL，每行即刷盘（崩溃不丢数据）。"""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "a", encoding="utf-8")

    def emit(self, record: dict) -> None:
        record.setdefault("wall_clock", time.time())
        self._fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._fh.flush()

    def turn(self, turn_id: int, gen_id: int, timestamps: dict, **kw) -> None:
        self.emit({"type": "turn", "turn_id": turn_id, "gen_id": gen_id,
                   "interrupted": kw.pop("interrupted", False),
                   "timestamps": dict(timestamps),
                   "meta": kw.pop("meta", {}), **kw})

    def vad_event(self, kind: str, t_ms: int, prob: float | None = None) -> None:
        rec = {"type": "vad_event", "kind": kind, "t_ms": t_ms}
        if prob is not None:
            rec["prob"] = round(prob, 3)
        self.emit(rec)

    def barge_in(self, turn_id: int, gen_id: int, t_ms: int, cancelled: dict,
                 **extra) -> None:
        self.emit({"type": "barge_in", "turn_id": turn_id, "gen_id": gen_id,
                   "t_ms": t_ms, "cancelled": cancelled, **extra})

    def close(self) -> None:
        self._fh.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def render_turn_waterfall(ts: dict) -> str:
    """单 turn 的分解瀑布（相对 t_last_user_audio 的毫秒偏移）。"""
    base = ts.get("t_last_user_audio")
    if base is None:
        return ""
    keys = ["t_endpoint", "t_asr_final", "t_llm_first_token",
            "t_first_sentence_out", "t_tts_first_chunk", "t_playback_start"]
    parts = [f"{k[2:]}=+{(ts[k] - base) * 1000:.0f}ms" for k in keys if k in ts]
    return " | ".join(parts)


def percentile(sorted_vals: list[float], p: float) -> float:
    """线性插值百分位；sorted_vals 必须已排序且非空。"""
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    k = (len(sorted_vals) - 1) * p / 100.0
    lo, hi = math.floor(k), math.ceil(k)
    if lo == hi:
        return sorted_vals[lo]
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (k - lo)


def _stats(vals: list[float]) -> dict | None:
    if not vals:
        return None
    s = sorted(vals)
    return {"n": len(s), "p50": percentile(s, 50), "p95": percentile(s, 95),
            "p99": percentile(s, 99), "mean": sum(s) / len(s)}


def load_records(path: str | Path) -> tuple[list[dict], list[dict], list[dict]]:
    turns, vad_events, barge_ins = [], [], []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            t = rec.get("type")
            if t == "turn":
                turns.append(rec)
            elif t == "vad_event":
                vad_events.append(rec)
            elif t == "barge_in":
                barge_ins.append(rec)
    return turns, vad_events, barge_ins


def render_report(turns: list[dict], barge_ins: list[dict], budget: dict) -> str:
    lines = []
    lines.append(f"turns: {len(turns)}  barge_ins: {len(barge_ins)}")
    interrupted = sum(1 for t in turns if t.get("interrupted"))
    if interrupted:
        lines.append(f"interrupted turns: {interrupted}")
    # M4 优化证据行
    spec = [t for t in turns if t.get("meta", {}).get("speculative")]
    if spec:
        tf = [t["meta"]["asr_true_final_ms"] for t in spec
              if isinstance(t.get("meta", {}).get("asr_true_final_ms"), (int, float))]
        note = f"，真 final 平均晚到 {sum(tf)/len(tf):.0f}ms" if tf else ""
        lines.append(f"投机派发（playbook ②）: {len(spec)}/{len(turns)} turns "
                     f"在 ASR final 前派发 LLM{note}")
    mute_vals = [b["mute_ms"] for b in barge_ins
                 if isinstance(b.get("mute_ms"), (int, float))]
    st = _stats(mute_vals)
    if st:
        # 口径：用户真实开口（VAD 回推起点，含 ~96ms VAD 确认）→ player.flush()
        # TRPG 非对称打断（项目 AGENTS.md §5）：叙述段阈值 500ms，等待段 200ms，
        # 事件带 narration 标记时分组统计，各按自己的验收窗判定。
        narr = [b["mute_ms"] for b in barge_ins
                if b.get("narration") and isinstance(b.get("mute_ms"), (int, float))]
        wait = [b["mute_ms"] for b in barge_ins
                if not b.get("narration") and isinstance(b.get("mute_ms"), (int, float))]
        has_marks = any("narration" in b for b in barge_ins)
        if has_marks and narr and wait:
            for label, vals, win in (("叙述段", narr, 700), ("等待段", wait, 400)):
                g = _stats(vals)
                if g:
                    ok = "OK" if g["p95"] <= win else f"OVER p95>{win}"
                    lines.append(f"barge-in 开口→静音（{label}）: n={len(vals)} "
                                 f"p50={g['p50']:.0f}ms p95={g['p95']:.0f}ms"
                                 f"（验收窗 ≤{win}ms）{ok}")
        else:
            ok = "OK" if st["p95"] <= 400 else "OVER p95>400"
            lines.append(f"barge-in 开口→静音: p50={st['p50']:.0f}ms p95={st['p95']:.0f}ms"
                         f"（验收窗 200–400ms）{ok}")
    lines.append("")
    header = f"{'segment':<20}{'budget':>8}{'p50':>9}{'p95':>9}{'p99':>9}{'mean':>9}  status"
    lines.append(header)
    lines.append("-" * len(header))
    for name, k0, k1, bkey in SEGMENTS:
        vals = []
        for t in turns:
            ts = t.get("timestamps", {})
            if k0 in ts and k1 in ts:
                vals.append((ts[k1] - ts[k0]) * 1000.0)
        st = _stats(vals)
        b = budget.get(bkey)
        if st is None:
            lines.append(f"{name:<20}{(str(b) if b else '-'):>8}  (no data)")
            continue
        ok = "OK" if (b is None or st["p95"] <= b) else f"OVER p95>{b}"
        lines.append(f"{name:<20}{(str(b) if b else '-'):>8}"
                     f"{st['p50']:>9.1f}{st['p95']:>9.1f}{st['p99']:>9.1f}{st['mean']:>9.1f}  {ok}")
    # 总延迟：用户说完 → 出声
    totals = []
    for t in turns:
        ts = t.get("timestamps", {})
        if "t_last_user_audio" in ts and "t_playback_start" in ts:
            totals.append((ts["t_playback_start"] - ts["t_last_user_audio"]) * 1000.0)
    st = _stats(totals)
    lines.append("-" * len(header))
    if st:
        b = budget.get("total")
        ok = "OK" if (b is None or st["p95"] <= b) else f"OVER p95>{b}"
        lines.append(f"{'TOTAL':<20}{(str(b) if b else '-'):>8}"
                     f"{st['p50']:>9.1f}{st['p95']:>9.1f}{st['p99']:>9.1f}{st['mean']:>9.1f}  {ok}")
    else:
        lines.append("TOTAL               (no data)")
    return "\n".join(lines)


def gen_fake_session(path: str | Path, n_turns: int, seed: int = 42) -> None:
    rng = random.Random(seed)
    with MetricsSink(path) as sink:
        t = time.monotonic()
        for i in range(1, n_turns + 1):
            t0 = t + rng.uniform(2, 6)
            ts = {"t_last_user_audio": t0}
            t += t0 - t
            # 各分段在预算内随机生成（少量超标模拟毛刺）
            for name, k0, k1, bkey in SEGMENTS:
                budget_ms = DEFAULT_BUDGET[bkey]
                dur = rng.uniform(0.3, 0.9) * budget_ms
                if rng.random() < 0.05:
                    dur *= 1.6
                ts[k1] = ts[k0] + dur / 1000.0
            # LLM 首 token 早于首句
            ts["t_llm_first_token"] = ts["t_asr_final"] + (ts["t_first_sentence_out"] - ts["t_asr_final"]) * rng.uniform(0.4, 0.8)
            sink.turn(i, i, ts, meta={"endpoint_ms": 400, "fake": True})
            sink.vad_event("speech_start", int(t0 * 1000) - 1500, 0.9)
            sink.vad_event("speech_end", int(t0 * 1000), 0.2)
            t = ts["t_playback_start"] + 1.0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m voice.metrics")
    sub = ap.add_subparsers(dest="cmd", required=True)

    rp = sub.add_parser("report", help="渲染 session JSONL 的延迟瀑布报告")
    rp.add_argument("session")
    rp.add_argument("--budget", help="configs/default.json 路径（读 budget_ms）")

    dp = sub.add_parser("demo", help="生成假 turn 并渲染报告（M0 验收用）")
    dp.add_argument("--turns", type=int, default=20)
    dp.add_argument("-o", "--out", default="reports/fake_session.jsonl")

    args = ap.parse_args(argv)

    if args.cmd == "demo":
        gen_fake_session(args.out, args.turns)
        print(f"wrote {args.out}")
        turns, _, barge_ins = load_records(args.out)
        print(render_report(turns, barge_ins, DEFAULT_BUDGET))
        return 0

    budget = dict(DEFAULT_BUDGET)
    if args.budget:
        with open(args.budget, encoding="utf-8") as fh:
            budget.update(json.load(fh).get("budget_ms", {}))
    turns, _, barge_ins = load_records(args.session)
    print(render_report(turns, barge_ins, budget))
    return 0


if __name__ == "__main__":
    sys.exit(main())
