"""三层记忆（AGENTS.md §4.4）：滑窗 + 滚动摘要 + 结构化事实。

- 滑动窗口：只有最近 window_turns 个用户回合的原文进 prompt（本模块）；
- 滚动摘要：窗口外的旧回合攒够间隔后，由一次独立的廉价 LLM 调用异步压缩，
  注入 system prompt 的摘要段，不阻塞语音管道（本模块）；
- 结构化事实：record_fact 写入的 world_facts 永不淘汰（在 engine，不在这里）。

history 与 TurnManager 共享同一列表对象。滑窗必须切在 user 消息边界：
assistant tool_calls 与它的 tool 结果不可分家，窗口也不能以 tool 消息
开头——否则 API 会以 400 拒绝整个请求。
"""

from __future__ import annotations

import asyncio

SUMMARY_PROMPT = (
    "你在为一位跑团主持人整理剧情摘要。把下面的对话记录（含判定与检索结果）"
    "压缩成 300 字以内的中文摘要，必须保留：玩家发现的关键线索与证物、"
    "做出的重要选择与后果、NPC 透露的信息、当前局势与未解之谜、"
    "玩家的生命值/理智值等重要状态变化。只输出摘要正文，不要标题或解释。"
)


class ConversationMemory:
    def __init__(self, history: list, window_turns: int = 18,
                 summary_interval_turns: int = 15):
        self.history = history            # 与 TurnManager.history 同一列表
        self.window_turns = window_turns
        self.summary_interval_turns = summary_interval_turns
        self.summary = ""
        self._summarized_len = 0          # history 前多少条已被摘要覆盖
        self._task: asyncio.Task | None = None

    # ---------- 滑窗 ----------

    def windowed(self) -> list[dict]:
        """最近 window_turns 个用户回合的消息（切点在 user 边界，tool 配对不拆）。"""
        msgs = self.history
        count = 0
        cut = 0
        for i in range(len(msgs) - 1, -1, -1):
            if msgs[i].get("role") == "user":
                count += 1
                if count >= self.window_turns:
                    cut = i
                    break
        return msgs[cut:]

    # ---------- 滚动摘要 ----------

    def maybe_summarize(self, llm) -> None:
        """窗口外的未摘要旧消息攒够间隔 → 后台 task 压缩，不阻塞语音管道。"""
        if self._task is not None and not self._task.done():
            return
        if getattr(llm, "stream", None) is None:
            return
        head_len = len(self.history) - len(self.windowed())
        new = self.history[self._summarized_len:head_len]
        user_turns = sum(1 for m in new if m.get("role") == "user")
        if user_turns < self.summary_interval_turns:
            return
        self._task = asyncio.create_task(
            self._run_summary(llm, self.summary, list(new), head_len))

    async def _run_summary(self, llm, prev: str, msgs: list[dict],
                           upto: int) -> None:
        try:
            lines = []
            for m in msgs:
                content = m.get("content")
                if not content:
                    continue
                name = {"user": "玩家", "assistant": "主持人",
                        "tool": "判定/检索结果"}.get(m.get("role"))
                if name:
                    lines.append(f"{name}: {content}")
            if not lines:
                self._summarized_len = upto
                return
            instruction = SUMMARY_PROMPT
            if prev:
                instruction += ("\n已有摘要如下，请把新对话合并进来，"
                                "仍保持 300 字以内：\n" + prev)
            req = [{"role": "system", "content": instruction},
                   {"role": "user", "content": "\n".join(lines)}]
            parts = []
            async for tok in llm.stream(req, 0):
                parts.append(tok)
            text = "".join(parts).strip()
            if text:
                self.summary = text
                print(f"[game] 滚动摘要更新（覆盖至第 {upto} 条历史）: "
                      f"{text[:60]}……")
            self._summarized_len = upto
        except asyncio.CancelledError:
            raise
        except Exception as e:
            # 失败不阻塞对局，下轮攒够间隔会重试
            print(f"[warn] 滚动摘要失败（下轮重试）: {e!r}")

    # ---------- 局间压缩 ----------

    def reset(self) -> None:
        """一局结束回 lobby：清摘要与计数，避免跨局污染（AGENTS.md §2.2）。"""
        self.summary = ""
        self._summarized_len = 0
