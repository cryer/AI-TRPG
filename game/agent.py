"""推荐员 / 主持人编排（AGENTS.md §2.1、§4.3）。

同一个 LLM，两套 system prompt，由 GameEngine 状态机切换：
- lobby 状态 → LOBBY_PROMPT（推荐员：介绍剧本、回答对比、start_adventure 入口）
- playing/ended 状态 → dm_prompt(engine)（主持人：每轮按世界状态动态拼装）

GameAgent.respond 对接 TurnManager 的 responder：实现工具调用循环，
content token 照常流式产出（送 TTS），tool_calls 在流尾收齐后由代码执行，
结果原子追加进共享 history 后再触发下一轮 completion。
"""

from __future__ import annotations

import json
from typing import AsyncIterator

from game.engine import GameEngine, LOBBY, PLAYING, ENDED
from game.memory import ConversationMemory
from game.script_loader import load_adventures
from game.tools import DM_TOOLS, LOBBY_TOOLS, SESSION_TOOLS, execute_tool

MAX_TOOL_ROUNDS = 5  # 防死循环

LOBBY_PROMPT = (
    "你是一家桌游店的老板，正在用语音电话接待一位想玩单人跑团剧本的顾客。"
    "你热情但克制，绝不剧透任何剧本的具体内容——只讲题材、时长、难度和"
    "一句话卖点这个级别的信息。"
    "你的职责：主动问候顾客，了解他的偏好（题材、恐怖程度、时长），介绍和"
    "对比店里的剧本，回答相关问题；顾客明确选定某本之后，调用 start_adventure 开本。"
    "工具使用规则：顾客问有什么本、要你推荐时，先调用 list_adventures；"
    "顾客问某个本的详情时，调用 describe_adventure；"
    "只有顾客明确表达了选定意图（比如「就玩这个」「开第三本」）才调用 start_adventure，"
    "顾客还在犹豫时绝不擅自调用。"
    "店里目前只有工具返回的这些剧本，不要编造任何不存在的剧本或设定。"
    "语音约束（非常重要）：你的所有回复都会被语音合成朗读出来，顾客用耳朵听，"
    "看不到任何文字。所以：像真正说话一样回答，简短口语化，每次不超过三句话；"
    "不要使用 Markdown、列表、编号、表情符号等任何视觉排版；"
    "永远不要说自己无法发声或只是文字助手。"
)

_DM_BASE_RULES = (
    "你是这场跑团的主持人（DM/Keeper），正在用语音带一位玩家完成一场冒险。"
    "扮演原则：你只描述世界、NPC 和行动的后果，行动选择权永远留给玩家，"
    "绝不替玩家做决定、绝不替玩家说话。"
    "判定流程：玩家的行动需要判定时，调用 roll_dice，点数由系统真实产生，"
    "必须如实向玩家报告点数，并按判定结果叙事——失败就编失败的后果，"
    "绝不许自己编点数或无视点数。"
    "防剧透：线索要埋在场景描述里让玩家自己发现，绝不直接念出 clues 的内容；"
    "结局的判定条件你知道，但不要向玩家透露。"
    "记录：玩家获得关键道具、触发关键事件、发现重要线索时，调用 record_fact 写进世界状态；"
    "玩家明确移动到新场景时调用 advance_scene；达成某个结局条件时调用 end_game。"
    "线索揭示：当前场景的线索（clues）你知道，但绝不能直接念出来——只有当玩家的"
    "具体行动覆盖了该线索的 discover_hint，才把线索内容自然地埋进你的描述里；"
    "玩家的行动不够具体时就只给表象，引导他说得更具体。"
    "检索：玩家问及或前往你不掌握细节的场景、或需要某个 NPC 的秘密资料时，"
    "调用 lookup_scene 查询，不要凭印象编造。"
    "打断处理：玩家可能随时打断你的叙述。被打断后不要重播整段，"
    "直接接住玩家的新意图继续（比如「好，你停在门口——你贴上门板仔细听……」）。"
    "语音约束：你的所有回复都会被语音合成朗读出来。口语化叙述，单次回复"
    "不超过五六句话（开场白和结局可以稍长）；不要使用 Markdown、列表、表情符号。"
)


def dm_prompt(engine: GameEngine, memory: ConversationMemory | None = None) -> str:
    """DM system prompt 动态拼装（每轮重建）。

    结构（AGENTS.md §4.3）：基础规则 + 剧本片段 + 世界状态（结构化事实，
    永不淘汰）+ 滚动摘要（滑窗外的早期剧情）+ 当前场景 + NPC 名册。
    最近 N 轮原文由 respond 里的滑窗注入，不进 system prompt。
    """
    adv = engine.adventure
    scene = engine.current_scene()
    parts = [
        "【基础 DM 规则】\n" + _DM_BASE_RULES,
        "【主持人设】\n" + adv["dm_persona"],
        "【玩家角色背景】（玩家已知此背景，不用再念给他听）\n"
        + adv["player_background"],
        "【判定规则】\n" + adv["dice_rules"],
        "【结局走向】（达成条件时调用 end_game；不要直接向玩家透露这些条件）\n"
        + "\n".join(f"- {e['id']}: {e['condition']}" for e in adv["endings"]),
        "【世界状态】\n事实: "
        + (json.dumps(engine.world_facts, ensure_ascii=False) or "{}")
        + "\n玩家状态: "
        + (json.dumps(engine.player_state, ensure_ascii=False) or "{}"),
    ]
    if scene is not None:
        scene_txt = (f"当前场景「{scene['title']}」(id={scene['id']}):\n"
                     f"{scene['description']}")
        if scene.get("exits"):
            exits = []
            for ex in scene["exits"]:
                target = engine.find_scene(ex)
                exits.append(f"{ex}（{target['title']}）" if target else ex)
            scene_txt += "\n可去的场景: " + "、".join(exits)
        if scene.get("clues"):
            clue_lines = "\n".join(
                f"- [{c['id']}] {c['content']}（揭示条件: {c['discover_hint']}）"
                for c in scene["clues"])
            scene_txt += ("\n本场景线索（DM 私信，绝不直接念出；玩家行动满足揭示条件"
                          "才埋进描述）:\n" + clue_lines)
        if scene.get("notes_for_dm"):
            scene_txt += f"\n（给 DM 的私信，不要念给玩家）: {scene['notes_for_dm']}"
        parts.append("【当前场景】\n" + scene_txt)
    if adv.get("npcs"):
        npc_lines = "\n".join(
            f"- {n['name']}：{n['personality']}；声线: {n.get('voice_hint', '无')}"
            f"；秘密（DM 私信，按剧本节奏透露）: "
            + "；".join(n.get("secrets", []))
            for n in adv["npcs"])
        parts.append("【NPC 名册】（扮演时恪守各自性格与秘密）\n" + npc_lines)
    if memory is not None and memory.summary:
        parts.append("【剧情回顾摘要】（早期剧情的压缩记录；细节可用 lookup_scene 补查）\n"
                     + memory.summary)
    if engine.state == ENDED:
        parts.append(
            "【本局已结束】结局已定，不要再推进新的剧情或判定。向玩家做完结局"
            "叙述后，简短复盘点评他本局的表现（做得好的地方、错过的关键线索），"
            "复盘完成后调用 back_to_lobby 工具结束整场冒险。")
    parts.append(
        "【对话历史说明】\n开场白之前的对话历史是玩家选剧本的过程（那时你是"
        "推荐员），从现在起你已经是主持人，不要再以推荐员自居。")
    return "\n\n".join(parts)


def _args_to_str(args) -> str:
    """assistant tool_calls 消息里的 arguments 字段：dict 重新序列化，原始字符串照用。"""
    if isinstance(args, dict):
        return json.dumps(args, ensure_ascii=False)
    return args or "{}"


class GameAgent:
    def __init__(self, cfg: dict, llm):
        game_cfg = cfg.get("game", {})
        self.adventures = load_adventures(
            game_cfg.get("adventures_dir", "adventures"))
        self.engine = GameEngine(self.adventures)
        self.llm = llm
        # 会被 bind_turn_manager 替换为 TurnManager.history（同一列表对象）
        self.history: list = []
        self.memory = ConversationMemory(
            self.history,
            window_turns=game_cfg.get("window_turns", 18),
            summary_interval_turns=game_cfg.get("summary_interval_turns", 15))
        self._narration_hook = lambda on: None   # bind 后接 TurnManager
        self.engine.on_session_reset = self._on_session_reset

    def bind_turn_manager(self, tm) -> None:
        """接线（AGENTS.md §2.2）：共享 history、打断模式回调、局间清理。"""
        self.history = tm.history
        self.memory.history = tm.history
        self._narration_hook = tm.set_narration
        tm.responder = self.respond

    def _on_session_reset(self) -> None:
        """一局结束回 lobby：清空 history 只留 session_note，避免跨局污染。"""
        self.memory.reset()
        self.history.clear()

    def system_prompt(self) -> str:
        """按引擎状态返回当前 system prompt（作为 callable 传给 TurnManager）。"""
        if self.engine.state == LOBBY:
            prompt = LOBBY_PROMPT
            if self.engine.session_note:
                prompt += ("\n【上一局回顾】" + self.engine.session_note +
                           "可以自然地提一句（比如他上局的表现），但不要展开细节。")
            return prompt
        return dm_prompt(self.engine, self.memory)

    async def respond(self, messages: list[dict],
                      gen_id: int) -> AsyncIterator[str]:
        stream_events = getattr(self.llm, "stream_events", None)
        if stream_events is None:
            # LLM 不支持 function calling（如 mock）：退化为纯流式，不用工具
            async for tok in self.llm.stream(messages, gen_id):
                yield tok
            return
        # 三层记忆：滑窗只取最近 N 轮原文进 prompt（tool 配对不拆）；
        # 窗口外的旧回合到间隔后异步压缩进滚动摘要
        self.memory.maybe_summarize(self.llm)
        msgs = [messages[0]] + self.memory.windowed()
        for _ in range(MAX_TOOL_ROUNDS):
            if self.engine.state == LOBBY:
                tools = LOBBY_TOOLS
            elif self.engine.state == ENDED:
                tools = SESSION_TOOLS
            else:
                tools = DM_TOOLS
            content_parts: list[str] = []
            tool_calls = None
            async for ev in stream_events(msgs, gen_id, tools=tools):
                if isinstance(ev, str):
                    content_parts.append(ev)
                    yield ev
                else:
                    tool_calls = ev["tool_calls"]
            if not tool_calls:
                return
            # 先收齐所有工具结果，再一次原子 append——history 里出现没有
            # tool 结果跟进的 assistant tool_calls 消息会让 API 400
            pending = []
            prompt_dirty = False
            for i, tc in enumerate(tool_calls):
                args = tc["arguments"] if isinstance(tc["arguments"], dict) else {}
                result = execute_tool(self.engine, tc["name"], args)
                print(f"[game] 工具 {tc['name']}("
                      f"{json.dumps(args, ensure_ascii=False)}) → "
                      f"{json.dumps(result, ensure_ascii=False)[:200]}")
                pending.append((tc, i, result))
                if result.get("ok") and tc["name"] in (
                        "start_adventure", "advance_scene", "end_game",
                        "back_to_lobby"):
                    prompt_dirty = True
                    if tc["name"] != "back_to_lobby":
                        # 开场白/场景描述/结局 = 长段叙述：提高打断确认阈值（§5）
                        self._narration_hook(True)
            extra = [{
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": tc["id"] or f"call_{gen_id}_{i}",
                    "type": "function",
                    "function": {"name": tc["name"],
                                 "arguments": _args_to_str(tc["arguments"])},
                } for tc, i, _ in pending],
            }] + [{
                "role": "tool",
                "tool_call_id": tc["id"] or f"call_{gen_id}_{i}",
                "content": json.dumps(result, ensure_ascii=False),
            } for tc, i, result in pending]
            msgs.extend(extra)
            self.history.extend(extra)
            if prompt_dirty:
                # 状态已迁移（开本/结局/回 lobby）：替换 system prompt，
                # 让续写的 completion 立即用新身份与新场景信息
                msgs[0] = {"role": "system", "content": self.system_prompt()}
