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
import re
from typing import AsyncIterator

from game.engine import GameEngine, LOBBY, PLAYING, ENDED
from game.memory import ConversationMemory
from game.script_loader import load_adventures
from game.tools import DM_TOOLS, LOBBY_TOOLS, SESSION_CONTROL_TOOLS, execute_tool

MAX_TOOL_ROUNDS = 5      # 防死循环
MAX_EMPTY_RETRIES = 2    # LLM 空回复（打断后易发）自动重试次数

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
    "玩家明确移动到新场景时调用 advance_scene。"
    "结局判定：只要剧情走到任何一个结局（包括玩家被逐出、死亡、放弃等失败结局），"
    "必须先调用 end_game 再叙述结局——只在叙述里宣布「你触发了结局」而不调用工具，"
    "这局在系统里就没有真正结束。"
    "会话控制：玩家想换剧本、重开当前剧本、或中途离场时，必须调用 start_adventure"
    "（换本/重开）或 back_to_lobby（回选本大厅）。只有调用工具才能真正切换，"
    "口头答应不算数；在工具调用成功前，绝不许编造其他剧本的开场或内容。"
    "线索揭示：当前场景的线索（clues）你知道，但绝不能直接念出来——只有当玩家的"
    "具体行动覆盖了该线索的 discover_hint，才把线索内容自然地埋进你的描述里；"
    "玩家的行动不够具体时就只给表象，引导他说得更具体。"
    "检索：玩家问及或前往你不掌握细节的场景、或需要某个 NPC 的秘密资料时，"
    "调用 lookup_scene 查询，不要凭印象编造。"
    "打断处理：玩家可能随时打断你的叙述。被打断后不要重播整段，"
    "直接接住玩家的新意图继续（比如「好，你停在门口——你贴上门板仔细听……」）。"
    "角色配音：系统会给不同角色配不同声线。NPC 开口说话时，在他的台词开头"
    "紧挨着写 ⟦s:NPC名字 e:情绪⟧（名字严格用名册里的名字，标记后不换行直接跟台词），"
    "台词必须用引号完整包住（「」或“”），说完及时合上引号；台词中间插了叙述、"
    "NPC 接着再说时，新引号里的台词不用重复标记。"
    "情绪从以下词里选最贴合当前语境的一个：{emotions}。不要编造名册之外的 NPC。"
    "情绪要连贯：同一个 NPC 在连续几句台词里沿用同一个情绪词，"
    "只有剧情明显转折才换，不要每句都换。标记只写 ⟦s:名字 e:情绪⟧ 这一种形式，"
    "不要写 ⟦v:...⟧ 或其他变体。"
    "旁白情绪：你的叙述旁白默认不写任何标记，会自动沿用当前的叙述氛围；"
    "只有当剧情氛围明显转变时（如发现尸体、危机降临、真相揭晓、转危为安），"
    "才在旁白段落开头写 ⟦s:旁白 e:情绪⟧ 切换氛围；同一氛围内不要每段都换，"
    "氛围恢复平常时用 ⟦s:旁白 e:平静⟧ 收回来。"
    "语音约束：你的所有回复都会被语音合成朗读出来。口语化叙述，单次回复"
    "不超过五六句话（开场白和结局可以稍长）；不要使用 Markdown、列表、表情符号。"
)

# 默认情绪词表；configs/trpg.json 的 cosyvoice.emotions 会覆盖它
# （那里是单一事实来源，TTS warmup 也按它预建模板）
_DEFAULT_EMOTIONS = ["平静", "紧张", "恐惧", "愤怒", "悲伤", "神秘",
                     "激动", "低语", "恭敬", "犹豫", "温柔", "威严"]


def dm_prompt(engine: GameEngine, memory: ConversationMemory | None = None,
              emotions: list | None = None) -> str:
    """DM system prompt 动态拼装（每轮重建）。

    结构（AGENTS.md §4.3）：基础规则 + 剧本片段 + 世界状态（结构化事实，
    永不淘汰）+ 滚动摘要（滑窗外的早期剧情）+ 当前场景 + NPC 名册。
    最近 N 轮原文由 respond 里的滑窗注入，不进 system prompt。
    """
    base_rules = _DM_BASE_RULES.format(
        emotions="、".join(emotions or _DEFAULT_EMOTIONS))
    adv = engine.adventure
    scene = engine.current_scene()
    parts = [
        "【基础 DM 规则】\n" + base_rules,
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
            "复盘完成后调用 back_to_lobby 工具结束整场冒险。如果玩家急着再开一局"
            "（同一本或换一本），直接调用 start_adventure，不必先回大厅。")
    parts.append(
        "【对话历史说明】\n开场白之前的对话历史是玩家选剧本的过程（那时你是"
        "推荐员），从现在起你已经是主持人，不要再以推荐员自居。")
    return "\n\n".join(parts)


def _args_to_str(args) -> str:
    """assistant tool_calls 消息里的 arguments 字段：dict 重新序列化，原始字符串照用。"""
    if isinstance(args, dict):
        return json.dumps(args, ensure_ascii=False)
    return args or "{}"


_TAG_V = re.compile(r"\bv:([\w一-鿿·'-]+)")
_TAG_S = re.compile(r"\bs:([^⟧\s]+)")
_TAG_E = re.compile(r"\be:([^⟧]+)")


def _npc_aliases(full_name: str) -> set[str]:
    """「周世海（老管家）」→ {全名, 周世海, 老管家}，覆盖 LLM 的各种写法。"""
    aliases = {full_name}
    m = re.match(r"^(.+?)（(.+?)）$", full_name)
    if m:
        aliases.update((m.group(1), m.group(2)))
    return aliases


# ⟦s:旁白 e:情绪⟧ 的名字别名：映射到旁白声线（DM 规则约定旁白氛围切换标记）
_NARRATOR_ALIASES = {"旁白", "叙述", "旁白者", "解说", "narrator"}


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
        # 情绪词表：与 TTS warmup 共用 configs 里的单一事实来源
        self.emotions = cfg.get("cosyvoice", {}).get("emotions")
        # 旁白声线 id：⟦s:旁白 e:情绪⟧ 改写成它（剧本 narrator_voice 优先）
        self._cfg_narrator_voice = cfg.get("cosyvoice", {}).get(
            "default_voice", "narrator_m")
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
        return dm_prompt(self.engine, self.memory, self.emotions)

    # ---------- 多音色标记改写（⟦s:NPC名 e:情绪⟧ → ⟦v:声线id e:情绪⟧） ----------

    def _npc_voice(self, name: str) -> str | None:
        adv = self.engine.adventure
        if adv is None:
            return None
        for n in adv.get("npcs", []):
            if name in _npc_aliases(n["name"]):
                return n.get("voice")
        # 模糊兜底：互为子串（『红鼻』杰克 vs 杰克）
        for n in adv.get("npcs", []):
            if len(name) >= 2 and name in n["name"]:
                return n.get("voice")
        return None

    def _rewrite_tag(self, body: str) -> str:
        em = _TAG_E.search(body)
        emotion = f" e:{em.group(1).strip()}" if em else ""
        vm = _TAG_V.search(body)
        if vm:
            return f"⟦v:{vm.group(1)}{emotion}⟧"
        sm = _TAG_S.search(body)
        if not sm:
            return ""                      # 无法识别的标记：剥掉，防念出
        name = sm.group(1).strip()
        if name in _NARRATOR_ALIASES:
            # 旁白氛围标记 → 旁白声线（剧本 narrator_voice 优先于全局默认）
            adv = self.engine.adventure
            voice = (adv or {}).get("narrator_voice") or self._cfg_narrator_voice
            return f"⟦v:{voice}{emotion}⟧"
        voice = self._npc_voice(name)
        return f"⟦v:{voice}{emotion}⟧" if voice else ""

    async def _voice_rewrite(self, stream) -> AsyncIterator:
        """流式过滤器：把 LLM 写的 ⟦s:...⟧ 标记改写成声线 id，未闭合标记丢弃。

        非 str 事件（tool_calls 字典）原样透传；改写后的文本才进 history，
        与玩家实际听到的一致。
        """
        buf = ""
        in_tag = False
        async for ev in stream:
            if not isinstance(ev, str):
                buf, in_tag = "", False
                yield ev
                continue
            out = []
            for ch in ev:
                if in_tag:
                    if ch == "⟧":
                        out.append(self._rewrite_tag(buf))
                        buf, in_tag = "", False
                    else:
                        buf += ch
                elif ch == "⟦":
                    if buf:
                        out.append(buf)
                        buf = ""
                    in_tag = True
                else:
                    buf += ch
            if buf and not in_tag:
                out.append(buf)
                buf = ""
            if out:
                yield "".join(out)
        if buf and not in_tag:
            yield buf

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
        empty_retries = 0
        for _ in range(MAX_TOOL_ROUNDS):
            if self.engine.state == LOBBY:
                tools = LOBBY_TOOLS
            elif self.engine.state == ENDED:
                tools = SESSION_CONTROL_TOOLS
            else:
                # 对局中途也要能换本/重开/离场，否则玩家会被困在当前剧本
                tools = DM_TOOLS + SESSION_CONTROL_TOOLS
            content_parts: list[str] = []
            tool_calls = None
            async for ev in self._voice_rewrite(
                    stream_events(msgs, gen_id, tools=tools)):
                if isinstance(ev, str):
                    content_parts.append(ev)
                    yield ev
                else:
                    tool_calls = ev["tool_calls"]
            if not tool_calls:
                # 空回复（无正文无工具调用）：打断后的下一轮容易触发，
                # 不逐字重试玩家只能面对沉默。temperature=1 下重试是新采样
                if not "".join(content_parts).strip() \
                        and empty_retries < MAX_EMPTY_RETRIES:
                    empty_retries += 1
                    print(f"[game] LLM 空回复，重试第 {empty_retries} 次")
                    continue
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
