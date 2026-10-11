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
MAX_CONTINUE_RETRIES = 2  # max_tokens 截断（finish_reason=length）自动续写次数

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
    "角色配音（重要）：系统会给不同角色配不同声线。NPC 开口说话时，必须在"
    "台词开头紧挨着写 ⟦s:NPC名字 e:情绪⟧——名字严格用名册里的名字，"
    "台词用引号（「」或“”）完整包住。"
    "示例——燕三娘紧张地说：⟦s:燕三娘 e:紧张⟧「走你的镖，少管闲事。」"
    "不要在标记外面再单写一个说话人名字当标签（「燕三娘⟦s:…⟧」这种形式"
    "会让玩家听到名字被念出来）。"
    "不写标记的台词玩家只能用旁白声线听到，所以每个 NPC 每次开口都要标记。"
    "情绪从以下词里选最贴合语境的一个：{emotions}；"
    "同一个 NPC 连续几句台词沿用同一个情绪词，剧情明显转折才换。"
    "标记只写 ⟦s:名字 e:情绪⟧ 这一种形式，不要写 ⟦v:...⟧ 或其他变体。"
    "旁白默认不写任何标记，会自动沿用当前叙述氛围；只在剧情氛围明显转变时"
    "（发现尸体、危机降临、真相揭晓、转危为安）在旁白段落开头写"
    "⟦s:旁白 e:情绪⟧ 切换氛围，氛围恢复平常时用 ⟦s:旁白 e:平静⟧ 收回来。"
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
        parts.append("【NPC 名册】（扮演时恪守各自性格与秘密；任何 NPC 开口说话时，"
                     "台词必须以 ⟦s:名册名字 e:情绪⟧ 开头，否则玩家听到的是旁白声线）\n"
                     + npc_lines)
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


# 未登记名字的性别/年龄关键词兜底：LLM 偶尔用叙述称呼写标记（⟦s:青衣女子⟧），
# 名册查不到就按称呼关键词猜一个声线，总比剥掉后掉成旁白强（真人局实锤：
# 整局标记全被剥光，所有 NPC 都变成了旁白音色）
_FEMALE_HINTS = ("女", "娘", "姐", "妹", "婆", "姑", "妃", "公主", "寡妇", "夫人", "小姐")
_OLD_HINTS = ("老", "翁", "叟", "丐", "爷", "伯", "掌柜", "长者")
_YOUNG_HINTS = ("少", "孩", "童", "小子", "青年", "姑娘", "丫头", "小姐")


def _guess_voice(name: str) -> str | None:
    if any(k in name for k in _FEMALE_HINTS):
        return ("npc_female_young" if any(k in name for k in _YOUNG_HINTS)
                else "npc_female")
    if any(k in name for k in _OLD_HINTS):
        return "npc_male_old"
    if any(k in name for k in _YOUNG_HINTS):
        return "npc_male_young"
    return None


# ⟦s:旁白 e:情绪⟧ 的名字别名：映射到旁白声线（DM 规则约定旁白氛围切换标记）
_NARRATOR_ALIASES = {"旁白", "叙述", "旁白者", "解说", "narrator"}

# ---------- 确定性引号台词归属 ----------
# LLM 经常整局不写 ⟦s:⟧ 标记（真人局多次实锤），prompt 约束不可靠。
# 这里在流式改写层兜底：开引号前若紧挨着的文本里出现「名册 NPC 名 + 说话
# 动词/冒号」（如「裴千户尖着嗓子喝道：」），自动注入该 NPC 的声线标记。
_QUOTE_OPEN_CHARS = "「『“"
_QUOTE_CLOSE_CHARS = "」』”"
_QUOTE_TOGGLE_CHARS = "\"＂"          # 直引号：同一字符开合切换
_SPEECH_VERB_RE = re.compile(
    r"(说|道|喊|喝|问|答|笑|骂|嚷|叹|叫|嘟囔|咕哝|嘀咕|呢喃|喃喃|低语|低声|"
    r"轻声|惊呼|尖叫|开口|出声|斥责|安慰|告诉|大喝|厉声)")
_ATTR_TAIL_CHARS = 24     # 在开引号前最多回看这么远找 NPC 名
_ATTR_TAG_GAP = 4         # 开引号前这么近内已有标记 → 不再注入


class _QuoteAttributor:
    """逐字符走查已改写文本，给无标记的引号台词自动归属 NPC 声线。"""

    def __init__(self, agent: "GameAgent"):
        self._agent = agent
        self._tail: list[str] = []        # 最近发出的普通字符（上限 64）
        self._since_tag = 999             # 距上一个标记之间隔了多少字符
        self._in_straight = False         # 直引号开合状态

    def _names(self) -> list[tuple[str, str]]:
        adv = self._agent.engine.adventure
        if adv is None:
            return []
        out = []
        for n in adv.get("npcs", []):
            voice = n.get("voice")
            if not voice:
                continue
            for alias in _npc_aliases(n["name"]):
                if len(alias) >= 2:
                    out.append((alias, voice))
        # 长名优先，避免「三娘」抢在「燕三娘」前命中
        out.sort(key=lambda t: -len(t[0]))
        return out

    def _attribute(self) -> str | None:
        tail = "".join(self._tail[-_ATTR_TAIL_CHARS:])
        best_pos, best_voice = -1, None
        for alias, voice in self._names():
            pos = tail.rfind(alias)
            if pos > best_pos:
                best_pos, best_voice = pos, voice
        if best_voice is None:
            return None
        # 名字之后要出现说话动词或冒号，否则可能只是提到 NPC（招牌「龙门客栈」）
        after = tail[best_pos:]
        if not (_SPEECH_VERB_RE.search(after) or "：" in after or ":" in after):
            return None
        return best_voice

    def feed(self, text: str) -> str:
        """处理一段已改写文本（标记之外的普通字符流），必要时注入 ⟦v:⟧。"""
        out: list[str] = []
        for ch in text:
            is_open = ch in _QUOTE_OPEN_CHARS
            if ch in _QUOTE_TOGGLE_CHARS:
                is_open = not self._in_straight
                self._in_straight = not self._in_straight
            if ch in _QUOTE_CLOSE_CHARS:
                self._in_straight = False
            if is_open and self._since_tag > _ATTR_TAG_GAP:
                voice = self._attribute()
                if voice:
                    print(f"[game] 无标记台词自动归属 → {voice}"
                          f"（上下文 …{''.join(self._tail[-12:])}）")
                    out.append(f"⟦v:{voice}⟧")
                    self._since_tag = 0
            out.append(ch)
            self._tail.append(ch)
            if len(self._tail) > 64:
                del self._tail[:32]
            self._since_tag += 1
        return "".join(out)

    def mark_tag(self) -> None:
        self._since_tag = 0


# ---------- NPC 身份标签剥离 ----------
# LLM 爱把台词写成「裴千户⟦s:…⟧「……」」「裴千户：「……」」——名字标签在
# 标记/引号外面，会被 TTS 原样念出来（真人局实锤）。确定性剥离规则：
# 句界（流开头/换行/句末标点）后、紧挨着标记或开引号的「名册名+冒号?」
# 是身份标签，从文本中删掉； narrative 里提及名字（裴千户守在门口）不满足
# 「句界 + 紧跟标记/引号」，不受影响。
_LABEL_BOUNDARY_CHARS = "\n。！？…!?"


def _label_names(adv: dict) -> list[str]:
    """名册全名/别名及其 ≥2 字纯中文后缀（陆文渊 ← 真钦差陆文渊、千户 ← 裴千户）。"""
    names: set[str] = set()
    for n in adv.get("npcs", []):
        for alias in _npc_aliases(n["name"]):
            for i in range(len(alias)):
                cand = alias[i:]
                if len(cand) >= 2 and re.fullmatch(r"[一-鿿]+", cand):
                    names.add(cand)
    return sorted(names, key=len, reverse=True)   # 长名优先匹配


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
        self._label_cache: tuple | None = None   # (adventure_id, strip_re, tail_re)

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
        if voice is None:
            voice = _guess_voice(name)
            if voice:
                print(f"[warn] NPC 标记 {name!r} 不在名册，按称呼关键词"
                      f"兜底声线 {voice}")
            else:
                print(f"[warn] NPC 标记 {name!r} 不在名册且无法猜测，标记被剥掉"
                      f"（该段台词将用旁白声线）")
        return f"⟦v:{voice}{emotion}⟧" if voice else ""

    def _label_regexes(self):
        """身份标签剥离用的 (strip_re, tail_re)；lobby/无名册时为 (None, None)。

        strip_re：句界后的「名册名+冒号?」紧跟标记或开引号 → 删标签；
        tail_re：文本尾巴上的「句界+名册名+冒号?」→ 先扣留，等下一个事件
        确认后面跟的是不是标记/引号（是则丢弃，否则放行）。
        """
        adv = self.engine.adventure
        if adv is None:
            return None, None
        if self._label_cache and self._label_cache[0] == adv["id"]:
            return self._label_cache[1], self._label_cache[2]
        names = _label_names(adv)
        if not names:
            strip_re = tail_re = None
        else:
            alt = "|".join(re.escape(x) for x in names)
            strip_re = re.compile(
                r'(?<=[' + _LABEL_BOUNDARY_CHARS + r'])\s*(?:' + alt +
                r')[：:]?(?=⟦|[「『“"＂])')
            tail_re = re.compile(
                r'(?<=[' + _LABEL_BOUNDARY_CHARS + r'])(\s*(?:' + alt +
                r')[：:]?)$')
        self._label_cache = (adv["id"], strip_re, tail_re)
        return strip_re, tail_re

    async def _voice_rewrite(self, stream) -> AsyncIterator:
        """流式过滤器：把 LLM 写的 ⟦s:...⟧ 标记改写成声线 id，未闭合标记丢弃。

        非 str 事件（tool_calls 字典）原样透传；改写后的文本才进 history，
        与玩家实际听到的一致。LLM 没写标记的引号台词由 _QuoteAttributor
        按「NPC 名 + 说话动词/冒号」确定性归属，不再依赖模型自觉。
        台词前的身份标签（裴千户⟦s:…⟧ / 裴千户：「…」）在此剥离，不念出来。
        """
        buf = ""
        in_tag = False
        attr = _QuoteAttributor(self)
        strip_re, tail_re = self._label_regexes()
        held = ""                 # 疑似身份标签的扣留尾巴（等下一事件确认）
        prev_boundary = True      # 上一产出字符是否句界（流开头视作句界）

        def _emit(text: str) -> str:
            """attr 归属 + 身份标签剥离/扣留，返回可立即产出的文本。"""
            nonlocal held, prev_boundary
            s = held + attr.feed(text)
            held = ""
            if not s:
                return ""
            if strip_re is not None:
                # 哨兵 \n：让句界 lookbehind 覆盖 s 开头（仅当上一产出在句界）
                if prev_boundary:
                    s = "\n" + s
                s = strip_re.sub("", s)
                m = tail_re.search(s)
                if m:
                    held = m.group(1)
                    s = s[:m.start(1)]
                if prev_boundary:
                    s = s[1:] if s.startswith("\n") else s
            if s:
                prev_boundary = s[-1] in _LABEL_BOUNDARY_CHARS
            return s

        async for ev in stream:
            if not isinstance(ev, str):
                # 工具调用事件：回合文本到此为止，扣留的尾巴不是标签，放行
                if held:
                    yield held
                    held = ""
                buf, in_tag = "", False
                yield ev
                continue
            out = []
            for ch in ev:
                if in_tag:
                    if ch == "⟧":
                        # 标签后面紧跟标记 → 确认是身份标签，丢弃扣留
                        held = ""
                        tag = self._rewrite_tag(buf)
                        if tag:
                            out.append(tag)
                            attr.mark_tag()
                            prev_boundary = False
                        buf, in_tag = "", False
                    else:
                        buf += ch
                elif ch == "⟦":
                    if buf:
                        out.append(_emit(buf))
                        buf = ""
                    in_tag = True
                else:
                    buf += ch
            if buf and not in_tag:
                out.append(_emit(buf))
                buf = ""
            if out:
                yield "".join(out)
        if buf and not in_tag:
            yield _emit(buf)
        if held:
            yield held

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
        continue_retries = 0
        full_text = ""
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
            finish_reason = None
            async for ev in self._voice_rewrite(
                    stream_events(msgs, gen_id, tools=tools)):
                if isinstance(ev, str):
                    content_parts.append(ev)
                    yield ev
                else:
                    tool_calls = ev.get("tool_calls")
                    finish_reason = ev.get("finish_reason")
            round_text = "".join(content_parts)
            full_text += round_text
            if not tool_calls:
                # 空回复（无正文无工具调用）：打断后的下一轮容易触发，
                # 不逐字重试玩家只能面对沉默。temperature=1 下重试是新采样
                if not round_text.strip() \
                        and empty_retries < MAX_EMPTY_RETRIES:
                    empty_retries += 1
                    print(f"[game] LLM 空回复，重试第 {empty_retries} 次")
                    continue
                # max_tokens 触顶截断：kimi-for-coding 强制 thinking 且
                # reasoning token 计入上限，长思考回合会把正文断在句子中间
                # （真人局实锤「燕三娘趁机」戛然而止）。续写只进本地 msgs，
                # 不进 history——整段回复由 TurnManager 汇总成一条 assistant
                # 消息入库，history 不受续写提示污染
                if finish_reason == "length" and round_text.strip() \
                        and continue_retries < MAX_CONTINUE_RETRIES:
                    continue_retries += 1
                    print(f"[game] LLM 回复触顶截断（finish_reason=length，"
                          f"已说 {len(full_text)} 字），自动续写第 "
                          f"{continue_retries} 次")
                    msgs = msgs + [
                        {"role": "assistant", "content": full_text},
                        {"role": "user", "content":
                         "（系统提示：你刚才的回复达到长度上限被截断了。请紧接"
                         "断点继续叙述，不要重复已经说过的内容，也不要回应这"
                         "条提示本身。）"}]
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
