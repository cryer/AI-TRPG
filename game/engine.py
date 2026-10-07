"""游戏引擎（AGENTS.md §4.1）：状态机 + 世界状态。

核心原则：游戏规则进代码，叙事扮演进 prompt。骰子点数、场景切换、
事实记录都由这里执行，LLM 只负责叙事与判定意图。
"""

from __future__ import annotations

import copy

LOBBY = "lobby"
PLAYING = "playing"
ENDED = "ended"


class GameEngine:
    def __init__(self, adventures: dict[str, dict]):
        self.adventures = adventures
        self.players: list = []          # 多人房预留（M1 只有一个玩家）
        self.state = LOBBY
        self.adventure: dict | None = None
        self.world_facts: dict = {}
        self.player_state: dict = {}
        self.scene_id: str | None = None

    # ---------- 状态迁移 ----------

    def start(self, adventure_id: str) -> dict:
        adv = self.adventures.get(adventure_id)
        if adv is None:
            raise KeyError(f"剧本不存在: {adventure_id!r}")
        self.adventure = adv
        self.world_facts = copy.deepcopy(adv["world"].get("facts", {}))
        self.player_state = copy.deepcopy(adv["world"].get("player_state", {}))
        self.scene_id = adv["scenes"][0]["id"]
        self.state = PLAYING
        return adv

    def end(self, ending_id: str) -> dict:
        if self.adventure is None:
            raise KeyError("当前没有进行中的剧本")
        for ending in self.adventure["endings"]:
            if ending["id"] == ending_id:
                self.state = ENDED  # M1 暂不自动回 lobby
                return ending
        raise KeyError(f"结局不存在: {ending_id!r}")

    def reset_to_lobby(self) -> None:
        self.state = LOBBY
        self.adventure = None
        self.world_facts = {}
        self.player_state = {}
        self.scene_id = None

    # ---------- 世界状态 ----------

    def record_fact(self, key: str, value) -> None:
        """结构化事实永不淘汰（AGENTS.md §4.4）：玩家 40 分钟前拿的钥匙必须还记得。"""
        self.world_facts[key] = value

    def advance_scene(self, scene_id: str) -> dict:
        scene = self.find_scene(scene_id)
        if scene is None:
            raise KeyError(f"场景不存在: {scene_id!r}")
        self.scene_id = scene["id"]
        return scene

    def current_scene(self) -> dict | None:
        if self.adventure is None or self.scene_id is None:
            return None
        return self.find_scene(self.scene_id)

    def find_scene(self, scene_id: str) -> dict | None:
        if self.adventure is None:
            return None
        for scene in self.adventure["scenes"]:
            if scene["id"] == scene_id:
                return scene
        return None

    def lookup_scene(self, query: str) -> dict | None:
        """按 scene id/title 子串匹配查场景，也支持查 NPC 名字。找不到返回 None。"""
        if self.adventure is None:
            return None
        q = (query or "").strip()
        if not q:
            return None
        for scene in self.adventure["scenes"]:
            if q in scene["id"] or q in scene["title"] or scene["title"] in q:
                return scene
        for npc in self.adventure["npcs"]:
            if q in npc["name"] or npc["name"] in q:
                return npc
        return None
