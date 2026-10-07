"""剧本加载与 schema 校验（AGENTS.md §3.2）。

缺字段直接 ValueError（带文件名和缺失字段路径），不做静默降级；
同时校验 scene exits 不悬空、ending id 唯一。
"""

from __future__ import annotations

import difflib
import json
from pathlib import Path

REQUIRED_FIELDS = ("id", "title", "genre", "duration_min", "difficulty",
                   "synopsis", "player_background", "dm_persona", "opening",
                   "world", "scenes", "npcs", "endings", "dice_rules")
WORLD_REQUIRED = ("facts", "player_state")
SCENE_REQUIRED = ("id", "title", "description", "clues", "exits")
CLUE_REQUIRED = ("id", "content", "discover_hint")
NPC_REQUIRED = ("name", "personality", "secrets")
ENDING_REQUIRED = ("id", "condition", "narration")


def _require(obj: dict, fields: tuple, where: str) -> None:
    missing = [f for f in fields if f not in obj]
    if missing:
        raise ValueError(f"{where}: 缺少必需字段 {', '.join(missing)}")


def _validate(adv: dict, fname: str) -> None:
    _require(adv, REQUIRED_FIELDS, f"{fname}")
    _require(adv["world"], WORLD_REQUIRED, f"{fname}: world")
    if not adv["scenes"]:
        raise ValueError(f"{fname}: scenes 不能为空")
    scene_ids = set()
    for i, scene in enumerate(adv["scenes"]):
        where = f"{fname}: scenes[{i}]"
        _require(scene, SCENE_REQUIRED, where)
        scene_ids.add(scene["id"])
        for j, clue in enumerate(scene["clues"]):
            _require(clue, CLUE_REQUIRED, f"{where}.clues[{j}]")
    for i, scene in enumerate(adv["scenes"]):
        for ex in scene["exits"]:
            if ex not in scene_ids:
                raise ValueError(
                    f"{fname}: scenes[{i}].exits 指向不存在的场景 {ex!r}")
    for i, npc in enumerate(adv["npcs"]):
        _require(npc, NPC_REQUIRED, f"{fname}: npcs[{i}]")
    ending_ids: set = set()
    for i, ending in enumerate(adv["endings"]):
        _require(ending, ENDING_REQUIRED, f"{fname}: endings[{i}]")
        if ending["id"] in ending_ids:
            raise ValueError(f"{fname}: ending id 重复 {ending['id']!r}")
        ending_ids.add(ending["id"])


def load_adventures(dir_path) -> dict[str, dict]:
    """加载目录下所有 .json 剧本，按 id 索引。目录不存在/为空抛 ValueError。"""
    d = Path(dir_path)
    if not d.is_dir():
        raise ValueError(f"剧本目录不存在: {d}")
    files = sorted(d.glob("*.json"))
    if not files:
        raise ValueError(f"剧本目录为空（没有 .json 文件）: {d}")
    adventures: dict[str, dict] = {}
    for fp in files:
        try:
            with open(fp, encoding="utf-8") as fh:
                adv = json.load(fh)
        except json.JSONDecodeError as e:
            raise ValueError(f"{fp.name}: JSON 解析失败: {e}") from e
        _validate(adv, fp.name)
        if adv["id"] in adventures:
            raise ValueError(f"{fp.name}: 剧本 id 重复 {adv['id']!r}")
        adventures[adv["id"]] = adv
    return adventures


def find_adventure(adventures: dict[str, dict], query: str) -> dict | None:
    """容错查找：先精确 id，再精确 title，再子串模糊匹配，最后相似度匹配。

    相似度兜底是为了接住 ASR 同音错字（如「庄园卫影」→「庄园魅影」）——
    语音链路里玩家说剧本名必然经过 ASR，逐字匹配一定会被同音字打败。
    """
    q = (query or "").strip()
    if not q:
        return None
    if q in adventures:
        return adventures[q]
    for adv in adventures.values():
        if adv["title"] == q:
            return adv
    for adv in adventures.values():
        if q in adv["id"] or q in adv["title"] or adv["title"] in q:
            return adv
    best, best_ratio = None, 0.0
    for adv in adventures.values():
        for cand in (adv["id"], adv["title"]):
            ratio = difflib.SequenceMatcher(None, q, cand).ratio()
            if ratio > best_ratio:
                best, best_ratio = adv, ratio
    return best if best_ratio >= 0.6 else None
