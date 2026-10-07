"""工具定义与执行（AGENTS.md §4.2，OpenAI function calling 格式）。

骰子点数必须由代码真实随机产生——这是防作弊硬要求，AI 不能自己编骰点。
所有工具执行失败返回 {"error": "人类可读的说明"}，不抛异常到 LLM 侧。
"""

from __future__ import annotations

import json
import random
import re

from game.script_loader import find_adventure

LOBBY_TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "list_adventures",
            "description": "列出店里目前所有可选剧本（题材、时长、难度、一句话卖点）",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "describe_adventure",
            "description": "查看某个剧本的详细介绍和玩家角色背景",
            "parameters": {
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "剧本 id 或标题"},
                },
                "required": ["id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "start_adventure",
            "description": "玩家明确选定剧本后，开始这场冒险（切换为主持人模式）",
            "parameters": {
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "剧本 id 或标题"},
                },
                "required": ["id"],
            },
        },
    },
]

DM_TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "roll_dice",
            "description": "掷骰判定。点数由系统真实随机产生，必须如实向玩家报告并按结果叙事",
            "parameters": {
                "type": "object",
                "properties": {
                    "spec": {"type": "string",
                             "description": "骰子表达式，如 1d20、1d20+3、2d6-1"},
                    "purpose": {"type": "string",
                                "description": "判定目的，如「撬锁」「察觉」"},
                },
                "required": ["spec", "purpose"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lookup_scene",
            "description": "按需检索剧本中某个场景的细节（含线索）或某个 NPC 的资料",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string",
                              "description": "场景 id/标题或 NPC 名字"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "record_fact",
            "description": "把重要事实写进世界状态（如「玩家拿到了书房钥匙」），之后永不遗忘",
            "parameters": {
                "type": "object",
                "properties": {
                    "key": {"type": "string", "description": "事实名称"},
                    "value": {"type": "string", "description": "事实内容"},
                },
                "required": ["key", "value"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "advance_scene",
            "description": "玩家移动到新场景时切换当前场景",
            "parameters": {
                "type": "object",
                "properties": {
                    "scene_id": {"type": "string", "description": "目标场景 id"},
                },
                "required": ["scene_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "end_game",
            "description": "达成某个结局的条件时结束本局，播放结局叙述",
            "parameters": {
                "type": "object",
                "properties": {
                    "ending_id": {"type": "string", "description": "结局 id"},
                },
                "required": ["ending_id"],
            },
        },
    },
]

SESSION_TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "back_to_lobby",
            "description": "结局叙述和复盘都完成后调用，结束整场冒险，回到选本模式",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]

_DICE_RE = re.compile(r"^\s*(\d+)\s*[dD]\s*(\d+)\s*(?:([+-])\s*(\d+))?\s*$")

START_INSTRUCTION = (
    "剧本已切换成功。你现在就是该剧本的主持人。立即以主持人身份、"
    "用下面 opening 字段的原文向玩家开场（可以口语化分段念出，"
    "但不要删减关键信息、不要概括省略——这是精心准备的开场白）。")


def _roll_dice(spec: str, purpose: str) -> dict:
    m = _DICE_RE.match(spec or "")
    if not m:
        return {"error": f"无法解析骰子表达式 {spec!r}（支持 NdM、NdM+K、NdM-K）"}
    n, faces = int(m.group(1)), int(m.group(2))
    if not (1 <= n <= 100) or faces < 2:
        return {"error": f"骰子表达式超出合理范围: {spec!r}"}
    rolls = [random.randint(1, faces) for _ in range(n)]
    modifier = 0
    if m.group(3):
        modifier = int(m.group(4)) * (1 if m.group(3) == "+" else -1)
    return {"spec": spec, "rolls": rolls, "modifier": modifier,
            "total": sum(rolls) + modifier, "purpose": purpose}


def _describe(adv: dict, detail: bool) -> dict:
    out = {"id": adv["id"], "title": adv["title"], "genre": adv["genre"],
           "duration_min": adv["duration_min"], "difficulty": adv["difficulty"],
           "synopsis": adv["synopsis"]}
    if detail:
        out["player_background"] = adv["player_background"]
    return out


def execute_tool(engine, name: str, args: dict) -> dict:
    """分发执行工具，返回可 JSON 序列化的结果 dict。失败返回 {"error": ...}。"""
    try:
        if name == "list_adventures":
            return {"adventures": [_describe(a, detail=False)
                                   for a in engine.adventures.values()]}
        if name == "describe_adventure":
            adv = find_adventure(engine.adventures, str(args.get("id", "")))
            if adv is None:
                return {"error": f"没有找到剧本 {args.get('id')!r}，"
                                 f"请先用 list_adventures 查看可选剧本"}
            return _describe(adv, detail=True)
        if name == "start_adventure":
            adv = find_adventure(engine.adventures, str(args.get("id", "")))
            if adv is None:
                return {"error": f"没有找到剧本 {args.get('id')!r}，"
                                 f"请先用 list_adventures 查看可选剧本"}
            engine.start(adv["id"])
            return {"ok": True, "id": adv["id"], "title": adv["title"],
                    "opening": adv["opening"],
                    "instruction": START_INSTRUCTION}
        if name == "roll_dice":
            return _roll_dice(str(args.get("spec", "")),
                              str(args.get("purpose", "")))
        if name == "lookup_scene":
            found = engine.lookup_scene(str(args.get("query", "")))
            if found is None:
                return {"error": f"没有找到匹配的场景或 NPC: "
                                 f"{args.get('query')!r}"}
            return found
        if name == "record_fact":
            engine.record_fact(str(args.get("key", "")), args.get("value"))
            return {"ok": True}
        if name == "advance_scene":
            scene = engine.advance_scene(str(args.get("scene_id", "")))
            return {"ok": True, "scene_id": scene["id"], "title": scene["title"],
                    "description": scene["description"]}
        if name == "end_game":
            ending = engine.end(str(args.get("ending_id", "")))
            return {"ok": True, "ending_id": ending["id"],
                    "narration": ending["narration"],
                    "instruction": "结局已达成。向玩家叙述结局（可口语化改写），"
                                   "然后简短复盘点评玩家本局表现。"}
        if name == "back_to_lobby":
            engine.reset_to_lobby()
            return {"ok": True,
                    "instruction": "你已回到桌游店老板身份。用一两句话跟玩家"
                                   "打个招呼（可以自然地点评一句他上一局的表现，"
                                   "但不要剧透其他剧本），然后等玩家选下一本或道别。"}
        return {"error": f"未知工具: {name}"}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


def tool_result_str(result: dict) -> str:
    return json.dumps(result, ensure_ascii=False)
