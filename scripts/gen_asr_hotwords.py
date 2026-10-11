"""从剧本库生成 sherpa ASR 热词表（configs/asr_hotwords.txt）。

热词解决语音链路的同音错字（「东盟夜雨」→「龙门夜雨」实锤）：
- 所有剧本标题、NPC 名字（含括号别名拆出的各段）；
- 跑团高频术语。
格式：每行一个词，中文字符空格分隔（该模型字级 token，sherpa-onnx
热词匹配按 token 序列）。热词只在 modified_beam_search 解码下生效，
配置见 configs/*.json 的 sherpa_asr 段。

用法：python scripts/gen_asr_hotwords.py
"""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "configs" / "asr_hotwords.txt"

# 跑团高频术语（剧本内容之外、玩家常说的）
TERMS = [
    "跑团", "剧本", "主持人", "掷骰", "判定", "开场白", "复盘",
    "冒险", "悬疑", "恐怖", "奇幻", "武侠", "科幻", "赛博朋克",
    "克苏鲁", "废土", "喜剧", "难度", "入门", "进阶",
    "换一本", "换一个", "重新开始", "回到大厅", "结束这局",
    "我要玩", "观察", "搜索", "撬锁", "潜行", "战斗", "逃跑",
]


def _words() -> list[str]:
    words: list[str] = []
    for fp in sorted((ROOT / "adventures").glob("*.json")):
        adv = json.loads(fp.read_text(encoding="utf-8"))
        words.append(adv["title"])
        for npc in adv.get("npcs", []):
            name = npc["name"]
            # 「柳三弦（说书人）」→ 柳三弦、说书人；「双刀」薇拉 → 双刀、薇拉
            for part in re.split(r"[（）()「」『』·]", name):
                part = part.strip()
                if len(part) >= 2:
                    words.append(part)
    words += TERMS
    # 去重保序；去掉含 ASCII/空白等不适合做字级热词的项
    seen: set[str] = set()
    out: list[str] = []
    for w in words:
        # 模型 tokens.txt 无「·」等符号，含非汉字的词会被热词编码器跳过
        if w in seen or not re.fullmatch(r"[一-鿿]+", w):
            continue
        seen.add(w)
        out.append(w)
    return out


def main() -> None:
    words = _words()
    lines = [" ".join(w) for w in words]  # 字级 token：字符空格分隔
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{OUT.relative_to(ROOT)}: {len(lines)} 个热词")
    for w in words[:10]:
        print(" ", w)


if __name__ == "__main__":
    sys.exit(main())
