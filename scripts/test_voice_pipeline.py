"""标记原子切分 / 跨句声线保持 / 句间合并 的离线验证（无需 GPU）。

覆盖三个真实对局暴露的 bug：
1. 分句器在 ⟦v:x e:恭敬⟧ 标记内部的冒号处切分 → 残缺标记被 TTS 念出；
2. NPC 多句台词只有首句带标记 → 后续句回落旁白声线；
3. 连续旁白逐句独立合成 → 语气/语速句间跳变（验证同 key 合并）。
"""

import asyncio
import sys

sys.path.insert(0, ".")
from voice.llm.splitter import split_sentences          # noqa: E402
from voice.tts.base import parse_tagged, strip_tags     # noqa: E402
from voice.tts.cosyvoice import CosyVoiceTTS            # noqa: E402

FAIL = 0


def check(name: str, cond: bool, detail: str = ""):
    global FAIL
    print(f"  [{'OK' if cond else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    if not cond:
        FAIL += 1


async def split_all(tokens: list[str], **kw) -> list[str]:
    async def gen():
        for t in tokens:
            yield t
    return [s async for s in split_sentences(gen(), **kw)]


async def main():
    print("== 1. 分句器：标记原子化 ==")
    # 逐 token 模拟 LLM 流：标记内含冒号，此前会在 e: 处被切
    text = "⟦v:npc_male_old e:恭敬⟧「您就是晚晴小姐请来的调查员吧，路上辛苦了。」门厅里很静，静得能听见挂钟的摆锤声。"
    sents = await split_all(list(text))
    check("标记未被切开", not any(("⟦" in s) != ("⟧" in s) for s in sents),
          detail=str(sents))
    check("标记与台词同句", any(s.startswith("⟦v:npc_male_old e:恭敬⟧「") for s in sents))
    # 无标点对局长句：硬切也要避开标记内部
    long_text = "⟦v:npc_x e:愤怒⟧" + "啊" * 60
    sents2 = await split_all(list(long_text))
    check("硬切避开标记内部", not any(("⟦" in s) != ("⟧" in s) for s in sents2),
          detail=str([s[:12] for s in sents2]))
    # 普通分句行为不回归
    sents3 = await split_all(list("雨下得很大。你推门进去，管家迎上来。"))
    check("硬断点切分", sents3[0] == "雨下得很大。", detail=str(sents3))

    print("== 2. 标记剥离兜底 ==")
    check("未闭合标记剥光", parse_tagged("⟦v:npc_male_old e:") == [])
    segs = parse_tagged("正文⟦v:x e:恭敬")
    check("尾部未闭合标记剥掉", segs == [("正文", "", None)], detail=str(segs))
    check("落单 ⟧ 剥掉", strip_tags("恭敬⟧「你好」") == "恭敬「你好」")

    print("== 3. _segment_stream：跨句声线 + 合并 ==")
    tts = CosyVoiceTTS()

    async def segments_of(sentences: list[str], trickle: bool = False):
        async def gen():
            for s in sentences:
                yield s
                if trickle:
                    await asyncio.sleep(0.2)   # 上游没就绪 → 不合并
        return [seg async for seg in tts._segment_stream(gen())]

    # 用户实锤场景：NPC 两句台词（引号跨句）→ 两句都应是 npc 声线
    segs = await segments_of([
        "⟦v:npc_male_old e:恭敬⟧「您就是晚晴小姐请来的调查员吧，",
        "路上辛苦了。」",
        "门厅里很静，",
        "静得能听见挂钟的摆锤声。",
    ])
    check("NPC 第二句继承声线", any(v == "npc_male_old" and "路上辛苦了" in t
                                    for t, v, e in segs), detail=str(segs))
    check("引号闭后旁白回默认声线", segs[-1][1] == "", detail=str(segs))
    check("旁白两句合并成一段", any("门厅里很静" in t and "摆锤声" in t
                                    for t, v, e in segs), detail=str(segs))
    check("情绪随标记携带", any(e == "恭敬" for t, v, e in segs))

    # 上游没就绪（trickle）→ 不合并，首音不被拖
    segs2 = await segments_of(["门厅里很静，", "静得能听见挂钟的摆锤声。"],
                              trickle=True)
    check("上游滞后时不合并", len(segs2) == 2, detail=str(segs2))

    # 情绪切换 → 不合并
    segs3 = await segments_of([
        "⟦v:npc_a e:平静⟧「别去书房。」",
        "⟦v:npc_a e:低语⟧「老爷昨夜走了。」",
    ])
    check("情绪变化不合并", len(segs3) == 2 and segs3[0][2] == "平静"
          and segs3[1][2] == "低语", detail=str(segs3))

    # 同 NPC 新一句台词（新引号 + 新标记）声线正确切换
    segs4 = await segments_of([
        "⟦v:npc_a⟧「你来了。」",
        "⟦v:npc_b⟧「谁在那里？」",
    ])
    check("声线切换分成两段", [v for _, v, _ in segs4] == ["npc_a", "npc_b"],
          detail=str(segs4))

    print("== 结果 ==")
    print("ALL OK" if FAIL == 0 else f"{FAIL} 项失败")
    return FAIL


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
