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

    async def segments_of(sentences: list[str], trickle: bool = False,
                          tts_inst=None, state=None):
        inst = tts_inst or tts

        async def gen():
            for s in sentences:
                yield s
                if trickle:
                    await asyncio.sleep(0.2)   # 上游没就绪 → 不合并
        return [seg async for seg in inst._segment_stream(gen(), state=state)]

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

    # 用户 turn 8 实锤场景：长台词跨 5 句，全部保持 NPC 声线
    segs5 = await segments_of([
        "唇角微微一弯。",
        "⟦v:female_young e:温柔⟧「您就是晚晴请来的侦探吧？",
        "真是一表人才。",
        "我叫沈曼丽，",
        "这两日庄里乱糟糟的，",
        "让您见笑了。」",
    ])
    npc_segs = [t for t, v, e in segs5 if v == "female_young"]
    check("长台词全程 NPC 声线", bool(npc_segs) and
          all(k in "".join(npc_segs) for k in ("一表人才", "沈曼丽", "见笑了")),
          detail=str(segs5))
    check("开头旁白是默认声线", segs5[0][1] == "", detail=str(segs5))

    # 真人局实锤场景（web_20261007_182521 turn 4）：LLM 用弯引号“”而非「」，

    # 多句台词必须同样全程保持 NPC 声线
    segs6 = await segments_of([
        "⟦v:npc_male_old e:低语⟧“老爷是昨夜十点四十走的，就在书房里。",
        "门从里头锁着，窗户也插得好好的。",
        "庄里人都说……是夫人回来索命了。”",
        "他顿了顿，朝走廊深处抬了抬下巴。",
    ])
    npc6 = [t for t, v, e in segs6 if v == "npc_male_old"]
    check("弯引号台词全程 NPC 声线", bool(npc6) and
          all(k in "".join(npc6) for k in ("十点四十", "窗户", "索命")),
          detail=str(segs6))
    check("弯引号闭合后旁白回默认", segs6[-1][1] == "" and
          "抬了抬下巴" in segs6[-1][0], detail=str(segs6))
    check("弯引号情绪随标记携带", any(e == "低语" for t, v, e in segs6))

    # NPC 两段台词中间插叙述/动作：第二段无标记引号必须继承上一个 NPC 声线
    segs7 = await segments_of([
        "⟦v:npc_male_old e:恭敬⟧“您路上辛苦了。”",
        "他压低声音，朝走廊瞟了一眼，",
        "“还有一件事，我只告诉您一个人。”",
        "说罢，他退回了阴影里。",
    ])
    npc7 = [t for t, v, e in segs7 if v == "npc_male_old"]
    check("无标记新引号继承上一 NPC 声线", bool(npc7) and
          "只告诉您一个人" in "".join(npc7), detail=str(segs7))
    check("段间叙述是默认声线", any(v == "" and "压低声音" in t
                                    for t, v, e in segs7), detail=str(segs7))
    check("结尾旁白是默认声线", segs7[-1][1] == "", detail=str(segs7))
    check("新引号情绪继承上一 NPC", any(v == "npc_male_old" and e == "恭敬"
                                        for t, v, e in segs7), detail=str(segs7))

    # 不同 NPC 新台词带标记 → 正确切换，不被上一个 NPC 继承逻辑污染
    segs8 = await segments_of([
        "⟦v:npc_male_old⟧“我没杀人。”",
        "⟦v:npc_female⟧“他说谎。”",
    ])
    check("标记切换优先于继承", [v for _, v, _ in segs8] ==
          ["npc_male_old", "npc_female"], detail=str(segs8))

    print("== 3b. 直引号台词 / 声线泄漏防护 / 旁白情绪记忆 ==")
    tts_n = CosyVoiceTTS(default_voice="narrator_m")

    # 服务器实锤（web turn 12）：LLM 用直引号 "…" 写台词，旧状态机不认 →
    # 标记声线经 explicit 泄漏，整段旁白变 NPC 音色
    segs9 = await segments_of([
        '⟦v:npc_female e:平静⟧"镖师不当心自己的镖，',
        '倒有闲心来管别人的酒。"',
        "她这才抬眼看你，一双眼又黑又亮。",
    ], tts_inst=tts_n, state={})
    npc9 = [t for t, v, e in segs9 if v == "npc_female"]
    check("直引号台词全程 NPC 声线", bool(npc9) and
          all(k in "".join(npc9) for k in ("镖师", "闲心")), detail=str(segs9))
    check("直引号闭合后旁白回默认", segs9[-1][1] == "" and
          "抬眼看你" in segs9[-1][0], detail=str(segs9))

    # 无引号台词：标记只影响本句，下一句旁白不继承 NPC 声线
    segs10 = await segments_of([
        "⟦v:npc_a e:愤怒⟧站住，把东西放下！",
        "你转身离开了巷子。",
    ], tts_inst=tts_n, state={})
    check("无引号台词本句是 NPC 声线", segs10[0][1] == "npc_a",
          detail=str(segs10))
    check("无引号台词后旁白不泄漏", segs10[-1][1] == "" and
          "转身离开" in segs10[-1][0], detail=str(segs10))

    # 旁白情绪记忆：⟦v:旁白声线 e:情绪⟧ 设置氛围后，无标记旁白沿用，
    # 跨回合（同一 state）延续，直到新旁白标记切换
    st: dict = {}
    segs11 = await segments_of([
        "⟦v:narrator_m e:悲伤⟧雨还在下。",
        "你裹紧斗篷，走进客栈。",
    ], tts_inst=tts_n, state=st)
    check("旁白氛围标记后旁白带情绪", all(v == "" and e == "悲伤"
                                        for _, v, e in segs11), detail=str(segs11))
    segs12 = await segments_of(["大堂里灯火昏黄。"], tts_inst=tts_n, state=st)
    check("旁白情绪跨回合延续", segs12[0] == ("大堂里灯火昏黄。", "", "悲伤"),
          detail=str(segs12))
    segs13 = await segments_of([
        "⟦v:narrator_m e:平静⟧天亮了。",
        "雪停了。",
    ], tts_inst=tts_n, state=st)
    check("旁白氛围可被新标记切换", all(e == "平静" for _, _, e in segs13),
          detail=str(segs13))

    # 旁白氛围标记不污染 NPC 引号继承（last_voice 仍是上一个 NPC）
    segs14 = await segments_of([
        "⟦v:npc_a e:平静⟧「你来了。」",
        "⟦v:narrator_m e:神秘⟧屋里忽然暗了下来。",
        "「还有谁在里面？」",
    ], tts_inst=tts_n, state={})
    check("旁白标记后新引号仍继承上一 NPC", segs14[-1][1] == "npc_a",
          detail=str(segs14))
    check("旁白句带氛围情绪", any(v == "" and e == "神秘" and "暗了下来" in t
                                for t, v, e in segs14), detail=str(segs14))

    print("== 4. _resolve_voice：声线 id 容错 ==")
    tts2 = CosyVoiceTTS(voices={v: {"prompt_wav": "x", "instruct": ""} for v in
                                ("narrator_m", "npc_female", "npc_female_young",
                                 "npc_male_old", "npc_male_young")},
                        default_voice="narrator_m")
    check("LLM 自创 id 补前缀", tts2._resolve_voice("female_young") == "npc_female_young")
    check("精确 id 原样", tts2._resolve_voice("npc_male_old") == "npc_male_old")
    check("唯一子串匹配", tts2._resolve_voice("male_old") == "npc_male_old")
    check("空 id 回默认", tts2._resolve_voice("") == "narrator_m")
    check("前缀规则优先于歧义", tts2._resolve_voice("female") == "npc_female")
    check("完全未知回默认", tts2._resolve_voice("robot") == "narrator_m")

    print("== 结果 ==")
    print("ALL OK" if FAIL == 0 else f"{FAIL} 项失败")
    return FAIL


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
