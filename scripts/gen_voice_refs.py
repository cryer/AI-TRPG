"""生成 CosyVoice2 克隆用参考音（火山 TTS 多 speaker → models/voices/*.wav）。

原理：CosyVoice2 零样本克隆只需要 3~10 秒干净参考音；用云端 TTS 的不同
speaker 各合成一段中性叙述文本，即可得到性别/年龄感各异的声线底版。
无效 speaker id 会被 API 拒绝，脚本自动跳过——CANDIDATES 是有意多放的候选池。

用法：python scripts/gen_voice_refs.py            # 生成全部缺失的参考音
     python scripts/gen_voice_refs.py --force     # 全部重新生成
"""

import asyncio
import sys
import wave
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from voice.tts.volcengine import VolcengineTTS

OUT_DIR = Path(__file__).resolve().parent.parent / "models" / "voices"

# 克隆用参考文本：约 12 秒，叙述口吻、情绪中性、含停顿与数字读音
REF_TEXT = ("夜幕低垂，庄园里静得只剩下壁炉中柴火轻微的噼啪声。"
            "老管家端着烛台，沿着长廊一步一步走来，"
            "墙上的影子被拉得很长。一九八七年之后，这里再没有人住过。")

# (voice_id, 火山 speaker 候选, 期望特征)——同 voice_id 多个候选按序尝试
# 仅 uranus/saturn 簇支持 seed-tts-2.0（音色列表见火山文档 Tonelist-1）
CANDIDATES = [
    ("narrator_m", ["zh_male_jieshuoxiaoming_uranus_bigtts",
                    "zh_male_yuanboxiaoshu_uranus_bigtts"], "低沉男声旁白（解说小明）"),
    ("npc_female", ["zh_female_vv_uranus_bigtts"], "成年女声"),
    ("npc_female_young", ["zh_female_zhixingnv_uranus_bigtts",
                          "zh_female_qinqienv_uranus_bigtts"], "知性年轻女声"),
    ("npc_male_old", ["zh_male_yuanboxiaoshu_uranus_bigtts",
                      "zh_male_youyoujunzi_uranus_bigtts"], "中年偏老男声（渊博小叔）"),
    ("npc_male_young", ["zh_male_ruyayichen_saturn_bigtts",
                        "zh_male_ruyaqingnian_uranus_bigtts"], "青年男声（儒雅）"),
]

SR = 16000


async def synth_one(tts: VolcengineTTS, speaker: str) -> bytes | None:
    async def one():
        yield REF_TEXT
    pcm = bytearray()
    try:
        async for chunk in tts.synth(one(), -1):
            pcm.extend(chunk.pcm)
    except Exception as e:
        print(f"  speaker {speaker} 失败: {e}")
        return None
    return bytes(pcm) if pcm else None


async def main(force: bool):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for voice_id, speakers, desc in CANDIDATES:
        out = OUT_DIR / f"{voice_id}.wav"
        if out.exists() and not force:
            print(f"{voice_id} 已存在，跳过（{desc}）")
            continue
        for speaker in speakers:
            print(f"{voice_id}（{desc}）尝试 speaker={speaker}")
            tts = VolcengineTTS(speaker=speaker)
            pcm = await synth_one(tts, speaker)
            await tts.close()
            if pcm:
                with wave.open(str(out), "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(SR)
                    wf.writeframes(pcm)
                print(f"  -> {out} ({len(pcm) / SR / 2:.1f}s)")
                break
        else:
            print(f"  !! {voice_id} 所有候选 speaker 均失败")


if __name__ == "__main__":
    asyncio.run(main("--force" in sys.argv))
