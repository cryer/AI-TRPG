# 本地 TTS（CosyVoice2）调优指南

本项目本地 TTS 用 CosyVoice2-0.5B（GPU），所有可调项都在 `configs/trpg.json`
的 `cosyvoice` 段。本文说明每个参数怎么调、有哪些声线和情绪、怎么自定义。

## 1. 延迟 / 卡顿怎么调（最常用）

卡顿的根因是合成速度跟不上播放速度（RTF≈1 时没有余量，任何波动都会
让播放器断粮）。**第一调节手段是 `nfe`**：

| `nfe` | 效果 | 适用 |
|---|---|---|
| 2 | 极限提速，音质明显发闷、有artifact | 实在跑不动时兜底 |
| 3 | 快，音质略糙但清晰自然（当前默认） | RTX 3050 等中低端卡 |
| 5 | 质量/速度平衡 | RTX 4070 级 |
| 10 | 官方默认，最好音质 | RTX 4090 或不在意延迟 |

`nfe` 是 flow matching 的采样步数，耗时和音质都近似随它线性变化。
RTX 3050 实测：nfe=5 时 RTF≈0.98（贴边，必卡）；nfe=3 时 RTF≈0.7 左右，
播放有足够余量。**迁移到 4090 后把它调回 6~10 换音质。**

其他性能相关参数（一般不用动）：

| 参数 | 默认 | 说明 |
|---|---|---|
| `fp16` | true | 半精度，LLM 部分提速约 3 倍、显存减半。关掉只会更慢 |
| `use_ort_estimator` | true | 解码器走 ONNX Runtime CUDA，比 torch 快约 10 倍。关掉必卡 |
| `cudnn_benchmark` | false | 真实对局句子长度每次都变，开了反而慢（3050 实测） |
| `speed` | 1.0 | **不要改**：非 1.0 会关闭流式合成，首音延迟暴增 |
| `load_jit` | （代码内 false） | 3050/WDDM 上实测慢一倍，保持关闭 |

句间合并参数（影响「语气连贯」与「首音速度」的权衡）：

| 参数 | 默认 | 说明 |
|---|---|---|
| `merge_max_chars` | 150 | 相邻同声线句合并成一批合成的字符上限。调小 → 首音稍快，但句间语气/语速更容易跳变；调大 → 更连贯但更可能憋长句。4090 建议 250 |
| `merge_wait_s` | 0.05 | 合并时等下一句的宽限秒数。调大更连贯，调小首音更快 |
| `sample_top_p` | （官方 0.8） | LM 采样 top_p 覆盖。每个合成批次独立采样，逐批的语气/音色会有漂移；调低（如 0.7）旁白更稳，调太低会发闷、语气变平 |

## 2. 声线（音色）

声线注册表在 `cosyvoice.voices`，当前共 5 条：

| 声线 id | 用途 | 参考音 | instruct（语气提示） |
|---|---|---|---|
| `narrator_m` | 旁白 / DM 叙述（默认声线） | `models/voices/narrator_m.wav` | 用低沉平稳、富有叙事感的语气讲述 |
| `npc_female` | 成年女性 NPC | `models/voices/npc_female.wav` | 用冷静优雅的语气说话 |
| `npc_female_young` | 年轻女性 NPC | `models/voices/npc_female_young.wav` | 用清亮自然的语气说话 |
| `npc_male_old` | 老年男性 NPC | `models/voices/npc_male_old.wav` | 用苍老缓慢的语气说话 |
| `npc_male_young` | 青年男性 NPC | `models/voices/npc_male_young.wav` | 用沉稳的青年语气说话 |

NPC 用哪个声线由**剧本文件**决定：`adventures/*.json` 里每个 NPC 的
`voice` 字段填上面的声线 id。DM 叙述和旁白永远用 `default_voice`。

声线 id 写错不会静音：解析按「精确匹配 → 补/去 `npc_` 前缀 → 唯一子串
→ 回退默认声线」兜底（如 `female_young` 会自动落到 `npc_female_young`）。

### 自定义声线

1. 准备参考音：3~10 秒、单人、干净无背景音的中文人声 wav，放到
   `models/voices/` 下。也可以跑 `scripts/gen_voice_refs.py` 用火山
   seed-tts-2.0 生成（需要 VOLCENGINE_API_KEY，仅 uranus/saturn 簇音色可用）。
2. 在 `cosyvoice.voices` 里加一条：

```json
"npc_female_child": {
  "prompt_wav": "models/voices/npc_female_child.wav",
  "instruct": "用稚嫩活泼的语气说话"
}
```

3. 到剧本 JSON 里把某个 NPC 的 `voice` 改成这个新 id。

`instruct` 是该声线的**固定语气底色**（区别于逐句的情绪，见下），用
「用……的语气说话/讲述」句式，写声线的基础人设即可。改完重启服务生效
（声线模板在启动时预建）。

## 3. 情绪

情绪是**逐句**变化的语气，由 DM（LLM）根据语境自动标注，玩家无需操作。
机制：DM 在 NPC 台词开头写 `⟦s:NPC名字 e:情绪⟧`，引擎改写成声线标记后
交给 TTS；旁白默认不带标记，用默认声线。

旁白氛围（M5 九轮起）：DM 可在剧情氛围明显转变时在旁白段落开头写
`⟦s:旁白 e:情绪⟧` 切换旁白情绪；一旦设置，后续所有旁白沿用该情绪
（跨回合延续），直到下一个旁白氛围标记切换——旁白情绪只随剧情氛围
有意变化，不会每段乱跳。

情绪词表的唯一事实来源是 `configs/trpg.json` 的 `cosyvoice.emotions`
（DM 的 prompt 和 TTS 的模板预建都读它）：

```
平静、紧张、恐惧、愤怒、悲伤、神秘、激动、低语、恭敬、犹豫、温柔、威严
```

自定义：

- **加词**：往 `emotions` 数组里加即可，重启生效。用简短的情绪名词
  （两三个字最好），LLM 会从中选最贴合语境的一个。
- **删词**：直接删，但注意剧本语境里高频情绪别删光，词太少语气会单调。
- 不在词表里的情绪也能工作（LLM 偶尔自由发挥时），只是该组合第一次
  合成要现场建模版，会有约 1 秒停顿——所以高频情绪务必放进词表预建。

## 4. 迁移到 4090 后建议

- `nfe` 调回 6~10（4090 上 RTF 预计 <0.3，音质优先）
- Linux 部署可尝试 `load_trt`（TensorRT 解码器，需额外转换），
  Windows 上保持当前 ONNX 方案即可
