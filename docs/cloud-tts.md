# 云端 TTS 配置指南（火山引擎豆包语音合成）

本项目支持两种多音色 TTS 后端，剧本与 DM 标记体系（`⟦v:声线id e:情绪⟧`）完全一致，只是声线 id 的解析目标不同：

| | 本地（默认） | 云端（本文） |
|---|---|---|
| provider | `cosyvoice` | `volcengine` |
| 声线注册表 | `configs/trpg.json` 的 `cosyvoice.voices`（零样本克隆参考音） | `configs/cloud.json` 的 `volcengine.voices`（火山官方音色 id） |
| 情绪机制 | instruct 自然语言（`请用{情绪}的语气说话。`） | `audio_params.emotion` 关键词 code |
| 参考配置 | `configs/trpg.json` | `configs/cloud.json` |

云端模式开箱示例：`python -m voice.web_server --config configs/cloud.json`（需 `.env` 里有 `VOLCENGINE_API_KEY`，ASR 走 `volcengine_asr` 段，LLM 走 `kimi` 段）。

## 1. 接口与参数结论

- 接口：`POST https://openspeech.bytedance.com/api/v3/tts/unidirectional`（HTTP chunked 单向流式，逐段一个请求，流式返回 base64 PCM）。
- 鉴权：Header `X-Api-Key` + `X-Api-Resource-Id`（官方音色用 `seed-tts-2.0`）。
- **emotion 参数位置：`req_params.audio_params.emotion`**，强度用 `req_params.audio_params.emotion_scale`（整数 1~5，不传时服务端默认 4）。
  依据：官方「WebSocket 双向流式-V3」文档明确列出 emotion/emotion_scale（[文档](https://docs.volcengine.com/docs/DoubaoVoice/WebSocketBidirectionalStreaming-V3)）；HTTP 单向 V3 文档的 audio_params 未列出 emotion，但豆包语音合成 2.0 插件文档与多家第三方接入（ZEGO、xiaozhi-esp32 等）都在单向流 audio_params 里传这两个字段，两条链路 audio_params 同构。**单向流的 emotion 支持请以实测/火山控制台为准**。
- **哪些音色支持 emotion code**：官方明确支持的是 1.0 多情感音色（voice_type 形如 `*_emo_*_mars_bigtts`，官方音色列表里标注了每个音色支持的情感子集）。2.0 大模型音色（`*_uranus_bigtts`）的官方情感机制是 `context_texts` 语音指令（放 `additions` 里，自然语言描述语气）；`emotion` code 在部分 2.0 音色上可用但官方未逐一承诺，不支持的音色会**静默忽略**（不报错、不掉回奇怪的声音）。本项目默认映射全部用 2.0 音色，情绪词命中时同时发 emotion code——属于「尽力而为」，要最稳的情绪控制可改用 1.0 多情感音色（需把 `resource_id` 换成 `seed-tts-1.0`）或在 provider 里扩展 context_texts。
- 2.0 中文 emotion code 表（官方 1.0 多情感音色文档列出，2.0 沿用同一套 code）：
  `happy`（开心）、`sad`（悲伤）、`angry`（生气）、`surprised`（惊讶）、`fear`（恐惧）、`hate`（厌恶）、`excited`（激动）、`coldness`（冷漠）、`neutral`（中性）、`depressed`（沮丧）、`lovey-dovey`（撒娇）、`shy`（害羞）、`comfort`（安慰鼓励）、`tension`（咆哮/焦急）、`tender`（温柔）、`storytelling`（讲故事）、`radio`（情感电台）、`magnetic`（磁性）、`advertising`（广告营销）、`vocal-fry`（气泡音）、`ASMR`（低语）、`news`（新闻播报）、`entertainment`（娱乐八卦）、`dialect`（方言）。

## 2. 官方音色库（2.0 中文音色节选）

完整列表以火山官方「[在线音色列表](https://www.volcengine.com/docs/DoubaoVoice/Tonelist-1)」/控制台音色库为准（100+ 个，持续上新）。以下按本项目需要分类节选（均为 `*_uranus_bigtts`，resource_id = `seed-tts-2.0`）：

**男声**

| voice_type | 名称 | 适用 |
|---|---|---|
| `zh_male_xuanyijieshuo_uranus_bigtts` | 悬疑解说 2.0 | 旁白/叙述（本项目默认 narrator） |
| `zh_male_shenyeboke_uranus_bigtts` | 深夜播客 2.0 | 低沉旁白备选 |
| `zh_male_yuanboxiaoshu_uranus_bigtts` | 渊博小叔 2.0 | 中老年男性 NPC |
| `zh_male_baqiqingshu_uranus_bigtts` | 霸气青叔 2.0 | 威严中年男性 |
| `zh_male_ruyaqingnian_uranus_bigtts` | 儒雅青年 2.0 | 斯文青年 NPC |
| `zh_male_yangguangqingnian_uranus_bigtts` | 阳光青年 2.0 | 青年男性 NPC（本项目默认） |
| `zh_male_m191_uranus_bigtts` | 云舟 2.0 | 通用男主声（S2S 主打） |
| `zh_male_taocheng_uranus_bigtts` | 小天 2.0 | 通用男主声 |
| `zh_male_shaonianzixin_uranus_bigtts` | 少年梓辛 2.0 | 少年 |
| `zh_male_gaolengchenwen_uranus_bigtts` | 高冷沉稳 2.0 | 冷面男性 NPC |
| `zh_male_qingcang_uranus_bigtts` | 擎苍 2.0 | 古风/奇幻长者感 |

**女声**

| voice_type | 名称 | 适用 |
|---|---|---|
| `zh_female_gaolengyujie_uranus_bigtts` | 高冷御姐 2.0 | 成熟女性 NPC（本项目默认 npc_female） |
| `zh_female_cancan_uranus_bigtts` | 知性灿灿 2.0 | 知性成年女性 |
| `zh_female_wuzetian_uranus_bigtts` | 武则天 2.0 | 威严年长女性 |
| `zh_female_popo_uranus_bigtts` | 婆婆 2.0 | 老年女性 |
| `zh_female_vv_uranus_bigtts` | Vivi 2.0 | 通用女主声（S2S 主打） |
| `zh_female_xiaohe_uranus_bigtts` | 小何 2.0 | 通用女主声 |
| `zh_female_linjianvhai_uranus_bigtts` | 邻家女孩 2.0 | 年轻女性 NPC（本项目默认 npc_female_young） |
| `zh_female_qingxinnvsheng_uranus_bigtts` | 清新女声 2.0 | 年轻女性备选 |
| `zh_female_tianmeitaozi_uranus_bigtts` | 甜美桃子 2.0 | 甜美少女 |
| `zh_female_gufengshaoyu_uranus_bigtts` | 古风少御 2.0 | 古风少女 |

**童声**

| voice_type | 名称 |
|---|---|
| `zh_male_tiancaitongsheng_uranus_bigtts` | 天才童声 2.0 |
| `zh_male_naiqimengwa_uranus_bigtts` | 奶气萌娃 2.0 |
| `zh_female_xiaoxue_uranus_bigtts` | 儿童绘本 2.0 |

**老年男性说明**：2.0 标准音色里没有严格意义的「老爷爷」音色；本项目用「渊博小叔」（中年偏低）顶替 `npc_male_old`。可选替代：角色扮演音色「幽默大爷 2.0」（`ICL_uranus_zh_male_youmodaye_tob`）——`ICL_*` 音色的 resource 路由未经本项目实测，**以火山控制台/文档为准**。

**多情感（emotion code 官方保证支持）**：只有 1.0 的 `*_emo_*_mars_bigtts` 音色（如 `zh_male_lengkugege_emo_v2_mars_bigtts`、`zh_female_roumeinvyou_emo_v2_mars_bigtts` 等，官方音色列表「多情感」场景），且每个音色只支持情感子集。用它们需 `resource_id: seed-tts-1.0`。

## 3. 本项目怎么配

### 3.1 剧本 → 声线 id（与本地模式完全一致）

剧本 JSON 的 `npcs[].voice` 写的是**项目声线 id**，不直接写火山音色：

```json
{"name": "老管家", "voice": "npc_male_old", ...}
```

DM 输出台词前写 `⟦s:NPC名 e:情绪⟧`，game 层按剧本 `voice` 字段改写为 `⟦v:npc_male_old e:恭敬⟧` 送进 TTS。声线 id 容错：LLM 自创 id（如漏 `npc_` 前缀的 `female_young`）会自动补前缀/子串匹配回注册表，查不到回退默认声线并打 warn，标记绝不会被念出来。

### 3.2 声线 id → 火山音色（`volcengine.voices`）

```json
"volcengine": {
  "speaker": "zh_male_xuanyijieshuo_uranus_bigtts",   // 兜底（旁白/未注册 id）
  "resource_id": "seed-tts-2.0",
  "emotion_scale": 4,                                  // 可选，1~5
  "voices": {
    "narrator_m":      {"speaker": "zh_male_xuanyijieshuo_uranus_bigtts"},
    "npc_female":      {"speaker": "zh_female_gaolengyujie_uranus_bigtts"},
    "npc_female_young":{"speaker": "zh_female_linjianvhai_uranus_bigtts"},
    "npc_male_old":    {"speaker": "zh_male_yuanboxiaoshu_uranus_bigtts"},
    "npc_male_young":  {"speaker": "zh_male_yangguangqingnian_uranus_bigtts"}
  }
}
```

- 本地 cosyvoice 的 5 个声线 id 与火山音色的对应关系即上表；自建新声线只需在两边各加一条映射（cosyvoice 加参考音，volcengine 加 speaker），剧本不用改。
- 不配 `voices` 时所有文本都用 `speaker` 兜底（退化为单音色，标记被自动剥离）。

### 3.3 情绪词 → emotion code（`volcengine.emotions`）

DM 情绪词表（12 个，与 DM prompt / cosyvoice 配置同源）的默认映射：

| DM 情绪词 | emotion code | 说明 |
|---|---|---|
| 平静 | `neutral` | |
| 紧张 | `tension` | 官方语义是「咆哮/焦急」，近似 |
| 恐惧 | `fear` | |
| 愤怒 | `angry` | |
| 悲伤 | `sad` | |
| 神秘 | `storytelling` | 「讲故事/自然讲述」，近似 |
| 激动 | `excited` | |
| 低语 | `ASMR` | |
| 恭敬 | `tender` | 无直接对应，借「温柔」 |
| 犹豫 | `shy` | 无直接对应，借「害羞」，可斟酌 |
| 温柔 | `tender` | |
| 威严 | `coldness` | 无直接对应，借「冷漠」，可斟酌 |

没把握的近义词映射可在配置里删改：词不在 `emotions` 里时该段不带 emotion 参数（自然语气），不会报错。所有 2.0 音色上的实际情绪效果**以实测为准**（见 §1 的说明）。

### 3.4 延迟预期

火山单向流首 chunk 实测通常在几百 ms 级（句级请求，网络好时可满足 `budget_ms.tts_first_chunk: 200` 附近量级）；逐段请求意味着一段对局会有多次 HTTP 调用，但 session 常驻复用 TLS，且每段独立流式返回，首音不被拖慢。barge-in 取消语义与本地一致（外层 cancel，迟到 chunk 由 gen_id 丢弃）。
