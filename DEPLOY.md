# AI 部署指南（DEPLOY.md）

> **这份文档是写给部署 AI 看的。** 如果你是人类用户：把本文件交给你的 AI 助手
> （Claude Code / Kimi Code / Cursor 等），说一句「读 DEPLOY.md 帮我部署这个项目」即可。
> 如果你是 AI：请严格按本文档执行，每一步都有验证命令；遇到与本文档冲突的现实情况，
> 以现实为准并向用户说明偏差。

项目形态：**纯 Python 服务端 + 浏览器客户端**（Web 端是哑终端，所有智能在服务端）。
不需要 Node.js、不需要数据库。

---

## 0. 先决定部署形态（必读）

本项目支持两种语音栈，**LLM 永远走云端 API**（Kimi），区别只在 ASR（语音识别）和 TTS（语音合成）：

| 形态 | ASR | TTS | 需要 GPU？ | 适合 |
|---|---|---|---|---|
| **A. 纯云端** | 火山引擎 API | 火山引擎 API | 不需要 | 无显卡的机器、快速体验、低配 VPS |
| **B. 本地+云端混合**（默认） | sherpa-onnx 本地（CPU） | CosyVoice2 本地（GPU） | 需要 N 卡 | 有显卡的机器；本地 TTS 免 API 费用、支持多 NPC 音色+情绪 |

无论哪种形态，浏览器打开网页就能玩。**形态 A 可以跳过本文档所有「模型下载」和 GPU 相关步骤。**

### 0.1 决策动作（AI 必做）

执行 `nvidia-smi`（没有此命令 = 无 N 卡）：

- **无 N 卡 / 命令不存在** → 形态 A
- **有 N 卡** → 看显存，按下表选形态 B 的 CosyVoice 参数档位：

| 显存 | 结论 | `cosyvoice` 配置档位 |
|---|---|---|
| < 6 GB | 本地 TTS 不现实，走形态 A（ASR 仍可本地 sherpa，它跑 CPU） | — |
| 6–10 GB（如 RTX 3050/3060） | 形态 B 低档位 | `fp16: true`, `nfe: 3`, `use_ort_estimator: true` |
| ≥ 12 GB（如 RTX 3080/4090） | 形态 B 高档位 | `fp16: true`, `nfe: 10`, `use_ort_estimator: true` |

多卡机器：用 `nvidia-smi` 选一张空闲卡，把卡号写进配置的 `cosyvoice.cuda_visible_devices`
（例如 `"6"`）。`nfe` 是音质/延迟旋钮：越高音质越好、首音越慢，范围 3–10。
断电风险小，先按上表给，用户体验后再调（调参细节见 `docs/local-tts.md`）。

---

## 1. 拉取代码

```bash
git clone --recursive <仓库地址> ai-TRPG
cd ai-TRPG
# 如果 clone 时忘了 --recursive：
git submodule update --init --recursive   # vendor/CosyVoice 是本地 TTS 的推理代码
```

形态 A 也建议初始化子模块（代码里 import 链会碰到它），但可以不下载 CosyVoice 模型权重。

## 2. Python 环境（3.10）

推荐 uv（没有就 `pip install uv` 或用 python -m venv 等价替代）：

```bash
uv venv --python 3.10 .venv
# Linux: source .venv/bin/activate    Windows: .venv\Scripts\activate
```

## 3. 安装依赖

**形态 A（纯云端，最小依赖）：**

```bash
uv pip install -r requirements-cloud.txt
```

**形态 B（含本地 TTS 的完整依赖）：**

```bash
uv pip install -r requirements.txt
```

注意（Linux + N 卡）：`requirements.txt` 已内置两个额外索引源——
PyTorch cu121 源和 aiinfra 的 onnxruntime-cuda-12 源。**不要用 PyPI 的
onnxruntime-gpu**（CUDA 11 构建，会静默回退 CPU，TTS 慢 10 倍）。
Windows 上装的是 CPU 版 onnxruntime + CUDA 版 torch，这是实测可用组合。

## 4. 模型文件（形态 A 只下第一个；形态 B 全下）

所有模型放项目内 `models/` 目录（配置里都是相对路径）：

| 模型 | 大小 | 形态 A | 形态 B | 获取方式 |
|---|---|---|---|---|
| silero_vad.onnx | ~2 MB | **必需** | 必需 | `curl -L -o models/silero_vad.onnx https://github.com/snakers4/silero-vad/raw/master/files/silero_vad.onnx` |
| sherpa ASR（中英双语流式） | ~200 MB | 跳过 | 必需 | 从 [sherpa-onnx releases](https://github.com/k2-fsa/sherpa-onnx/releases) 下载 `sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20.tar.bz2`，解压重命名为 `models/sherpa-asr-bilingual-zh-en/` |
| CosyVoice2-0.5B | ~5.3 GB | 跳过 | 必需 | `python -c "from modelscope import snapshot_download; snapshot_download('iic/CosyVoice2-0.5B', local_dir='models/CosyVoice2-0.5B')"` |
| 声线参考音 ×5 | 各 ~1 MB | 跳过 | 必需 | 见 §4.1 |
| sherpa VITS TTS | ~100 MB | 跳过 | 可选（备用引擎，默认不用） | sherpa-onnx releases 的 `vits-zh-hf-fanchen-C` 解压到 `models/vits-zh-hf-fanchen-C/` |

### 4.1 声线参考音（CosyVoice 零样本克隆的底版）

`models/voices/` 下需要 5 个 wav：`narrator_m / npc_female / npc_female_young / npc_male_old / npc_male_young`。

两种获得方式（任选）：

1. **脚本生成（推荐）**：先配好 `.env` 里的 `VOLCENGINE_API_KEY`（见 §6），然后
   `python scripts/gen_voice_refs.py` —— 它用火山 TTS 的不同 speaker 各合成一段
   中性文本当克隆底版。
2. **自己录音/找素材**：每个声线准备 3–10 秒干净人声（16kHz 单声道 wav 最佳，
   情绪中性、无背景音乐），按上面的名字放进 `models/voices/`。

## 5. 选择并检查配置文件

- 形态 A → `configs/cloud.json`（providers: asr=volcengine, tts=volcengine）
- 形态 B → `configs/trpg.json`（providers: asr=sherpa, tts=cosyvoice），
  按 §0.1 的档位表核对 `cosyvoice` 段的 `nfe` / `cuda_visible_devices`

配置里所有路径都是相对路径，开箱即用；模型没下载齐会在启动时报明确错误。

## 6. 配置 .env（最后一步）

在项目根目录新建 `.env`：

```dotenv
# 必需：LLM（Kimi，https://platform.moonshot.cn/ 或 coding 订阅的 api.kimi.com）
KIMI_API_KEY=sk-...

# 形态 A 必需 / 形态 B 用于生成声线参考音：火山引擎语音服务
# （console.volcengine.com → 语音技术 → 开通「语音识别」和「语音合成」→ 创建 API Key）
VOLCENGINE_API_KEY=...

# 可选：Deepgram ASR（另一个云端 ASR 选项，默认不用）
# DEEPGRAM_API_KEY=...
```

`.env` 已在 `.gitignore` 里，不会入库。**不要把 key 写进任何 json 配置文件。**

## 7. 启动与验证

```bash
# 自检（不需要 GPU/网络的管道测试，应全部 OK）
PYTHONIOENCODING=utf-8 python scripts/test_voice_pipeline.py

# 启动 Web 服务（形态 A 把配置换成 configs/cloud.json）
python -m voice.web_server --config configs/trpg.json --port 16891 --https-port 16892
```

- 本机玩：浏览器打开 `http://localhost:16891`
- 可选 CLI 调试（不需要浏览器，本机麦克风直接对话）：
  `python -m voice.duplex --config configs/trpg.json`

### 7.1 HTTPS 证书（服务器/局域网部署必做）

**为什么必须 HTTPS**：浏览器只允许在「安全上下文」里调麦克风
（`getUserMedia`）。`localhost`/`127.0.0.1` 豁免，但用服务器 IP 或局域网
IP 访问时**必须 HTTPS**，否则页面打不开麦克风、无法通话。

**生成自签证书（本项目自带脚本，推荐）**：

```bash
./deploy/gen_cert.sh <服务器IP>
# 例：./deploy/gen_cert.sh 203.0.113.10
```

脚本做的事（`deploy/gen_cert.sh`，依赖 openssl，Linux/Git Bash 自带）：

- 用 `openssl req -x509` 生成 2048 位 RSA 自签证书，有效期 825 天
  （浏览器对自签证书接受的上限）；
- **SAN 里写入服务器 IP** + `127.0.0.1` + `localhost`——SAN 必须包含访问
  用的那个 IP，否则浏览器会直接拒绝，这是最容易漏的一点；
- 输出 `certs/cert.pem` 和 `certs/key.pem`（`certs/` 已 gitignore，
  不会入库）。

`web_server` 启动时检测到 `certs/cert.pem` 存在就会自动在 `--https-port`
（默认 16892）上启用 TLS，无需额外配置。随后：

1. **放行端口**：云服务器去控制台安全组/防火墙放行 HTTPS 端口（如
   16892），只放行 22 是不够的；本机防火墙同理；
2. 浏览器访问 `https://<服务器IP>:16892`，首次会警告证书不受信任，
   点「高级 → 继续前往」即可（自签证书的正常现象）；
3. IP 变了要重新跑一次 `gen_cert.sh`（SAN 里写死了旧 IP）。

**有域名时的升级方案（可选）**：用 Caddy 反代自动签发受信任的
Let's Encrypt 证书，免点「继续前往」。仓库 `caddy/` 下有 `Caddyfile`
模板和 `deploy/start_caddy.sh` / `stop_caddy.sh`，把模板里的域名改成
自己的、放行 443 即可。没有域名就用上面的自签方案，功能完全一样。

启动日志里应看到 `provider 上线：asr=... tts=...`；本地 TTS 首次加载模型需要
10–60 秒（GPU 预热 + 声线模板预建），看到 `声线模板预建完成` 才算就绪。

## 8. 排障速查

| 症状 | 处理 |
|---|---|
| 端口被占用 | 换 `--port` / `--https-port`；Windows 用 `netstat -ano \| findstr :16891` 找占用进程 |
| TTS 极慢（首音 > 10s） | 八成是 onnxruntime 回退 CPU 了：检查是否装错 PyPI 版 onnxruntime-gpu；Linux 还需 `LD_LIBRARY_PATH` 带上 venv 内 torch 自带的 `nvidia/*/lib`（`deploy/start.sh` 有现成写法） |
| 启动报模型路径不存在 | 回到 §4 检查模型目录名是否与配置一致 |
| 日志 `no frontend is avaliable` | wetext 文本规整前端下载被 modelscope 限流，重启服务一般恢复；影响仅是数字/符号读法 |
| 说完话 AI 沉默 | 看日志 `[game] LLM 空回复`（系统会自动重试 2 次）；频繁出现检查 KIMI_API_KEY 额度 |
| 语音念着奇怪标记 | 不该发生；把日志里对应 sentence 行发给维护者 |
| 播放卡顿 | 降 `cosyvoice.nfe`（如 10→6→3）、确认 `use_ort_estimator: true`；详见 `docs/local-tts.md` |

## 9. 部署 AI 完成前的核对清单

- [ ] 形态已按 `nvidia-smi` 结果选定，配置档位匹配显存
- [ ] `scripts/test_voice_pipeline.py` 全部 OK
- [ ] `.env` 已建且含 KIMI_API_KEY（形态 A 还需 VOLCENGINE_API_KEY）
- [ ] 服务启动日志出现 `provider 上线`（形态 B 还有 `声线模板预建完成`）
- [ ] 服务器/局域网部署：已跑 `./deploy/gen_cert.sh <服务器IP>` 且安全组/防火墙放行了 HTTPS 端口
- [ ] 浏览器能打开页面并看到跑团风格 UI；点击「开启」按钮能连上并听到 AI 说话
- [ ] 告知用户：访问地址、用的哪个配置文件、形态 A/B、调参文档位置（`docs/local-tts.md` / `docs/cloud-tts.md`）
