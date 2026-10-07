"""M5w Web 应用服务端：浏览器拨通即通话。

用法：
    python -m voice.web_server [--config configs/local.json] [--port 6789]
                               [--asr sherpa|volcengine] [--tts sherpa|volcengine]

- GET /            → web/index.html（通话界面：拨通/挂断 + 频率波纹）
- GET /pcm-worklet.js → 采集端 AudioWorklet
- GET /ws          → 通话 WebSocket：上行二进制 = 16kHz PCM16 mono 32ms 帧；
                     下行二进制 = agent TTS 音频；下行 JSON = flush/eos 控制帧

每次 WS 连接 = 一通电话：独立 VAD/TurnManager/MetricsSink（reports/web_*.jsonl），
provider 实例服务级共享（ASR/TTS 的本地/云端选择由 config 或命令行独立指定）。
挂断（WS 断开）即结束会话并打印延迟报告。演示按单路通话设计。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import ssl
import sys
from datetime import datetime
from pathlib import Path

from aiohttp import web
from dotenv import load_dotenv

from voice.asr import create_asr
from voice.llm import create_llm
from voice.metrics import MetricsSink, load_records, render_report
from voice.transport.web_ws import WsPlayer, ws_frames
from voice.tts import create_tts
from voice.turn_manager import TurnManager
from voice.vad import SileroVAD

SYSTEM_PROMPT = (
    "你是一个语音助手，正在和用户进行实时的语音通话。"
    "你的所有回复都会被语音合成朗读出来，用户用耳朵听，看不到任何文字。"
    "所以：像真正说话一样回答，简短口语化，每次不超过三句话；"
    "永远不要说自己无法发声、无法播放音频、没有声音或'只是文字助手'——你的回复本身就是声音；"
    "被要求唱歌、讲故事、讲笑话、模仿声音时，直接用口语演绎出来"
    "（唱歌就直接把歌词唱出来，可以带语气词、重复和延长音）；"
    "不要使用 Markdown、列表、编号、表情符号等任何视觉排版。"
)
WEB_DIR = Path(__file__).resolve().parent.parent / "web"


def _make_vad(cfg: dict) -> SileroVAD:
    vad_cfg = cfg.get("vad", {})
    return SileroVAD(vad_cfg.get("model_path", "models/silero_vad.onnx"),
                     threshold=vad_cfg.get("threshold", 0.5),
                     neg_threshold=vad_cfg.get("neg_threshold", 0.35),
                     endpoint_ms=cfg.get("endpoint_ms", 400),
                     min_speech_ms=vad_cfg.get("min_speech_ms", 96))


# ---- provider 运行时切换（页面上"识别/合成 × 本地/云端"开关） ----
#
# TurnManager 每回合才调 asr.stream / tts.synth，代理对象在调用瞬间解析
# 当前活动实例 → 切换下一回合即生效，通话不中断。provider 实例池化，
# 本地模型首次切换时懒加载 + 预热（避免开机就加载两套栈）。

CHOICES = {"asr": ("volcengine", "sherpa"), "tts": ("volcengine", "sherpa", "cosyvoice")}


def _create_provider(cfg: dict, slot: str, name: str):
    c = dict(cfg)
    c["providers"] = dict(cfg.get("providers", {}))
    c["providers"][slot] = name
    return create_asr(c) if slot == "asr" else create_tts(c)


async def _get_provider(app: web.Application, slot: str, name: str):
    inst = app["pool"][slot].get(name)
    if inst is None:
        loop = asyncio.get_running_loop()
        inst = await loop.run_in_executor(None, _create_provider,
                                          app["cfg"], slot, name)
        await _warm(slot, inst)
        app["pool"][slot][name] = inst
        print(f"[web] provider 上线：{slot}={name}")
    return inst


async def _warm(slot: str, inst) -> None:
    warmup = getattr(inst, "warmup", None)
    if warmup is None:
        return
    try:
        if slot == "tts":
            await warmup("嗯。")
        else:
            await warmup()
    except Exception as e:
        print(f"[warn] {slot} 预热失败：{e}")


class _SlotProxy:
    """按当前活动 provider 转发；只暴露 TurnManager 用到的方法。"""

    def __init__(self, app: web.Application, slot: str):
        self._app = app
        self._slot = slot

    def _active(self):
        return self._app["pool"][self._slot][self._app["active"][self._slot]]

    def stream(self, *args, **kwargs):      # asr
        return self._active().stream(*args, **kwargs)

    def synth(self, *args, **kwargs):       # tts
        return self._active().synth(*args, **kwargs)


async def api_get_providers(request: web.Request) -> web.Response:
    app = request.app
    return web.json_response({
        slot: {"active": app["active"][slot], "choices": list(CHOICES[slot])}
        for slot in CHOICES
    })


async def api_set_providers(request: web.Request) -> web.Response:
    app = request.app
    try:
        body = await request.json()
    except Exception:
        return web.json_response({"error": "invalid json"}, status=400)
    for slot, name in body.items():
        if slot not in CHOICES or name not in CHOICES[slot]:
            return web.json_response({"error": f"bad choice {slot}={name}"},
                                     status=400)
    try:
        for slot, name in body.items():
            await _get_provider(app, slot, name)   # 懒加载 + 预热，失败不切换
            app["active"][slot] = name
            print(f"[web] 切换：{slot} → {name}")
    except Exception as e:
        return web.json_response({"error": f"{type(e).__name__}: {e}"},
                                 status=500)
    return await api_get_providers(request)


async def ws_handler(request: web.Request) -> web.WebSocketResponse:
    app = request.app
    cfg = app["cfg"]
    ws = web.WebSocketResponse(heartbeat=30)
    await ws.prepare(request)
    peer = request.remote
    print(f"[web] 拨入：{peer}")

    player = WsPlayer(ws)
    # 游戏模式：GameAgent 持有每玩家的 GameEngine，必须每连接新建（llm 无状态可共享）
    agent = None
    if "game" in cfg:
        from game.agent import GameAgent
        agent = GameAgent(cfg, app["llm"])
    tm = TurnManager(vad=_make_vad(cfg), asr=_SlotProxy(app, "asr"),
                     llm=app["llm"], tts=_SlotProxy(app, "tts"),
                     player=player, cfg=cfg,
                     system_prompt=(agent.system_prompt if agent is not None
                                    else cfg.get("system_prompt", SYSTEM_PROMPT)))
    if agent is not None:
        agent.bind_turn_manager(tm)   # 共享 history、记忆、打断模式回调
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    jsonl_path = Path(app["out"]) / f"web_{stamp}.jsonl"
    try:
        with MetricsSink(jsonl_path) as sink:
            await tm.run(ws_frames(ws), sink)
    except Exception as e:
        print(f"[web] 会话异常结束：{e!r}")
    finally:
        player.close()

    st = player.stats()
    print(f"[web] 挂断：{peer}，会话落盘 {jsonl_path}\n"
          f"[泄露检查] stale chunk 丢弃 {st['dropped_stale_chunks']} 个"
          f"（{st['dropped_stale_bytes']}B）；flush 清除未播 {st['flushed_bytes']}B")
    turns, _, barge_ins = load_records(jsonl_path)
    if turns:
        print(render_report(turns, barge_ins, cfg.get("budget_ms", {})))
    return ws


async def on_startup(app: web.Application) -> None:
    cfg = app["cfg"]
    prov = cfg.get("providers", {})
    app["pool"] = {"asr": {}, "tts": {}}
    app["active"] = {"asr": prov.get("asr") or "volcengine",
                     "tts": prov.get("tts") or "volcengine"}
    app["llm"] = create_llm(cfg)
    for slot in ("asr", "tts"):
        await _get_provider(app, slot, app["active"][slot])   # 初始项预热
    print(f"[web] providers: asr={app['active']['asr']} llm={prov.get('llm')} "
          f"tts={app['active']['tts']}（页面开关可实时切换）")


async def on_cleanup(app: web.Application) -> None:
    providers = [app.get("llm")]
    for slot_pool in (app.get("pool") or {}).values():
        providers.extend(slot_pool.values())
    for p in providers:
        close = getattr(p, "close", None)
        if close is not None:
            try:
                await close()
            except Exception:
                pass


def main(argv=None) -> int:
    load_dotenv()
    ap = argparse.ArgumentParser(prog="python -m voice.web_server")
    ap.add_argument("--config", default="configs/default.json")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=6789)
    ap.add_argument("--https-port", type=int, default=6790)
    ap.add_argument("--ssl-cert", default="certs/cert.pem")
    ap.add_argument("--ssl-key", default="certs/key.pem")
    ap.add_argument("--out", default="reports")
    ap.add_argument("--asr", help="覆盖 config 的 ASR provider（sherpa=本地）")
    ap.add_argument("--llm", help="覆盖 config 的 LLM provider")
    ap.add_argument("--tts", help="覆盖 config 的 TTS provider（sherpa=本地）")
    args = ap.parse_args(argv)

    with open(args.config, encoding="utf-8") as fh:
        cfg = json.load(fh)
    for slot in ("asr", "llm", "tts"):
        override = getattr(args, slot)
        if override:
            cfg.setdefault("providers", {})[slot] = override

    app = web.Application()
    app["cfg"] = cfg
    app["out"] = args.out
    app.router.add_get("/", lambda r: web.FileResponse(WEB_DIR / "index.html"))
    app.router.add_get("/pcm-worklet.js",
                       lambda r: web.FileResponse(WEB_DIR / "pcm-worklet.js"))
    app.router.add_get("/ws", ws_handler)
    app.router.add_get("/api/providers", api_get_providers)
    app.router.add_post("/api/providers", api_set_providers)
    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)

    async def _serve() -> None:
        runner = web.AppRunner(app)
        await runner.setup()
        await web.TCPSite(runner, args.host, args.port).start()
        print(f"[web] HTTP : http://localhost:{args.port}"
              f"（本机/局域网调试；getUserMedia 仅 localhost 可用）")
        cert, key = Path(args.ssl_cert), Path(args.ssl_key)
        if cert.exists() and key.exists():
            ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            ctx.load_cert_chain(str(cert), str(key))
            await web.TCPSite(runner, args.host, args.https_port,
                              ssl_context=ctx).start()
            print(f"[web] HTTPS: https://<服务器IP>:{args.https_port}"
                  f"（手机/浏览器入口；自签证书首次访问需点「继续前往」）")
        else:
            print(f"[web] 未找到 {cert}，HTTPS 未启用；"
                  f"手机浏览器要麦克风请先运行 deploy/gen_cert.sh <服务器IP>")
        await asyncio.Event().wait()   # 常驻

    try:
        asyncio.run(_serve())
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
