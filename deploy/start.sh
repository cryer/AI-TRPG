#!/usr/bin/env bash
# 启动 ai-TRPG web 服务：HTTP :16891；certs/ 有自签证书时 HTTPS :16892。
# 生产 HTTPS 走 caddy 反代（见 caddy/Caddyfile），HTTP 端口只绑回环外的本机即可。
set -e
cd "$(dirname "$0")/.."
mkdir -p reports
# torch wheel 自带的 CUDA/cuDNN 8 运行库：onnxruntime-gpu（cu12 构建）需要
# libcudnn.so.8 等，系统路径里没有，从这里补
NVLIB="$PWD/.venv/lib/python3.10/site-packages/nvidia"
for d in "$NVLIB"/*/lib; do
  LD_LIBRARY_PATH="$d${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
done
export LD_LIBRARY_PATH
if [ -f server.pid ] && kill -0 "$(cat server.pid)" 2>/dev/null; then
  echo "已在运行 (pid $(cat server.pid))"
  exit 0
fi
nohup .venv/bin/python -u -m voice.web_server --config configs/prod.json \
  --host 0.0.0.0 --port 16891 --https-port 16892 \
  > reports/server.log 2>&1 &
echo $! > server.pid
sleep 2
echo "started pid $(cat server.pid), log: reports/server.log"
