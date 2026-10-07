#!/usr/bin/env bash
# 启动 ai-TRPG web 服务：HTTP :16891；certs/ 有自签证书时 HTTPS :16892。
# 生产 HTTPS 走 caddy 反代（见 caddy/Caddyfile），HTTP 端口只绑回环外的本机即可。
set -e
cd "$(dirname "$0")/.."
mkdir -p reports
if [ -f server.pid ] && kill -0 "$(cat server.pid)" 2>/dev/null; then
  echo "已在运行 (pid $(cat server.pid))"
  exit 0
fi
nohup .venv/bin/python -m voice.web_server --config configs/prod.json \
  --host 0.0.0.0 --port 16891 --https-port 16892 \
  > reports/server.log 2>&1 &
echo $! > server.pid
sleep 2
echo "started pid $(cat server.pid), log: reports/server.log"
