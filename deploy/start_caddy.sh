#!/usr/bin/env bash
# 启动 caddy 反代（生产 HTTPS）：trpg.116-169-215-214.sslip.io → 127.0.0.1:6891
# caddy 二进制部署时从 voice-agent 拷贝（cp /opt/voice-agent/caddy/caddy caddy/）
set -e
cd "$(dirname "$0")/.."
if [ -f caddy/caddy.pid ] && kill -0 "$(cat caddy/caddy.pid)" 2>/dev/null; then
  echo "caddy 已在运行 (pid $(cat caddy/caddy.pid))"
  exit 0
fi
[ -x caddy/caddy ] || { echo "caddy/caddy 不存在，先从 voice-agent 拷贝二进制"; exit 1; }
nohup ./caddy/caddy run --config caddy/Caddyfile > caddy/caddy.log 2>&1 &
echo $! > caddy/caddy.pid
sleep 2
echo "caddy started pid $(cat caddy/caddy.pid), log: caddy/caddy.log"
