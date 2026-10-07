#!/usr/bin/env bash
# 启动 caddy 反代（生产 HTTPS）：trpg.<YOUR-SERVER-IP-DASHED>.sslip.io → 127.0.0.1:6891
# 域名见 caddy/Caddyfile（部署时替换为你的服务器 IP，sslip.io 用法见该文件注释）。
# caddy 二进制：从任意已装 caddy 的机器拷贝到 caddy/，或从 caddyserver.com 下载。
set -e
cd "$(dirname "$0")/.."
if [ -f caddy/caddy.pid ] && kill -0 "$(cat caddy/caddy.pid)" 2>/dev/null; then
  echo "caddy 已在运行 (pid $(cat caddy/caddy.pid))"
  exit 0
fi
[ -x caddy/caddy ] || { echo "caddy/caddy 不存在，先放入 caddy 二进制（见上方注释）"; exit 1; }
nohup ./caddy/caddy run --config caddy/Caddyfile > caddy/caddy.log 2>&1 &
echo $! > caddy/caddy.pid
sleep 2
echo "caddy started pid $(cat caddy/caddy.pid), log: caddy/caddy.log"
