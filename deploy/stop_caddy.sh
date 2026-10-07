#!/usr/bin/env bash
# 停止 caddy 反代
cd "$(dirname "$0")/.."
if [ -f caddy/caddy.pid ] && kill -0 "$(cat caddy/caddy.pid)" 2>/dev/null; then
  kill "$(cat caddy/caddy.pid)"
  rm -f caddy/caddy.pid
  echo "caddy stopped"
else
  echo "caddy 未在运行"
  rm -f caddy/caddy.pid
fi
