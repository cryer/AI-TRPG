#!/usr/bin/env bash
# 停止 ai-TRPG web 服务
cd "$(dirname "$0")/.."
if [ -f server.pid ] && kill -0 "$(cat server.pid)" 2>/dev/null; then
  kill "$(cat server.pid)"
  rm -f server.pid
  echo "stopped"
else
  echo "未在运行"
  rm -f server.pid
fi
