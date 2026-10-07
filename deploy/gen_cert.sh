#!/usr/bin/env bash
# 生成自签 HTTPS 证书（手机浏览器调用麦克风必须 HTTPS；caddy 反代之外的兜底方案）。
# 用法: ./deploy/gen_cert.sh <服务器IP>
set -e
cd "$(dirname "$0")/.."
IP=${1:?usage: ./deploy/gen_cert.sh <server-ip>}
mkdir -p certs
openssl req -x509 -newkey rsa:2048 -nodes \
  -keyout certs/key.pem -out certs/cert.pem -days 825 \
  -subj "/CN=ai-trpg" \
  -addext "subjectAltName=IP:${IP},IP:127.0.0.1,DNS:localhost"
echo "已生成 certs/cert.pem / certs/key.pem (SAN: ${IP})"
echo "浏览器首次访问 https://${IP}:16892 时选择「继续前往」信任该自签证书。"
