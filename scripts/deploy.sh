#!/usr/bin/env bash
# =============================================================================
# deploy.sh — Incremental deploy (called by GitHub Actions on every push to main)
#
# Assumes setup-vm.sh has already been run once.
# Safe to run manually too:
#   ssh user@vm 'bash ~/ocr-project/scripts/deploy.sh'
# =============================================================================
set -euo pipefail

APP_DIR="${APP_DIR:-$HOME/ocr-project}"   # override by exporting APP_DIR before calling

echo "[deploy] $(date '+%Y-%m-%d %H:%M:%S') — starting"

echo "[deploy] pulling latest code…"
git -C "$APP_DIR" fetch origin main
git -C "$APP_DIR" reset --hard origin/main

echo "[deploy] installing Python deps…"
"$APP_DIR/backend/.venv/bin/pip" install -r "$APP_DIR/backend/requirements.txt" --quiet

echo "[deploy] building frontend…"
cd "$APP_DIR/frontend"
npm ci --prefer-offline
npm run build

echo "[deploy] restarting backend service…"
sudo systemctl restart ocr-backend

echo "[deploy] reloading nginx…"
sudo nginx -t && sudo systemctl reload nginx

echo "[deploy] done — $(date '+%Y-%m-%d %H:%M:%S')"
sudo systemctl status ocr-backend --no-pager | head -5
