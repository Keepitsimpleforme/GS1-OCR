#!/usr/bin/env bash
# =============================================================================
# setup-vm.sh — One-time provisioning for Ubuntu 22.04 LTS
#
# Spec: CURSOR_SPEC.md v2.0
#   - LightOn OCR-2B  (Ollama) — image → raw text
#   - Qwen2.5:7b      (Ollama) — raw text → structured JSON fields
#   - pyzbar + OpenCV (system) — barcode decode (3-stage preprocessing)
#   - FastAPI backend           — served by uvicorn behind nginx
#   - React frontend            — served as static files by nginx
#
# VM REQUIREMENTS (minimum):
#   RAM  : 16 GB  (LightOn ~4 GB + Qwen2.5:7b ~5 GB, plus OS headroom)
#   Disk : 40 GB  (models ~15 GB, OS + app ~10 GB, buffer)
#   GPU  : Strongly recommended — NVIDIA GPU with 8 GB+ VRAM
#          (CPU-only works but OCR+extraction can take 3–5 min per image)
#   OS   : Ubuntu 22.04 LTS
#
# Usage (run once as a sudo-capable user):
#   chmod +x scripts/setup-vm.sh
#   ./scripts/setup-vm.sh
# =============================================================================
set -euo pipefail

# ── Edit these before running ─────────────────────────────────────────────────
REPO_URL="https://github.com/Keepitsimpleforme/GS1-OCR.git"
APP_DIR="/home/ubuntu/ocr-project"   # change "ubuntu" if your VM user is different
APP_USER="${SUDO_USER:-ubuntu}"      # your VM login username (ubuntu / ec2-user / etc.)
OCR_MODEL="maternion/LightOnOCR-2"
EXTRACT_MODEL="qwen2.5:7b"
# ─────────────────────────────────────────────────────────────────────────────

echo "━━━ [1/8] System packages ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
sudo apt-get update -y
sudo apt-get install -y \
    git curl nginx \
    python3 python3-pip python3-venv python3-dev \
    build-essential \
    libzbar0 \
    libgl1-mesa-glx libglib2.0-0   # required by opencv-python-headless

# Node.js 20 (skip if already >= 18)
NODE_MAJOR=$(node --version 2>/dev/null | grep -oP '(?<=v)\d+' | head -1 || echo 0)
if [ "$NODE_MAJOR" -lt 18 ]; then
    echo "Installing Node.js 20 LTS…"
    curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
    sudo apt-get install -y nodejs
fi
echo "Node $(node --version)  |  npm $(npm --version)"

echo ""
echo "━━━ [2/8] Ollama ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if ! command -v ollama &>/dev/null; then
    curl -fsSL https://ollama.com/install.sh | sh
fi
sudo systemctl enable --now ollama
echo "Waiting for Ollama to start…"
sleep 6

# ── Model pulls ───────────────────────────────────────────────────────────────
echo ""
echo "Pulling OCR model: $OCR_MODEL"
echo "(~2–4 GB — may take 5–15 min on first run)"
ollama pull "$OCR_MODEL"

echo ""
echo "Pulling extraction model: $EXTRACT_MODEL"
echo "(~5 GB — may take 10–20 min on first run)"
ollama pull "$EXTRACT_MODEL"

echo "Installed Ollama models:"
ollama list

echo ""
echo "━━━ [3/8] Clone / update project ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if [ -d "$APP_DIR/.git" ]; then
    echo "Repo already exists — pulling latest"
    sudo git -C "$APP_DIR" pull origin main
else
    sudo git clone "$REPO_URL" "$APP_DIR"
fi
sudo chown -R "$APP_USER":"$APP_USER" "$APP_DIR"

echo ""
echo "━━━ [4/8] .env file ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if [ ! -f "$APP_DIR/.env" ]; then
    cp "$APP_DIR/.env.example" "$APP_DIR/.env"
    echo "Created .env from .env.example — edit if you need custom values"
else
    echo ".env already exists — skipping (not overwriting)"
fi

echo ""
echo "━━━ [5/8] Python venv + pip deps ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
cd "$APP_DIR/backend"
python3 -m venv .venv
.venv/bin/pip install --upgrade pip --quiet
.venv/bin/pip install -r requirements.txt
echo "Backend deps installed"

# Smoke-test key imports
.venv/bin/python -c "
import cv2, pyzbar, PIL, ollama, fastapi
print('cv2', cv2.__version__)
print('All imports OK')
"

echo ""
echo "━━━ [6/8] React frontend build ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
cd "$APP_DIR/frontend"
npm ci --prefer-offline
npm run build
echo "Frontend built → $APP_DIR/frontend/dist"

echo ""
echo "━━━ [7/8] systemd service ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
# Replace placeholders __APP_DIR__ and __APP_USER__ with real values
sed \
    -e "s|__APP_DIR__|$APP_DIR|g" \
    -e "s|__APP_USER__|$APP_USER|g" \
    "$APP_DIR/backend/ocr-backend.service" \
    | sudo tee /etc/systemd/system/ocr-backend.service > /dev/null
sudo systemctl daemon-reload
sudo systemctl enable --now ocr-backend
echo "Backend service status:"
sudo systemctl status ocr-backend --no-pager | head -8

echo ""
echo "━━━ [8/8] nginx ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
# Replace __APP_DIR__ placeholder in nginx config
sed "s|__APP_DIR__|$APP_DIR|g" "$APP_DIR/nginx.conf" \
    | sudo tee /etc/nginx/sites-available/ocr-project > /dev/null
sudo ln -sf /etc/nginx/sites-available/ocr-project /etc/nginx/sites-enabled/ocr-project
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t
sudo systemctl enable --now nginx
sudo systemctl reload nginx

echo ""
echo "━━━ Done! ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
VM_IP=$(curl -sf http://checkip.amazonaws.com || hostname -I | awk '{print $1}')
echo ""
echo "  Frontend  →  http://$VM_IP"
echo "  API docs  →  http://$VM_IP/api/docs"
echo "  Parse     →  POST http://$VM_IP/api/parse"
echo ""
echo "  Logs:     sudo journalctl -u ocr-backend -f"
echo "  Restart:  sudo systemctl restart ocr-backend"
echo ""
echo "  Local dev tunnel (run on your Mac to use VM's Ollama locally):"
echo "  ssh -L 11434:localhost:11434 $APP_USER@$VM_IP"
