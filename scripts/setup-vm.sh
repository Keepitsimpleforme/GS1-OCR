#!/usr/bin/env bash
# =============================================================================
# setup-vm.sh — One-time provisioning for Ubuntu 22.04 LTS
#
# Run once on a fresh VM as a sudo-capable user:
#   chmod +x scripts/setup-vm.sh
#   ./scripts/setup-vm.sh
#
# What it does
# ────────────
#  1. System packages: Python 3, Node 20, nginx, libzbar0, git, curl
#  2. Ollama (local LLM runtime) + pulls required models
#  3. Clones the project to /opt/ocr-project (or uses existing clone)
#  4. Creates Python venv + installs pip deps
#  5. Builds the React frontend
#  6. Installs the systemd service for the FastAPI backend
#  7. Configures nginx as a reverse proxy
# =============================================================================
set -euo pipefail

# ── Config (edit these before running) ────────────────────────────────────────
REPO_URL="https://github.com/YOUR_ORG/ocr-project.git"   # ← change
APP_DIR="/opt/ocr-project"
APP_USER="${SUDO_USER:-ubuntu}"   # owner of the app files
OCR_MODEL="maternion/LightOnOCR-2"
# ──────────────────────────────────────────────────────────────────────────────

echo "━━━ [1/7] System packages ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
sudo apt-get update -y
sudo apt-get install -y \
    git curl nginx \
    python3 python3-pip python3-venv python3-dev \
    libzbar0 build-essential

# Node.js 20 (skip if already >= 18)
NODE_MAJOR=$(node --version 2>/dev/null | grep -oP '(?<=v)\d+' | head -1 || echo 0)
if [ "$NODE_MAJOR" -lt 18 ]; then
    echo "Installing Node.js 20 LTS…"
    curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
    sudo apt-get install -y nodejs
fi
echo "Node $(node --version)  |  npm $(npm --version)"

echo ""
echo "━━━ [2/7] Ollama ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if ! command -v ollama &>/dev/null; then
    curl -fsSL https://ollama.com/install.sh | sh
fi
sudo systemctl enable --now ollama
echo "Waiting for Ollama to start…"
sleep 5
echo "Pulling OCR model — this can take several minutes on first run"
ollama pull "$OCR_MODEL"

echo ""
echo "━━━ [3/7] Clone / update project ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if [ -d "$APP_DIR/.git" ]; then
    echo "Repo already exists — pulling latest"
    sudo git -C "$APP_DIR" pull origin main
else
    sudo git clone "$REPO_URL" "$APP_DIR"
fi
sudo chown -R "$APP_USER":"$APP_USER" "$APP_DIR"

echo ""
echo "━━━ [4/7] Python venv + pip deps ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
cd "$APP_DIR/backend"
python3 -m venv .venv
.venv/bin/pip install --upgrade pip --quiet
.venv/bin/pip install -r requirements.txt --quiet
echo "Backend deps installed"

echo ""
echo "━━━ [5/7] React frontend build ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
cd "$APP_DIR/frontend"
npm ci --prefer-offline
npm run build
echo "Frontend built → $APP_DIR/frontend/dist"

echo ""
echo "━━━ [6/7] systemd service ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
# Patch User= to the real app user before installing
sed "s/^User=ubuntu/User=$APP_USER/" \
    "$APP_DIR/backend/ocr-backend.service" \
    | sudo tee /etc/systemd/system/ocr-backend.service > /dev/null
sudo systemctl daemon-reload
sudo systemctl enable --now ocr-backend
echo "Backend service status:"
sudo systemctl status ocr-backend --no-pager | head -8

echo ""
echo "━━━ [7/7] nginx ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
sudo cp "$APP_DIR/nginx.conf" /etc/nginx/sites-available/ocr-project
sudo ln -sf /etc/nginx/sites-available/ocr-project /etc/nginx/sites-enabled/ocr-project
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t
sudo systemctl enable --now nginx
sudo systemctl reload nginx

echo ""
echo "━━━ Done! ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
VM_IP=$(curl -sf http://checkip.amazonaws.com || hostname -I | awk '{print $1}')
echo "  App:      http://$VM_IP"
echo "  API docs: http://$VM_IP/api/docs    (FastAPI Swagger UI — via nginx)"
echo "  Parse:    POST http://$VM_IP/api/parse  (multipart/form-data, field: file)"
echo ""
echo "  View backend logs:  sudo journalctl -u ocr-backend -f"
echo "  Restart backend:    sudo systemctl restart ocr-backend"
