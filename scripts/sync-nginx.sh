#!/usr/bin/env bash
# Copy repo nginx.conf → system site config (__APP_DIR__ → real path), test, reload.
# Run after `git pull` if you did not use deploy.sh (which calls this).
set -euo pipefail
APP_DIR="${APP_DIR:-$HOME/ocr-project}"
if [[ ! -f "$APP_DIR/nginx.conf" ]]; then
  echo "error: missing $APP_DIR/nginx.conf" >&2
  exit 1
fi
sed "s|__APP_DIR__|$APP_DIR|g" "$APP_DIR/nginx.conf" | sudo tee /etc/nginx/sites-available/ocr-project > /dev/null
sudo ln -sf /etc/nginx/sites-available/ocr-project /etc/nginx/sites-enabled/ocr-project
sudo nginx -t
sudo systemctl reload nginx
echo "nginx: installed site from $APP_DIR/nginx.conf and reloaded."
