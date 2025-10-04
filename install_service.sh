#!/usr/bin/env bash
set -euo pipefail

# This script installs the WhatsApp Agent as a systemd service on Ubuntu 22.04+
# It creates a dedicated user, sets up a venv, installs deps, and configures systemd.

REPO_DIR=${REPO_DIR:-/opt/whatsapp-agent}
SERVICE_USER=${SERVICE_USER:-whatsagent}
SERVICE_GROUP=${SERVICE_GROUP:-whatsagent}
ENV_DIR=${ENV_DIR:-/etc/whatsapp-agent}
PYTHON_BIN=${PYTHON_BIN:-python3}
PORT=${PORT:-8001}

if [[ $EUID -ne 0 ]]; then
  echo "Please run as root (sudo)." >&2
  exit 1
fi

# Create service user
if ! id -u "$SERVICE_USER" >/dev/null 2>&1; then
  adduser --system --group --home "$REPO_DIR" "$SERVICE_USER"
fi

mkdir -p "$REPO_DIR" "$ENV_DIR"
chown -R "$SERVICE_USER":"$SERVICE_GROUP" "$REPO_DIR"
chmod 750 "$ENV_DIR"

# Copy repository into place (assumes current directory is repo root)
rsync -a --delete --exclude '.git' --exclude '.venv' ./ "$REPO_DIR"/

# Create venv and install deps
sudo -u "$SERVICE_USER" "$PYTHON_BIN" -m venv "$REPO_DIR/.venv"
"$REPO_DIR/.venv/bin/pip" install --upgrade pip
"$REPO_DIR/.venv/bin/pip" install -r "$REPO_DIR/requirements.txt"

# Create environment file (edit manually or supply via ENV_SRC)
if [[ -n "${ENV_SRC:-}" && -f "$ENV_SRC" ]]; then
  install -m 600 -o "$SERVICE_USER" -g "$SERVICE_GROUP" "$ENV_SRC" "$ENV_DIR/env"
else
  if [[ ! -f "$ENV_DIR/env" ]]; then
    cat > "$ENV_DIR/env" <<EOF
ENV=production
PORT=$PORT
APP_NAME=WhatsApp AI Agent
SECRET_KEY=change-me
ALLOWED_HOSTS=*
DISABLE_DB=1
EOF
    chown "$SERVICE_USER":"$SERVICE_GROUP" "$ENV_DIR/env"
    chmod 600 "$ENV_DIR/env"
  fi
fi

# Install systemd unit
install -m 644 systemd/whatsapp-agent.service /etc/systemd/system/whatsapp-agent.service
systemctl daemon-reload
systemctl enable whatsapp-agent
systemctl restart whatsapp-agent

# Firewall (UFW): allow HTTP/HTTPS
if command -v ufw >/dev/null 2>&1; then
  ufw allow 80/tcp || true
  ufw allow 443/tcp || true
  ufw allow ${PORT}/tcp || true
fi

echo "Service installed. View logs: journalctl -u whatsapp-agent -f"
