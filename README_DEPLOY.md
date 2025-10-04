## WhatsApp Agent - Production Deployment Guide

This guide covers two production deployment options for Ubuntu 22.04+:
- Docker Compose (recommended)
- systemd + Python virtualenv

Both approaches support environment variables, logs, healthchecks, and CI/CD.

### Requirements
- Ubuntu 22.04+
- A domain (recommended) pointing to your server IP
- WhatsApp provider credentials (Meta Business / Twilio)
- Optional: paid ngrok account for reserved domain if you must use ngrok

---

## 1) Deploy with Docker Compose (recommended)

### 1.1. Install Docker & Compose
```bash
sudo apt-get update -y
sudo apt-get install -y ca-certificates curl gnupg
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt-get update -y
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
sudo usermod -aG docker $USER
newgrp docker
```

### 1.2. Clone repo and prepare env
```bash
sudo mkdir -p /opt/whatsapp-agent && sudo chown -R $USER:$USER /opt/whatsapp-agent
cd /opt/whatsapp-agent
# Copy your repository here
# git clone <your-repo-url> .

cp .env.example .env
# Edit .env and set secrets (WHATSAPP_ACCESS_TOKEN, PHONE_NUMBER_ID, WEBHOOK_VERIFY_TOKEN, SECRET_KEY, etc.)
$EDITOR .env
```

### 1.3. Configure reverse proxy (Caddy)
- `docker-compose.yml` includes a `proxy` service using Caddy.
- Update DNS A record for your domain to your server IP.
- Edit `Caddyfile` to use your domain and enable HTTPS with Let’s Encrypt:
```text
{
  email you@example.com
}

# Replace example.com with your domain
example.com {
    encode zstd gzip
    reverse_proxy agent:8001
}

# Optional local HTTP
:80 {
    encode zstd gzip
    reverse_proxy agent:8001
}
```

### 1.4. Start services
```bash
docker compose up -d --build
# Verify health
curl -fsS http://localhost:8001/healthz
```

If using your domain and HTTPS, verify via:
```bash
curl -I https://example.com/healthz
```

### 1.5. Logs and maintenance
- View logs: `docker compose logs -f agent`
- Restart: `docker compose restart agent`
- Update: `docker compose pull && docker compose up -d`
- Rollback: deploy a previous image tag and `docker compose up -d`

### 1.6. Register webhook URL
- Preferred: use your domain `https://example.com/api/webhook`
- Alternative (ngrok paid): use `NGROK_AUTHTOKEN` and `NGROK_DOMAIN` and run the app with `USE_NGROK=1`.
- For Meta WhatsApp Cloud API: set the callback in your App dashboard or via API. Ensure `WEBHOOK_VERIFY_TOKEN` matches.

---

## 2) Deploy with systemd + virtualenv

### 2.1. Install system packages
```bash
sudo apt-get update -y
sudo apt-get install -y python3-venv rsync ufw
```

### 2.2. Copy repo and install service
```bash
# On the server, from the repo root
sudo bash ./install_service.sh
# Or specify variables
# sudo REPO_DIR=/opt/whatsapp-agent PORT=8001 SERVICE_USER=whatsagent bash ./install_service.sh
```

The script:
- Creates user `whatsagent`
- Copies repo to `/opt/whatsapp-agent`
- Creates venv and installs requirements
- Creates `/etc/whatsapp-agent/env` (chmod 600) for environment variables
- Installs and enables `whatsapp-agent.service`

### 2.3. Configure environment
Edit `/etc/whatsapp-agent/env` and set:
```text
ENV=production
PORT=8001
SECRET_KEY=...
WHATSAPP_ACCESS_TOKEN=...
WHATSAPP_PHONE_NUMBER_ID=...
WEBHOOK_VERIFY_TOKEN=...
ALLOWED_HOSTS=yourdomain.com
DISABLE_DB=1
```
Then restart:
```bash
sudo systemctl restart whatsapp-agent
```

### 2.4. Logs and healthchecks
- Logs: `journalctl -u whatsapp-agent -f`
- Health: `curl http://localhost:8001/healthz`

### 2.5. Firewall and SSH hardening
```bash
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw allow 8001/tcp
sudo ufw enable
sudo sed -i 's/^#\?PasswordAuthentication yes/PasswordAuthentication no/' /etc/ssh/sshd_config
sudo systemctl reload sshd
```

---

## CI/CD (GitHub Actions)
- Workflow `.github/workflows/ci-cd.yml` runs lint/tests, builds, and pushes an image to GHCR.
- Optional deploy job SSHes to the server and runs `docker compose up -d` using the remote `docker-compose.yml`.

Required secrets:
- `SSH_PRIVATE_KEY`: private key for the deploy user
- `SERVER_HOST`: server IP or hostname
- Optional: `SERVER_USER`, `APP_DIR`

---

## Observability
- Logs to stdout; structured JSON logs can be enabled with `LOG_FORMAT=json`.
- `/healthz` and `/health` endpoints for health checks; `/status` returns version and uptime.
- Optional `/metrics` if `ENABLE_METRICS=1` and `prometheus_fastapi_instrumentator` is installed.

---

## Rollbacks & Backups
- Keep previous image tags. To rollback: retag and `docker compose up -d --no-deps --build agent`.
- For systemd: `sudo systemctl restart whatsapp-agent` after reverting code.
- If using Mongo/Redis in Docker, snapshot volumes or use managed services.

---

## Ngrok in production?
- Prefer a real domain with TLS via Caddy.
- If you must use ngrok, use a paid plan with `NGROK_AUTHTOKEN` and `NGROK_DOMAIN` for a stable URL; set `USE_NGROK=1`.
