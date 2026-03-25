# Stylomex Deployment Quickstart

This repository includes a production deployment bundle for Ubuntu + Nginx + systemd.

## 1) One-time server setup

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y git nginx python3.11 python3.11-venv python3-pip curl
curl -fsSL https://ollama.com/install.sh | sh
ollama pull gemma3:1b
```

## 2) Clone repository

```bash
sudo mkdir -p /opt/stylomex
sudo chown -R $USER:$USER /opt/stylomex
cd /opt/stylomex
git clone https://github.com/alwynrejiar/style-transfer-ai.git
cd style-transfer-ai
cp .env.example .env
```

Fill `.env` with real values for:

- `SUPABASE_URL`
- `SUPABASE_ANON_KEY`
- `GEMINI_API_KEY`

## 3) Install systemd units

```bash
sudo cp deploy/systemd/stylomex.service /etc/systemd/system/stylomex.service
sudo cp deploy/systemd/ollama.service /etc/systemd/system/ollama.service
sudo systemctl daemon-reload
sudo systemctl enable ollama stylomex
sudo systemctl start ollama stylomex
```

## 4) Install Nginx config

```bash
sudo cp deploy/nginx/stylomex.conf /etc/nginx/sites-available/stylomex.conf
sudo ln -sf /etc/nginx/sites-available/stylomex.conf /etc/nginx/sites-enabled/stylomex.conf
sudo nginx -t
sudo systemctl reload nginx
```

## 5) GitHub Actions auto-deploy

Create repository secrets:

- `VPS_HOST`
- `VPS_USER`
- `VPS_SSH_KEY`
- `VPS_PORT` (optional)

Every push to `main` triggers `.github/workflows/deploy.yml`.

## 6) Manual deploy command on server

```bash
cd /opt/stylomex/style-transfer-ai
chmod +x deploy/deploy.sh
PROJECT_DIR=/opt/stylomex/style-transfer-ai ./deploy/deploy.sh
```

## 7) Runtime checks

```bash
curl -s http://127.0.0.1:8000/api/health
sudo journalctl -u stylomex -f
sudo journalctl -u ollama -f
sudo tail -f /var/log/nginx/error.log
```