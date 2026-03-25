#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/opt/stylomex/style-transfer-ai}"

echo "[stylomex] Deploy started"
cd "$PROJECT_DIR"

if [ ! -d .git ]; then
  echo "[stylomex] ERROR: $PROJECT_DIR is not a git checkout"
  exit 1
fi

echo "[stylomex] Pulling latest main"
git fetch --all --prune
git checkout main
git pull --ff-only origin main

if [ ! -d .venv ]; then
  echo "[stylomex] Creating virtualenv"
  python3.11 -m venv .venv
fi

echo "[stylomex] Installing Python dependencies"
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r install/requirements.txt

echo "[stylomex] Ensuring spaCy model"
python -m spacy download en_core_web_sm

echo "[stylomex] Generating frontend config"
python scripts/generate_frontend_config.py

echo "[stylomex] Restarting services"
sudo systemctl restart stylomex
sudo systemctl reload nginx

echo "[stylomex] Deploy complete"