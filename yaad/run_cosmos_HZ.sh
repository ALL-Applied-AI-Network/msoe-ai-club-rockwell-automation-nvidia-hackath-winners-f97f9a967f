#!/bin/bash
set -euo pipefail
# Execute on the YAAD login host; /workspace maps to the team's workspace.
work=/home/yt-winners7506/workspace/cosmos_HZ
test -f "$work/cosmos_service.py"
mkdir -p "$work/cache"
if docker container inspect hz-cosmos >/dev/null 2>&1; then
  docker start hz-cosmos >/dev/null
  echo 'Cosmos container is running; check /health for model readiness.'
  exit 0
fi
docker run -d --name hz-cosmos --gpus all --memory 40g --cpus 4 \
  -p 127.0.0.1:8766:8766 -v "$work:/app" -w /app \
  -e HF_HOME=/app/cache -e PYTHONUNBUFFERED=1 \
  python:3.12-slim bash -lc \
  'set -eu; apt-get update -qq; apt-get install -y -qq gcc >/app/compiler-install.log 2>&1; pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu130; pip install -q "transformers==4.57.3" accelerate pillow fastapi uvicorn; python -m uvicorn cosmos_service:app --host 0.0.0.0 --port 8766'
