#!/bin/bash
# YAAD pool job: train the vial classifier on /job/in/crops (made by make_crops.py in the workspace).
#   submit_job image=python:3.12-slim input_path=vial_classifier command="bash /job/in/train_job.sh [EPOCHS]"
# Writes model.pt, metrics.json, train_log.csv to /job/out.
set -euo pipefail
EPOCHS=${1:-15}
cd /job/in
pip install -q --root-user-action ignore numpy opencv-python-headless
pip install -q --root-user-action ignore torch torchvision --index-url https://download.pytorch.org/whl/cu130
python - <<'EOF'
import torch
assert torch.cuda.is_available(), "CUDA not available"
print("torch", torch.__version__, "|", torch.cuda.get_device_name(0), "capability", torch.cuda.get_device_capability(0))
EOF
python train_classifier.py --crops crops --out /job/out --epochs "$EPOCHS"
