#!/bin/bash
# S-tray damage preview in the team workspace (pool slots are busy with the dataset run).
set -uo pipefail
OUT=/workspace/sdg_s_preview
mkdir -p "$OUT"
docker run --rm -u 0 --gpus all -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y \
  -v /workspace/csi_digital_twin_v1:/job/in -v "$OUT":/job/out \
  --entrypoint bash nvcr.io/nvidia/isaac-sim:5.1.0 -c '
    cd /job/in
    /isaac-sim/python.sh -c "import cv2" 2>/dev/null || /isaac-sim/python.sh -m pip install --quiet opencv-python-headless
    /isaac-sim/python.sh tools/sdg_vials.py --trays G,S --frames 24 --seed 3 --out /job/out --preview \
      --weights pass=0.4,missing_cap=0.2,crooked_cap=0.2,damaged_vial=0,no_vial=0.2
  '
echo "exit=$?"
