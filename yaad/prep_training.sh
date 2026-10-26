#!/bin/bash
# Build crops and layout.json from the four dataset jobs, then clean up test leftovers.
set -euo pipefail
JOBS="d8fbf0a5-9775-4666-a4c5-1b3c2612e483 d1c13abd-b43c-4f04-9193-680b28fbcf04 8d12b83b-9f5f-4fd7-b378-5aadefe19abc f8436a4b-f49a-4ca4-bd7a-bccc19923bd0"
cd /workspace/vial_classifier
bash build_crops.sh $JOBS
runs=""
for j in $JOBS; do runs="$runs /workspace/results/$j/out"; done
docker run --rm -v /workspace:/workspace python:3.12-slim bash -c "
  pip install -q --root-user-action ignore numpy opencv-python-headless >/dev/null 2>&1
  cd /workspace/vial_classifier && python predict.py layout --out layout.json $runs"
rm -rf pred_test frames_test layout_test.json
echo PREP_DONE
