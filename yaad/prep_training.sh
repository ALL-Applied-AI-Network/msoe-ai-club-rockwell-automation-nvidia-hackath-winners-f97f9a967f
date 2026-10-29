#!/bin/bash
# Build crops and layout.json from finished dataset jobs, then clean up test leftovers.
#   bash prep_training.sh JOB_ID [JOB_ID ...]
set -euo pipefail
cd /workspace/vial_classifier
bash build_crops.sh "$@"
runs=""
for j in "$@"; do runs="$runs /workspace/results/$j/out"; done
docker run --rm -v /workspace:/workspace python:3.12-slim bash -c "
  pip install -q --root-user-action ignore numpy opencv-python-headless >/dev/null 2>&1
  cd /workspace/vial_classifier && python predict.py layout --out layout.json $runs"
rm -rf pred_test frames_test layout_test.json damage_eval crackpick.tgz
echo PREP_DONE
