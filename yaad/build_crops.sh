#!/bin/bash
# Build /workspace/vial_classifier/crops from finished sdg_vials.py jobs (CPU container in the workspace).
#   bash /workspace/vial_classifier/build_crops.sh JOB_ID [JOB_ID ...]
set -euo pipefail
cd /workspace/vial_classifier
rm -rf crops crops_test smoke_model sample*.tgz
runs=""
for j in "$@"; do
  test -d "/workspace/results/$j/out/labels" || { echo "no labels for job $j"; exit 1; }
  runs="$runs results/$j/out"
done
docker run --rm -v /workspace:/workspace python:3.12-slim bash -c "
  pip install -q --root-user-action ignore numpy opencv-python-headless >/dev/null 2>&1
  cd /workspace && python vial_classifier/make_crops.py --out vial_classifier/crops $runs"
du -sh crops
