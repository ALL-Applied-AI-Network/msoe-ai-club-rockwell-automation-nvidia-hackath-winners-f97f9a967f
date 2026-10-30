#!/bin/bash
set -euo pipefail
cd /job/in
pip install -q numpy opencv-python-headless pillow
pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu130
python - <<'PY'
import json,torch,torchvision
assert torch.cuda.is_available(), 'DGX CUDA is required'
open('/job/out/runtime.json','w').write(json.dumps(dict(torch=torch.__version__,torchvision=torchvision.__version__,device=torch.cuda.get_device_name(0))))
PY
python classifier/prepare_HZ_data.py --train synthetic/train --val synthetic/val --test synthetic/test --real real --out /job/out/crops
python classifier/train_classifier.py --device cuda --init v2.pt --crops /job/out/crops --out /job/out/v3_HZ --epochs 20 --batch 128 --lr 0.0001 --threads 4 --seed 301026 --model-version v3_HZ --crop-mode review_bbox_v1
python classifier/evaluate_HZ.py --real real --model /job/out/v3_HZ/model.pt --version v3_HZ --out /job/out/v3_HZ/real-evaluation.json
python classifier/evaluate_HZ.py --real real --model v2.pt --version v2 --out /job/out/v3_HZ/v2-real-evaluation.json
python classifier/evaluate_synthetic_HZ.py --crops /job/out/crops --model /job/out/v3_HZ/model.pt --out /job/out/v3_HZ/synthetic-test.json
python classifier/evaluate_synthetic_HZ.py --crops /job/out/crops --model v2.pt --out /job/out/v3_HZ/v2-synthetic-test.json
python classifier/calibrate_acceptance.py --model /job/out/v3_HZ/model.pt --crops /job/out/crops --out /job/out/v3_HZ/optional-calibration
cp /job/out/crops/manifest.json /job/out/v3_HZ/data-manifest.json
cp /job/out/runtime.json /job/out/v3_HZ/runtime.json
cp twin_public_view.json /job/out/v3_HZ/camera-fit.json
cp v3_HZ-inputs.json /job/out/v3_HZ/input-provenance.json
