#!/usr/bin/env bash
set -euo pipefail
cd /job/in
python -m pip install --quiet numpy opencv-python-headless pillow
python -m pip install --quiet torch torchvision --index-url https://download.pytorch.org/whl/cu130
python - <<'PY'
import json,torch
assert torch.cuda.is_available()
json.dump({'device':torch.cuda.get_device_name(0),'torch_version':torch.__version__},open('/job/out/runtime.json','w'),indent=2)
PY
python classifier/train_classifier.py --device cuda --init v3_HZ.pt --crops crops --out /job/out/v4_HZ \
  --epochs 20 --batch 512 --lr 0.0002 --threads 8 --seed 30102644 --model-version v4_HZ \
  --crop-mode review_bbox_v1 --real-repeat 64
python classifier/evaluate_HZ.py --real real --model /job/out/v4_HZ/model.pt \
  --out /job/out/v4_HZ/reviewed-real-fit.json --version v4_HZ
python classifier/evaluate_synthetic_HZ.py --crops crops --model /job/out/v4_HZ/model.pt --out /job/out/v4_HZ/synthetic-test.json
python - <<'PY'
import hashlib,json,torch
from pathlib import Path
p=Path('/job/out/v4_HZ/model.pt');ck=torch.load(p,map_location='cpu',weights_only=True)
assert ck['version']=='v4_HZ' and ck['training_source']=='synthetic+reviewed-real' and ck['real_repeat']==64
json.dump({'model_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
           'review_sha256':'22de39ca98a254f9e65fa03b889fd5915c5d342faa3bf40a138818107fcaff6c',
           'training_source':ck['training_source'],'real_repeat':ck['real_repeat'],
           'limitation':'The two reviewed real images were used for training; their fit metrics are not independent accuracy.'},
          open('/job/out/v4_HZ/provenance.json','w'),indent=2)
PY
cp /job/out/runtime.json /job/out/v4_HZ/runtime.json
