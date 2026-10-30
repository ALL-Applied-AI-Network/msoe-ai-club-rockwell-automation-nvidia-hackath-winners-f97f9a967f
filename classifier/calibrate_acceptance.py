"""DGX-only internal calibration. Existing validation is NOT an independent final test.

Fit pass confidence on half the validation frames; report the other half separately.
The checkpoint was previously selected using validation, so new-seed and real tests are required.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import torch
import torchvision

from decision import decide


def metrics(rows, probabilities, classes, threshold):
    counts = dict(positions=len(rows), defects=0, defects_accepted=0, good=0, good_held=0,
                  uncertain=0, correct=0)
    for row, p in zip(rows, probabilities):
        cls, _, _ = decide(classes, p.tolist(), .5, threshold)
        truth = row['class']
        defect = truth in ('missing_cap', 'crooked_cap', 'damaged_vial')
        counts['defects'] += int(defect)
        counts['defects_accepted'] += int(defect and cls == 'pass')
        counts['good'] += int(truth == 'pass')
        counts['good_held'] += int(truth == 'pass' and cls != 'pass')
        counts['uncertain'] += int(cls == 'uncertain')
        counts['correct'] += int(cls == truth)
    return counts


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', type=Path, required=True)
    ap.add_argument('--crops', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    assert torch.cuda.is_available(), 'Run this on a DGX Spark GPU'
    torch.set_num_threads(4)
    a.out.mkdir(parents=True, exist_ok=False)
    rows = [r for r in csv.DictReader((a.crops/'index.csv').open()) if r['split'] == 'val']
    assert rows
    ck = torch.load(a.model, map_location='cpu', weights_only=True)
    classes = ck['classes']
    model = torchvision.models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, len(classes))
    model.load_state_dict(ck['state_dict'])
    model.cuda().eval()
    mean = torch.tensor(ck['mean'], device='cuda').view(1,3,1,1)
    std = torch.tensor(ck['std'], device='cuda').view(1,3,1,1)
    def read(row):
        img = cv2.imread(str(a.crops/row['path']))
        assert img is not None and img.shape == (ck['size'], ck['size'], 3)
        return img[:,:,::-1].copy()
    probs = []
    with torch.inference_mode(), ThreadPoolExecutor(4) as pool:
        for i in range(0,len(rows),128):
            x = torch.from_numpy(np.stack(list(pool.map(read,rows[i:i+128])))).permute(0,3,1,2).cuda().float()/255
            probs.append(model((x-mean)/std).softmax(1).cpu().numpy())
    probs = np.concatenate(probs)
    calibration = np.array([int(hashlib.sha256(f"{r['run']}/{r['frame']}".encode()).hexdigest(),16)%2 == 0 for r in rows])
    good = calibration & np.array([r['class']=='pass' for r in rows])
    # A predeclared 1% empirical hold budget; never select a threshold on audit/test errors.
    sorted_good = np.sort(probs[good, classes.index('pass')])
    assert len(sorted_good) >= 100 and (~calibration).any()
    threshold = max(.5, float(sorted_good[int(len(sorted_good)*.01)]))
    policy = dict(pass_min_conf=threshold, model_sha256=hashlib.sha256(a.model.read_bytes()).hexdigest(),
                  calibration_good_hold_budget=.01, status='synthetic-only; requires real calibration',
                  rule='argmax pass below threshold becomes uncertain/review; never relabel as a defect')
    report = dict(device=torch.cuda.get_device_name(0), torch_version=torch.__version__, policy=policy,
                  scope='Internal frame split of existing model-selection validation; not independent test', splits={})
    for name, mask in [('calibration',calibration),('audit',~calibration),('all',np.ones(len(rows),bool))]:
        subset = [r for r,m in zip(rows,mask) if m]
        report['splits'][name] = dict(baseline=metrics(subset,probs[mask],classes,.5),
                                    candidate=metrics(subset,probs[mask],classes,threshold))
    np.savez_compressed(a.out/'probabilities.npz',probabilities=probs,calibration=calibration)
    (a.out/'policy.json').write_text(json.dumps(policy,indent=2)+'\n')
    (a.out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__ == '__main__':
    main()
