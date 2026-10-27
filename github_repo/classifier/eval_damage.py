"""Per-damage-type recall of a trained model on the validation S crops, plus a sheet of missed crops.

    python eval_damage.py --crops crops --model model.pt --out damage_eval
"""
import argparse, csv
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision

ap = argparse.ArgumentParser()
ap.add_argument("--crops", type=Path, required=True)
ap.add_argument("--model", type=Path, required=True)
ap.add_argument("--out", type=Path, required=True)
a = ap.parse_args()
a.out.mkdir(parents=True, exist_ok=True)

ck = torch.load(a.model, map_location="cpu", weights_only=False)
classes = ck["classes"]
model = torchvision.models.resnet18()
model.fc = torch.nn.Linear(model.fc.in_features, len(classes))
model.load_state_dict(ck["state_dict"])
model.eval()
mean = torch.tensor(ck["mean"]).view(1, 3, 1, 1)
std = torch.tensor(ck["std"]).view(1, 3, 1, 1)

rows = [r for r in csv.DictReader(open(a.crops / "index.csv")) if r["split"] == "val" and r["tray"] == "S"]
imgs = np.stack([cv2.imread(str(a.crops / r["path"]))[..., ::-1] for r in rows])
x = (torch.from_numpy(imgs.copy()).permute(0, 3, 1, 2).float() / 255 - mean) / std
with torch.no_grad():
    prob = torch.cat([torch.softmax(model(x[i:i + 128]), 1) for i in range(0, len(x), 128)]).numpy()
pred = prob.argmax(1)
dv = classes.index("damaged_vial")

by_kind = defaultdict(Counter)
missed = defaultdict(list)
for r, p, pr in zip(rows, pred, prob):
    if r["class"] != "damaged_vial":
        continue
    by_kind[r["damage"]][classes[p]] += 1
    if p != dv:
        missed[r["damage"]].append((r["path"], float(pr[dv])))
print("S validation damaged_vial by damage type -> predicted class counts")
for k, c in sorted(by_kind.items()):
    n = sum(c.values())
    print(f"  {k:14s} n={n:3d} recall {c['damaged_vial'] / n:.2f}  {dict(c)}")
others = [pr[dv] for r, pr in zip(rows, prob) if r["class"] != "damaged_vial"]
dam = [pr[dv] for r, pr in zip(rows, prob) if r["class"] == "damaged_vial"]
print(f"P(damaged) on intact S vials: median {np.median(others):.3f}, 95th pct {np.percentile(others, 95):.3f}")
print(f"P(damaged) on damaged S vials: median {np.median(dam):.3f}")
for t in (0.5, 0.3, 0.2, 0.1, 0.05):
    rec = np.mean(np.array(dam) >= t)
    fp = np.mean(np.array(others) >= t)
    print(f"  flag damaged if P >= {t:.2f}: damaged recall {rec:.2f}, intact vials flagged {fp:.2f}")

# Sheet: up to 8 missed crops per damage type (rows), and 8 caught ones for comparison.
tiles = []
for k in sorted(by_kind):
    m = [cv2.imread(str(a.crops / pth)) for pth, _ in missed[k][:8]]
    caught = [cv2.imread(str(a.crops / r["path"])) for r, p in zip(rows, pred)
              if r["damage"] == k and p == dv][:8]
    for label, lst in ((f"{k} MISSED", m), (f"{k} caught", caught)):
        row = np.zeros((128, 160 + 8 * 130, 3), np.uint8)
        cv2.putText(row, label, (4, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        for i, im in enumerate(lst):
            row[:, 160 + i * 130:160 + i * 130 + 128] = im
        tiles.append(row)
cv2.imwrite(str(a.out / "damage_sheet.png"), np.vstack(tiles))
print("sheet ->", a.out / "damage_sheet.png")
