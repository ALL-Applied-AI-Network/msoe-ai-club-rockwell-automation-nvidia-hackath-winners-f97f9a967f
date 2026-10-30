"""Train the per-position vial classifier on crops from make_crops.py.

    python train_classifier.py --crops crops --out model [--epochs 15]

Fine-tunes an ImageNet-pretrained ResNet-18 (5 classes: pass, missing_cap, crooked_cap, damaged_vial,
no_vial) with strong photometric and small geometric augmentation, since the model is trained on renders
and will be used on real camera images. The loss is class-weighted, and the checkpoint with the best
validation macro-F1 is kept. Loss weights balance every (tray, class) pair. Writes <out>/model.pt (weights + class list + input size), metrics.json
(per-class and per-tray precision/recall, confusion matrices; rewritten after every epoch) and train_log.csv.

All crops are decoded once into memory (threads) and augmented on the GPU per sample, so training needs no
DataLoader worker processes (they hang in the YAAD containers, and decoding in-process took 17 min/epoch).
"""
import argparse, csv, json, math, time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn
import torchvision

CLASSES = ["pass", "missing_cap", "crooked_cap", "damaged_vial", "no_vial"]
ap = argparse.ArgumentParser()
ap.add_argument("--crops", type=Path, required=True)
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--epochs", type=int, default=15)
ap.add_argument("--batch", type=int, default=256)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--size", type=int, default=128)
ap.add_argument("--threads", type=int, default=16, help="threads for decoding the crops")
ap.add_argument("--workers", type=int, default=0, help="ignored (kept for older job scripts)")
ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
ap.add_argument("--limit", type=int, default=0, help="use at most N crops per split (smoke tests)")
ap.add_argument("--no-pretrained", action="store_true")
ap.add_argument("--init", type=Path, help="fine-tune an existing checkpoint into a NEW output folder")
ap.add_argument("--seed", type=int, default=301026)
ap.add_argument("--damage-weight", type=float, default=1.0, help="additional damaged-vial loss weight")
ap.add_argument("--model-version", default="v2", help="checkpoint release name")
ap.add_argument("--crop-mode", default="legacy_position_v1")
a = ap.parse_args()
if a.epochs < 1 or a.batch < 2 or a.damage_weight <= 0:
    ap.error("epochs >= 1, batch >= 2 and positive damage weight required")
if a.out.exists() and any(a.out.iterdir()):
    ap.error("output directory must be empty; existing models are never overwritten")
a.out.mkdir(parents=True, exist_ok=True)
torch.manual_seed(a.seed)
np.random.seed(a.seed)
dev = torch.device(a.device)

rows = list(csv.DictReader(open(a.crops / "index.csv")))
split_rows = {s: [r for r in rows if r["split"] == s] for s in ("train", "val")}
assert all(r.get('source','synthetic') == 'synthetic' for rs in split_rows.values() for r in rs), 'real evaluation rows must never enter training/selection'
groups = {s: {(r['run'], r['frame']) for r in rs} for s, rs in split_rows.items()}
assert groups['train'] and groups['val'] and not groups['train'] & groups['val'], 'invalid frame split'
if a.limit:
    rng = np.random.default_rng(0)
    split_rows = {s: [r[i] for i in rng.permutation(len(r))[:a.limit]] for s, r in split_rows.items()}
for s, r in split_rows.items():
    print(f"{s}: {len(r)} crops {dict(Counter(x['class'] for x in r))}", flush=True)


def load(split):
    """All crops of a split as one uint8 NCHW tensor on the device, plus labels."""
    t0 = time.time()

    def read(r):
        img = cv2.imread(str(a.crops / r["path"]), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"unreadable crop: {r['path']}")
        if img.shape[:2] != (a.size, a.size):
            img = cv2.resize(img, (a.size, a.size), interpolation=cv2.INTER_AREA)
        return img[..., ::-1]                                  # BGR -> RGB

    with ThreadPoolExecutor(a.threads) as ex:
        imgs = np.stack(list(ex.map(read, split_rows[split])))
    x = torch.from_numpy(imgs).permute(0, 3, 1, 2).contiguous().to(dev)
    y = torch.tensor([CLASSES.index(r["class"]) for r in split_rows[split]], device=dev)
    print(f"loaded {split}: {tuple(x.shape)} in {time.time() - t0:.0f}s", flush=True)
    return x, y


xtr, ytr = load("train")
xva, yva = load("val")
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
mean_t = torch.tensor(MEAN, device=dev).view(1, 3, 1, 1)
std_t = torch.tensor(STD, device=dev).view(1, 3, 1, 1)
LUM = torch.tensor([0.299, 0.587, 0.114], device=dev).view(1, 3, 1, 1)
g = torch.exp(-torch.arange(-2, 3, device=dev, dtype=torch.float32) ** 2 / 2)
BLUR = (g[:, None] * g[None, :] / g.sum() ** 2).expand(3, 1, 5, 5).contiguous()


def augment(x):
    """Per-sample GPU augmentation of a uint8 NCHW batch -> normalized float batch."""
    n = x.shape[0]
    x = x.float() / 255

    def u(lo, hi):
        return torch.empty(n, device=dev).uniform_(lo, hi)

    # Geometry: zoom, small rotation (half the samples), shift, horizontal flip.
    s, ang = u(0.87, 1.15), u(-8, 8) * (torch.rand(n, device=dev) < 0.5)
    flip = torch.where(torch.rand(n, device=dev) < 0.5, -1.0, 1.0)
    c, si = torch.cos(torch.deg2rad(ang)) / s, torch.sin(torch.deg2rad(ang)) / s
    theta = torch.stack([torch.stack([c * flip, -si, u(-0.06, 0.06)], 1),
                         torch.stack([si * flip, c, u(-0.06, 0.06)], 1)], 1)
    x = Fn.grid_sample(x, Fn.affine_grid(theta, x.shape, align_corners=False), padding_mode="border",
                       align_corners=False)
    # Photometric: brightness, contrast, saturation, grayscale.
    x = x * u(0.6, 1.4).view(n, 1, 1, 1)
    m = x.mean((1, 2, 3), keepdim=True)
    x = (x - m) * u(0.6, 1.4).view(n, 1, 1, 1) + m
    gray = (x * LUM).sum(1, keepdim=True)
    sat = u(0.6, 1.4).view(n, 1, 1, 1) * (torch.rand(n, 1, 1, 1, device=dev) >= 0.1)
    x = gray + (x - gray) * sat
    # Lower effective resolution, blur and sensor noise, each on about a third of the samples.
    sel = torch.rand(n, device=dev) < 0.3
    if sel.any():
        f = float(np.random.uniform(0.5, 0.8))
        small = Fn.interpolate(x[sel], scale_factor=f, mode="bilinear", antialias=True, align_corners=False)
        x[sel] = Fn.interpolate(small, size=x.shape[-2:], mode="bilinear", align_corners=False)
    sel = torch.rand(n, device=dev) < 0.3
    if sel.any():
        x[sel] = Fn.conv2d(Fn.pad(x[sel], (2, 2, 2, 2), mode="replicate"), BLUR, groups=3)
    sel = (torch.rand(n, 1, 1, 1, device=dev) < 0.3).float()
    x = x + sel * 0.03 * torch.randn_like(x)
    return (x.clamp(0, 1) - mean_t) / std_t


def prep(x):
    return (x.float() / 255 - mean_t) / std_t


weights = None if a.no_pretrained or a.init else torchvision.models.ResNet18_Weights.IMAGENET1K_V1
model = torchvision.models.resnet18(weights=weights)
model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
if a.init:
    ck = torch.load(a.init, map_location='cpu', weights_only=True)
    assert ck['classes'] == CLASSES and ck['size'] == a.size
    assert ck['mean'] == MEAN and ck['std'] == STD
    model.load_state_dict(ck['state_dict'])
model = model.to(dev).to(memory_format=torch.channels_last)
print(f"device: {dev}; GPU: {torch.cuda.get_device_name(dev) if dev.type == 'cuda' else 'none'}", flush=True)

# Per-sample weights by inverse frequency of (tray, class). Weighting by class alone let damaged_vial
# (S tray only, rare overall) outweigh S-tray pass/missing_cap so much that every lying vial became "damaged".
freq = Counter(r["class"] for r in split_rows["train"])
group = Counter((r["tray"], r["class"]) for r in split_rows["train"])
gw = {k: min(10.0, len(split_rows["train"]) / (len(group) * n)) for k, n in group.items()}   # cap tiny groups
wtr = torch.tensor([gw[(r["tray"], r["class"])] * (a.damage_weight if r['class'] == 'damaged_vial' else 1)
                    for r in split_rows["train"]], device=dev)
print("loss weight per (tray, class):", {f"{t}/{c}": round(w, 2) for (t, c), w in sorted(gw.items())}, flush=True)
ce = nn.CrossEntropyLoss(reduction="none", label_smoothing=0.05)


def loss_fn(logits, y, w):
    return (ce(logits, y) * w).sum() / w.sum()
opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.05)
steps_per_epoch = max(1, math.ceil(len(ytr) / a.batch))
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.epochs * steps_per_epoch, pct_start=0.1)
amp = dev.type == "cuda"


def evaluate():
    model.eval()
    preds = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
        for i in range(0, len(yva), a.batch):
            preds.append(model(prep(xva[i:i + a.batch]).contiguous(memory_format=torch.channels_last)).argmax(1))
    preds = torch.cat(preds).cpu().numpy()
    y = yva.cpu().numpy()
    trays = np.array([r["tray"] for r in split_rows["val"]])

    def report(mask):
        cm = np.zeros((len(CLASSES), len(CLASSES)), int)
        np.add.at(cm, (y[mask], preds[mask]), 1)
        per = {}
        for k, c in enumerate(CLASSES):
            tp, n_true, n_pred = cm[k, k], cm[k].sum(), cm[:, k].sum()
            if n_true == 0:
                continue
            p, r = (tp / n_pred if n_pred else 0.0), tp / n_true
            per[c] = {"precision": round(float(p), 4), "recall": round(float(r), 4),
                      "f1": round(float(2 * p * r / (p + r)) if p + r else 0.0, 4), "support": int(n_true)}
        return {"accuracy": round(float((y[mask] == preds[mask]).mean()), 4) if mask.any() else None,
                "macro_f1": round(float(np.mean([v["f1"] for v in per.values()])), 4) if per else None,
                "per_class": per, "confusion": cm.tolist()}

    return {"all": report(np.ones(len(y), bool)), **{f"tray_{t}": report(trays == t) for t in sorted(set(trays))}}


log = open(a.out / "train_log.csv", "w", newline="")
logw = csv.writer(log)
logw.writerow(["epoch", "train_loss", "val_acc", "val_macro_f1", "seconds"])
best, best_metrics, best_epoch = -1.0, None, 0
for ep in range(1, a.epochs + 1):
    model.train()
    t0, total, n = time.time(), 0.0, 0
    perm = torch.randperm(len(ytr), device=dev)
    for b in range(steps_per_epoch):
        idx = perm[b * a.batch:(b + 1) * a.batch]
        x = augment(xtr[idx]).contiguous(memory_format=torch.channels_last)
        y = ytr[idx]
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            loss = loss_fn(model(x).float(), y, wtr[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        total, n = total + loss.item() * len(y), n + len(y)
    m = evaluate()
    f1 = m["all"]["macro_f1"] or 0.0
    logw.writerow([ep, round(total / max(n, 1), 4), m["all"]["accuracy"], f1, round(time.time() - t0, 1)]); log.flush()
    print(f"epoch {ep}/{a.epochs} loss {total / max(n, 1):.4f} val acc {m['all']['accuracy']} macro-F1 {f1} "
          f"({time.time() - t0:.0f}s)", flush=True)
    if f1 > best:
        best, best_metrics, best_epoch = f1, m, ep
        torch.save({"state_dict": model.state_dict(), "classes": CLASSES, "size": a.size, "mean": MEAN, "std": STD,
                    "arch": "resnet18", "epoch": ep, "version": a.model_version, "crop_mode": a.crop_mode,
                    "training_source": "synthetic-only"}, a.out / "model.pt")
    json.dump({"best_macro_f1": best, "best_epoch": best_epoch, "epochs_done": ep, "classes": CLASSES,
               "metrics": best_metrics, "train_counts": dict(freq), "args": {k: str(v) for k, v in vars(a).items()}},
              open(a.out / "metrics.json", "w"), indent=1)

print(f"best validation macro-F1 {best} (epoch {best_epoch})")
for scope, m in best_metrics.items():
    print(f"[{scope}] acc {m['accuracy']} macro-F1 {m['macro_f1']}")
    for c, v in m["per_class"].items():
        print(f"   {c:13s} P {v['precision']:.3f} R {v['recall']:.3f} F1 {v['f1']:.3f} (n={v['support']})")
    print("   confusion (rows true, cols predicted):", m["confusion"])
