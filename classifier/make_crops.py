"""Cut one crop per labeled tray position out of sdg_vials.py output, for training a per-position classifier.

    python make_crops.py --out crops RUN_DIR [RUN_DIR ...]

Each RUN_DIR is an sdg_vials.py output folder (frames/ and labels/), e.g. results/<job_id>/out.
G positions are cropped as a square around the aperture (neighbouring vials are partly in view, like the
real labeled crops). S positions (lying vials, about 30 px apart) are cropped along the intact vial's axis,
rotated so the vial lies horizontal, at a fixed scale whatever the damage; outside a band around the target
vial the crop is dimmed, so the label clearly belongs to one vial. The crop code lives in vial_crops.py and
is shared with predict.py. Crops are written to <out>/<split>/<class>/<run>_<frame>_<pos>.png, and
<out>/index.csv lists every crop with its labels. The train/val split is by frame, so crops of one frame
never end up in both.
"""
import argparse, csv, json, sys, zlib
from collections import Counter
from pathlib import Path

import cv2

import vial_crops

ap = argparse.ArgumentParser()
ap.add_argument("runs", nargs="+", type=Path)
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--size", type=int, default=128, help="crop size in pixels after resizing")
ap.add_argument("--g-scale", type=float, default=vial_crops.G_SCALE, help="G crop side as a multiple of the aperture radius")
ap.add_argument("--s-scale", type=float, default=vial_crops.S_SCALE, help="S crop side as a multiple of the vial length")
ap.add_argument("--s-band", type=float, default=vial_crops.S_BAND, help="S target band half-height as a multiple of the vial diameter")
ap.add_argument("--s-dim", type=float, default=vial_crops.S_DIM, help="brightness outside the S target band")
ap.add_argument("--val-every", type=int, default=10, help="every Nth frame (by hash) goes to the validation split")
a = ap.parse_args()

rows, counts = [], Counter()
for run in a.runs:
    labels = sorted((run / "labels").glob("*.json"))
    if not labels:
        sys.exit(f"no labels in {run}")
    run_id = run.parent.name if run.name == "out" else run.name
    for lp in labels:
        lab = json.loads(lp.read_text())
        img = cv2.imread(str(run / lab["image"]))
        if img is None:
            sys.exit(f"cannot read {run / lab['image']}")
        padded = vial_crops.pad(img)
        frame = f"{lab['frame']:06d}"
        split = "val" if zlib.crc32(f"{run_id}/{frame}".encode()) % a.val_every == 0 else "train"
        for pos, info in lab["positions"].items():
            if not info.get("in_frame"):
                continue
            if pos.startswith("G"):
                crop = vial_crops.g_crop(padded, info, a.size, a.g_scale)
            else:
                crop = vial_crops.s_crop(padded, info, a.size, a.s_scale, a.s_band, a.s_dim)
            if crop is None:
                continue
            cls = info["class"]
            rel = Path(split) / cls / f"{run_id[:8]}_{frame}_{pos}.png"
            (a.out / rel).parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(a.out / rel), crop)
            rows.append({"path": rel.as_posix(), "class": cls, "tray": pos[0], "pos": pos, "frame": frame,
                         "run": run_id, "split": split, "damage": info.get("damage", "")})
            counts[(split, pos[0], cls)] += 1

a.out.mkdir(parents=True, exist_ok=True)
with open(a.out / "index.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print(f"{len(rows)} crops from {len(a.runs)} runs -> {a.out}")
for (split, tray, cls), n in sorted(counts.items()):
    print(f"  {split:5s} {tray} {cls:13s} {n}")
