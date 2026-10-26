"""Classify every tray position in cap-seal camera frames with the trained vial classifier.

1. Build a layout once: the nominal pixel position of every aperture (G) and lying vial (S), averaged over
   synthetic labels (camera jitter and vial placement average out):

       python predict.py layout --out layout.json RUN_DIR [RUN_DIR ...]

2. Classify frames (files or folders of .png/.jpg, 640x480 like the real camera):

       python predict.py run --model model.pt --layout layout.json --out preds IMAGE_OR_DIR [...]

   If the real camera sits slightly differently from the twin, move the whole layout with --shift DX DY,
   --scale S and --rotate DEG (about the image centre) and check the overlays. --check-layout only draws
   the layout on the frames, without the model, for aligning it.

   With --labels DIR, each frame's labels are read from DIR/<frame name>.json and accuracy is reported. The
   file can be an sdg_vials.py label ({"positions": {"G01": {"class": ...}}}) or a plain {"G01": "pass", ...}.

Writes <out>/<frame>.json (class, confidence and all class probabilities per position), <out>/predictions.csv,
and <out>/overlays/<frame>.png. Positions whose top probability is below --min-conf are marked uncertain.
"""
import argparse, csv, json, math, sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

import vial_crops

COLORS = {"pass": (80, 200, 60), "missing_cap": (0, 150, 255), "crooked_cap": (230, 60, 230),
          "damaged_vial": (40, 40, 255), "no_vial": (255, 160, 40), "uncertain": (200, 200, 200)}
SHORT = {"pass": "P", "missing_cap": "MC", "crooked_cap": "CC", "damaged_vial": "DV", "no_vial": "NV", "uncertain": "?"}


def build_layout(a):
    acc = defaultdict(lambda: defaultdict(list))
    n_frames = 0
    for run in a.runs:
        for lp in sorted((run / "labels").glob("*.json")):
            n_frames += 1
            for pos, info in json.loads(lp.read_text())["positions"].items():
                for key in ("aperture_px", "radius_px", "axis_px", "diam_px", "bbox_px"):
                    if info.get(key) is not None:
                        acc[pos][key].append(np.asarray(info[key], dtype=float))
    if not n_frames:
        sys.exit("no label files found")
    layout = {"source_frames": n_frames, "positions": {}}
    for pos in sorted(acc):
        layout["positions"][pos] = {k: np.round(np.mean(v, axis=0), 2).tolist() for k, v in acc[pos].items()}
    a.out.write_text(json.dumps(layout, indent=1))
    kinds = Counter(p[0] for p in layout["positions"])
    print(f"layout from {n_frames} frames: {dict(kinds)} positions -> {a.out}")
    if any(p.startswith("S") and "axis_px" not in v for p, v in layout["positions"].items()):
        print("note: S positions have no axis_px in these labels; crops fall back to the box (older generator)")


def transform_layout(positions, a, w, h):
    """Apply --scale/--rotate about the image centre, then --shift, to every pixel coordinate."""
    c, s = math.cos(math.radians(a.rotate)), math.sin(math.radians(a.rotate))

    def pt(u, v):
        x, y = (u - w / 2) * a.scale, (v - h / 2) * a.scale
        return [x * c - y * s + w / 2 + a.shift[0], x * s + y * c + h / 2 + a.shift[1]]

    out = {}
    for pos, info in positions.items():
        q = dict(info)
        if "aperture_px" in q:
            q["aperture_px"] = pt(*q["aperture_px"])
        if "axis_px" in q:
            q["axis_px"] = [pt(*p) for p in q["axis_px"]]
        if "bbox_px" in q:                                     # only used when axis_px is missing
            x0, y0, x1, y1 = q["bbox_px"]
            (x0, y0), (x1, y1) = pt(x0, y0), pt(x1, y1)
            q["bbox_px"] = [min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)]
        for k in ("radius_px", "diam_px"):
            if k in q:
                q[k] = q[k] * a.scale
        out[pos] = q
    return out


def true_labels(path):
    if not path.exists():
        return None
    d = json.loads(path.read_text())
    d = d.get("positions", d)
    return {p: (v["class"] if isinstance(v, dict) else v) for p, v in d.items()}


def draw(ov, pos, info, cls, conf, truth):
    col = COLORS[cls]
    tag = f"{pos} {SHORT[cls]}" + (f" {conf:.2f}" if conf is not None else "")
    if truth and truth != cls:
        tag += f" (true {SHORT.get(truth, truth)})"
    if pos.startswith("G"):
        u, v = (int(round(x)) for x in info["aperture_px"])
        r = int(round(info["radius_px"]))
        cv2.circle(ov, (u, v), r, col, 2 if not truth or truth == cls else 3, cv2.LINE_AA)
        cv2.putText(ov, tag, (u - 18, v - r - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.3, col, 1, cv2.LINE_AA)
    else:
        ax = vial_crops.s_axis(info)
        if ax:
            u0, v0, u1, v1, d = ax
            cv2.line(ov, (int(u0), int(v0)), (int(u1), int(v1)), col, max(1, int(d * 0.15)), cv2.LINE_AA)
            cv2.putText(ov, tag, (int(min(u0, u1)), int(min(v0, v1) - d / 2 - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                        col, 1, cv2.LINE_AA)


def run(a):
    layout = json.loads(a.layout.read_text())["positions"]
    trays = {t.strip().upper() for t in a.trays.split(",")}
    layout = {p: v for p, v in layout.items() if p[0] in trays}
    images = []
    for item in a.images:
        images += sorted(x for x in item.iterdir() if x.suffix.lower() in (".png", ".jpg", ".jpeg")) if item.is_dir() else [item]
    if not images:
        sys.exit("no images")

    model = None
    if not a.check_layout:
        import torch, torchvision
        ck = torch.load(a.model, map_location="cpu", weights_only=False)
        classes, size = ck["classes"], ck["size"]
        model = torchvision.models.resnet18()
        model.fc = torch.nn.Linear(model.fc.in_features, len(classes))
        model.load_state_dict(ck["state_dict"])
        device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device).eval()
        mean = torch.tensor(ck["mean"]).view(1, 3, 1, 1)
        std = torch.tensor(ck["std"]).view(1, 3, 1, 1)
    else:
        size = 128

    (a.out / "overlays").mkdir(parents=True, exist_ok=True)
    csv_rows, hits, total, per_cls = [], 0, 0, defaultdict(lambda: [0, 0])
    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"skip unreadable {img_path}")
            continue
        h, w = img.shape[:2]
        if (w, h) != (640, 480):
            print(f"warning: {img_path.name} is {w}x{h}; the layout is for 640x480 frames")
        pos_info = transform_layout(layout, a, w, h)
        padded = vial_crops.pad(img)
        names, crops = [], []
        for pos, info in pos_info.items():
            c = vial_crops.crop_position(padded, pos, info, size)
            if c is not None:
                names.append(pos)
                crops.append(c)
        ov = img.copy()
        truth = true_labels(a.labels / f"{img_path.stem}.json") if a.labels else None
        result = {}
        if model is not None and crops:
            import torch
            x = torch.from_numpy(np.stack(crops)[..., ::-1].copy()).permute(0, 3, 1, 2).float() / 255
            x = ((x - mean) / std).to(device)
            with torch.no_grad():
                prob = torch.softmax(model(x), 1).cpu().numpy()
            for pos, p in zip(names, prob):
                k = int(p.argmax())
                cls = classes[k] if p[k] >= a.min_conf else "uncertain"
                t = truth.get(pos) if truth else None
                result[pos] = {"class": cls, "confidence": round(float(p[k]), 4),
                               "probs": {c: round(float(v), 4) for c, v in zip(classes, p)}}
                if t:
                    result[pos]["true"] = t
                    total += 1
                    hits += int(t == classes[k])
                    per_cls[t][0] += int(t == classes[k])
                    per_cls[t][1] += 1
                csv_rows.append({"frame": img_path.name, "pos": pos, "class": cls, "confidence": round(float(p[k]), 4),
                                 "true": t or ""})
                draw(ov, pos, pos_info[pos], cls, float(p[k]), t)
        else:
            for pos in names:
                draw(ov, pos, pos_info[pos], "uncertain", None, None)
        if result:
            (a.out / f"{img_path.stem}.json").write_text(json.dumps(result, indent=1))
        cv2.imwrite(str(a.out / "overlays" / f"{img_path.stem}.png"), ov)
        if model is not None:
            counts = Counter(v["class"] for v in result.values())
            print(f"{img_path.name}: {dict(counts)}")

    if csv_rows:
        with open(a.out / "predictions.csv", "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(csv_rows[0]))
            wr.writeheader()
            wr.writerows(csv_rows)
    if total:
        print(f"accuracy against labels: {hits}/{total} = {hits / total:.3f}")
        for c, (ok, n) in sorted(per_cls.items()):
            print(f"   {c:13s} recall {ok / n:.3f} (n={n})")
    print(f"{len(images)} frames -> {a.out}")


ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
sub = ap.add_subparsers(dest="cmd", required=True)
lp = sub.add_parser("layout", help="build layout.json from sdg_vials.py runs")
lp.add_argument("runs", nargs="+", type=Path)
lp.add_argument("--out", type=Path, default=Path("layout.json"))
rp = sub.add_parser("run", help="classify frames")
rp.add_argument("images", nargs="+", type=Path)
rp.add_argument("--model", type=Path, default=Path("model.pt"))
rp.add_argument("--layout", type=Path, default=Path("layout.json"))
rp.add_argument("--out", type=Path, default=Path("preds"))
rp.add_argument("--trays", default="G,S")
rp.add_argument("--min-conf", type=float, default=0.5, help="below this top probability a position is 'uncertain'")
rp.add_argument("--shift", type=float, nargs=2, default=[0.0, 0.0], metavar=("DX", "DY"))
rp.add_argument("--scale", type=float, default=1.0)
rp.add_argument("--rotate", type=float, default=0.0, help="degrees, about the image centre")
rp.add_argument("--labels", type=Path, help="folder of per-frame label JSON files, to report accuracy")
rp.add_argument("--check-layout", action="store_true", help="only draw the layout on the frames (no model)")
rp.add_argument("--device", default=None)
a = ap.parse_args()
build_layout(a) if a.cmd == "layout" else run(a)
