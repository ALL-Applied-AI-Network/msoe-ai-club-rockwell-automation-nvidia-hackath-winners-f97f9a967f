"""List the tray-inspection stage hierarchy: prim tree (trimmed), and world positions of vial/cap/hole-like prims."""
import re, sys
from collections import Counter
from pxr import Usd, UsdGeom

stage = Usd.Stage.Open(sys.argv[1])
out = open(sys.argv[2], "w")
xc = UsdGeom.XformCache()
types = Counter()
interesting = re.compile(r"vial|cap|hole|slot|aperture|tray|fixture|plate|pos|well", re.I)

def walk(prim, depth):
    types[prim.GetTypeName()] += 1
    kids = list(prim.GetChildren())
    if depth <= 6:
        out.write(f"{'  ' * depth}{prim.GetName()} [{prim.GetTypeName()}] ({len(kids)} children)\n")
    # Collapse long runs of same-pattern siblings.
    shown = 0
    for k in kids:
        if shown < 8 or interesting.search(k.GetName()):
            walk(k, depth + 1); shown += 1
        else:
            types[k.GetTypeName()] += 1

walk(stage.GetPseudoRoot(), 0)
out.write("\n=== prim type counts ===\n")
for t, n in types.most_common(30):
    out.write(f"{n:6d}  {t or '(none)'}\n")

out.write("\n=== interesting prims: path | type | world xyz | custom attrs ===\n")
n = 0
for p in stage.Traverse():
    if interesting.search(p.GetName()) and p.IsA(UsdGeom.Xformable):
        t = xc.GetLocalToWorldTransform(p).ExtractTranslation()
        attrs = [a.GetName() for a in p.GetAttributes() if a.IsCustom() or a.GetName().startswith("userProperties")]
        out.write(f"{p.GetPath()} | {p.GetTypeName()} | {t[0]:.4f} {t[1]:.4f} {t[2]:.4f} | {attrs[:6]}\n")
        n += 1
        if n > 600:
            out.write("... truncated\n"); break
out.write(f"\nmetersPerUnit={UsdGeom.GetStageMetersPerUnit(stage)} upAxis={UsdGeom.GetStageUpAxis(stage)}\n")
out.close()
print("done", n, "interesting prims")
