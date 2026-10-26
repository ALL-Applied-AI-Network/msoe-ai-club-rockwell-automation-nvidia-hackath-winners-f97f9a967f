"""Geometry details for vial tube/cap meshes, materials and lights in the tray-inspection stage."""
import sys
import numpy as np
from pxr import Usd, UsdGeom, UsdShade, UsdLux

stage = Usd.Stage.Open(sys.argv[1])
xc = UsdGeom.XformCache()

def mesh_info(path):
    prim = stage.GetPrimAtPath(path)
    m = UsdGeom.Mesh(prim)
    pts = np.array(m.GetPointsAttr().Get())
    fvc = np.array(m.GetFaceVertexCountsAttr().Get())
    print(f"\n## {path}")
    print("  points", len(pts), "faces", len(fvc), "face sizes", sorted(set(fvc.tolist()))[:6])
    print("  local min", pts.min(0).round(5), "max", pts.max(0).round(5))
    r = np.hypot(pts[:, 0] - pts[:, 0].mean(), pts[:, 1] - pts[:, 1].mean())
    print("  radial min/max", r.min().round(5), r.max().round(5))
    zs = np.unique(pts[:, 2].round(4))
    print("  distinct z levels", len(zs), zs[:12], "...", zs[-6:])
    print("  subdivisionScheme", m.GetSubdivisionSchemeAttr().Get(), "doubleSided", m.GetDoubleSidedAttr().Get())
    print("  normals", m.GetNormalsAttr().HasAuthoredValue(), "primvars", [p.GetName() for p in UsdGeom.PrimvarsAPI(prim).GetPrimvars()])
    print("  instanceable/ref", prim.IsInstance(), prim.HasAuthoredReferences(), "parent instanceable", prim.GetParent().IsInstance())
    mat, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
    print("  material", mat.GetPath() if mat else None)
    for p in (prim.GetParent(), prim):
        x = UsdGeom.Xformable(p)
        print(f"  xformOps {p.GetName()}:", [(op.GetOpName(), op.Get()) for op in x.GetOrderedXformOps()])
    w = xc.GetLocalToWorldTransform(prim)
    print("  world scale-ish", [round(w.GetRow3(i).GetLength(), 5) for i in range(3)])

for p in ["/TrayInspection/G01_tube/vial_tube", "/TrayInspection/G01_cap/vial_cap",
          "/TrayInspection/G16_tube/vial_tube", "/TrayInspection/S01_tube/vial_tube", "/TrayInspection/S01_cap/vial_cap",
          "/TrayInspection/tray_good_top/tray_good_top"]:
    mesh_info(p)

# Are all G tubes identical geometry?
ref = np.array(UsdGeom.Mesh(stage.GetPrimAtPath("/TrayInspection/G01_tube/vial_tube")).GetPointsAttr().Get())
same = sum(np.allclose(np.array(UsdGeom.Mesh(stage.GetPrimAtPath(f"/TrayInspection/G{i:02d}_tube/vial_tube")).GetPointsAttr().Get()).shape, ref.shape) for i in range(1, 31))
print("\nG tubes with same point count as G01:", same, "/ 30")

print("\n## materials")
for p in stage.Traverse():
    if p.IsA(UsdShade.Material):
        sh = [c for c in p.GetChildren() if c.IsA(UsdShade.Shader)]
        for s in sh:
            sd = UsdShade.Shader(s)
            ins = {i.GetBaseName(): i.Get() for i in sd.GetInputs() if i.Get() is not None}
            print(f"  {p.GetPath()} :: {sd.GetIdAttr().Get()} {ins}")

print("\n## lights")
for p in stage.Traverse():
    if p.HasAPI(UsdLux.LightAPI) or "Light" in p.GetTypeName():
        L = UsdLux.LightAPI(p)
        t = xc.GetLocalToWorldTransform(p).ExtractTranslation()
        print(f"  {p.GetPath()} [{p.GetTypeName()}] at {tuple(round(v, 3) for v in t)} intensity={L.GetIntensityAttr().Get()} exposure={L.GetExposureAttr().Get()} color={L.GetColorAttr().Get()}")
print("\ntimeCodesPerSecond", stage.GetTimeCodesPerSecond(), "start/end", stage.GetStartTimeCode(), stage.GetEndTimeCode())
