"""Synthetic training data for the cap-seal camera: one class per tray position (Isaac Sim, headless).

    ./python.sh tools/sdg_vials.py --frames 20 --seed 1 --out out --preview
    ./python.sh tools/sdg_vials.py --trays G,S --frames 20 --seed 1 --out out --preview

For every frame, each position of the selected trays (G01-G30 upright on the good tray, S01-S05 lying on
the scrap tray) gets one class (pass, missing_cap, crooked_cap, damaged_vial, no_vial). The scene is edited
to match in the session layer only, so the USD file is never changed.
Lights, materials, background, camera pose and harmless vial lean / cap seating are randomized, and
CapSealCam renders the frame like render_example.py does: a pinhole render with overscan, then the real
lens distortion and a crop to 640x480.

Writes frames/NNNNNN.png and labels/NNNNNN.json (class and pixel position of each aperture in the
final image). With --preview it also writes overlays/NNNNNN.png with the labels drawn on, plus a contact sheet.

damaged_vial on the G tray is limited to damage the near-top-down camera can see: a broken neck, or a
chipped rim with the cap off. Body cracks on upright capped vials are hidden by the cap. On the S tray the
vials lie on their side, so damage is a chipped rim, a broken neck, or a star fracture or body crack on the
camera-facing side, beside the specular streak and drawn 1-2.3 mm wide so the camera can resolve it.
"""
import argparse, json, math, os, traceback
from pathlib import Path
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLASSES = ["pass", "missing_cap", "crooked_cap", "damaged_vial", "no_vial"]
ap = argparse.ArgumentParser()
ap.add_argument("--frames", type=int, default=20)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--start", type=int, default=0, help="index of the first frame (to split one dataset over several jobs)")
ap.add_argument("--out", default="sdg_out")
ap.add_argument("--subframes", type=int, default=16, help="path-tracing subframes per render step")
ap.add_argument("--stage", default=os.path.join(HERE, "stations", "tray_inspection", "tray_inspection.usdc"))
ap.add_argument("--trays", default="G", help="trays to label: G, S or G,S")
ap.add_argument("--weights", default="pass=0.36,missing_cap=0.16,crooked_cap=0.16,damaged_vial=0.16,no_vial=0.16",
                help="class sampling weights per G-tray position")
ap.add_argument("--weights-s", default="pass=0.25,missing_cap=0.15,crooked_cap=0.05,damaged_vial=0.45,no_vial=0.10",
                help="class sampling weights per S-tray position")
ap.add_argument("--preview", action="store_true", help="also write label overlays and a contact sheet")
ap.add_argument("--public-view", type=Path, help="calibrated 1920x1080, 60-position geometry configuration")
ap.add_argument("--keep-usd-fov", action="store_true",
                help="render with the camera's authored focal length instead of matching focal_px_final "
                     "(Isaac Sim 5.1 renders CapSealCam about 1.36x zoomed in)")
a = ap.parse_args()
def class_probs(spec):
    w = dict(kv.split("=") for kv in spec.split(","))
    p = np.array([float(w[c]) for c in CLASSES])
    return p / p.sum()
PROBS = {"G": class_probs(a.weights), "S": class_probs(a.weights_s)}
TRAYS = [t.strip().upper() for t in a.trays.split(",")]

cam_cfg = json.load(open(os.path.join(HERE, "cameras", "CapSealCam.json")))
public_view = json.loads(a.public_view.read_text()) if a.public_view else None
if public_view:
    cam_cfg.update(public_view)
RW, RH = cam_cfg["render_px"]
W, H = cam_cfg["final_image_px"]
F, K1 = cam_cfg["focal_px_final"], cam_cfg["lens_k1"]
out = Path(a.out)
for d in ["frames", "labels"] + (["overlays"] if a.preview else []):
    (out / d).mkdir(parents=True, exist_ok=True)

from isaacsim import SimulationApp
kit = SimulationApp({"headless": True, "renderer": "PathTracing", "width": RW, "height": RH})
try:
    from isaacsim.core.utils.extensions import enable_extension
    enable_extension("omni.replicator.core")
    import cv2
    import omni.usd
    import omni.replicator.core as rep
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade, Vt

    # Lens distortion maps, as in apply_lens.py: final (distorted) pixel -> pinhole render pixel.
    Kd = np.array([[F, 0, W / 2], [0, F, H / 2], [0, 0, 1.0]])
    gu, gv = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    und = cv2.undistortPoints(np.stack([gu.ravel(), gv.ravel()], 1)[:, None, :], Kd,
                              np.array([K1, 0, 0, 0, 0.0])).reshape(H, W, 2)
    MAPX = (und[..., 0] * F + RW / 2).astype(np.float32)
    MAPY = (und[..., 1] * F + RH / 2).astype(np.float32)

    def to_final(u, v):
        """Pinhole render pixel -> final distorted 640x480 pixel (forward model of apply_lens.py)."""
        x, y = (u - RW / 2) / F, (v - RH / 2) / F
        s = 1 + K1 * (x * x + y * y)
        return x * s * F + W / 2, y * s * F + H / 2

    ctx = omni.usd.get_context()
    if not ctx.open_stage(str(Path(a.stage).resolve())):
        raise RuntimeError(f"could not open stage: {a.stage}")
    while ctx.get_stage_loading_status()[2] > 0:
        kit.update()
    for _ in range(10):
        kit.update()
    stage = ctx.get_stage()
    stage.SetEditTarget(stage.GetSessionLayer())

    ROOT = "/TrayInspection"
    POS_G = [f"G{i:02d}" for i in range(1, 31)]
    POS_S = [f"S{i:02d}" for i in range(1, 31 if public_view else 6)]
    POS = (POS_G if "G" in TRAYS else []) + (POS_S if "S" in TRAYS else [])
    PLATE_TOP = 1.76     # tray top surface in tube-local inches (tube origin = vial bottom)
    CAP_SEAT = 3.6       # cap origin (cap bottom) in tube-local inches
    TUBE_H, TUBE_R = 4.1, 0.57

    def matrix_op(path):
        """Replace a prim's transform with one matrix op (session layer); return (op, original local matrix)."""
        x = UsdGeom.Xformable(stage.GetPrimAtPath(path))
        m0 = x.GetLocalTransformation()
        x.ClearXformOpOrder()
        op = x.AddTransformOp()
        op.Set(m0)
        return op, m0

    def rot(axis, deg):
        return Gf.Matrix4d(1.0).SetRotate(Gf.Rotation(Gf.Vec3d(*(float(x) for x in axis)), float(deg)))

    def trans(v):
        return Gf.Matrix4d(1.0).SetTranslate(Gf.Vec3d(*(float(x) for x in v)))

    sess = stage.GetSessionLayer()
    if public_view:
        Sdf.CreatePrimInLayer(sess, ROOT)
        spec = stage.GetPrimAtPath(f"{ROOT}/S01_tube").GetPrimStack()[-1]
        for p in POS_S[5:]:
            if not Sdf.CopySpec(spec.layer, spec.path, sess, f"{ROOT}/{p}_tube"):
                raise RuntimeError(f"cannot create {p}")
    tube_op, tube0, cap_op, cap0 = {}, {}, {}, {}
    for p in POS_G + POS_S:
        tube_op[p], tube0[p] = matrix_op(f"{ROOT}/{p}_tube")
    # Only S01 and S04 have caps in the station file: give the other lying vials a copy of S01's cap,
    # seated the same way (session layer), so every S position can be pass, crooked or missing_cap.
    sess = stage.GetSessionLayer()
    s01_spec = stage.GetPrimAtPath(f"{ROOT}/S01_cap").GetPrimStack()[-1]
    for p in POS_G + POS_S:
        path = f"{ROOT}/{p}_cap"
        if not stage.GetPrimAtPath(path).IsValid():
            Sdf.CreatePrimInLayer(sess, ROOT)
            if not Sdf.CopySpec(s01_spec.layer, s01_spec.path, sess, path):
                raise RuntimeError(f"could not create {path}")
            cap_op[p], _ = matrix_op(path)
            cap0[p] = cap0["S01"] * tube0["S01"].GetInverse() * tube0[p]
            cap_op[p].Set(cap0[p])
        else:
            cap_op[p], cap0[p] = matrix_op(path)
    cam_xf_op, cam_xf0 = matrix_op(f"{ROOT}/CapSealCam")
    cam_prim = stage.GetPrimAtPath(f"{ROOT}/CapSealCam/CapSealCam")
    root_world = UsdGeom.Xformable(stage.GetPrimAtPath(ROOT)).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    if public_view:
        # The bundled robot/carrier is one merged mesh and occludes two fixture columns
        # from the supplied camera pose. Keep fixture/vial geometry, omit this mismatched context.
        UsdGeom.Imageable(stage.GetPrimAtPath(f"{ROOT}/AU_001436920200716")).MakeInvisible()
        for p in POS_S:
            matrix = Gf.Matrix4d(*np.asarray(public_view['scrap_tube_matrices'][p]).flatten().tolist())
            cap0[p] = cap0[p] * tube0[p].GetInverse() * matrix
            tube0[p] = matrix
            tube_op[p].Set(matrix); cap_op[p].Set(cap0[p])
        cam_xf0 = Gf.Matrix4d(*np.asarray(public_view['camera_local_matrix']).flatten().tolist())
        cam_xf_op.Set(cam_xf0)
        camera = UsdGeom.Camera(cam_prim)
        camera.GetHorizontalApertureAttr().Set(.432)
        camera.GetVerticalApertureAttr().Set(.432 * RH / RW)
        camera.GetFocalLengthAttr().Set(.432 * F / RW)

    plastic = UsdShade.Material(stage.GetPrimAtPath(f"{ROOT}/_materials/tw_vial_plastic"))
    def material_shader(mat):
        return UsdShade.Shader(next(c for c in stage.GetPrimAtPath(f"{ROOT}/_materials/{mat}").GetChildren()
                                    if c.IsA(UsdShade.Shader)))
    cap_sh, alu_sh = material_shader("tw_cap_yellow"), material_shader("tw_alu_brushed")
    plastic_sh = material_shader("tw_vial_plastic")
    cap_col0 = Gf.Vec3f(cap_sh.GetInput("diffuseColor").Get())
    alu_col0, alu_rough0 = Gf.Vec3f(alu_sh.GetInput("diffuseColor").Get()), alu_sh.GetInput("roughness").Get()
    lights = []
    for lp in [f"{ROOT}/key_overhead/key_overhead", f"{ROOT}/fill_side/fill_side"]:
        L = UsdLux.LightAPI(stage.GetPrimAtPath(lp))
        lights.append((L, L.GetIntensityAttr().Get(), Gf.Vec3f(L.GetColorAttr().Get())))

    # Background below the station (the station file has no floor, so the edge of the frame renders black).
    bg = UsdGeom.Mesh.Define(stage, "/SDG/background")
    bg.CreatePointsAttr([(-2.5, -2.5, 1.0), (1.5, -2.5, 1.0), (1.5, 1.5, 1.0), (-2.5, 1.5, 1.0)])
    bg.CreateFaceVertexCountsAttr([4]); bg.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    bg_mat = UsdShade.Material.Define(stage, "/SDG/looks/bg")
    bg_sh = UsdShade.Shader.Define(stage, "/SDG/looks/bg/shader")
    bg_sh.CreateIdAttr("UsdPreviewSurface")
    bg_col = bg_sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f)
    bg_rough = bg_sh.CreateInput("roughness", Sdf.ValueTypeNames.Float)
    bg_mat.CreateSurfaceOutput().ConnectToSource(bg_sh.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(bg.GetPrim()).Bind(bg_mat)
    dome = UsdLux.DomeLight.Define(stage, "/SDG/dome")

    # Fracture surfaces and stress whitening: broken plastic edges scatter light and read as white.
    frac_mat = UsdShade.Material.Define(stage, "/SDG/looks/fracture")
    frac_sh = UsdShade.Shader.Define(stage, "/SDG/looks/fracture/shader")
    frac_sh.CreateIdAttr("UsdPreviewSurface")
    frac_sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.92, 0.93, 0.95))
    frac_rough = frac_sh.CreateInput("roughness", Sdf.ValueTypeNames.Float)
    frac_opacity = frac_sh.CreateInput("opacity", Sdf.ValueTypeNames.Float)
    frac_mat.CreateSurfaceOutput().ConnectToSource(frac_sh.ConnectableAPI(), "surface")
    # Body cracks and star fractures: nearly opaque white, so a 1-2 px line still reads against clear plastic.
    crack_mat = UsdShade.Material.Define(stage, "/SDG/looks/crack")
    crack_sh = UsdShade.Shader.Define(stage, "/SDG/looks/crack/shader")
    crack_sh.CreateIdAttr("UsdPreviewSurface")
    crack_sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.95, 0.96, 0.97))
    crack_rough = crack_sh.CreateInput("roughness", Sdf.ValueTypeNames.Float)
    crack_opacity = crack_sh.CreateInput("opacity", Sdf.ValueTypeNames.Float)
    crack_mat.CreateSurfaceOutput().ConnectToSource(crack_sh.ConnectableAPI(), "surface")

    # Render product, as in render_example.py (including the clipping-range re-sync for the scaled camera).
    rp = rep.create.render_product(str(cam_prim.GetPath()), (RW, RH))
    ann = rep.AnnotatorRegistry.get_annotator("rgb"); ann.attach([rp])
    cam_ann = rep.AnnotatorRegistry.get_annotator("camera_params"); cam_ann.attach([rp])
    rep.orchestrator.set_capture_on_play(False)
    kit.update()
    clip = UsdGeom.Camera(cam_prim).GetClippingRangeAttr()
    near_far = clip.Get()
    clip.Set(Gf.Vec2f(near_far[0] * 0.5, near_far[1])); kit.update()
    clip.Set(near_far); kit.update()
    ucam = UsdGeom.Camera(cam_prim)
    focal_attr = ucam.GetFocalLengthAttr()
    fx = focal_attr.Get() / ucam.GetHorizontalApertureAttr().Get() * RW
    print(f"camera fx from USD {fx:.2f} px vs focal_px_final {F:.2f} px", flush=True)

    def renderer_fx():
        # camera_params stays empty until the renderer has produced frames; two steps per reading so the
        # value cannot lag behind a focal-length change.
        for _ in range(10):
            for _ in range(2):
                rep.orchestrator.step(rt_subframes=1, delta_time=0.0)
            proj = np.asarray(cam_ann.get_data().get("cameraProjection", []), dtype=np.float64)
            if proj.size == 16 and proj[0] != 0:
                return float(proj[0]) * RW / 2
        raise RuntimeError("camera_params annotator returned no projection")

    # Isaac Sim 5.1 clamps focalLength to at least 0.5, and CapSealCam authors 0.367 (with apertures 0.432 x
    # 0.324), so it renders zoomed in (fx 888.9 px instead of 652.9). Scaling focal length and apertures
    # together keeps the field of view and gets above the clamp (session layer only).
    fov_log = [{"focalLength": float(focal_attr.Get()), "renderer_fx": round(renderer_fx(), 2)}]
    if not a.keep_usd_fov:
        for attr in (focal_attr, ucam.GetHorizontalApertureAttr(), ucam.GetVerticalApertureAttr()):
            attr.Set(float(attr.Get() * 100))
        kit.update()
        fov_log.append({"focalLength": float(focal_attr.Get()), "renderer_fx": round(renderer_fx(), 2)})
    print(f"renderer focal check: {fov_log} (calibration {F:.2f} px)", flush=True)
    if not a.keep_usd_fov and abs(fov_log[-1]["renderer_fx"] / F - 1) > 0.002:
        raise RuntimeError(f"renderer fx {fov_log[-1]['renderer_fx']} px does not match the calibration {F} px")

    def tube_mesh(path, top, fractured):
        """Vial tube with a per-angle top height (inches, tube-local); top: array over the angle columns.
        Columns where `fractured` is true get the fracture material on the rim and the top wall band."""
        n, rows, ri, zb = len(top), 24, 0.525, 0.06
        th = np.linspace(0, 2 * np.pi, n, endpoint=False)
        t = np.linspace(0, 1, rows)[:, None]
        outer = np.stack([np.broadcast_to(TUBE_R * np.cos(th), (rows, n)), np.broadcast_to(TUBE_R * np.sin(th), (rows, n)),
                          t * top[None, :]], -1).reshape(-1, 3)
        inner = np.stack([np.broadcast_to(ri * np.cos(th), (rows, n)), np.broadcast_to(ri * np.sin(th), (rows, n)),
                          zb + t * (top[None, :] - zb)], -1).reshape(-1, 3)
        pts = np.concatenate([outer, inner, [[0, 0, 0], [0, 0, zb]]])
        io, ii, co, ci = 0, rows * n, 2 * rows * n, 2 * rows * n + 1
        cnt, idx = [], []
        for r in range(rows - 1):
            for j in range(n):
                k = (j + 1) % n
                idx += [io + r * n + j, io + r * n + k, io + (r + 1) * n + k, io + (r + 1) * n + j]
                idx += [ii + r * n + j, ii + (r + 1) * n + j, ii + (r + 1) * n + k, ii + r * n + k]
                cnt += [4, 4]
        for j in range(n):
            k = (j + 1) % n
            top_o, top_i = io + (rows - 1) * n, ii + (rows - 1) * n
            idx += [top_o + j, top_i + j, top_i + k, top_o + k]          # rim
            idx += [co, io + k, io + j, ci, ii + j, ii + k]              # outer and inner bottoms
            cnt += [4, 3, 3]
        m = UsdGeom.Mesh.Define(stage, path)
        m.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(pts.astype(np.float32)))
        m.CreateFaceVertexCountsAttr(cnt); m.CreateFaceVertexIndicesAttr(idx)
        m.CreateSubdivisionSchemeAttr("none"); m.CreateDoubleSidedAttr(True)
        UsdShade.MaterialBindingAPI.Apply(m.GetPrim()).Bind(plastic)
        # Face order above: 2 wall quads per (row, column), then rim + 2 bottom triangles per column.
        band = 3                                               # top wall rows that show stress whitening
        frac_faces = []
        for j in np.flatnonzero(fractured):
            frac_faces.append(2 * (rows - 1) * n + 3 * j)      # rim
            for r in range(rows - 1 - band, rows - 1):
                frac_faces += [2 * (r * n + j), 2 * (r * n + j) + 1]
        if frac_faces:
            sub = UsdGeom.Subset.CreateGeomSubset(m, "fracture", UsdGeom.Tokens.face,
                                                  Vt.IntArray(sorted(int(f) for f in frac_faces)),
                                                  familyName=UsdShade.Tokens.materialBind)
            UsdShade.MaterialBindingAPI.Apply(sub.GetPrim()).Bind(frac_mat)
        return m

    def shard(path, cx, cy, size, rng):
        ang = rng.uniform(0, 2 * np.pi, 3) + np.array([0, 2.1, 4.2])
        r = size * rng.uniform(0.5, 1.0, 3)
        tri = np.stack([cx + r * np.cos(ang), cy + r * np.sin(ang)], 1)
        z0, z1 = PLATE_TOP + 0.002, PLATE_TOP + 0.03
        pts = [(x, y, z0) for x, y in tri] + [(x, y, z1) for x, y in tri]
        m = UsdGeom.Mesh.Define(stage, path)
        m.CreatePointsAttr(pts)
        m.CreateFaceVertexCountsAttr([3, 3, 4, 4, 4])
        m.CreateFaceVertexIndicesAttr([0, 2, 1, 3, 4, 5, 0, 1, 4, 3, 1, 2, 5, 4, 2, 0, 3, 5])
        m.CreateDoubleSidedAttr(True)
        UsdShade.MaterialBindingAPI.Apply(m.GetPrim()).Bind(plastic)

    def cap_tilt(deg, az, lift, rc=0.62):
        """Cap rotated about a point on its bottom rim (cap-local inches): one side stays down, the other lifts."""
        p = (rc * math.cos(az), rc * math.sin(az), 0.0)
        return trans((-p[0], -p[1], 0)) * rot((-math.sin(az), math.cos(az), 0), deg) * trans((p[0], p[1], lift))

    def set_visible(path, vis):
        img = UsdGeom.Imageable(stage.GetPrimAtPath(path))
        img.MakeVisible() if vis else img.MakeInvisible()

    cam_pos_world = UsdGeom.Xformable(cam_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()).ExtractTranslation()

    def facing_angle(tube_m):
        """Angle around the vial axis (tube-local) of the side that faces the camera."""
        q = (tube_m * root_world).GetInverse().Transform(cam_pos_world)
        return math.atan2(q[1], q[0])

    def crack_walk(rng, th0, z0, heading, length, step=0.04, wobble=0.35):
        """Random walk on the outer surface in (arc length, height); returns (k, 2) of (angle, height)."""
        n = max(2, int(length / step))
        hd = heading + np.cumsum(rng.normal(0, wobble, n))
        s = np.concatenate([[0.0], np.cumsum(step * np.cos(hd))])
        z = np.concatenate([[z0], z0 + np.cumsum(step * np.sin(hd))])
        return np.stack([th0 + s / TUBE_R, np.clip(z, 0.1, 3.5)], 1)

    def star_fracture(rng, thc, zc):
        return [crack_walk(rng, thc, zc, rng.uniform(0, 2 * np.pi), rng.uniform(0.15, 0.45), step=0.03, wobble=0.25)
                for _ in range(int(rng.integers(4, 9)))]

    def body_crack(rng, thc, zc):
        # Across or diagonal to the vial axis, never along it: the specular streak of an intact lying vial
        # runs along the axis and must not look like a crack.
        heading = rng.uniform(-np.pi / 2 + 0.45, np.pi / 2 - 0.45) + rng.choice([0.0, np.pi])
        main = crack_walk(rng, thc, zc, heading, rng.uniform(0.4, 1.2), step=0.03, wobble=0.5)
        lines = [main]
        for _ in range(int(rng.integers(1, 5))):
            i = int(rng.integers(1, len(main)))
            lines.append(crack_walk(rng, main[i, 0], main[i, 1], rng.uniform(0, 2 * np.pi), rng.uniform(0.08, 0.4),
                                    step=0.03, wobble=0.5))
        return lines

    def crack_mesh(path, lines, width, spot=None):
        """White ribbons just outside the vial surface along each crack line (tube-local inches), plus an
        optional whitened impact spot (thc, zc, radius) for star fractures."""
        r = TUBE_R + 0.004
        pts, idx, cnt = [], [], []
        if spot:
            thc, zc, rr = spot
            ang = np.linspace(0, 2 * np.pi, 16, endpoint=False)
            rim = [(r * math.cos(thc + rr * math.cos(t) / TUBE_R), r * math.sin(thc + rr * math.cos(t) / TUBE_R),
                    zc + rr * math.sin(t)) for t in ang]
            pts += [(r * math.cos(thc), r * math.sin(thc), zc)] + rim
            for k in range(16):
                idx += [0, 1 + k, 1 + (k + 1) % 16]
                cnt.append(3)
        for ln in lines:
            th, z = ln[:, 0], ln[:, 1]
            P3 = np.stack([r * np.cos(th), r * np.sin(th), z], 1)
            for i in range(len(P3) - 1):
                normal = np.array([math.cos(th[i]), math.sin(th[i]), 0.0])
                side = np.cross(normal, P3[i + 1] - P3[i])
                side *= (width / 2) / (np.linalg.norm(side) + 1e-9)
                b = len(pts)
                pts += [P3[i] - side, P3[i] + side, P3[i + 1] + side, P3[i + 1] - side]
                idx += [b, b + 1, b + 2, b + 3]
                cnt.append(4)
        m = UsdGeom.Mesh.Define(stage, path)
        m.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.asarray(pts, np.float32)))
        m.CreateFaceVertexCountsAttr(cnt); m.CreateFaceVertexIndicesAttr(idx)
        m.CreateDoubleSidedAttr(True)
        UsdShade.MaterialBindingAPI.Apply(m.GetPrim()).Bind(crack_mat)

    def apply_position(p, cls, rng):
        tray = p[0]
        info = {"class": cls}
        tube_path, cap_path = f"{ROOT}/{p}_tube", f"{ROOT}/{p}_cap"
        for c in stage.GetPrimAtPath(tube_path).GetChildren():
            if c.GetName().startswith("sdg_"):
                stage.RemovePrim(c.GetPath())
        set_visible(f"{tube_path}/vial_tube", cls != "no_vial")
        set_visible(cap_path, cls not in ("no_vial", "missing_cap"))
        if tray == "G":
            lean = Gf.Matrix4d(1.0)
            if cls != "no_vial" and rng.random() < 0.35:      # vial lean alone is not a defect
                b, d = rng.uniform(0, 2 * np.pi), rng.uniform(0.5, 3.0)
                lean = rot((math.cos(b), math.sin(b), 0), d)
                info["vial_lean_deg"] = round(d, 2)
            tube_m = lean * tube0[p]
        else:                                                  # lying vial: slide and turn it a little on the tray
            c = tube0[p].Transform(Gf.Vec3d(0, 0, TUBE_H / 2))
            yaw, d = rng.uniform(-8, 8), rng.uniform(-0.15, 0.15, 2)
            tube_m = tube0[p] * trans(-c) * rot((0, 0, 1), yaw) * trans(c + Gf.Vec3d(float(d[0]), float(d[1]), 0))
            info["yaw_deg"] = round(yaw, 2)
        tube_op[p].Set(tube_m)
        info["_top"] = TUBE_H
        cap_local = Gf.Matrix4d(1.0)
        if cls == "pass" and rng.random() < 0.3:               # acceptable seating variation
            d = rng.uniform(0.3, 2.0)
            cap_local = cap_tilt(d, rng.uniform(0, 2 * np.pi), rng.uniform(0, 0.01))
            info["cap_tilt_deg"] = round(d, 2)
        elif cls == "crooked_cap":
            d = rng.uniform(8, 25)
            cap_local = cap_tilt(d, rng.uniform(0, 2 * np.pi), rng.uniform(0.03, 0.2))
            info["cap_tilt_deg"] = round(d, 2)
        elif cls == "damaged_vial":
            n = 96
            th = np.linspace(0, 2 * np.pi, n, endpoint=False)
            face = facing_angle(tube_m)
            if tray == "G":
                kind = "broken_neck" if rng.random() < 0.55 else "chipped_rim"
            else:
                kind = ["star_fracture", "body_crack", "chipped_rim", "broken_neck"][int(rng.choice(4, p=[0.3, 0.3, 0.2, 0.2]))]
            if kind in ("star_fracture", "body_crack"):
                # Cracks sized to be resolvable (1-2.3 mm wide; the camera resolves about 0.7 px/mm) and set
                # 20-50 degrees off the top of the vial: still facing the camera, but out of the specular streak.
                thc = face + rng.choice([-1, 1]) * rng.uniform(0.35, 0.9)
                zc = rng.uniform(0.5, 3.0)
                width = rng.uniform(0.04, 0.09)
                lines = star_fracture(rng, thc, zc) if kind == "star_fracture" else body_crack(rng, thc, zc)
                spot = (thc, zc, rng.uniform(0.06, 0.14)) if kind == "star_fracture" else None
                crack_mesh(f"{tube_path}/sdg_crack", lines, width, spot)
                info["crack_width_mm"] = round(width * 25.4, 2)
                if rng.random() < 0.4:                         # damage is on the body; the cap may be on or off
                    set_visible(cap_path, False)
                    info["cap_off"] = True
            else:
                if kind == "chipped_rim":
                    c = face + rng.uniform(-1.0, 1.0) if tray == "S" else rng.uniform(0, 2 * np.pi)
                    hw, depth = rng.uniform(0.3, 0.9), rng.uniform(0.1, 0.35)
                    s = np.clip(1 - (np.abs(np.angle(np.exp(1j * (th - c)))) / hw) ** 1.5, 0, 1)
                    top = TUBE_H - depth * s * rng.uniform(0.7, 1.3, n)
                    fractured = s > 0.02
                    set_visible(cap_path, False)
                else:
                    base = rng.uniform(2.3, 3.5) if tray == "G" else rng.uniform(1.2, 3.2)
                    amp = rng.uniform(0.08, 0.35)
                    wave = sum(rng.uniform(0.3, 1) * np.cos(k * th + rng.uniform(0, 2 * np.pi)) for k in range(2, 7))
                    top = np.clip(base + amp * wave / 2.5 + rng.uniform(0, 0.05, n), 1.0 if tray == "S" else 1.95, 4.0)
                    fractured = np.ones(n, bool)
                    if tray == "G":
                        for s_i in range(int(rng.integers(2, 6))):
                            ang, rad = rng.uniform(0, 2 * np.pi), rng.uniform(0.75, 1.3)
                            shard(f"{tube_path}/sdg_shard{s_i}", rad * math.cos(ang), rad * math.sin(ang), rng.uniform(0.06, 0.2), rng)
                    if tray == "S" or rng.random() < 0.5:      # cap knocked off
                        set_visible(cap_path, False)
                    else:                                      # cap resting askew on the broken neck
                        cap_local = trans((0, 0, float(top.min()) - TUBE_H)) * cap_tilt(rng.uniform(8, 30), rng.uniform(0, 2 * np.pi), 0.05)
                        info["cap_resting_on_break"] = True
                set_visible(f"{tube_path}/vial_tube", False)
                tube_mesh(f"{tube_path}/sdg_damaged", top, fractured)
                info["_top"] = float(top.max())
            info["damage"] = kind
        cap_op[p].Set(cap_local * cap0[p] * tube0[p].GetInverse() * tube_m)
        return info

    def vec3f(v):
        return Gf.Vec3f(*(float(x) for x in v))

    def randomize_globals(rng):
        tint = rng.uniform([0.85, 0.85, 0.8], [1.0, 1.0, 1.1])
        for L, i0, c0 in lights:
            L.GetIntensityAttr().Set(float(i0 * rng.uniform(0.55, 1.6)))
            L.GetColorAttr().Set(vec3f(np.minimum(np.array(c0) * tint, 1.0)))
        dome.GetIntensityAttr().Set(float(rng.uniform(0, 300)))
        cap_sh.GetInput("diffuseColor").Set(vec3f(np.clip(np.array(cap_col0) + rng.uniform(-0.06, 0.06, 3), 0, 1)))
        alu_sh.GetInput("diffuseColor").Set(vec3f(np.array(alu_col0) * rng.uniform(0.85, 1.15)))
        alu_sh.GetInput("roughness").Set(float(np.clip(alu_rough0 + rng.uniform(-0.2, 0.15), 0.2, 0.9)))
        bg_col.Set(vec3f(rng.uniform([0.28, 0.26, 0.23], [0.65, 0.60, 0.53]) if public_view else rng.uniform([0.08, 0.06, 0.04], [0.55, 0.45, 0.4])))
        bg_rough.Set(float(rng.uniform(0.5, 0.95)))
        frac_rough.Set(float(rng.uniform(0.4, 0.8)))
        # Vial gloss: from sharp specular streaks to a soft sheen, so highlights carry no class information.
        plastic_sh.GetInput("roughness").Set(float(rng.uniform(0.03, 0.3)))
        plastic_sh.GetInput("specular").Set(float(rng.uniform(0.25, 0.6)))
        frac_opacity.Set(float(rng.uniform(0.5, 0.9)))
        crack_rough.Set(float(rng.uniform(0.5, 0.9)))
        crack_opacity.Set(float(rng.uniform(0.85, 1.0)))
        jt = rng.uniform(-0.12, 0.12, 3)                       # about +-3 mm and +-0.4 deg of camera pose jitter
        ax = rng.normal(size=3); ax /= np.linalg.norm(ax)
        cam_xf_op.Set(rot(ax, rng.uniform(0, 0.4)) * trans(jt) * cam_xf0)

    def project(world_pt, view_proj):
        """World point (stage meters) -> final image pixel, using the view/projection the renderer used."""
        h = np.array([world_pt[0], world_pt[1], world_pt[2], 1.0]) @ view_proj
        if h[3] <= 0:
            return None
        x, y = h[0] / h[3], h[1] / h[3]
        return to_final((x + 1) / 2 * RW, (1 - y) / 2 * RH)

    COLORS = {"pass": (80, 200, 60), "missing_cap": (0, 150, 255), "crooked_cap": (230, 60, 230),
              "damaged_vial": (40, 40, 255), "no_vial": (255, 160, 40)}
    json.dump({"classes": CLASSES, "trays": TRAYS, "seed": a.seed,
               "weights": {t: dict(zip(CLASSES, PROBS[t].round(4).tolist())) for t in TRAYS},
               "image_size": [W, H], "camera": "CapSealCam", "fov_matched_to_calibration": not a.keep_usd_fov,
               "fov_log": fov_log, "public_view": bool(public_view),
               "geometry_fit_rms_px": public_view.get('geometry_fit_rms_px') if public_view else None}, open(out / "dataset.json", "w"), indent=1)

    overlays, s_zooms = [], []
    for fi in range(a.start, a.start + a.frames):
        rng = np.random.default_rng([a.seed, fi])
        labels = {p: apply_position(p, CLASSES[rng.choice(len(CLASSES), p=PROBS[p[0]])], rng) for p in POS}
        randomize_globals(rng)
        kit.update()
        # Two steps per frame so the capture cannot show the previous frame's scene.
        for _ in range(2):
            rep.orchestrator.step(rt_subframes=a.subframes, delta_time=0.0)
        data = np.asarray(ann.get_data())
        if data.ndim != 3 or data.shape[:2] != (RH, RW):
            raise RuntimeError(f"frame {fi}: RGB capture returned shape {data.shape}")
        final = cv2.remap(np.ascontiguousarray(data[..., :3][..., ::-1]), MAPX, MAPY, cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT)

        # Pixel positions in the final image: aperture centre at the tray surface (fixed), and the
        # cap top centre (follows lean and tilt) when a cap is shown.
        cp = cam_ann.get_data()
        view_proj = (np.asarray(cp["cameraViewTransform"], dtype=np.float64).reshape(4, 4) @
                     np.asarray(cp["cameraProjection"], dtype=np.float64).reshape(4, 4))
        if fi == a.start:
            eff_f = np.asarray(cp["cameraProjection"]).reshape(4, 4)[0, 0] * RW / 2
            print(f"renderer camera: focal {cp['cameraFocalLength']} aperture {cp['cameraAperture']} "
                  f"-> effective fx {eff_f:.1f} px (calibration {F:.1f} px)", flush=True)
        cam_inv = view_proj
        for p, info in labels.items():
            top = info.pop("_top")
            cap_shown = UsdGeom.Imageable(stage.GetPrimAtPath(f"{ROOT}/{p}_cap")).ComputeVisibility() != UsdGeom.Tokens.invisible
            if cap_shown:
                k = project((cap_op[p].Get() * root_world).Transform(Gf.Vec3d(0, 0, 0.75)), cam_inv)
                info["cap_px"] = [round(k[0], 1), round(k[1], 1)] if k else None
            if p[0] == "G":
                to_world = tube0[p] * root_world
                c = project(to_world.Transform(Gf.Vec3d(0, 0, PLATE_TOP)), cam_inv)
                e = project(to_world.Transform(Gf.Vec3d(TUBE_R + 0.05, 0, PLATE_TOP)), cam_inv)
                info["aperture_px"] = [round(c[0], 1), round(c[1], 1)] if c else None
                info["radius_px"] = round(math.hypot(e[0] - c[0], e[1] - c[1]), 1) if c and e else None
                info["in_frame"] = bool(c and 0 <= c[0] < W and 0 <= c[1] < H)
            else:
                # Lying vial: box around the body (and cap when shown), and the body centre.
                m = tube_op[p].Get() * root_world
                z_end = CAP_SEAT + 0.75 if cap_shown else top
                ring = [Gf.Vec3d(TUBE_R * math.cos(t), TUBE_R * math.sin(t), z)
                        for z in (0.0, z_end / 2, z_end) for t in np.linspace(0, 2 * np.pi, 8, endpoint=False)]
                uv = np.array([q for q in (project(m.Transform(v), cam_inv) for v in ring) if q])
                c = project(m.Transform(Gf.Vec3d(0, 0, min(top, TUBE_H) / 2)), cam_inv)
                info["body_px"] = [round(c[0], 1), round(c[1], 1)] if c else None
                info["bbox_px"] = [round(float(v), 1) for v in (*uv.min(0), *uv.max(0))] if len(uv) else None
                # Axis of the intact vial (closed end -> cap top), whatever the damage, for fixed-scale crops.
                full = CAP_SEAT + 0.75
                ends = [project(m.Transform(Gf.Vec3d(0, 0, z)), cam_inv) for z in (0.0, full)]
                mid = project(m.Transform(Gf.Vec3d(0, 0, full / 2)), cam_inv)
                sides = [project(m.Transform(Gf.Vec3d(*v)), cam_inv) for v in ((TUBE_R, 0, full / 2), (0, TUBE_R, full / 2))]
                if all(ends) and mid and all(sides):
                    info["axis_px"] = [[round(ends[0][0], 1), round(ends[0][1], 1)], [round(ends[1][0], 1), round(ends[1][1], 1)]]
                    info["diam_px"] = round(2 * max(math.hypot(s[0] - mid[0], s[1] - mid[1]) for s in sides), 1)
                info["in_frame"] = bool(len(uv) and uv[:, 0].max() < W and uv[:, 1].max() < H and uv.min() >= 0)

        if public_view:
            for p, info in labels.items():
                c = project(public_view['aperture_world'][p], cam_inv)
                if not c:
                    raise RuntimeError(f"{p}: public aperture is outside camera frustum")
                offsets = public_view['crop_offsets']['good' if p[0]=='G' else 'scrap']
                info['review_bbox'] = [round(c[0]+offsets[0],1), round(c[1]+offsets[1],1),
                                       round(c[0]+offsets[2],1), round(c[1]+offsets[3],1)]
                info['target_px'] = [round(c[0],1),round(c[1],1)]
                info['in_frame'] = bool(0<=c[0]<W and 0<=c[1]<H)
        name = f"{fi:06d}"
        if not cv2.imwrite(str(out / "frames" / f"{name}.png"), final):
            raise OSError(f"could not write frame {name}")
        json.dump({"frame": fi, "seed": a.seed, "image": f"frames/{name}.png", "positions": labels},
                  open(out / "labels" / f"{name}.json", "w"), indent=1)
        if a.preview:
            ov = final.copy()
            for p, info in labels.items():
                tag = p + " " + "".join(w[0] for w in info["class"].split("_")).upper()
                if p[0] == "S":
                    if info["bbox_px"]:
                        x0, y0, x1, y1 = (int(round(v)) for v in info["bbox_px"])
                        cv2.rectangle(ov, (x0, y0), (x1, y1), COLORS[info["class"]], 1, cv2.LINE_AA)
                        cv2.putText(ov, tag + (" " + info["damage"] if "damage" in info else ""), (x0, max(8, y0 - 3)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, COLORS[info["class"]], 1, cv2.LINE_AA)
                    continue
                if not info["aperture_px"]:
                    continue
                c = tuple(int(round(x)) for x in info["aperture_px"])
                col = COLORS[info["class"]]
                cv2.circle(ov, c, int(info["radius_px"] or 12), col, 2, cv2.LINE_AA)
                if info.get("cap_px"):
                    cv2.circle(ov, tuple(int(round(x)) for x in info["cap_px"]), 3, col, -1, cv2.LINE_AA)
                cv2.putText(ov, tag, (c[0] - 18, c[1] - int(info["radius_px"] or 12) - 3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.32, col, 1, cv2.LINE_AA)
            cv2.imwrite(str(out / "overlays" / f"{name}.png"), ov)
            overlays.append(ov)
            boxes = np.array([i["bbox_px"] for p, i in labels.items() if p[0] == "S" and i.get("bbox_px")])
            if len(boxes):                                     # 3x zoom of the scrap tray, unannotated
                x0, y0 = np.clip(boxes[:, :2].min(0) - 12, 0, None).astype(int)
                x1, y1 = np.minimum(boxes[:, 2:].max(0) + 12, [W, H]).astype(int)
                s_zooms.append(cv2.resize(final[y0:y1, x0:x1], ((x1 - x0) * 3, (y1 - y0) * 3),
                                          interpolation=cv2.INTER_CUBIC))
        counts = {c: sum(1 for i in labels.values() if i["class"] == c) for c in CLASSES}
        print(f"frame {name} written {counts}", flush=True)

    if a.preview and overlays:
        cols = 4
        thumbs = [cv2.resize(o, (W // 2, H // 2), interpolation=cv2.INTER_AREA) for o in overlays]
        thumbs += [np.zeros_like(thumbs[0])] * (-len(thumbs) % cols)
        sheet = np.vstack([np.hstack(thumbs[r:r + cols]) for r in range(0, len(thumbs), cols)])
        cv2.imwrite(str(out / "preview_sheet.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
    if s_zooms:
        hh = max(z.shape[0] for z in s_zooms)
        tiles = [cv2.copyMakeBorder(z, 0, hh - z.shape[0], 0, 4, cv2.BORDER_CONSTANT) for z in s_zooms]
        rows = [np.hstack(tiles[r:r + 6]) for r in range(0, len(tiles), 6)]
        ww = max(r.shape[1] for r in rows)
        sheet = np.vstack([cv2.copyMakeBorder(r, 0, 4, 0, ww - r.shape[1], cv2.BORDER_CONSTANT) for r in rows])
        cv2.imwrite(str(out / "s_tray_zoom.png"), sheet)
    # Everything is written before close(): Kit's fast shutdown exits the process.
    print("done", flush=True)
except BaseException:
    traceback.print_exc()
    # Isaac Sim 5.1's close() takes no exit code, and its fast shutdown would exit 0: fail hard instead.
    os._exit(1)
kit.close()
