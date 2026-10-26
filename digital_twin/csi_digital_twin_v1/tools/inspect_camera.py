"""Print every authored attribute of CapSealCam (and its parent Xform)."""
import sys
from pxr import Usd

stage = Usd.Stage.Open(sys.argv[1])
for path in ["/TrayInspection/CapSealCam", "/TrayInspection/CapSealCam/CapSealCam"]:
    prim = stage.GetPrimAtPath(path)
    print(f"## {path} [{prim.GetTypeName()}] apiSchemas={prim.GetAppliedSchemas()}")
    for attr in prim.GetAttributes():
        if attr.HasAuthoredValue() or attr.GetName().startswith(("omni", "cameraProjection", "fthetaP", "physical")):
            print(f"  {attr.GetName()} ({attr.GetTypeName()}) = {attr.Get()}")
    print("  customData:", prim.GetCustomData())
