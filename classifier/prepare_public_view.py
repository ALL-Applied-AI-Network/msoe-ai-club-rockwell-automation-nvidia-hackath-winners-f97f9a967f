"""Fit the twin's camera to supplied fixture geometry, never to defect labels."""
import json
from pathlib import Path
import cv2
import numpy as np
from pxr import Usd, UsdGeom, Gf

ROOT = Path(__file__).resolve().parents[1]


def main():
    calibration = json.loads((ROOT/'inspection_app/calibration.json').read_text())
    profile = calibration['profiles'][0]
    stage = Usd.Stage.Open(str(ROOT/'digital_twin/csi_digital_twin_v1/stations/tray_inspection/tray_inspection.usdc'))
    cache = UsdGeom.XformCache()
    def world(name):
        return cache.GetLocalToWorldTransform(stage.GetPrimAtPath('/TrayInspection/'+name))
    rw = cache.GetLocalToWorldTransform(stage.GetPrimAtPath('/TrayInspection'))
    gplate, splate = world('tray_good_top'), world('tray_scrap_top')
    obj, image, apertures, scrap_matrices = [], [], {}, {}
    base = world('S01_tube')
    def center(grid,i):
        col,row=divmod(i-1,5); u,v=1-col/5,row/4
        tl,tr,bl,br=np.asarray(grid,float)
        return (1-v)*((1-u)*tl+u*tr)+v*((1-u)*bl+u*br)
    for i in range(1,31):
        p=f'G{i:02d}'
        point=world(p+'_tube').Transform(Gf.Vec3d(0,0,1.76))
        apertures[p]=list(point)
        sp=splate.Transform(gplate.GetInverse().Transform(point))
        apertures[f'S{i:02d}']=list(sp)
        matrix=Gf.Matrix4d(base)
        matrix.SetRow(3,Gf.Vec4d(sp[0],sp[1],sp[2]-.008,1))
        local=matrix*rw.GetInverse()
        scrap_matrices[f'S{i:02d}']=[list(local[j]) for j in range(4)]
        for name,point2 in [(p,point),(f'S{i:02d}',sp)]:
            obj.append(list(point2)); image.append(center(profile['areas']['good' if name[0]=='G' else 'scrap']['grid'],i))
    obj=np.asarray(obj,np.float32); image=np.asarray(image,np.float32)
    k=np.array([[1500.,0,960],[0,1500.,540],[0,0,1.]])
    flags=cv2.CALIB_USE_INTRINSIC_GUESS|cv2.CALIB_FIX_PRINCIPAL_POINT|cv2.CALIB_FIX_ASPECT_RATIO|cv2.CALIB_ZERO_TANGENT_DIST|cv2.CALIB_FIX_K2|cv2.CALIB_FIX_K3
    rms,k,dist,rvecs,tvecs=cv2.calibrateCamera([obj],[image],(1920,1080),k,np.zeros(5),flags=flags)
    rotation,_=cv2.Rodrigues(rvecs[0]); position=-rotation.T@tvecs[0].ravel()
    camera=np.eye(4);camera[:3,:3]=rotation.T@np.diag([1.,-1.,-1.])*.0254;camera[:3,3]=position
    local=Gf.Matrix4d(*camera.T.flatten().tolist())*rw.GetInverse()
    config=dict(final_image_px=[1920,1080],render_px=[2112,1188],focal_px_final=float(k[0,0]),lens_k1=float(dist.ravel()[0]),
                camera_local_matrix=[list(local[i]) for i in range(4)],aperture_world=apertures,scrap_tube_matrices=scrap_matrices,
                crop_offsets={t:profile['areas'][t]['crop_offsets'] for t in ['good','scrap']},
                geometry_fit_rms_px=float(rms),source='supplied public calibration geometry; no appearance labels or pixels used in training')
    target=ROOT/'inspection_app/twin_public_view.json';target.write_text(json.dumps(config,indent=2)+'\n')
    print(f'Camera fit RMS {rms:.2f}px, f={k[0,0]:.1f}, k1={dist.ravel()[0]:.4f}')


if __name__=='__main__':main()
