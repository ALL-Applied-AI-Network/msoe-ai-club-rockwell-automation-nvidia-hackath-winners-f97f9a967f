import hashlib,json,sys,threading,time
from pathlib import Path
import cv2,numpy as np,torch,torchvision

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'classifier'))
from public_geometry import PublicGeometry,review_crop
from decision import decide
from .sequence import summarize

COLORS={'pass':(90,210,95),'missing_cap':(0,170,255),'crooked_cap':(210,90,230),
        'damaged_vial':(70,70,245),'no_vial':(220,170,90),'uncertain':(160,160,160)}


class Engine:
    def __init__(self):
        torch.set_num_threads(4)
        self.device='cuda' if torch.cuda.is_available() else 'cpu'
        self.geometry=PublicGeometry();self.models={};self.lock=threading.Lock()

    @staticmethod
    def model_path(version):
        if version=='v2':return ROOT/'model/model.pt'
        if version=='v3_HZ':return ROOT/'model/v3_HZ/model.pt'
        if version=='v4_HZ':return ROOT/'model/v4_HZ/model.pt'
        raise ValueError('Unknown model version')

    def load(self,version):
        path=self.model_path(version)
        if not path.exists():raise FileNotFoundError(f'{version} has not finished training')
        stamp=path.stat().st_mtime_ns
        if version not in self.models or self.models[version]['stamp']!=stamp:
            ck=torch.load(path,map_location='cpu',weights_only=True)
            if version in {'v3_HZ','v4_HZ'} and ck.get('crop_mode')!='review_bbox_v1':raise ValueError('Model crop contract mismatch')
            model=torchvision.models.resnet18(weights=None);model.fc=torch.nn.Linear(model.fc.in_features,len(ck['classes']))
            model.load_state_dict(ck['state_dict']);model.to(self.device).eval()
            self.models[version]=dict(model=model,checkpoint=ck,stamp=stamp,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                     mean=torch.tensor(ck['mean'],device=self.device).view(1,3,1,1),
                                     std=torch.tensor(ck['std'],device=self.device).view(1,3,1,1))
        return self.models[version]

    def inspect(self,image,version='v4_HZ',counts=None,complete=False,geometry_override=None):
        tick=time.perf_counter()
        slots,alignment=geometry_override if geometry_override else self.geometry.align(image)
        with self.lock,torch.inference_mode():
            entry=self.load(version);ck=entry['checkpoint']
            crops=[review_crop(image,s['bbox'],ck['size']) for s in slots]
            x=torch.from_numpy(np.stack(crops)[...,::-1].copy()).permute(0,3,1,2).to(self.device).float()/255
            probs=entry['model']((x-entry['mean'])/entry['std']).softmax(1).cpu().numpy()
        for s,p in zip(slots,probs):
            cls,raw,confidence=decide(ck['classes'],p.tolist())
            s.update({'class':cls,'raw_class':raw,'confidence':confidence,'probabilities':dict(zip(ck['classes'],p.tolist()))})
        groups=summarize(slots,counts,complete)
        return dict(model=version,model_sha256=entry['sha256'],device=self.device,slots=slots,groups=groups,alignment=alignment,
                    elapsed_ms=round((time.perf_counter()-tick)*1000,1),
                    validation_status='experimental; independent real-camera validation is still required',
                    baseline_domain_warning=version=='v2')


def overlay(image,result):
    out=image.copy()
    for s in result['slots']:
        x,y=[round(v) for v in s['center']];color=COLORS[s['class']]
        if s['action']=='pending':color=(160,160,160)
        cv2.circle(out,(x,y),18,color,3,cv2.LINE_AA)
        tag=f"{s['id']} {s['class']} {s['confidence']:.2f}"
        cv2.putText(out,tag,(x-20,y-24),cv2.FONT_HERSHEY_SIMPLEX,.42,(15,15,15),3,cv2.LINE_AA)
        cv2.putText(out,tag,(x-20,y-24),cv2.FONT_HERSHEY_SIMPLEX,.42,color,1,cv2.LINE_AA)
    return out
