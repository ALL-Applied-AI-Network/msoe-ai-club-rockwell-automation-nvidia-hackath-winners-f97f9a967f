"""Evaluate the frozen synthetic test split on the DGX GPU."""
import argparse,csv,hashlib,json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import cv2,numpy as np,torch,torchvision
from evaluate_HZ import report


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--crops',type=Path,required=True);ap.add_argument('--model',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    assert torch.cuda.is_available();torch.set_num_threads(4)
    rows=[r for r in csv.DictReader((a.crops/'index.csv').open()) if r['split']=='test'];assert rows
    assert all(r['source']=='synthetic' for r in rows)
    ck=torch.load(a.model,map_location='cpu',weights_only=True);classes=ck['classes']
    model=torchvision.models.resnet18(weights=None);model.fc=torch.nn.Linear(model.fc.in_features,len(classes))
    model.load_state_dict(ck['state_dict']);model.cuda().eval()
    mean=torch.tensor(ck['mean'],device='cuda').view(1,3,1,1);std=torch.tensor(ck['std'],device='cuda').view(1,3,1,1)
    def read(row):
        image=cv2.imread(str(a.crops/row['path']));assert image is not None and image.shape==(128,128,3)
        return image[...,::-1].copy()
    probabilities=[]
    with torch.inference_mode(),ThreadPoolExecutor(4) as pool:
        for i in range(0,len(rows),128):
            x=torch.from_numpy(np.stack(list(pool.map(read,rows[i:i+128])))).permute(0,3,1,2).cuda().float()/255
            probabilities.append(model((x-mean)/std).softmax(1).cpu().numpy())
    probabilities=np.concatenate(probabilities)
    result=dict(model_sha256=hashlib.sha256(a.model.read_bytes()).hexdigest(),device=torch.cuda.get_device_name(0),
                all=report(rows,probabilities,classes),by_tray={},by_damage={})
    for name,key,groups in [('by_tray','tray',['G','S']),('by_damage','damage',sorted({r['damage'] for r in rows if r['damage']}))]:
        for group in groups:
            mask=np.array([r[key]==group for r in rows]);result[name][group]=report([r for r,m in zip(rows,mask) if m],probabilities[mask],classes)
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2)+'\n')
    print('synthetic test',{k:result['all'][k] for k in ['accuracy','positions','defects_accepted','good_held']},flush=True)


if __name__=='__main__':main()
