"""GPU evaluation of confirmed real crops; never feeds real pixels to training."""
import argparse,csv,hashlib,json,time
from collections import Counter
from pathlib import Path
import cv2,numpy as np,torch,torchvision
from public_geometry import review_crop
from decision import decide


def report(rows,probabilities,classes):
    truth=np.array([classes.index(r['class']) for r in rows]);raw=probabilities.argmax(1)
    decisions=np.array([classes.index(c) if c in classes else len(classes) for c,_,_ in
                        [decide(classes,p.tolist()) for p in probabilities]])
    cm=np.zeros((5,6),int);np.add.at(cm,(truth,decisions),1)
    defects=np.isin(truth,[classes.index(c) for c in ['missing_cap','crooked_cap','damaged_vial']]);good=truth==classes.index('pass')
    pidx=classes.index('pass');scores=1-probabilities[:,pidx]
    threshold=float(scores[defects].min()) if defects.any() else None
    per_class={}
    for i,c in enumerate(classes):
        support=int((truth==i).sum());predicted=int((decisions==i).sum());tp=int(cm[i,i])
        precision=tp/predicted if predicted else 0.;recall=tp/support if support else None
        per_class[c]=dict(support=support,precision=precision,recall=recall,
                          f1=2*precision*recall/(precision+recall) if recall is not None and precision+recall else 0.)
    return dict(positions=len(rows),accuracy=float((truth==decisions).mean()),raw_accuracy=float((truth==raw).mean()),
        confusion=cm.tolist(),classes=classes+['uncertain'],defects=int(defects.sum()),
        defects_accepted=int((defects&(decisions==pidx)).sum()),
        defects_as_empty=int((defects&(decisions==classes.index('no_vial'))).sum()),
        occupied_as_empty=int(((truth!=classes.index('no_vial'))&(decisions==classes.index('no_vial'))).sum()),
        empty_as_occupied=int(((truth==classes.index('no_vial'))&np.isin(decisions,[0,1,2,3])).sum()),
        uncertain=int((decisions==5).sum()),good=int(good.sum()),good_held=int((good&(decisions!=pidx)).sum()),
        per_class=per_class,macro_f1=float(np.mean([v['f1'] for v in per_class.values() if v['support']])),
        diagnostic_threshold_to_catch_all_known_defects=threshold,
        diagnostic_false_alarm_rate_at_that_threshold=float((scores[good]>=threshold).mean()) if threshold is not None and good.any() else None,
        threshold_note='Post-hoc diagnostic only; not deployed or an unbiased accuracy claim',
        errors=[dict(**r,prediction=classes[d] if d<5 else 'uncertain',probabilities=dict(zip(classes,p.tolist())))
                for r,t,d,p in zip(rows,truth,decisions,probabilities) if t!=d])


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--real',type=Path,required=True);ap.add_argument('--model',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True);ap.add_argument('--version',required=True)
    a=ap.parse_args();assert torch.cuda.is_available(),'Use the DGX Spark GPU';torch.set_num_threads(4)
    ck=torch.load(a.model,map_location='cpu',weights_only=True);classes=ck['classes'];rows=[];crops=[]
    source=json.loads((a.real/'labels.json').read_text())
    for scene in source['results']:
        path=a.real/scene['source_path'];assert hashlib.sha256(path.read_bytes()).hexdigest()==scene['source_sha256']
        im=cv2.imread(str(path));assert im.shape[:2]==(1080,1920)
        for s in scene['slots']:
            if s['review_status']!='confirmed' or s['class'] is None:continue
            rows.append(dict(image=scene['filename'],pos=s['id'],tray=s['id'][0],**{'class':s['class']}))
            crops.append(review_crop(im,s['bbox'],ck['size']))
    assert rows
    model=torchvision.models.resnet18(weights=None);model.fc=torch.nn.Linear(model.fc.in_features,len(classes))
    model.load_state_dict(ck['state_dict']);model.cuda().eval()
    mean=torch.tensor(ck['mean'],device='cuda').view(1,3,1,1);std=torch.tensor(ck['std'],device='cuda').view(1,3,1,1)
    x=torch.from_numpy(np.stack(crops)[...,::-1].copy()).permute(0,3,1,2).cuda().float()/255
    tick=time.perf_counter()
    with torch.inference_mode():probabilities=model((x-mean)/std).softmax(1).cpu().numpy()
    results=dict(version=a.version,model_sha256=hashlib.sha256(a.model.read_bytes()).hexdigest(),device=torch.cuda.get_device_name(0),
        inference_ms=(time.perf_counter()-tick)*1000,training_source=ck.get('training_source','synthetic-only'),
        scope='Two supplied public development reference images; no independent real-line generalization claim',
        all=report(rows,probabilities,classes),by_image={},by_tray={})
    for name in sorted({r['image'] for r in rows}):
        mask=np.array([r['image']==name for r in rows]);results['by_image'][name]=report([r for r,m in zip(rows,mask) if m],probabilities[mask],classes)
    for tray in 'GS':
        mask=np.array([r['tray']==tray for r in rows]);results['by_tray'][tray]=report([r for r,m in zip(rows,mask) if m],probabilities[mask],classes)
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(results,indent=2)+'\n')
    np.savez_compressed(a.out.with_suffix('.npz'),probabilities=probabilities,rows=np.array([json.dumps(r) for r in rows]))
    print(a.version,{k:v for k,v in results['all'].items() if k not in ['errors','confusion','per_class']},flush=True)


if __name__=='__main__':main()
