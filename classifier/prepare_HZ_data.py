"""Create synthetic-only train/val crops and separate confirmed real evaluation crops."""
import argparse,csv,hashlib,json
from pathlib import Path
import cv2
from public_geometry import review_crop


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--train',type=Path,required=True);ap.add_argument('--val',type=Path,required=True)
    ap.add_argument('--test',type=Path,required=True)
    ap.add_argument('--real',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--expected-frames',type=int,nargs=3,default=[96,24,24],metavar=('TRAIN','VAL','TEST'))
    a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    rows=[];seen={};seeds={}
    def save(image,box,split,cls,name,metadata):
        rel=Path(split)/cls/(name+'.png');p=a.out/rel;p.parent.mkdir(parents=True,exist_ok=True)
        if not cv2.imwrite(str(p),review_crop(image,box)):raise OSError(p)
        rows.append(dict(path=rel.as_posix(),class_=cls,split=split,**metadata))
    for (split,run),expected in zip([('train',a.train),('val',a.val),('test',a.test)],a.expected_frames):
        labels=sorted((run/'labels').glob('*.json'))
        if len(labels)!=expected:raise ValueError(f'{split}: expected {expected} complete frames, found {len(labels)}')
        for lp in labels:
            d=json.loads(lp.read_text());seed=d['seed'];frame=d['frame'];key=(seed,frame)
            if key in seen:raise ValueError('Duplicate seed/frame across splits')
            seen[key]=split;seeds.setdefault(split,set()).add(seed)
            im=cv2.imread(str(run/d['image']));assert im is not None and im.shape[:2]==(1080,1920) and im.std()>8
            assert len(d['positions'])==60
            assert all(v['in_frame'] for v in d['positions'].values())
            for pos,v in d['positions'].items():
                save(im,v['review_bbox'],split,v['class'],f'{seed}_{frame:06d}_{pos}',
                     dict(tray=pos[0],pos=pos,frame=str(frame),run=str(seed),damage=v.get('damage',''),source='synthetic'))
    assert all(seeds[x].isdisjoint(seeds[y]) for x,y in [('train','val'),('train','test'),('val','test')]), 'use different rendering seeds'
    real=json.loads((a.real/'labels.json').read_text())
    for scene in real['results']:
        src=a.real/scene['source_path'];assert hashlib.sha256(src.read_bytes()).hexdigest()==scene['source_sha256']
        im=cv2.imread(str(src));assert im.shape[:2]==(1080,1920)
        for s in scene['slots']:
            if s['review_status']!='confirmed' or s['class'] is None:continue
            save(im,s['bbox'],'real_eval',s['class'],f"{Path(scene['filename']).stem}_{s['id']}",
                 dict(tray=s['id'][0],pos=s['id'],frame=scene['filename'],run=scene['source_sha256'],damage='',source='real_evaluation_only'))
    with (a.out/'index.csv').open('w',newline='') as f:
        fields=['path','class','split','tray','pos','frame','run','damage','source'];w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for r in rows:r['class']=r.pop('class_');w.writerow(r)
    manifest=dict(crop_mode='review_bbox_v1',real_pixels_used_for_training=False,
                  seed_groups={k:sorted(v) for k,v in seeds.items()},counts={s:sum(r['split']==s for r in rows) for s in ['train','val','test','real_eval']})
    (a.out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(manifest),flush=True)


if __name__=='__main__':main()
