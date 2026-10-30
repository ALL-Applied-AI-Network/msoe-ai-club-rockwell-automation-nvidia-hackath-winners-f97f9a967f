"""Inspect calibrated public-camera images with the same engine as the website."""
import argparse,json,sys
from pathlib import Path
import cv2
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from inspection_app.engine import Engine,overlay


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('images',nargs='+',type=Path);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--version',choices=['v2','v3_HZ'],default='v3_HZ')
    ap.add_argument('--good-count',type=int);ap.add_argument('--scrap-count',type=int)
    ap.add_argument('--order-complete',action='store_true');a=ap.parse_args()
    if len({p.stem for p in a.images})!=len(a.images):ap.error('image stems must be unique')
    engine=Engine();decoded=[]
    for path in a.images:
        image=cv2.imread(str(path))
        if image is None:ap.error(f'unreadable image: {path}')
        slots,alignment=engine.geometry.align(image);decoded.append((path,image,(slots,alignment)))
    a.out.mkdir(parents=True,exist_ok=False)
    counts={k:v for k,v in [('good',a.good_count),('scrap',a.scrap_count)] if v is not None}
    for path,image,geometry in decoded:
        result=engine.inspect(image,a.version,counts,a.order_complete,geometry_override=geometry)
        (a.out/(path.stem+'.json')).write_text(json.dumps(result,indent=2)+'\n')
        if not cv2.imwrite(str(a.out/(path.stem+'-overlay.jpg')),overlay(image,result)):raise OSError('cannot save overlay')
        print(path.name,a.version,f"{result['elapsed_ms']}ms",result['groups'])


if __name__=='__main__':main()
