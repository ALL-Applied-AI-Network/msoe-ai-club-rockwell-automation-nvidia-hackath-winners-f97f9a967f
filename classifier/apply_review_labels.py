"""Replace real_eval classes in a crop index with a validated participant review."""
import argparse,csv,json
from pathlib import Path

ap=argparse.ArgumentParser();ap.add_argument('--index',type=Path,required=True)
ap.add_argument('--review',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
rows=list(csv.DictReader(a.index.open(newline='',encoding='utf-8'))); review=json.loads(a.review.read_text(encoding='utf-8'))
labels={(r['source_sha256'],s['id']):s['class'] for r in review['results'] for s in r['slots']
        if s['review_status']=='confirmed'}
assert len(review['results'])==2 and len(labels)==120 and all(labels.values()),'review must contain 120 confirmed labels'
seen=set()
for row in rows:
    if row['split']!='real_eval':continue
    key=(row['run'],row['pos']);assert key in labels,f'missing reviewed label: {key}'
    row['class']=labels[key];seen.add(key)
assert seen==set(labels),'crop index and reviewed labels differ'
a.out.parent.mkdir(parents=True,exist_ok=True)
with a.out.open('w',newline='',encoding='utf-8') as f:
    w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
print(f'{len(seen)} reviewed labels applied -> {a.out}')
