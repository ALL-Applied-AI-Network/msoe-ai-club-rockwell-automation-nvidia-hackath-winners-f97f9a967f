"""Calibrated 60-position geometry and crop contract shared by training and serving.

Geometry/anchor matching follows the participant labeler's supplied calibration.
"""
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

APP = Path(__file__).resolve().parents[1] / 'inspection_app'


def review_crop(image, bbox, size=128):
    left, top, right, bottom = [int(round(x)) for x in bbox]
    h,w=image.shape[:2]
    if not (0 <= left < right <= w and 0 <= top < bottom <= h):
        raise ValueError('Calibrated crop extends outside the image')
    return cv2.resize(image[top:bottom,left:right],(size,size),interpolation=cv2.INTER_AREA)


def slots_for_profile(profile, offset=(0,0)):
    slots=[]
    for area,prefix in [('good','G'),('scrap','S')]:
        tl,tr,bl,br=np.asarray(profile['areas'][area]['grid'],float)
        left,top,right,bottom=profile['areas'][area]['crop_offsets']
        for i in range(1,31):
            column,row=divmod(i-1,5);u,v=1-column/5,row/4
            x,y=(1-v)*((1-u)*tl+u*tr)+v*((1-u)*bl+u*br)+np.asarray(offset)
            slots.append(dict(id=f'{prefix}{i:02d}',area=area,position=i,center=[float(x),float(y)],
                              bbox=[round(x+left),round(y+top),round(x+right),round(y+bottom)]))
    return slots


class PublicGeometry:
    def __init__(self, config=APP/'calibration.json', references=APP/'references'):
        self.config=json.loads(Path(config).read_text())
        self.references={}
        for p in self.config['profiles']:
            file=Path(references)/p['reference_image']
            if hashlib.sha256(file.read_bytes()).hexdigest()!=p['reference_sha256']:
                raise ValueError('Camera reference hash mismatch')
            self.references[p['id']]=self.anchor(Image.open(file).convert('RGB'),p,(0,0))

    @staticmethod
    def anchor(im,profile,offset):
        dx,dy=offset
        values=[]
        for l,t,r,b in profile['alignment_anchors']:
            crop=im.crop((l+dx,t+dy,r+dx,b+dy)).convert('L').resize((48,24))
            values.append(np.asarray(crop,dtype=float).ravel()/255)
        values=np.concatenate(values)
        return (values-values.mean())/max(values.std(),.05)

    def align(self,image):
        if image.shape[:2]!=(1080,1920):
            raise ValueError('Camera requires calibrated 1920x1080 frames; select that camera resolution')
        im=Image.fromarray(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
        candidates=[]
        for p in self.config['profiles']:
            reference=self.references[p['id']]
            for dy in range(-18,19,3):
                for dx in range(-18,19,3):
                    error=float(np.mean((self.anchor(im,p,(dx,dy))-reference)**2))
                    candidates.append((error,p['id'],dx,dy))
        error,profile_id,dx,dy=min(candidates)
        if error>self.config['max_alignment_error'] or max(abs(dx),abs(dy))>=18:
            raise ValueError('Unfamiliar camera alignment. Recalibrate before inspecting this view.')
        profile=next(p for p in self.config['profiles'] if p['id']==profile_id)
        return slots_for_profile(profile,(dx,dy)),dict(profile=profile_id,offset=[dx,dy],residual=error,
                                                     geometry_requires_visual_check=True)
