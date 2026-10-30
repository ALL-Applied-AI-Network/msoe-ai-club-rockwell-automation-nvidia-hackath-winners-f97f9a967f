import json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import cv2,numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'classifier'))
from evaluate_HZ import report
from prepare_HZ_data import main as prepare


class HZMetricTests(unittest.TestCase):
    def test_escape_occupancy_and_uncertainty_are_distinct(self):
        classes=['pass','missing_cap','crooked_cap','damaged_vial','no_vial']
        rows=[{'class':c} for c in ['damaged_vial','damaged_vial','pass','no_vial']]
        p=np.array([[.6,.005,.005,0,.39],[.02,0,0,.08,.9],[.4,0,.35,0,.25],[.7,.1,0,0,.2]])
        r=report(rows,p,classes)
        self.assertEqual(r['defects_accepted'],1);self.assertEqual(r['defects_as_empty'],1)
        self.assertEqual(r['empty_as_occupied'],1);self.assertEqual(r['uncertain'],1)
        self.assertEqual(r['accuracy'],0);self.assertEqual(r['good_held'],1)

    def test_seed_frame_overlap_is_rejected_before_real_evaluation_import(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            image=np.repeat(np.tile((np.arange(1920)%256).astype(np.uint8),(1080,1))[...,None],3,axis=2)
            for split in ['train','val','test']:
                run=root/split;(run/'frames').mkdir(parents=True);(run/'labels').mkdir()
                cv2.imwrite(str(run/'frames/000000.png'),image)
                positions={f'{t}{i:02d}':{'class':'pass','in_frame':True,'review_bbox':[10,10,100,100]} for t in 'GS' for i in range(1,31)}
                (run/'labels/000000.json').write_text(json.dumps(dict(seed=1,frame=0,image='frames/000000.png',positions=positions)))
            args=['prepare','--train',str(root/'train'),'--val',str(root/'val'),'--test',str(root/'test'),
                  '--real',str(root/'nonexistent-real'),'--out',str(root/'crops'),'--expected-frames','1','1','1']
            with patch.object(sys,'argv',args),self.assertRaisesRegex(ValueError,'Duplicate seed/frame'):
                prepare()
