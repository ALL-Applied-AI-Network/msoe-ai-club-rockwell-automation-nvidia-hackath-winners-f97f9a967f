import sys,unittest
from pathlib import Path
import cv2,numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'classifier'))
from public_geometry import PublicGeometry,review_crop,APP


class PublicGeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.geometry=PublicGeometry()
    def test_both_real_references_match_sixty_ordered_targets(self):
        for name in ['tray_000001.jpg','tray_000122.jpg']:
            image=cv2.imread(str(APP/'references'/name));slots,alignment=self.geometry.align(image)
            self.assertEqual(len({s['id'] for s in slots}),60)
            self.assertEqual(alignment['residual'],0)
            self.assertGreater(slots[0]['center'][0],slots[5]['center'][0])
            self.assertLess(slots[0]['center'][1],slots[4]['center'][1])
            for slot in slots:self.assertEqual(review_crop(image,slot['bbox']).shape,(128,128,3))
    def test_unfamiliar_camera_and_wrong_resolution_are_rejected(self):
        for image in [np.zeros((480,640,3),np.uint8),np.zeros((1080,1920,3),np.uint8)]:
            with self.assertRaises(ValueError):self.geometry.align(image)
    def test_outside_crop_fails_instead_of_silent_padding(self):
        with self.assertRaises(ValueError):review_crop(np.zeros((20,20,3),np.uint8),[-1,0,10,10])


if __name__=='__main__':unittest.main()
