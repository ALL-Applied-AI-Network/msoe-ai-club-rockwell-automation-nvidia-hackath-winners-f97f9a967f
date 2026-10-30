import base64
import copy
import io
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from fastapi.testclient import TestClient

from inspection_app import server
from inspection_app.sequence import summarize,TemporalTracker


def slots(default='no_vial'):
    return [dict(id=f'{prefix}{i:02d}',area=area,position=i,center=[100+i*5,100],bbox=[90,90,130,130],
                 **{'class':default},confidence=.95,probabilities={default:1.})
            for prefix,area in [('G','good'),('S','scrap')] for i in range(1,31)]


def fake_inspect(image,version,counts=None,complete=False):
    items=slots();items[0]['class']='pass'
    return dict(model=version,model_sha256='test-checkpoint',device='test',slots=items,
                groups=summarize(items,counts,complete),alignment=dict(profile='test',offset=[0,0],residual=0),elapsed_ms=1)


class FillPatternTests(unittest.TestCase):
    def test_trailing_empty_is_pending_and_internal_gap_is_only_possible_without_signal(self):
        items=slots();items[0]['class']='pass';items[2]['class']='pass'
        result=summarize(items)
        self.assertEqual(result['good']['possible_gaps'],[2])
        self.assertEqual(result['good']['missing'],[])
        self.assertEqual(result['good']['pending'],list(range(4,31)))
        result=summarize(items,{'good':4})
        self.assertEqual(result['good']['missing'],[2,4])

    def test_order_completion_does_not_require_scrap_fixture_to_be_full(self):
        result=summarize(slots(),complete=True)
        self.assertEqual(result['good']['missing'],list(range(1,31)))
        self.assertEqual(result['scrap']['missing'],[])
        self.assertEqual(result['scrap']['pending'],list(range(1,31)))
        self.assertIn('upstream',result['scrap']['routing_verdict'])

    def test_unknown_is_not_absence_and_invalid_inputs_fail(self):
        items=slots();items[0]['class']='uncertain'
        result=summarize(items,{'good':1})
        self.assertEqual(result['good']['uncertain'],[1]);self.assertEqual(result['good']['missing'],[])
        with self.assertRaises(ValueError):summarize(items,{'good':True})
        with self.assertRaises(ValueError):summarize(items[:-1])

    def test_temporal_placement_requires_two_observations_and_does_not_invent_initial_events(self):
        tracker=TemporalTracker();empty=slots()
        self.assertEqual(tracker.update(copy.deepcopy(empty)),[])
        self.assertEqual(tracker.update(copy.deepcopy(empty)),[])
        placed=slots();placed[0]['class']='pass'
        self.assertEqual(tracker.update(copy.deepcopy(placed)),[])
        event=tracker.update(copy.deepcopy(placed))
        self.assertEqual(len(event),1);self.assertEqual(event[0]['position'],'G01')
        self.assertEqual(event[0]['event'],'placement_observed')


class WebBoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client=TestClient(server.app)
        cls.image=(server.APP/'references/tray_000001.jpg').read_bytes()

    def test_pages_and_both_examples_are_served(self):
        for route in ['/','/live','/assets/app.js','/assets/style.css','/api/examples/tray_000001.jpg','/api/examples/tray_000122.jpg']:
            self.assertEqual(self.client.get(route).status_code,200,route)
        self.assertEqual(self.client.get('/api/examples/secret.txt').status_code,404)

    def test_status_uses_cached_health_without_contacting_dgx(self):
        with patch.object(server.cosmos_client,'call',side_effect=AssertionError('network on status path')):
            data=self.client.get('/api/status').json()
        self.assertIn('v3_HZ',data['models']);self.assertIn('ready',data['cosmos'])

    def test_invalid_counts_and_image_are_rejected_before_inference(self):
        with patch.object(server.engine,'inspect',side_effect=AssertionError('invalid input reached inference')):
            for value in ['-1','31','2.5','٢']:
                response=self.client.post('/api/inspect',data={'good':value},files={'file':('image.jpg',self.image,'image/jpeg')})
                self.assertEqual(response.status_code,422,value)
            response=self.client.post('/api/inspect',files={'file':('bad.jpg',b'not an image','image/jpeg')})
            self.assertEqual(response.status_code,422)

    def test_real_wrong_resolution_is_refused(self):
        _,encoded=cv2.imencode('.png',np.zeros((480,640,3),np.uint8))
        response=self.client.post('/api/inspect',data={'version':'v2'},files={'file':('wrong.png',encoded.tobytes(),'image/png')})
        self.assertEqual(response.status_code,422);self.assertIn('1920x1080',response.json()['detail'])

    def test_inspection_contract_and_model_session_isolation(self):
        session='3d5ba681-2aa3-4f8e-b60e-3936a3e14cc5'
        with patch.object(server.engine,'inspect',side_effect=fake_inspect):
            for version in ['v2','v3_HZ']:
                response=self.client.post('/api/inspect',data={'version':version,'session':session},files={'file':('image.jpg',self.image,'image/jpeg')})
                self.assertEqual(response.status_code,200,response.text[:200]);data=response.json()
                self.assertEqual(len(data['slots']),60);self.assertTrue(data['overlay'].startswith('data:image/jpeg;base64,'))
                self.assertEqual(data['events'],[])
            self.assertIn((session,'v2'),server.sessions);self.assertIn((session,'v3_HZ'),server.sessions)

    def test_video_samples_and_only_final_frame_receives_completion_signal(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'sample.avi';writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'MJPG'),1,(1920,1080))
            self.assertTrue(writer.isOpened())
            for i in range(3):writer.write(np.full((1080,1920,3),i*20,np.uint8))
            writer.release()
            with patch.object(server.engine,'inspect',side_effect=fake_inspect):
                response=self.client.post('/api/video',data={'version':'v2','complete':'true'},files={'file':('sample.avi',path.read_bytes(),'video/x-msvideo')})
        self.assertEqual(response.status_code,200,response.text[:200]);frames=response.json()['frames']
        self.assertEqual(len(frames),3);self.assertEqual(frames[0]['groups']['good']['missing'],[])
        self.assertEqual(frames[-1]['groups']['good']['missing'],list(range(2,31)))
        self.assertEqual(frames[-1]['groups']['scrap']['missing'],[])
        self.assertIn('source_frame',frames[-1]);self.assertLess(frames[0]['time_seconds'],frames[-1]['time_seconds'])

    def test_cosmos_adapter_is_real_and_unavailable_is_not_a_fabricated_answer(self):
        with patch.object(server.cosmos_client,'call',return_value={'model':'test-cosmos','answer':'Visible evidence','advisory_only':True}) as call:
            response=self.client.post('/api/reason',files={'file':('image.jpg',self.image,'image/jpeg')})
            self.assertEqual(response.status_code,200);self.assertEqual(call.call_args.args[0],'/reason')
            self.assertIn('images',call.call_args.args[1])
        with patch.object(server.cosmos_client,'call',side_effect=RuntimeError('SSH unavailable')):
            response=self.client.post('/api/reason',files={'file':('image.jpg',self.image,'image/jpeg')})
        self.assertEqual(response.status_code,503);self.assertNotIn('answer',response.json())


if __name__=='__main__':unittest.main()
