import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'classifier'))
from evaluate_predictions import evaluate


class MetricsTests(unittest.TestCase):
    def test_missing_and_uncertain_are_not_correct_or_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            labels, preds = root/'labels', root/'preds'
            labels.mkdir(); preds.mkdir()
            (labels/'a.json').write_text(json.dumps({'G01':'pass','S01':'damaged_vial','S02':'damaged_vial'}))
            (preds/'a.json').write_text(json.dumps({'G01':{'class':'uncertain'},'S01':{'class':'pass'}}))
            result = evaluate(preds,labels)['all']
            self.assertEqual(result['positions'],3)
            self.assertEqual(result['correct'],0)
            self.assertEqual(result['defects_accepted'],1)
            self.assertEqual(result['good_held'],1)
            self.assertEqual(result['missing'],1)
