import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'classifier'))
from release_HZ import gate


class ReleaseGateTests(unittest.TestCase):
    def test_accuracy_does_not_hide_defects_misclassified_as_empty(self):
        old=dict(accuracy=.3,defects_accepted=3,defects_as_empty=0,good_held=47)
        unsafe=dict(accuracy=.9,defects_accepted=0,defects_as_empty=4,good_held=1)
        self.assertFalse(gate(old,unsafe)['passed'])
        better=dict(accuracy=.8,defects_accepted=1,defects_as_empty=0,good_held=5)
        self.assertTrue(gate(old,better)['passed'])
        better['good_held']=48
        self.assertFalse(gate(old,better)['passed'])
