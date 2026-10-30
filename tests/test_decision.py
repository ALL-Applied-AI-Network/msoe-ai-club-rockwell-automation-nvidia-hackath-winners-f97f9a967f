import sys
import unittest
import subprocess
import tempfile
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "classifier"))
from decision import decide


class DecisionTests(unittest.TestCase):
    def test_uncertain_pass_is_held_without_inventing_a_defect(self):
        self.assertEqual(decide(["pass", "damaged_vial"], [.8, .2], pass_min_conf=.95),
                         ("uncertain", "pass", .8))

    def test_defect_and_boundary(self):
        self.assertEqual(decide(["pass", "damaged_vial"], [.1, .9], pass_min_conf=.95)[0], "damaged_vial")
        self.assertEqual(decide(["pass", "damaged_vial"], [.95, .05], pass_min_conf=.95)[0], "pass")

    def test_invalid_inputs_fail(self):
        for p in ([float("nan"), .1], [.2, .1], [-.1, 1.1]):
            with self.assertRaises(ValueError):
                decide(["pass", "damaged_vial"], p)
        with self.assertRaises(ValueError):
            decide(["pass"], [1.], pass_min_conf=2)

    def test_wrong_resolution_produces_no_predictions(self):
        import cv2
        import numpy as np
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            image = directory / 'wrong.png'
            cv2.imwrite(str(image), np.zeros((1080, 1920, 3), dtype=np.uint8))
            out = directory / 'predictions'
            result = subprocess.run([sys.executable, str(root/'classifier/predict.py'), 'run',
                                     '--check-layout', '--layout', str(root/'model/layout.json'),
                                     '--out', str(out), str(image)], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('requires readable 640x480', result.stderr)
            self.assertFalse(out.exists())

    def test_policy_cannot_be_used_with_a_different_checkpoint(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            model, policy = directory/'model.pt', directory/'policy.json'
            model.write_bytes(b'different checkpoint')
            policy.write_text(json.dumps({'model_sha256':'wrong', 'pass_min_conf':.9}))
            result = subprocess.run([sys.executable, str(root/'classifier/predict.py'), 'run',
                                     '--model', str(model), '--policy', str(policy), 'unused.png'],
                                    capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('different model checkpoint', result.stderr)


if __name__ == "__main__":
    unittest.main()
