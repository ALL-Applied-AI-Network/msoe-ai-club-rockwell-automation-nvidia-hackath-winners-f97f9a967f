"""Evaluate emitted decisions, including missing predictions and uncertain holds."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path


def evaluate(predictions, labels):
    counts = Counter()
    by_group = defaultdict(Counter)
    for label in sorted(labels.glob('*.json')):
        truth = json.loads(label.read_text())
        truth = truth.get('positions', truth)
        prediction_path = predictions / label.name
        predicted = json.loads(prediction_path.read_text()) if prediction_path.exists() else {}
        for position, info in truth.items():
            if isinstance(info, dict) and not info.get('in_frame', True):
                continue
            actual = info['class'] if isinstance(info, dict) else info
            decision = predicted.get(position, {}).get('class', 'missing_prediction')
            defect = actual in ('missing_cap', 'crooked_cap', 'damaged_vial')
            values = dict(positions=1, correct=int(actual == decision), defects=int(defect),
                          defects_accepted=int(defect and decision == 'pass'), good=int(actual == 'pass'),
                          good_held=int(actual == 'pass' and decision != 'pass'),
                          uncertain=int(decision == 'uncertain'), missing=int(decision == 'missing_prediction'))
            counts.update(values)
            by_group[f'tray_{position[0]}'].update(values)
            by_group[actual].update(values)
            if isinstance(info, dict) and info.get('damage'):
                by_group[f"{position[0]}/{info['damage']}"].update(values)
    if not counts['positions']:
        raise ValueError('no labeled positions')
    return dict(all=dict(counts), groups={k:dict(v) for k,v in sorted(by_group.items())})


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--predictions', type=Path, required=True)
    ap.add_argument('--labels', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    report = evaluate(a.predictions, a.labels)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))
