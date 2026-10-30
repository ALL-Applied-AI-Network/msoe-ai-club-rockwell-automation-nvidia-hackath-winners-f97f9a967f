# Improvement experiments — October 30, 2026

Baseline: dataset-1 commit `aaf0efc`, v2 SHA-256
`453be3f451fe5e6f801ffd6b20e7096aa8cc1724244db13711c9ce4566c456af`.
No baseline weights were overwritten. Previous work remains in the existing git stashes.

## Acceptance rule

YAAD job `b8edd623-b98c-4fe9-82c5-0fac077dc17c` completed on spark-8191 / NVIDIA GB10,
torch 2.14.1+cu130. Model and input tensors explicitly use CUDA.

Half the existing validation frames, grouped by run/frame hash, select a pass-confidence
threshold under a predeclared 1% calibration good-vial hold budget. The other half is an
internal audit. Because v2 was originally selected on this validation set, neither half is
an independent final model test. No threshold was selected using audit errors.

Frozen threshold: `0.9265182018280029`. Lower-confidence argmax pass predictions become
`uncertain` with action `review`. They are not claimed as correctly classified defects.
The policy verifies the checkpoint SHA-256 before use.

| Check | Baseline accepted defects | Policy accepted defects | Baseline good held | Policy good held |
|---|---:|---:|---:|---:|
| Calibration, 2,309 defects / 2,122 good | 4 | 0 | 0 | 21 |
| Internal audit, 2,263 defects / 2,102 good | 10 | 5 | 0 | 26 |
| All existing validation, 4,572 defects / 4,224 good | 14 | 5 | 0 | 47 |
| Older-generator 24 frames, 391 defects / 291 good | 19 | 12 | 3 | 22 |
| Older-generator 2 frames, 33 defects / 23 good | 2 | 1 | 0 | 1 |

This reduces escapes by holding ambiguous decisions, at the cost of lower strict
classification accuracy. Distribution shift matters: on the older 24-frame set, the
policy holds 16 of 29 good S-tray vials. Therefore it is an optional experiment, not an
automatic production default or a guaranteed 1% real false-hold rate.

Raw results: `outputs/improvement-20261030/calibration-job/calibration/report.json`,
`probabilities.npz`, per-frame JSON/CSV/overlays, and the `*-policy-report.json` files.

## Model experiment and new-seed test

- Fine-tuning job: `33ede9fc-9dee-40f3-b1f2-1ae1ad50566f`.
- Configuration fixed before new-seed evaluation: initialize v2, 3 epochs, learning rate
  0.00001, seed 301026, damaged-vial loss multiplier 2. Checkpoint selection uses existing
  validation macro-F1. Training uses only the existing synthetic training split.
- Render job: `931bc2a1-95b6-4897-b2d0-7fa2aa60113a`, Isaac Sim 5.1.0, seed 30102691,
  40 frames, both trays, original v2 generator. This includes G-tray damage absent from
  the existing training distribution; report it separately as a stress test.
- Both jobs succeeded on spark-8191 / NVIDIA GB10. Training used 84,080 synthetic crops;
  all 40 rendered images passed size/nonblank/label checks (1,400 positions).
- Final comparison job `3504e649-4732-47a4-be09-ef305addec78` also succeeded on the GB10,
  using FP32 CUDA inference for the matched-checkpoint comparison.

### Final results and decision

| Existing validation (10,920 crops) | V2 | Fine-tuned candidate |
|---|---:|---:|
| Accuracy | 99.7619% | 99.7894% |
| Macro-F1 | 0.990529 | 0.991510 |
| Body-crack recall | 53/74 (71.62%) | 57/74 (77.03%) |
| Defects accepted | 14/4,572 | 11/4,572 |
| Good vials flagged as defects | 0/4,224 | 2/4,224 |

| New-seed test, 40 frames | Defects accepted / 708 | Good held / 500 | Strict correct / 1,400 |
|---|---:|---:|---:|
| V2 default | 4 | 9 | 1,190 |
| Fine-tuned default | 4 | 11 | 1,187 |
| V2 optional review policy | 0 | 59 | 1,140 |
| Fine-tuned optional review policy | 0 | 54 | 1,144 |

**Keep v2 as the default; do not promote the candidate.** Its small validation gain does
not generalize to the new-seed frames. The optional review policy reduces escapes in this
sample but substantially increases operator review. Zero observed escapes is not proof of
zero risk; only 40 independent scenes were tested and positions within a scene are correlated.

The 40-frame stress test contains 193 G-tray damage examples, a category absent from
training. Both models classify zero of these as `damaged_vial`; most are flagged as other
defects, but three chipped rims are accepted. This explains much of the lower overall
classification accuracy. For S-tray damage, both get 83/87 correct, including 16/20 body
cracks and 33/33 star fractures. Do not conceal G-tray results by reporting only S-tray recall.

On the older 24-frame set, accepted defects remain 19 for both models; good holds increase
from 3 to 4 with the candidate. The older two-frame set remains at two accepted defects.

The candidate is preserved at `outputs/improvement-20261030/candidate/model.pt` for
reproducibility, SHA-256 `0687284380c516d232028ee97983450692f4acf733136c3c2cb4e9446ed77ae6`.
The baseline `model/model.pt` remains unchanged. Complete fresh-frame predictions, metrics
and overlays are in `outputs/improvement-20261030/comparison-job/`. The exact comparison
script is `outputs/improvement-20261030/compare_candidate.py`. A compact evidence snapshot
is versionable at `docs/validation/improvement-20261030.json`.

## Software corrections

- Refuse unreadable/wrong-resolution input before writing a batch.
- Refuse duplicate image stems, empty selected layouts, and missing crops.
- Compute reported accuracy from emitted classes; uncertain is not correct.
- Save raw class and accept/review/flag action separately.
- Evaluate every labeled position, counting missing predictions explicitly.
- Fine-tune into a new folder; preserve existing models. Seed random generators, check
  run/frame split separation, train on final partial batches, and fail on missing weights.
- Six local regression tests pass: decision boundaries, invalid probabilities, resolution
  refusal, checkpoint-policy mismatch, and honest counting of missing/uncertain predictions.
  Python compilation and `git diff --check` also pass.

## Actions needed from the team

The user confirmed no separate dataset was supplied. The local package
`../vial-labeler-participants-v1` contains two public reference images, geometry for 60
positions at 1920x1080, and no reviewed labels. Its README mentions a separate release, but
we do not assume the team has it. All 11 package tests pass.

Prepared `outputs/real-review-20261030/index.html` with both images and 120 unresolved
positions. Open this page, visually check rings against the target holes, label only visible
conditions, confirm reviewed positions, and export the review JSON. Tell the agent the
downloaded file path; ambiguous positions should stay unresolved. This provides initial
development labels, not an independent final benchmark or synthetic training material.

The package's grid mapping is available; it still needs a matching synthetic camera/tray
and integration with our model. Additional real captures and a placement-count/completion
interface require access to the actual line. If organizers provide that access, collect
multiple lighting/fill/defect conditions with an operator and reserve a separate test set.

The current 35-position cap-seal model cannot yet certify the requested 30+30 fixtures.
Invisible/subpixel cracks cannot be recovered reliably by training alone. Real accuracy,
camera alignment and a supervised live-line demonstration remain unverified.
