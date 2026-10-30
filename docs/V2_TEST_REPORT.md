# V2 checkpoint verification — October 30, 2026

Verdict: v2 materially improves synthetic inspection, but is not ready to claim validated real-line performance. No model, production code, thresholds, or stashed improvements were changed during this test.

## Verified DGX execution

- YAAD job: `2f8f29ea-a66e-402b-a3c0-8cb447f14224`; succeeded, exit 0.
- Host: `spark-8191`; GPU: NVIDIA GB10; torch 2.14.1+cu130.
- The comparison explicitly asserts CUDA availability and puts both networks, inputs, and normalization tensors on CUDA. All full-frame checks were rerun with `--device cuda`.
- The initial local comparison was stopped when DGX execution was requested. The numbers below come from the completed DGX job.
- Checkpoint SHA-256 matches the pulled repository model: `453be3f451fe5e6f801ffd6b20e7096aa8cc1724244db13711c9ce4566c456af`.

## Both models on the SAME v2 validation crops

10,920 crops from the existing v2 synthetic validation set. This reproduces validation, not an independent final test. Neither model was retrained or tuned. Both use argmax for the five-class metrics, with the existing 0.5 confidence rule separately checked.

| Metric | V1 | V2 |
|---|---:|---:|
| Five-class accuracy | 98.53% | 99.76% |
| Macro-F1 | 0.9184 | 0.9905 |
| Damaged-vial recall | 44.88% | 92.23% |
| Visible defects incorrectly emitted as pass (of 4,572) | 93 | 14 |
| Passing vials flagged as a visible defect (of 4,224) | 2 | 0 |
| Uncertain positions | 1 | 1 |

| Damage subtype | Samples | V1 recall | V2 recall |
|---|---:|---:|---:|
| star_fracture | 85 | 2.35% | 98.82% |
| chipped_rim | 66 | 100.00% | 100.00% |
| body_crack | 74 | 1.35% | 71.62% |
| broken_neck | 58 | 100.00% | 100.00% |

The v2 confusion matrix matches the rounded metrics shipped in `model/metrics.json`. Of the 14 visible defects emitted as pass, 11 were damaged vials and 3 were crooked caps. Ten further body cracks were classified as missing caps: their subtype is wrong, but they still raise a defect alarm.

## Complete-frame checks on DGX

| Input | Positions | Strict five-class accuracy | Defects emitted as pass |
|---|---:|---:|---:|
| legacy: older-generator synthetic frames | 840 | 95.83% | 19/391 |
| fresh: older-generator synthetic frames | 70 | 85.71% | 2/33 |

These older renders include the previous faint crack appearance. They expose remaining distribution sensitivity; the v2 validation score should not be assumed to apply to all rendering conditions.

## Remaining failures and limitations

- Real references: both public 640x480 images completed GPU inference (70 positions). Overlays show several apparently empty G apertures called cap defects. This is qualitative evidence of a transfer problem; no confirmed real labels exist here, so no real accuracy is claimed.
- The supplied 1920x1080 participant image was also tested with `--check-layout`: the pulled predictor warned about resolution, still wrote an overlay, and exited 0. It does not enforce camera/layout compatibility.
- The pulled predictor still computes its printed accuracy from pre-abstention argmax. This report recomputes emitted-decision accuracy independently from CSV, treating uncertainty as incorrect.
- Current geometry remains G01-G30 plus S01-S05, rather than the full official 60 positions.
- No live CSI/PLC test, fresh v2 rendering, threshold calibration, or independent final real benchmark was performed. The visible-crack generator assumes damage large enough for the camera to resolve.

## Saved evidence

- [Full GPU comparison and missed-crop identities](../outputs/v2-verification/dgx-results/comparison.json)
- [GPU device diagnostic](../outputs/v2-verification/dgx-results/gpu.txt)
- [Full-frame decision counts](../outputs/v2-verification/dgx-results/frame-checks.json)
- [Real-image overlays](../outputs/v2-verification/dgx-results/real/overlays/)
- [CUDA comparison script](../outputs/v2-verification/compare_dgx.py)

Earlier documentation, evaluation logic, and safety fixes remain preserved in the named Oct 30 stash. They have not been reapplied to the pulled branch.
