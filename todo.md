# v3_HZ real-camera training and inspection application — October 30, 2026

Parallel ownership: main chat owns model/data/geometry/DGX/Cosmos. Branched chat
`01a18dfb-7f1e-7d73-b766-743ea1a743d0` owns website, camera/video UI, web API/sequence tests,
launcher and `docs/WEB_APP_HZ.md`. Both share this directory and preserve the current branch.
Delivery chat `01a18e13-ddbe-7c93-b79c-2bb60e90c279` owns the verified transfer to
`../Files from DGX Spark` and `docs/DGX_DELIVERY_HZ.md`. Main chat now focuses on training/readiness.

- [x] Validate Harsh's review export: 120 confirmed labels, two source images, hashes/template verified.
- [x] Read NVIDIA DEFT/TAO and Cosmos guidance; clone TAO Skill Bank 7.3.0; choose matched synthetic ResNet18 and GPU Cosmos Reason.
- [x] Prepare calibrated 60-position crop contract and immutable synthetic train/val/test seeds; real examples remain evaluation-only.
- [x] Establish existing v2 baseline on reviewed real data on DGX (32/120 strict correct).
- [ ] Train and save the next model as v3_HZ on DGX Spark, preserving v2; compare on held-out image groups where possible.
- [ ] Evaluate, diagnose errors, calibrate on development data, and record promotion criteria and limitations.
- [ ] Build a working image-upload/camera capture inspection page with the trained model backend.
- [ ] Build a camera/video inspection page with overlays and placement/completion-aware fill-pattern monitoring.
- [x] Run actual Cosmos Reason1-7B on DGX GB10; successful generated response saved at outputs/v3_HZ/cosmos-smoke.json.
- [ ] Finish website integration of Cosmos readiness/errors and separate advisory results (branched chat).
- [ ] Verify backend, frontend, model loading, image/video flows, and critical fill-pattern/annotation boundaries.
- [ ] Write launch commands, model/data provenance, measured results, and any required user actions.
- [ ] After generation/training succeeds, transfer completed DGX datasets, crops, models, reports and run sources to `../Files from DGX Spark` and verify SHA-256 manifests.

## Prior work and outstanding station validation

- [x] Preserve v2 baseline checkpoint and prior stashed work.
- [x] Reject incompatible image resolution before writing predictions.
- [x] Count emitted uncertain predictions as incorrect in classification accuracy.
- [x] Add checkpoint-bound, optional pass acceptance policy; uncertainty means review, not a detected defect.
- [x] Add reproducible fine-tuning, protect existing outputs, include final partial training batches.
- [x] Complete DGX calibration and compare audit frames without selecting on audit results.
- [x] Complete DGX fine-tuning candidate; reject promotion because the new-seed test does not improve.
- [x] Evaluate frozen candidates on 40 new-seed synthetic frames; report missed defects AND good-vial holds.
- [ ] Address G-tray damage observability and missing training coverage; current stress test confirms this gap.
- [x] Inspect supplied labeler: two real references, 1920x1080 geometry, G01–G30/S01–S30; no reviewed labels.
- [x] Prepare both supplied images as `outputs/real-review-20261030/index.html` (120 positions).
- [ ] User reviews visible positions and exports JSON; ambiguous positions stay unresolved.
- [ ] Import confirmed review without treating these development references as an independent benchmark.
- [ ] Integrate supplied 60-position geometry with a matching synthetic camera/tray; the current 640x480 twin differs.
- [ ] Arrange additional real camera captures/CSI validation if the event provides access; user reports no separate dataset supplied.
- [ ] Integrate placement count/order completion; distinguish trailing empty slots from missed placements.
- [ ] Measure real per-class recall, defect escapes, false holds and end-to-end latency on held-out real data.
- [ ] Demonstrate the complete inspection loop on CSI's line with an operator.

The existing cap-seal model covers G01–G30 and S01–S05. It is not the final 60-position solution.
Real examples already viewed for geometry are development references, not an untouched test set.
Do not train on the real evaluation release or invent labels for ambiguous cracks.
