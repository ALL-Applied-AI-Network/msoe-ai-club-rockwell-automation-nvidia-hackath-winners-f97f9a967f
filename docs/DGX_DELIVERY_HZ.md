# DGX delivery · v3_HZ

Status: **model-first transfer only**, per the user's latest instruction. Wait for successful
training, then deliver only the newly created checkpoint as `v3_HZ.pt` in the destination.
All dataset, crop, website/source, log and bulk-package transfers are deferred until that is done.
No final DGX artifact transfer has started.

Destination: `C:\Users\zodeh\OneDrive\Desktop\NVIDIA_HACKATHON\Files from DGX Spark`.

The delivery will contain:

- Synthetic train/validation/test frames and per-frame JSON labels, plus the successful preview.
- All generated crops with `train`, `val`, `test`, and `real_eval` kept separate.
- The completed DGX training artifacts and locally published v3_HZ checkpoint/report.
- Original v2 baseline where needed to run the application.
- The finalized camera/video website, classifier, digital twin, references, launch script, and run documentation.
- Original real-review export and its validated portable review package.
- Raw job logs, source mappings, SHA-256 manifests, and download verification results.

Source data will remain on the DGX. The public Cosmos foundation weights are retained in
the DGX service cache; this delivery records their model ID/revision and service dependency.
The newly trained inspection checkpoint is included.

Prepared tooling lives in `outputs/delivery_HZ/`. Its archive verification and traversal
boundary self-check passed. Final packaging requires exact render/crop counts and a trained
checkpoint. Archive SHA-256 is checked before extraction, every manifest-listed file is
checked after extraction, and conflicting existing destination files are never overwritten.

Final job ID and model readiness are read from the parent-owned handoff files:
`docs/validation/v3_HZ-training-job.json` and `model/v3_HZ/report.json`.
