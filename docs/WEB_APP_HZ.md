# Local inspection website — HZ

## Status

Implemented in the shared checkout by the website chat. The parent chat owns training,
model release/evaluation, synthetic generation, and the actual DGX Cosmos service.

- Image upload, drag/drop, supplied examples, and camera snapshots.
- Live camera capture with sampled inference, 60-position overlays, and stable-change events.
- Uploaded video with up to 30 unique sampled frames and a selectable result timeline.
- Model selection defaults to **v2 · GitHub model.pt**; `v3_HZ · new model` is also
  selectable and stays unavailable until its checkpoint exists. The v2 domain warning remains visible.
- Good/scrap counts and a **good-order** completion signal; classifications and placement
  findings remain separate. Completion does not require the scrap fixture to be full.
- Position details, raw context crops, JSON/CSV exports, and Cosmos advisory assessment.
- Checkpoint/readiness detection: missing models produce an unavailable state; no simulated
  classifier results, generated confidence, or fallback Cosmos answers.

## Launch on Windows

From the repository root, run:

```powershell
.\Start-Inspection-HZ.ps1 -Install
```

Subsequent launches:

```powershell
.\Start-Inspection-HZ.ps1
```

The launcher prepares `outputs/hz-web-env`, waits for the local HTTP page to respond, then
opens [http://127.0.0.1:8000/](http://127.0.0.1:8000/). Stop with Ctrl+C. Use `-NoBrowser`
or `-Port 8001` when needed. It binds only to loopback.

The project environment inherits existing Torch packages to avoid copying the large GPU/CPU
runtime and pins a compatible FastAPI/Starlette pair. The pre-existing global environment
had FastAPI 0.119.0 with an incompatible Starlette version; launching through the project
environment avoids modifying global packages. Other inherited optional packages can have
unrelated dependency warnings; this app does not import `sse-starlette`.

Equivalent direct command after setup:

```powershell
outputs\hz-web-env\Scripts\python.exe -m uvicorn inspection_app.server:app --host 127.0.0.1 --port 8000
```

## Use

### Image inspection

1. Select the model. The existing trained `v2` is the default. `v3_HZ` becomes available when
   `model/v3_HZ/model.pt` is present. The v2 baseline is explicitly marked as a different camera/crop domain.
2. Upload a calibrated 1920×1080 image, choose a supplied example, or connect the camera and
   capture a snapshot. Other resolutions are refused; a different viewpoint needs calibration.
3. Optionally enter an authoritative placement count for each tray. Mark good-order completion
   only when that signal actually applies to this frame.
4. Run inspection, check the alignment rings, and inspect the position map/details.
5. Download JSON or CSV. JSON exports omit embedded image bytes to keep the result portable;
   CSV records every position, model/hash, class/confidence and placement action.

An empty position preceding a later occupied one is a **possible gap**. A count/completion
signal can establish a missing placement. Trailing empty slots remain pending. A visibly
passing vial in the scrap tray does not establish a routing error; that needs upstream data.

### Live monitor

Open [http://127.0.0.1:8000/live](http://127.0.0.1:8000/live), connect a camera, then start
monitoring. The browser requests actual 1920×1080 capture and reports permission, camera,
resolution and alignment failures. Camera permission is a user action.

Only one inspection request is active at a time. Sampling targets at most one request per
second and displays the measured sampling rate; raw video is not claimed to be classified
on every frame. Overlays expire when a new result has not arrived for four seconds.
Monitoring stops when the tab becomes hidden; reconnect to resume. Stop releases all tracks.

Stable change events need two consecutive matching observed classifications. They are visual
observations, not an authoritative robot placement counter. No PLC or robot command is sent.

### Recorded video

Use **Upload video** on the live page. Videos are limited to 200 MB and inspected at up to
30 unique frames, including the final frame when decoding permits. Placements between
samples can be missed. The sampling interval and each sample time are returned and shown.

Video-level placement counts and the good-order completion signal apply only to the final
successfully decoded sample. Earlier samples use observed occupancy. This avoids applying a
final completion signal retroactively to the whole fill sequence. Uploaded temporary files
are removed when the request finishes. Timeline frames include the raw source for unbiased
crop viewing/Cosmos, separately from the annotated display.

### Cosmos Reason

The site calls the parent-owned `inspection_app/cosmos_client.py` adapter. By default it
uses bounded SSH requests to `hz-cosmos` on `yt-winners7506@172.30.102.95`, whose GPU
service is loopback-only. No persistent HTTP tunnel or browser-side API credential is needed.
Connect CSI VPN and retain the authorized SSH key. `HZ_DGX_SSH` overrides the SSH target;
`COSMOS_URL` optionally points to an explicitly configured HTTP service.

Cosmos health is refreshed in a background thread. The page and classifier remain usable
when VPN, SSH or Cosmos is unavailable. The advisory panel is disabled until the service is
ready and reports real request failures without inventing an answer. Choosing a position
asks Cosmos about its raw context crop; otherwise it receives the current raw frame.
Cosmos never changes the classifier's classes, placement findings, or confidence scores.

The parent confirmed actual NVIDIA GB10 Cosmos inference in
`outputs/v3_HZ/cosmos-smoke.json`. Model/service setup belongs to the parent chat.

## API

- `GET /api/status`: local device, checkpoint availability, cached Cosmos readiness.
- `POST /api/inspect`: multipart `file`, `version`, optional `good`, `scrap`, `complete`, `session`.
  Use a new UUID for an independent live stream. Sessions expire after 15 minutes; model
  versions have separate temporal state. At most 64 active model/session pairs are retained.
- `POST /api/video`: multipart video plus model and optional final-placement signals.
- `POST /api/reason`: multipart image and optional question; server uses the DGX adapter.
- `GET /api/examples/tray_000001.jpg` and `tray_000122.jpg`: supplied real references.
- `GET /api/docs`: interactive API schema.

## Verification

Run web-specific checks through the project environment:

```powershell
outputs\hz-web-env\Scripts\python.exe -m unittest discover -s tests -p test_web_HZ.py -v
node --check inspection_app/static/app.js
```

Eleven backend/sequence tests pass. They cover both pages/assets/examples, cached status
without a DGX network call, invalid counts/images, actual wrong-resolution refusal,
model/session isolation, temporal debounce, pending/internal-gap/count semantics, good-order
completion without false scrap completion, unique sampled video and final-only signals,
and the real Cosmos adapter contract/error path. Backend-boundary tests mock model inference
where needed; they do not claim classification accuracy.

Browser QA verified actual image loading, the v2 diagnostic warning, a real 60-position v2
inspection, and alignment residuals through the interface. A four-frame reference-image
recording exists at `outputs/web-HZ-verification/station-sample.avi` solely for UI testing;
it is not a live-line recording. Final browser/export/Cosmos/video results are recorded below.

### Completed browser checks

- Real v2 inference through the image page returned all 60 positions, a labeled overlay,
  model hash/device, and automatic `view_001` alignment with zero residual on its reference.
  These are functional checks, not a claim that the v2 classifications are correct.
- The video page accepted the four-frame reference recording, showed four unique samples,
  and scrubbed to 3.0 seconds with `view_122` alignment and observed-change events.
- **Ask Cosmos** completed an actual DGX request through the browser: model
  `nvidia/Cosmos-Reason1-7B`, NVIDIA GB10 readiness, 48.26-second full-frame response.
  Its text remains advisory and can contain visual interpretation errors.
- JSON/CSV buttons were exercised. The in-app browser did not provide a download file path;
  disk-save behavior should also be checked in the user's normal browser. Actual serialization
  and download invocation are verified by `node tests/test_web_exports_HZ.cjs`, covering
  image JSON, CSV, formula escaping, and all-frame video export without embedded image bytes.
- Responsive layouts were visually inspected in the narrow in-app panel and the 1265-pixel
  desktop preview. Proof images: `outputs/web-HZ-verification/live-monitor.png` and
  `outputs/web-HZ-verification/video-timeline.png`.

The preview currently runs on loopback port 8000. It can be restarted with the launcher
after parent-owned model/engine updates. No git branch change, stash, commit, or push was made.

## Remaining checks requiring hardware/model completion

- Physical station camera capture and actual robot placements require the camera/line and
  operator permission. No live hardware quality or latency claim is made from sample files.
- `v3_HZ` inference/quality gates are completed by the parent when its checkpoint arrives.
- Camera geometry is verified only for the supplied view and small accepted translations;
  a moved camera needs recalibration.
- Local inference can use CPU when the website runs on this laptop. Training and Cosmos
  inference use DGX through the parent's jobs/service; the UI reports the actual device.

## GitHub checkpoint identity

The dropdown entry `v2 · GitHub model.pt` loads `model/model.pt`. Its bytes were verified
against `origin/dataset-1:model/model.pt` at commit `aaf0efc` (identical SHA-256
`453be3f451fe5e6f801ffd6b20e7096aa8cc1724244db13711c9ce4566c456af`).
This is the existing v2 checkpoint, so no duplicate model entry or copy is needed.
