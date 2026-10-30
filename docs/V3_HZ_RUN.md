# v3_HZ training and inspection run

## Inputs and rules

Harsh supplied `Harsh_Vial_Inspection_labeled_Data_for_real_image/labels-2026-10-30T172529Z.json`.
It was validated by the participant package's importer against the original immutable review
template and source image hashes. The result is `outputs/real-confirmed-HZ/`: two images,
120 confirmed positions, with all five classes represented. There are only four reviewed
damaged-vial positions in total. These examples support development diagnostics, not a
reliable claim about performance on unseen production scenes.

The [Dataset 1 instructions](https://msoe-maic.com/library/?article=Rockwell-NVIDIA-Fall-2026-Vial-Inspection)
ask participants to generate training images with the twin and avoid training on the real
sample images. Real images and labels are therefore evaluation/reference inputs only.
No real image pixels or crops enter gradient training. Supplied fixture geometry is used
to match the synthetic camera and crop layout.

## Evaluate, diagnose, augment, retrain, gate

The existing v2 ResNet18 provides the five-class baseline. The principal diagnosed mismatches
are camera resolution/pose, a five-position scrap layout versus the supplied 30-position
scrap fixture, differing crop context, and G-tray damage absent from the original training
data. A fresh synthetic set addresses these mismatches before changing model architecture.

`classifier/prepare_public_view.py` fits camera intrinsics/extrinsics to the supplied 60
aperture coordinates and the twin's physical fixture coordinates. Geometry RMS is 1.41 px.
The fit is approximate; it is not a physical camera calibration certificate. The twin now
renders 1920x1080 images and labels G01–G30 and S01–S30. The inherited merged robot/carrier
mesh occludes fixture columns from this view and is omitted in the public-view training
variant; fixture/vial geometry remains. This is a known background/domain difference.

`review_bbox_v1` is the shared crop contract: calibrated contextual rectangles resized to
128x128 with identical OpenCV interpolation in synthetic preparation and serving. This
replaces applying the old 640x480 fixed layout to the public view. Runtime alignment checks
both supplied reference profiles and rejects unfamiliar views and image sizes.

Frozen synthetic splits use distinct RNG seeds:

| Split | Seed | Frames | Target positions |
|---|---:|---:|---:|
| Training | 30102641 | 96 | 5,760 |
| Validation | 30102642 | 24 | 1,440 |
| Test | 30102643 | 24 | 1,440 |

Training/validation/test frames are never shared. Real review rows are marked `real_eval`
and are excluded by the trainer, which loads only `train` and `val` rows. Checkpoint selection
uses synthetic validation macro-F1. Final evaluation reports synthetic test and real
development-reference results separately. The real references are already known development
material, so do not describe them as an untouched independent test.

New model version: **v3_HZ**, initialized from the original v2 checkpoint and saved separately
at `model/v3_HZ/model.pt`. All final run arguments, hashes, results and promotion decisions
will be recorded with the checkpoint. V2 is preserved.

Development gate, fixed before v3_HZ fitting: the sum of visible defects emitted as `pass`
or `no_vial` must decrease against v2 on the supplied reviewed references; strict accuracy
must improve and good-vial holds must not increase. Separately compare both models on the
new synthetic test crops. Real-reference labels remain Harsh's reviewed judgments; they
have not been independently adjudicated. Passing this small development gate does not
establish production readiness. A supervised real-line test on additional scenes is still
required for that claim.

The v2 baseline using the public camera's contextual crop contract achieved 32/120 strict
correct decisions (26.67%), accepted 3/24 visible defects as pass, and held 47/49 labeled-good
vials. This is a domain/crop-transfer baseline; it is not comparable to v2's 99.76% score on
its original synthetic validation crops. The new model must earn improvements on the actual
public view rather than reusing the old synthetic score.

## DGX jobs

- Successful 60-position preview: `79385a5e-22bd-4572-b45d-bf8c08bad40e`.
- Existing v2 real-reference baseline: `f8230d3e-d9ad-4cb5-9be3-eb46d8d27dd7`.
- Synthetic training render: `7c40c0fb-bf5f-4cee-8a94-ba29751b462c`.
- Synthetic validation and test render: `6a58b65e-ee25-4bf3-85e9-66231ed8e928`.
- Isaac Sim image: `nvcr.io/nvidia/isaac-sim:5.1.0`, GPU enabled, 48 GB / 8 CPUs per render job.
- Two initial preview attempts failed during new scene cloning/JSON serialization. Both
  defects were corrected before the successful preview and dataset jobs. Failed attempts
  are not training inputs.

## Cosmos and TAO

The [event resources](https://msoe-maic.com/library/?article=Rockwell-NVIDIA-Fall-2026-Resources)
list local DGX inference as an option. The initial Reason2 checkpoint required gated access;
the application instead runs the public NVIDIA `Cosmos-Reason1-7B` model on the DGX's GB10.
Its revision is pinned to `375e24000b24baed78f4618d3dd779e47cd96323`.
The [official model repository](https://huggingface.co/nvidia/Cosmos-Reason1-7B) supplies the
weights and architecture. This is actual model inference, not a text-only substitute.

`inspection_app/cosmos_service.py` serves the model inside the team's `hz-cosmos` container.
The host listener binds only to 127.0.0.1. `inspection_app/cosmos_client.py` performs bounded
SSH requests to it; the website does not require a public GPU endpoint or cloud API key.
The CSI network/VPN and the user's SSH key must work. Cosmos output is advisory and is
kept separate from classifier decisions and robot placement-count signals.

The NVIDIA TAO Skill Bank was cloned at release `7.3.0` in
`outputs/research/tao-skill-bank` and the Visual ChangeNet/DEFT guidance inspected.
Visual ChangeNet is an alternative binary paired-image architecture; the shipped model
retains the five-class ResNet18 pipeline. No TAO/Visual ChangeNet training or formal DEFT
orchestrator completion is claimed. This run follows the evaluate/diagnose/augment/retrain/
gate structure described by the event, using our existing classifier.

## Website

The branched chat owns `docs/WEB_APP_HZ.md`, the launch script, and the local camera/video
interface. The local website serves model inference, calibrated overlays, manual
placement-count/completion inputs, temporal observations and a separate Cosmos advisory.
Uncertain observations remain review cases; trailing empty slots remain pending until
placement/completion evidence makes them expected. This software does not command the robot.
