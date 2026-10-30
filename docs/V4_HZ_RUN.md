# v4_HZ reviewed-real adaptation

`v4_HZ` starts from `v3_HZ` and mixes the existing synthetic training crops with 120 confirmed
real crops from `labels-2026-10-30T195155Z_v4_HZ.json`. The reviewed crops are repeated with
GPU augmentation while the synthetic crops act as rehearsal data.

The two real images are training inputs, so their reported fit is a regression check, not an
independent accuracy estimate. Production accuracy still requires new labeled camera frames that
were not used during training.

Training runs on one YAAD DGX Spark GB10 with CUDA/bfloat16, batch size 512, and checkpoint version
`v4_HZ`. The frozen synthetic test split is evaluated after fine-tuning to measure catastrophic forgetting.
