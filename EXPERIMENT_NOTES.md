# Current Experiment Notes

## H200 fixed 0.375 from 66000

Date: 2026-08-05

This handoff package is for the fixed-ratio 0.375 Router-selection run from the clean 66000 checkpoint.

Key setup:

```text
checkpoint: weights/step_00066000.pt
output_dir: results/h200_fixed0375_from66000_20epoch_bs24
8 H200 GPUs
batch_size: 24 per GPU
accum_steps: 1
epochs: 20
Router selects exactly 96 / 256 grids per image
Router-only: first 0.5 epoch
Full training without GAN: 0.5 to 1.5 epoch
GAN/D warmup starts after 1.5 epoch
D warmup: 0.01 epoch
lambda_gan: 0.12
lambda_mix: 2.0
lambda_dino_feat: 0.5
lambda_clip_feat: 0.5
lambda_disc_feature_matching: 0.5
EMA enabled
save_epoch_every: 10
```

The copied training script includes the 2026-08-04 memory fixes:

```text
- zero-weight base/native/mix-native paths avoid unnecessary grad graphs
- Router-only phase avoids DDP unused-parameter errors
- D update forwards real/fake separately instead of concatenating a doubled batch
```

## Smoke Test

A 4-GPU smoke test was run in this MoT directory on 2026-08-05. Because this local machine keeps DINO/CLIP caches under `/home/heyefei/.cache` instead of the repo-relative `../.cache`, the smoke command overrode `--dino-repo` and `--clip-cache-dir`. On H200, running `download_public_weights.py` will create the repo-relative cache expected by the YAML.

Smoke command shape:

```text
torchrun --standalone --nproc_per_node=4 train_titok_llamagen_decoder_adapt_router_f2d_e2e_dynamic.py
  --config configs/h200_fixed0375_from66000_20epoch.yaml
  --batch-size 1 --accum-steps 1 --limit-samples 8 --epochs 2.5
  --resume /home/.../version4/results/.../step_00066000.pt
  --adapter-init /home/.../version4/results/.../step_00066000.pt
  --dino-repo /home/heyefei/.cache/torch/hub/facebookresearch_dinov2_main
  --clip-cache-dir /home/heyefei/.cache/open_clip
  --output-dir /tmp/mot_h200_fixed0375_smoke
```

Result:

```text
passed
router_only_steps=1
gan_start_step=66004
d_warmup_steps=1
step 66001: phase=router, gan=0, d=0
step 66002: phase=joint, gan=0, d=0
step 66003: phase=joint, gan=0, d=0
step 66004: phase=joint D warmup, gan=0, d=1.007
step 66005: phase=joint, gan=0.027, g%=2.1, gr%=5.2, d=1.078
```

Smoke output `/tmp/mot_h200_fixed0375_smoke` was deleted after verification.
