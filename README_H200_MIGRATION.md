# H200 migration notes

Current package source: `../Mixture-of-Tokenizer/version4`

Main resume config:

```bash
configs/titok_llamagen_mix_ae_unfreeze_encoder_gan_router_f2d_e2e_dynamic_freeze1d_patchdinoD_gan012_dino050_ema0999_from127360_h200_8gpu_resume.yaml
```

Launch command for 8 H200 GPUs:

```bash
cd MoT
wandb login
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  torchrun --standalone --nproc_per_node=8 train_titok_llamagen_decoder_adapt_router_f2d_e2e_dynamic.py \
  --config configs/titok_llamagen_mix_ae_unfreeze_encoder_gan_router_f2d_e2e_dynamic_freeze1d_patchdinoD_gan012_dino050_ema0999_from127360_h200_8gpu_resume.yaml
```

The config assumes 8 GPUs, `batch_size=32`, `accum_steps=1`, global batch 256, and resumes from `weights/epoch_0005_step_00127360.pt`. `max_steps=144050` continues the original from-94000 H200 schedule; raise it if more epochs are needed.

Expected local checkpoint layout:

```text
weights/step_00066000.pt
weights/epoch_0005_step_00127360.pt
```

Public pretrained weights download:

```bash
cd MoT
python download_public_weights.py --project-root .. --torch-cache-root ../.cache/torch --hf-endpoint https://hf-mirror.com
```

Trained checkpoint download:

```bash
HF_HUB_DISABLE_XET=1 hf download Chloeeeeeeee123/MoT-1 weights/step_00066000.pt --repo-type model --local-dir .
HF_HUB_DISABLE_XET=1 hf download sophiaa/MoT-1-checkpoints epoch_0005_step_00127360.pt --repo-type model --local-dir weights
```

Wandb is enabled in the resume yaml. If the H200 machine cannot access wandb, either set `WANDB_MODE=offline` before launch or pass `--no-wandb`.

Download script test mode:

```bash
python download_public_weights.py --test --only lpips_vgg
```

`--test` downloads into `/tmp`, validates the file, and deletes the temporary test directory automatically. For large weights, use the same flag with `--only dinov2_vits14` or `--only llamagen_vq_ds16_c2i`; network quality may dominate runtime.
