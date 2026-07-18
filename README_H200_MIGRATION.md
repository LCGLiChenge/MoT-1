# H200 migration notes

Current package source: `../Mixture-of-Tokenizer/version4`

Main resume config:

```bash
configs/titok_llamagen_mix_ae_unfreeze_encoder_gan_router_f2d_e2e_dynamic_freeze1d_patchdinoD_gan012_dino050_ema0999_from127360_h200_8gpu_resume.yaml
```

Launch command for 8 H200 GPUs:

```bash
cd MoT
read -rsp "WANDB_API_KEY: " WANDB_API_KEY; echo
export WANDB_API_KEY
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  torchrun --standalone --nproc_per_node=8 train_titok_llamagen_decoder_adapt_router_f2d_e2e_dynamic.py \
  --config configs/titok_llamagen_mix_ae_unfreeze_encoder_gan_router_f2d_e2e_dynamic_freeze1d_patchdinoD_gan012_dino050_ema0999_from127360_h200_8gpu_resume.yaml
unset WANDB_API_KEY
wandb logout
```

The config assumes 8 GPUs, `batch_size=32`, `accum_steps=1`, global batch 256, and resumes from `weights/epoch_0005_step_00127360.pt`. `max_steps=142375` continues for about 3 epochs from step 127360, updates `latest.pt` every epoch, and saves an extra epoch checkpoint at epoch 3.

Expected local checkpoint layout:

```text
weights/step_00066000.pt
weights/epoch_0005_step_00127360.pt
```

If any public pretrained weight is missing, run:

```bash
cd MoT
python download_public_weights.py --project-root .. --torch-cache-root ../.cache/torch --hf-endpoint https://hf-mirror.com
```

This covers the TiTok checkpoint, LlamaGen VQ checkpoint, DINOv2 checkpoint, and LPIPS VGG checkpoint.

If either trained checkpoint is missing locally, pull both from Hugging Face:

```bash
HF_HUB_DISABLE_XET=1 hf download Chloeeeeeeee123/MoT-1 \
  weights/step_00066000.pt \
  weights/epoch_0005_step_00127360.pt \
  --repo-type model \
  --local-dir .
```

Wandb is enabled in the resume yaml. Keep the API key out of yaml/git and pass it through `WANDB_API_KEY`. If a specific team/user is needed, set `wandb_entity` in the yaml or pass `--wandb-entity ENTITY`. If the H200 machine cannot access wandb, either set `WANDB_MODE=offline` before launch or pass `--no-wandb`.

Download script test mode:

```bash
python download_public_weights.py --test --only lpips_vgg
```

`--test` downloads into `/tmp`, validates the file, and deletes the temporary test directory automatically. For large weights, use the same flag with `--only dinov2_vits14` or `--only llamagen_vq_ds16_c2i`; network quality may dominate runtime.
