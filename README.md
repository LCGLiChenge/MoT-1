# MoT: End-to-End Top-p Dynamic Token Allocation

This repository contains only the current MoT reconstruction pipeline: an end-to-end Router selects a variable number of 2D refinement grids with a differentiable top-p mask, while the tokenizer and decoder are jointly optimized.

Older fixed-ratio, E83 search, and E117/AR experiments are intentionally excluded from this branch.

## Verified configuration

The current checkpoint was trained with top-p temperature `2.0` and `p=0.91`. Hard inference is calibrated separately with temperature `1.0` and `p=0.994897` to obtain an average budget of approximately 96 refinement tokens.

On ImageNet validation (50,000 images, 256 x 256, EMA weights, Inception-2048 FID without input normalization), the verified checkpoint gives:

| Average 2D tokens | rFID | PSNR | LPIPS |
|---:|---:|---:|---:|
| 96.124 | 1.198612 | 19.5080 | 0.222875 |

## Repository layout

```text
configs/
  top_p_train.yaml          # current 4-epoch continuation recipe
  top_p_eval_5000.yaml      # quick checkpoint screening
  top_p_eval_50000.yaml     # final ImageNet validation protocol
models/
  titok_to_llamagen.py      # shared TiTok/LlamaGen representation module
train_titok_llamagen_recon.py
train_titok_llamagen_decoder_adapt_global_gain_texture.py
train_titok_llamagen_decoder_adapt_router_f2d_e2e_dynamic.py
eval_titok_llamagen_mix_metrics_router_f2d_e2e_dynamic.py
download_public_weights.py
environment.yml
```

`train_titok_llamagen_decoder_adapt_global_gain_texture.py` is retained because the evaluator imports shared spatial-score and feature-conversion utilities from it. The active allocation mode is `router_e2e_top_p`.

## Environment

```bash
conda env create -n MoT -f environment.yml
conda activate MoT
```

The default configs assume the following sibling layout:

```text
MoT-1/
1d-tokenizer/
  modeling/titok.py
  configs/infer/TiTok/titok_l32.yaml
  tokenizer_titok_l32.bin
LlamaGen/
  tokenizer/tokenizer_image/vq_model.py
  pretrained_models/vq_ds16_c2i.pt
ImageNet/
  train/
  validation/
.cache/
  torch/hub/facebookresearch_dinov2_main/
  open_clip/
```

Public weights and feature backbones can be prepared with:

```bash
python download_public_weights.py \
  --project-root .. \
  --torch-cache-root ../.cache/torch
```

## Checkpoint

Training and evaluation checkpoints are not stored in Git. Place an exact-resume checkpoint at:

```text
weights/top_p_resume.pt
```

For evaluation, place the selected checkpoint at:

```text
weights/top_p_checkpoint.pt
```

The continuation config restores the raw model, optimizer, discriminator, discriminator optimizer, EMA, and Router state. Override `resume`, data paths, or output paths in the YAML when using another layout.

## Train

The current recipe uses 8 GPUs, batch size 20 per GPU, full GAN loss from the first resumed step, top-p temperature 2, and one checkpoint per epoch:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --standalone --nproc_per_node=8 \
  train_titok_llamagen_decoder_adapt_router_f2d_e2e_dynamic.py \
  --config configs/top_p_train.yaml
```

## Evaluate

Screen a checkpoint on 5,000 validation images:

```bash
CUDA_VISIBLE_DEVICES=0 python \
  eval_titok_llamagen_mix_metrics_router_f2d_e2e_dynamic.py \
  --config configs/top_p_eval_5000.yaml
```

Run the final 50,000-image evaluation:

```bash
CUDA_VISIBLE_DEVICES=0 python \
  eval_titok_llamagen_mix_metrics_router_f2d_e2e_dynamic.py \
  --config configs/top_p_eval_50000.yaml
```

Both evaluation configs use EMA weights and the calibrated hard-inference setting (`T=1`, `p=0.994897`).
