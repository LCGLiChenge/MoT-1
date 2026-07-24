# MoT-1 Current H200 Run

This repo is a minimal handoff package for the current MoT experiment only. It resumes the projected ConvNeXt discriminator branch from `epoch_0005_step_00127360.pt` and trains on 8 H200 GPUs.

## 1. Clone

```bash
git clone https://github.com/LCGLiChenge/MoT-1.git
cd MoT-1
```

## 2. Environment

```bash
conda env create -n MoT1 -f environment.yml
conda activate MoT1
```

## 3. External Code And Public Weights

The default layout is relative to this repo:

```text
../1d-tokenizer/modeling/titok.py
../1d-tokenizer/tokenizer_titok_l32.bin
../LlamaGen/tokenizer/tokenizer_image/vq_model.py
../LlamaGen/pretrained_models/vq_ds16_c2i.pt
../ImageNet/train
../ImageNet/validation
```

If public weights or cached backbones are missing, run:

```bash
python download_public_weights.py \
  --project-root .. \
  --torch-cache-root ../.cache/torch \
  --hf-endpoint https://hf-mirror.com
```

## 4. MoT Checkpoints

The H200 config expects these private checkpoints:

```text
weights/step_00066000.pt
weights/epoch_0005_step_00127360.pt
```

If they are not already local, download them from the private Hugging Face checkpoint repo into this repo root. Example:

```bash
HF_HUB_DISABLE_XET=1 hf download sophiaa/MoT-1-checkpoints \
  weights/step_00066000.pt \
  weights/epoch_0005_step_00127360.pt \
  --repo-type model \
  --local-dir .
```

If the checkpoint repo uses different filenames, keep the local paths above or update `adapter_init` and `resume` in `configs/h200_projectedconvnext_from_epoch5.yaml`.

## 5. Wandb

The train config has wandb enabled. Use the project-provided key on the training server:

```bash
read -rsp "WANDB_API_KEY: " WANDB_API_KEY; echo
export WANDB_API_KEY
```

Training should print a wandb run URL. To disable wandb for a smoke test, pass `--no-wandb`. After training, clear the key:

```bash
unset WANDB_API_KEY
wandb logout
```

## 6. Smoke Test

Use one GPU and a tiny subset. This checks imports, dataloader, checkpoint loading, D reset, and saving.

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0 \
torchrun --standalone --nproc_per_node=1 train_titok_llamagen_decoder_adapt_router_f2d_e2e_dynamic.py \
  --config configs/h200_projectedconvnext_from_epoch5.yaml \
  --batch-size 1 \
  --accum-steps 1 \
  --limit-samples 8 \
  --num-workers 0 \
  --max-steps 127361 \
  --log-every 1 \
  --sample-every 0 \
  --sample-images 0 \
  --save-every 0 \
  --no-latest-every-epoch \
  --save-epoch-every 0 \
  --no-save-step-checkpoints \
  --no-wandb \
  --output-dir results/smoke_projectedconvnext_from_epoch5
```

Delete only `results/smoke_projectedconvnext_from_epoch5` after the smoke test passes.

## 7. Train On 8 H200 GPUs

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --standalone --nproc_per_node=8 train_titok_llamagen_decoder_adapt_router_f2d_e2e_dynamic.py \
  --config configs/h200_projectedconvnext_from_epoch5.yaml
```

This uses `batch_size=32`, `accum_steps=1`, resets optimizer and discriminator for the projected ConvNeXt branch, keeps the 1D adapter frozen, effectively freezes Router with `lr_router=0`, trains the 2D tokenizer/decoder at low LR, uses EMA, updates `latest.pt` every epoch, and saves explicit step checkpoints listed in the yaml.

## 8. Eval

For the latest checkpoint:

```bash
CUDA_VISIBLE_DEVICES=0 python eval_titok_llamagen_mix_metrics_router_f2d_e2e_dynamic.py \
  --config configs/eval_projectedconvnext_50000.yaml
```

For a specific checkpoint:

```bash
CUDA_VISIBLE_DEVICES=0 python eval_titok_llamagen_mix_metrics_router_f2d_e2e_dynamic.py \
  --config configs/eval_projectedconvnext_50000.yaml \
  --ckpt results/projectedconvnext_from_epoch5_h200_8gpu/step_00145000.pt \
  --output-json results/projectedconvnext_from_epoch5_h200_8gpu/eval_step_00145000_50000.json
```
