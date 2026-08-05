# MoT-1 H200 Fixed 0.375 Run

This repo is a minimal handoff package for the current MoT experiment. It trains the fixed-ratio 0.375 Router-selection version from the clean 66000 checkpoint on 8 H200 GPUs.

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
../.cache/torch/hub/facebookresearch_dinov2_main
../.cache/open_clip
```

If public weights or cached backbones are missing, run:

```bash
python download_public_weights.py \
  --project-root .. \
  --torch-cache-root ../.cache/torch \
  --hf-endpoint https://hf-mirror.com
```

## 4. Private Checkpoint

This run needs the clean 66000 checkpoint at:

```text
weights/step_00066000.pt
```

If it is not local, download it from the private Hugging Face checkpoint repo:

```bash
HF_HUB_DISABLE_XET=1 hf download Chloeeeeeeee123/MoT-1 \
  weights/step_00066000.pt \
  --repo-type model \
  --local-dir .
```

## 5. Wandb

The train config has wandb enabled. On the remote server, set the key provided by the project owner:

```bash
export WANDB_API_KEY=PASTE_KEY_HERE
```

To disable wandb for a smoke test, pass `--no-wandb`.

## 6. Smoke Test

Use a tiny run before launching the full job:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0 \
torchrun --standalone --nproc_per_node=1 train_titok_llamagen_decoder_adapt_router_f2d_e2e_dynamic.py \
  --config configs/h200_fixed0375_from66000_20epoch.yaml \
  --batch-size 1 \
  --accum-steps 1 \
  --limit-samples 8 \
  --num-workers 0 \
  --epochs 0.01 \
  --save-every 0 \
  --save-epoch-fraction-every 0 \
  --no-save-step-checkpoints \
  --no-latest-every-epoch \
  --save-epoch-every 0 \
  --sample-every 0 \
  --no-wandb \
  --log-every 1 \
  --output-dir results/smoke_h200_fixed0375_from66000
```

Delete only `results/smoke_h200_fixed0375_from66000` after the smoke test passes.

## 7. Train On 8 H200 GPUs

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --standalone --nproc_per_node=8 train_titok_llamagen_decoder_adapt_router_f2d_e2e_dynamic.py \
  --config configs/h200_fixed0375_from66000_20epoch.yaml
```

Main settings:

```text
resume: weights/step_00066000.pt
batch_size: 24 per GPU
accum_steps: 1
epochs: 20
ratio: fixed 0.375, exactly 96 grids per image
router_only_epochs: 0.5
full training starts after 0.5 epoch
gan_start_epoch: 1.5
d_warmup_epochs: 0.01
lambda_gan: 0.12
save: latest.pt every epoch, plus epoch checkpoints every 5 epochs
```

## 8. Eval

```bash
CUDA_VISIBLE_DEVICES=0 python eval_titok_llamagen_mix_metrics_router_f2d_e2e_dynamic.py \
  --config configs/eval_h200_fixed0375_50000.yaml
```

For a specific checkpoint:

```bash
CUDA_VISIBLE_DEVICES=0 python eval_titok_llamagen_mix_metrics_router_f2d_e2e_dynamic.py \
  --config configs/eval_h200_fixed0375_50000.yaml \
  --ckpt results/h200_fixed0375_from66000_20epoch_bs24/latest.pt \
  --output-json results/h200_fixed0375_from66000_20epoch_bs24/eval_latest_50000.json
```
