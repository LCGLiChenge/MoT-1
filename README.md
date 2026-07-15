# MoT H200 Training

## 1. Clone

```bash
git clone https://github.com/LCGLiChenge/MoT-1.git
cd MoT-1
```

## 2. Create environment

Create a fresh environment from the checked-in file. The `-n MoT1` flag overrides the
name stored in `environment.yml`, so the same file can be reused for other local
environment names.

```bash
conda env create -n MoT1 -f environment.yml
conda activate MoT1
```

## 3. Prepare external code and public weights

The training script imports TiTok and LlamaGen source code at runtime. Before
training, make sure the default layout contains both source trees and the public
weights:

```text
../1d-tokenizer/modeling/titok.py
../1d-tokenizer/tokenizer_titok_l32.bin
../LlamaGen/tokenizer/tokenizer_image/vq_model.py
../LlamaGen/pretrained_models/vq_ds16_c2i.pt
```

If the source trees live elsewhere, keep the downloaded weights in the paths
above and pass `--titok-root`, `--titok-config`, and `--llamagen-root` to the
training command.

Download the public weights:

```bash
python download_public_weights.py \
  --project-root .. \
  --torch-cache-root ../.cache/torch \
  --hf-endpoint https://hf-mirror.com
```

## 4. Pull MoT checkpoints

```bash
HF_HUB_ENABLE_HF_TRANSFER=1 hf download Chloeeeeeeee123/MoT-1 \
  weights/step_00066000.pt \
  weights/step_00094000.pt \
  --repo-type model \
  --local-dir .
```

If the network is unstable, retry with resumable HTTP:

```bash
HF_HUB_DISABLE_XET=1 hf download Chloeeeeeeee123/MoT-1 \
  weights/step_00066000.pt \
  weights/step_00094000.pt \
  --repo-type model \
  --local-dir .
```

## 5. Smoke test

ImageNet train should be available at `../ImageNet/train` relative to `MoT-1`;
use a symlink if needed. This command runs one training step on one GPU and
checks that the downloaded weights, imports, dataloader, resume path, and save
path all work.

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0 \
torchrun --standalone --nproc_per_node=1 train_titok_llamagen_decoder_adapt_router_f2d_e2e_dynamic.py \
  --config configs/titok_llamagen_mix_ae_unfreeze_encoder_gan_router_f2d_e2e_dynamic_freeze1d_patchdinoD_gan012_dino050_ema0999_from94000_h200_8gpu_10epoch.yaml \
  --batch-size 1 \
  --limit-samples 8 \
  --num-workers 0 \
  --max-steps 94001 \
  --log-every 1 \
  --sample-every 0 \
  --sample-images 0 \
  --save-every 0 \
  --no-latest-every-epoch \
  --save-epoch-every 0 \
  --output-dir results/smoke_mot1
```

## 6. Train on 8 H200 GPUs

ImageNet train should be available at `../ImageNet/train` relative to `MoT-1`;
use a symlink if needed.

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --standalone --nproc_per_node=8 train_titok_llamagen_decoder_adapt_router_f2d_e2e_dynamic.py \
  --config configs/titok_llamagen_mix_ae_unfreeze_encoder_gan_router_f2d_e2e_dynamic_freeze1d_patchdinoD_gan012_dino050_ema0999_from94000_h200_8gpu_10epoch.yaml
```
