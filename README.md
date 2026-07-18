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

The resume config expects the 66k adapter init and the epoch5 resume checkpoint here:

```text
weights/step_00066000.pt
weights/epoch_0005_step_00127360.pt
```

Download them with:

```bash
HF_HUB_DISABLE_XET=1 hf download Chloeeeeeeee123/MoT-1 \
  weights/step_00066000.pt \
  --repo-type model \
  --local-dir .

HF_HUB_DISABLE_XET=1 hf download sophiaa/MoT-1-checkpoints \
  epoch_0005_step_00127360.pt \
  --repo-type model \
  --local-dir weights
```

## 5. Smoke test

ImageNet train should be available at `../ImageNet/train` relative to `MoT-1`;
use a symlink if needed. This command runs one training step on one GPU and
checks that the downloaded weights, imports, dataloader, resume path, and save
path all work.

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0 \
torchrun --standalone --nproc_per_node=1 train_titok_llamagen_decoder_adapt_router_f2d_e2e_dynamic.py \
  --config configs/titok_llamagen_mix_ae_unfreeze_encoder_gan_router_f2d_e2e_dynamic_freeze1d_patchdinoD_gan012_dino050_ema0999_from127360_h200_8gpu_resume.yaml \
  --batch-size 1 \
  --limit-samples 8 \
  --num-workers 0 \
  --max-steps 127361 \
  --log-every 1 \
  --sample-every 0 \
  --sample-images 0 \
  --save-every 0 \
  --no-latest-every-epoch \
  --save-epoch-every 0 \
  --no-wandb \
  --output-dir results/smoke_mot1
```

## 6. Train on 8 H200 GPUs

ImageNet train should be available at `../ImageNet/train` relative to `MoT-1`;
use a symlink if needed. The config enables wandb and names the run
`mot_h200_epoch5_resume_from127360`. Do not put the wandb API key into yaml or git.

Use your wandb key through an environment variable so the remote server logs to
your account:

```bash
read -rsp "WANDB_API_KEY: " WANDB_API_KEY; echo
export WANDB_API_KEY

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --standalone --nproc_per_node=8 train_titok_llamagen_decoder_adapt_router_f2d_e2e_dynamic.py \
  --config configs/titok_llamagen_mix_ae_unfreeze_encoder_gan_router_f2d_e2e_dynamic_freeze1d_patchdinoD_gan012_dino050_ema0999_from127360_h200_8gpu_resume.yaml

unset WANDB_API_KEY
wandb logout
```

If wandb should write to a specific team/user entity, set `wandb_entity` in the
yaml or pass `--wandb-entity ENTITY`. If the server cannot access wandb, set
`WANDB_MODE=offline` before launch or pass `--no-wandb`.

This resumes model, optimizer, discriminator, and EMA from `weights/epoch_0005_step_00127360.pt`. It updates `latest.pt` every epoch and saves an extra epoch checkpoint every 5 epochs. To train beyond the original schedule, raise `max_steps` in the yaml.
