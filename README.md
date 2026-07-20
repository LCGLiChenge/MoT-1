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

If any public weight file is missing, run the download script:

```bash
python download_public_weights.py \
  --project-root .. \
  --torch-cache-root ../.cache/torch \
  --hf-endpoint https://hf-mirror.com
```

This covers the TiTok checkpoint, LlamaGen VQ checkpoint, DINOv2 checkpoint, and
LPIPS VGG checkpoint. It does not download our trained MoT checkpoints.

## 4. Pull MoT checkpoints

The resume config expects the 66k adapter init and the epoch5 resume checkpoint here:

```text
weights/step_00066000.pt
weights/epoch_0005_step_00127360.pt
```

If either trained checkpoint is missing locally, pull both from Hugging Face:

```bash
HF_HUB_DISABLE_XET=1 hf download Chloeeeeeeee123/MoT-1 \
  weights/step_00066000.pt \
  weights/epoch_0005_step_00127360.pt \
  --repo-type model \
  --local-dir .
```

## 5. Configure wandb

The training config has wandb enabled:

```yaml
wandb: true
wandb_project: MoT
wandb_name: mot_h200_epoch5_resume_5epoch_unfreeze2dquant
```

Use the wandb API key provided by the project owner so runs from the remote
server appear in the owner's wandb account. On the training server, enter the key
without printing it to the terminal:

```bash
read -rsp "WANDB_API_KEY: " WANDB_API_KEY; echo
export WANDB_API_KEY
```

Do not write the API key into yaml, README, shell scripts, GitHub, or shared
logs. If the run should go to a specific team/user entity, set `wandb_entity` in
the yaml or pass `--wandb-entity ENTITY`.

When training starts, wandb should print a project URL and a run URL. Open that
run URL in a browser to watch curves and console logs. After training finishes,
clear the key from the shell:

```bash
unset WANDB_API_KEY
wandb logout
```

If the server cannot access wandb, use offline logging:

```bash
export WANDB_MODE=offline
```

Then sync later with `wandb sync path/to/wandb/offline-run-*`. To disable wandb
for smoke tests, pass `--no-wandb`.

## 6. Smoke test

ImageNet train should be available at `../ImageNet/train` relative to `MoT-1`;
use a symlink if needed. This command runs one training step on one GPU and
checks that the downloaded weights, imports, dataloader, resume path, and save
path all work. It disables wandb so the smoke test does not create a real run.

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

## 7. Train on 8 H200 GPUs

ImageNet train should be available at `../ImageNet/train` relative to `MoT-1`;
use a symlink if needed. Run the wandb setup in step 5 first, then launch:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --standalone --nproc_per_node=8 train_titok_llamagen_decoder_adapt_router_f2d_e2e_dynamic.py \
  --config configs/titok_llamagen_mix_ae_unfreeze_encoder_gan_router_f2d_e2e_dynamic_freeze1d_patchdinoD_gan012_dino050_ema0999_from127360_h200_8gpu_resume.yaml
```

The terminal should print a wandb run URL after resume loading. Keep that URL for
monitoring the run from another machine.

This resumes model, discriminator, and EMA from `weights/epoch_0005_step_00127360.pt` and resets the optimizer because the 2D quantizer/codebook is newly trainable, keeps the 1D adapter and Router frozen, enables low-lr 2D quantizer/codebook tuning, then trains about 5 more epochs to `max_steps=152385`. It updates `latest.pt` every epoch and saves one extra epoch checkpoint at the end because `save_epoch_every=5`.
