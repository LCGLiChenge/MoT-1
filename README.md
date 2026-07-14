# MoT H200 Training

## 1. Clone

```bash
git clone https://github.com/LCGLiChenge/MoT-1.git /home/heyefei/lichenge/MoT
cd /home/heyefei/lichenge/MoT
```

## 2. Download public weights

```bash
python download_public_weights.py \
  --project-root /home/heyefei/lichenge \
  --torch-cache-root /home/heyefei/.cache/torch \
  --hf-endpoint https://hf-mirror.com
```

## 3. Pull MoT checkpoints

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

## 4. Train on 8 H200 GPUs

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --standalone --nproc_per_node=8 train_titok_llamagen_decoder_adapt_router_f2d_e2e_dynamic.py \
  --config configs/titok_llamagen_mix_ae_unfreeze_encoder_gan_router_f2d_e2e_dynamic_freeze1d_patchdinoD_gan012_dino050_ema0999_from94000_h200_8gpu_10epoch.yaml
```
