# H200 migration notes

Current package source: `../Mixture-of-Tokenizer/version4`

Main training config:

```bash
configs/titok_llamagen_mix_ae_unfreeze_encoder_gan_router_f2d_e2e_dynamic_freeze1d_patchdinoD_gan012_dino050_ema0999_from94000_h200_8gpu_10epoch.yaml
```

Launch command for 8 H200 GPUs:

```bash
cd MoT
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  torchrun --standalone --nproc_per_node=8 train_titok_llamagen_decoder_adapt_router_f2d_e2e_dynamic.py \
  --config configs/titok_llamagen_mix_ae_unfreeze_encoder_gan_router_f2d_e2e_dynamic_freeze1d_patchdinoD_gan012_dino050_ema0999_from94000_h200_8gpu_10epoch.yaml
```

The config assumes 8 GPUs, `batch_size=32`, `accum_steps=1`, global batch 256. With 1,281,167 train images, 10 epochs are about 50,050 optimizer steps, so from step 94,000 the target `max_steps` is 144,050.

Expected local checkpoint layout:

```text
weights/step_00066000.pt
weights/step_00094000.pt
```

External dependencies and weights must also exist on the H200 machine, or the paths in the config must be changed.

Public pretrained weights download:

```bash
cd MoT
python download_public_weights.py --project-root .. --torch-cache-root ../.cache/torch --hf-endpoint https://hf-mirror.com
```

This downloads only public pretrained dependencies:

```text
1d-tokenizer/tokenizer_titok_l32.bin
LlamaGen/pretrained_models/vq_ds16_c2i.pt
.cache/torch/hub/checkpoints/dinov2_vits14_pretrain.pth
LlamaGen/tokenizer/tokenizer_image/cache/vgg.pth
```

It does not download our trained checkpoints. Those still need to be copied manually:

```text
weights/step_00066000.pt
weights/step_00094000.pt
```

Download script test mode:

```bash
python download_public_weights.py --test --only lpips_vgg
```

`--test` downloads into `/tmp`, validates the file, and deletes the temporary test directory automatically. For large weights, use the same flag with `--only dinov2_vits14` or `--only llamagen_vq_ds16_c2i`; network quality may dominate runtime.
