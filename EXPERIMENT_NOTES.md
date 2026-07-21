# Experiment Notes

## 2026-07-19 - 3-GPU smoke test for 2D quantizer unfreeze

Context:
- Resume checkpoint: `results/titok_llamagen_mix_ae_unfreeze_encoder_gan_router_f2d_e2e_dynamic_freeze1d_patchdinoD_gan012_from129360_to137360_local4gpu_bs4_accum6/step_00132000.pt`.
- Probe setting: `--train-llamagen-quantizer`, `--lr-llamagen-encoder 2e-7`, `--lambda-gan 0.12`, `--reset-optimizer`.
- 3-GPU global batch adaptation for training command: previous 4 GPU x bs4 x accum6 = 96; use 3 GPU x bs4 x accum8 = 96.

Smoke result:
- Initial 3-GPU smoke reproduced DDP unused-parameter failure after step 132001 when the LlamaGen 2D quantizer was trainable.
- Enabling DDP `find_unused_parameters=True` avoided the first failure but caused Router `ratio_head` ready-twice errors, so that path was not used.
- Final fix keeps DDP `find_unused_parameters=False` and adds a zero-valued dummy loss over trainable `llamagen_vq.quantize` parameters. This does not change the numeric loss, but keeps DDP reduction consistent when quantizer parameters are trainable.
- Final 3-GPU smoke test completed steps 132001-132002 successfully with `CUDA_VISIBLE_DEVICES=4,5,7`, `batch_size=1`, `accum_steps=1`, wandb disabled.
- Temporary smoke outputs under `/tmp` were removed after the test.

## 2026-07-22 - DINO feature reconstruction loss probe

Context:
- Goal: test whether a frozen DINOv2 feature reconstruction loss can push FID down without using validation/test statistics.
- Added `FrozenDINOFeatureLoss` in `train_titok_llamagen_recon.py`. The DINO backbone is frozen; target image features are detached, while generated image features keep gradients to the generator image.
- Added router training args: `lambda_dino_feat`, `dino_feat_loss`, `dino_feat_input_size`, `dino_feat_use_patch_tokens`, and `dino_feat_normalize`. Defaults keep old behavior unchanged (`lambda_dino_feat=0`).
- Probe config: `configs/titok_llamagen_mix_ae_unfreeze_encoder_gan_router_f2d_e2e_dynamic_freeze1d_patchdinoD_gan012_dino050_dinofeat010_from132000_4gpu_probe.yaml`.

Probe setting:
- Resume checkpoint: `results/titok_llamagen_mix_ae_unfreeze_encoder_gan_router_f2d_e2e_dynamic_freeze1d_patchdinoD_gan012_from129360_to137360_local4gpu_bs4_accum6/step_00132000.pt`.
- `lambda_dino_feat=0.1`, `dino_feat_loss=l1`, normalized cls+patch DINO tokens.
- Keep 1D adapter frozen, Router frozen (`lr_router=0`), and 2D quantizer/codebook frozen (`train_llamagen_quantizer=false`).
- Keep Patch+DINO discriminator and GAN weight unchanged (`lambda_gan=0.12`, `dino_loss_weight=0.50`).

Smoke result:
- `python3 -m py_compile` passed for the modified training scripts.
- Config parser accepted the new yaml keys.
- Single-GPU 1-step smoke completed on GPU1 with batch size 1. It loaded adapter init, EMA, discriminator, and resumed from step 132000; the run header confirmed `dino_feat:0.1/l1`.
- Temporary smoke output `/tmp/mot_smoke_dinofeat010` was removed.
- 4-GPU smoke was not run because GPUs 4-7 were occupied by an active MoT training job at the time.
