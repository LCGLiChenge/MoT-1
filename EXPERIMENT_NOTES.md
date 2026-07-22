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

Eval result on 50k validation images, EMA:
- step 133000: FID 2.5786, PSNR 20.3861, LPIPS 0.19957.
- step 134000: FID 2.5788, PSNR 20.3863, LPIPS 0.19959.
- step 135000: FID 2.5797, PSNR 20.3845, LPIPS 0.19944.
- step 136000: FID 2.5924, PSNR 20.3832, LPIPS 0.19926.
- Conclusion: DINO feature loss at weight 0.1 does not improve FID; it slightly helps LPIPS late but pushes FID worse than the clean 132000 baseline.

## 2026-07-22 - Tail-only 2D decoder unfreeze FID probe

Context:
- Goal: test a more FID-friendly adaptation path after DINO feature loss failed to reduce FID.
- Instead of training the whole LlamaGen 2D decoder or 2D codebook, train only the last two LlamaGen decoder up blocks plus `conv_out`.
- Keep 1D adapter, Router, 2D encoder, quant conv, quantizer/codebook, and post-quant conv frozen.
- Keep the inherited Patch+DINO discriminator and GAN settings unchanged.

Implementation:
- Added `--llamagen-decoder-train-last-n` to `train_titok_llamagen_decoder_adapt_router_f2d_e2e_dynamic.py`.
- Default `-1` preserves old behavior; `0` disables decoder training; positive values train only the last N `decoder.conv_blocks` plus `decoder.conv_out`.
- New probe config: `configs/titok_llamagen_mix_ae_unfreeze_encoder_gan_router_f2d_e2e_dynamic_freeze1d_patchdinoD_gan012_tail2_from132000_4gpu_probe.yaml`.

Probe setting:
- Resume checkpoint: `results/titok_llamagen_mix_ae_unfreeze_encoder_gan_router_f2d_e2e_dynamic_freeze1d_patchdinoD_gan012_from129360_to137360_local4gpu_bs4_accum6/step_00132000.pt`.
- `llamagen_decoder_train_last_n=2`, `train_post_quant_conv=false`, `train_llamagen_quantizer=false`.
- `lambda_gan=0.12`, `dino_loss_weight=0.50`, `reset_optimizer=true`, `reset_discriminator=false`.
- 4-GPU local probe runs from 132000 to 134000 and saves step checkpoints every 1000 steps.

Smoke result:
- `python3 -m py_compile train_titok_llamagen_decoder_adapt_router_f2d_e2e_dynamic.py` passed.
- Config parser accepted the new yaml and resolved `llamagen_decoder_train_last_n=2`.
- 4-GPU 1-step smoke completed with local path overrides for data, adapter init, and DINO repo.
- Run header confirmed `trainable_params=2881541`, `train_post_quant_conv=False`, `train_llamagen_quantizer=False`, and `llamagen_decoder_train_last_n=2`.
- The smoke step completed G/D update and saved `/tmp/mot_smoke_tail2/latest.pt`; the temporary output directory was removed after the test.

Eval result on 50k validation images, EMA:
- step 133000: FID 2.5807, PSNR 20.3887, LPIPS 0.19953, L1 0.13804, SSIM 0.50912, tokens 133.51.
- step 134000: FID 2.5851, PSNR 20.3896, LPIPS 0.19954, L1 0.13803, SSIM 0.50918, tokens 133.51.
- Conclusion: tail-only decoder unfreeze slightly improves PSNR/L1/SSIM but worsens FID versus the clean 132000 baseline, so it is not a good FID-compression direction.

## 2026-07-22 - D-only warmup + discriminator feature matching probe

Context:
- Goal: make GAN updates more FID-friendly after DINO feature loss and tail-only decoder unfreeze both worsened FID.
- Hypothesis: first recalibrate the inherited Patch+DINO discriminator on the current mix distribution, then use discriminator feature matching to stabilize G updates.
- This does not use validation/test statistics as a training target.

Implementation:
- Added `--g-freeze-steps`: during the first N resumed steps, the generator optimizer step and EMA update are skipped, while D still trains. G forward/backward is still executed to keep DDP reductions consistent.
- Added `--lambda-disc-feature-matching`: G receives an L1 feature matching loss from the current discriminator intermediate features.
- Feature matching supports Patch, MultiScale Patch, and Patch+DINO discriminators; DDP-wrapped discriminators are unwrapped before feature extraction.

Probe setting:
- New config: `configs/titok_llamagen_mix_ae_unfreeze_encoder_gan_router_f2d_e2e_dynamic_freeze1d_patchdinoD_gan012_fm010_gfreeze1k_from132000_4gpu_probe.yaml`.
- Resume checkpoint: `results/titok_llamagen_mix_ae_unfreeze_encoder_gan_router_f2d_e2e_dynamic_freeze1d_patchdinoD_gan012_from129360_to137360_local4gpu_bs4_accum6/step_00132000.pt`.
- Run from 132000 to 135000, save every 1000 steps.
- First 1000 steps: G frozen / D only (`g_freeze_steps=1000`).
- Joint phase: `lambda_disc_feature_matching=0.1`, `lambda_gan=0.12`, inherited Patch+DINO D, full low-lr 2D tokenizer+decoder trainable; 1D adapter and Router effectively frozen (`lr_router=0`).

Smoke result:
- `python3 -m py_compile train_titok_llamagen_recon.py train_titok_llamagen_decoder_adapt_router_f2d_e2e_dynamic.py` passed.
- Config parser resolved `lambda_disc_feature_matching=0.1`, `g_freeze_steps=1000`, `max_steps=135000`.
- 4-GPU g-freeze 1-step smoke completed; progress bar showed `phase=g_freeze`, `fm=0.000`, and D update ran.
- 4-GPU joint 1-step smoke with `--g-freeze-steps 0` completed; progress bar showed `phase=joint` and nonzero `fm=0.024`.
- Temporary smoke outputs `/tmp/mot_smoke_fm_gfreeze` and `/tmp/mot_smoke_fm_joint` were removed.


Eval result on 50k validation images, EMA:
- step 133000: FID 2.57319, PSNR 20.3849, LPIPS 0.19949, L1 0.13808, SSIM 0.50897, tokens 133.51.
- step 134000: FID 2.57316, PSNR 20.3892, LPIPS 0.19952, L1 0.13804, SSIM 0.50914, tokens 133.51.
- step 135000: FID 2.57339, PSNR 20.3862, LPIPS 0.19936, L1 0.13807, SSIM 0.50919, tokens 133.51.
- Interpretation: step 133000 matches the 132000 EMA baseline because the first 1000 resumed steps freeze G and skip EMA updates; only D is recalibrated.
- Conclusion: D-only warmup plus discriminator feature matching is stable, but does not materially improve FID beyond the clean 132000 baseline. The best measured checkpoint is 134000 by a negligible FID margin, so this is not enough to claim a real FID gain.

## 2026-07-22 - Direct GAN pressure probe from clean 132000

Context:
- The D-only warmup plus discriminator feature matching branch was stable but did not materially improve FID.
- New probe removes both mechanisms and directly increases GAN pressure, to test whether FID is limited by weak generator-side adversarial signal.

Probe setting:
- New config: `configs/titok_llamagen_mix_ae_unfreeze_encoder_gan_router_f2d_e2e_dynamic_freeze1d_patchdinoD_gan016_from132000_4gpu_probe.yaml`.
- Resume checkpoint: `results/titok_llamagen_mix_ae_unfreeze_encoder_gan_router_f2d_e2e_dynamic_freeze1d_patchdinoD_gan012_from129360_to137360_local4gpu_bs4_accum6/step_00132000.pt`.
- Run from 132000 to 136000, save every 1000 steps.
- `lambda_gan=0.16`, `lambda_disc_feature_matching=0.0`, `g_freeze_steps=0`.
- Keep inherited Patch+DINO D, `dino_loss_weight=0.50`, `lr_d=2e-5`, and full low-lr 2D tokenizer+decoder training unchanged.
- Local 4-GPU setting uses `batch_size=4`, `accum_steps=6`, global batch 96.

Smoke result:
- 4-GPU 1-step smoke completed from step 132000 to 132001 with local path overrides.
- Run header confirmed `global_batch=96`, `gan:0.16`, `disc_fm:0.0`, `g_freeze:0`, and `phase=joint`.
- The smoke step completed G/D update and saved `/tmp/mot_smoke_gan016_bs4/latest.pt`; temporary smoke outputs `/tmp/mot_smoke_gan016` and `/tmp/mot_smoke_gan016_bs4` were removed.

Eval result on 50k validation images, EMA:
- step 133000: FID 2.55916, PSNR 20.3476, LPIPS 0.19922, L1 0.13875, SSIM 0.50717, tokens 133.50.
- step 134000: FID 2.57741, PSNR 20.3306, LPIPS 0.19922, L1 0.13909, SSIM 0.50638, tokens 133.50.
- step 135000: FID 2.55541, PSNR 20.3184, LPIPS 0.19907, L1 0.13928, SSIM 0.50598, tokens 133.50.
- step 136000: FID 2.55801, PSNR 20.3115, LPIPS 0.19893, L1 0.13937, SSIM 0.50566, tokens 133.50.
- Interpretation: stronger GAN pressure can reduce FID versus the clean 132000 baseline (FID 2.57319), with best measured FID at 135000, but reconstruction metrics degrade steadily.
- Conclusion: this is a real but small FID improvement, not enough for the 2.2 target. Treat 135000 as the best gan0.16 checkpoint so far; do not continue this branch blindly unless accepting further PSNR/L1 drift.

## 2026-07-22 - High-frequency GAN plus low-frequency anchor probe

Context:
- The direct `lambda_gan=0.16` probe improved FID slightly but steadily degraded PSNR/L1/SSIM, suggesting adversarial gradients were changing low-frequency structure or color, not only texture.
- Deleted the generated gan0.16 probe weights (`latest.pt`, `step_00133000.pt`, `step_00134000.pt`, `step_00135000.pt`, `step_00136000.pt`) after preserving log and eval json files.

Implementation:
- Added `--gan-input-filter` with default `none` and new mode `highfreq_composite`.
- In `highfreq_composite`, discriminator fake input is `low(real) + high(fake)`, clipped to `[0, 1]`; this keeps D input image-like while forcing the adversarial gradient to mainly affect fake high-frequency content.
- Added `--gan-highpass-size` to control the low/high split resolution.
- Added `--lambda-lowfreq-anchor` and `--lowfreq-anchor-size`; the anchor is an L1 loss between downsampled `x_mix` and downsampled real image, intended to protect low-frequency color/structure and PSNR.
- Defaults keep old behavior unchanged: `gan_input_filter=none`, `lambda_lowfreq_anchor=0`.

Probe setting:
- New config: `configs/titok_llamagen_mix_ae_unfreeze_encoder_gan_router_f2d_e2e_dynamic_freeze1d_patchdinoD_gan016_hfgan_lfanchor_from132000_4gpu_probe.yaml`.
- Resume checkpoint: `results/titok_llamagen_mix_ae_unfreeze_encoder_gan_router_f2d_e2e_dynamic_freeze1d_patchdinoD_gan012_from129360_to137360_local4gpu_bs4_accum6/step_00132000.pt`.
- Run from 132000 to 136000, save every 1000 steps.
- `lambda_gan=0.16`, `gan_input_filter=highfreq_composite`, `gan_highpass_size=64`, `lambda_lowfreq_anchor=1.0`, `lowfreq_anchor_size=32`.
- Keep inherited Patch+DINO D, `dino_loss_weight=0.50`, `lr_d=2e-5`, and full low-lr 2D tokenizer+decoder training unchanged.
- Local 4-GPU setting uses `batch_size=4`, `accum_steps=6`, global batch 96.

Smoke result:
- `python3 -m py_compile train_titok_llamagen_decoder_adapt_router_f2d_e2e_dynamic.py` passed.
- Config parser resolved `lambda_gan=0.16`, `gan_input_filter=highfreq_composite`, `gan_highpass_size=64`, `lambda_lowfreq_anchor=1.0`, `lowfreq_anchor_size=32`, `batch_size=4`, `accum_steps=6`, `max_steps=136000`.
- First 4-GPU smoke on GPUs 4,5,6,7 failed because GPU7 was occupied by another process using about 28.8GB; no code error.
- Re-ran 4-GPU smoke on GPUs 0,4,5,6; it completed one G/D update from step 132000 to 132001.
- Run header confirmed `global_batch=96`, `lowfreq_anchor:1.0@32`, `gan:0.16@72000+ramp0/highfreq_composite@64`, and `phase=joint`.
- Progress bar showed nonzero `gan=0.021` and `lf=0.029`, confirming both new loss paths were active.
- Temporary smoke output `/tmp/mot_smoke_hfgan_lfanchor` was removed.

Failure update:
- Training saved `step_00133000.pt`, `step_00134000.pt`, `step_00135000.pt`, and `latest.pt`; these checkpoint files were deleted after preserving log/eval json.
- Eval on 50k validation images, EMA, step 133000: FID 2.72009, PSNR 20.2852, LPIPS 0.19878, L1 0.13950, SSIM 0.50607, tokens 133.54.
- Training log also deteriorated: mix PSNR moved from about 19.72 at 132050 to about 19.53 at 133000 and about 19.41 at 135000; low-frequency anchor rose from about 0.0293 to about 0.0313.
- Diagnosis: `highfreq_composite` changed the discriminator fake distribution to `low(real) + high(fake)`. This hid fake low-frequency errors from D, created a new D task unlike the inherited Patch+DINO full-image discriminator, and made the adversarial signal poorly aligned with full-image FID. The low-frequency anchor was too weak and redundant with existing L1 to counteract this.
- Conclusion: do not continue this branch. A safer variant should keep D inputs as normal real/fake full images and only detach fake low-frequency content on the generator path if high-frequency-only GAN gradients are desired.

