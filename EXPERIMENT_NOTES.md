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



## 2026-07-22 - High-frequency generator-gradient-only GAN probe

Context:
- The previous `highfreq_composite + lowfreq_anchor` branch failed because it changed the discriminator fake input distribution to `low(real) + high(fake)`, which made D optimize a task misaligned with full-image FID.
- This probe keeps D input as the normal full fake/real images, and only removes the low-frequency component from the generator-side GAN gradient.

Implementation:
- Added `gan_input_filter=highfreq_grad_only`.
- For generator GAN loss only, fake input is computed as `fake + low(fake).detach() - low(fake)`, so the forward value remains the full fake image while the GAN gradient becomes high-pass filtered.
- For discriminator updates, fake and real inputs remain full images exactly as in `gan_input_filter=none`.
- Kept `highfreq_composite` available for reproducibility, but it should not be used for this probe.

Probe setting:
- New config: `configs/titok_llamagen_mix_ae_unfreeze_encoder_gan_router_f2d_e2e_dynamic_freeze1d_patchdinoD_gan014_hfgrad_from132000_4gpu_probe.yaml`.
- Resume checkpoint: `results/titok_llamagen_mix_ae_unfreeze_encoder_gan_router_f2d_e2e_dynamic_freeze1d_patchdinoD_gan012_from129360_to137360_local4gpu_bs4_accum6/step_00132000.pt`.
- Run from 132000 to 133000, save every 1000 steps.
- `lambda_gan=0.14`, `gan_input_filter=highfreq_grad_only`, `gan_highpass_size=64`, `lambda_lowfreq_anchor=0.0`.
- Keep inherited Patch+DINO D, `dino_loss_weight=0.50`, `lr_d=2e-5`, and full low-lr 2D tokenizer+decoder training unchanged.
- Local 4-GPU setting uses `batch_size=4`, `accum_steps=6`, global batch 96.

Smoke result:
- `python3 -m py_compile train_titok_llamagen_decoder_adapt_router_f2d_e2e_dynamic.py` passed.
- Config parser resolved `lambda_gan=0.14`, `gan_input_filter=highfreq_grad_only`, `gan_highpass_size=64`, `lambda_lowfreq_anchor=0.0`, `max_steps=133000`, `batch_size=4`, `accum_steps=6`.
- 4-GPU smoke on GPUs 0,3,4,5 completed one G/D update from step 132000 to 132001.
- Run header confirmed `global_batch=96`, `gan:0.14@72000+ramp0/highfreq_grad_only@64`, `lowfreq_anchor:0.0@32`, and `phase=joint`.
- Progress bar showed `mix_l1=0.130`, `mix_lp=0.299`, `psnr=19.75`, `base=16.57`, `mask=0.51`, `tok=130`, `gan=0.051`, `lf=0.000`, `d=0.652`.
- Temporary smoke output `/tmp/mot_smoke_gan014_hfgrad` was removed.

Eval result on 50k validation images, EMA:
- step 133000: FID 2.61921, PSNR 20.4116, LPIPS 0.19979, L1 0.13772, SSIM 0.50985, tokens 133.51.
- Compared with the direct full-image `lambda_gan=0.16` probe at step 133000 (FID 2.55916, PSNR 20.3476), `highfreq_grad_only + lambda_gan=0.14` preserves reconstruction better but gives worse FID.
- Compared with the clean 132000 baseline region (FID about 2.57), this does not show a useful FID improvement. Do not continue this branch unless the priority shifts to preserving PSNR over FID.


## 2026-07-22 - Full-image GAN 0.16 with stronger reconstruction weight probe

Context:
- `highfreq_grad_only + lambda_gan=0.14` preserved PSNR but worsened FID, so the next probe returns to full-image GAN input.
- Direct full-image `lambda_gan=0.16` gave the best FID among recent probes, but PSNR/L1/SSIM drifted steadily. This probe keeps that FID-oriented adversarial signal and increases the mix reconstruction weight.

Probe setting:
- New config: `configs/titok_llamagen_mix_ae_unfreeze_encoder_gan_router_f2d_e2e_dynamic_freeze1d_patchdinoD_gan016_mix3_from132000_4gpu_probe.yaml`.
- Resume checkpoint: `results/titok_llamagen_mix_ae_unfreeze_encoder_gan_router_f2d_e2e_dynamic_freeze1d_patchdinoD_gan012_from129360_to137360_local4gpu_bs4_accum6/step_00132000.pt`.
- Run from 132000 to 133000, save every 1000 steps.
- `lambda_gan=0.16`, `gan_input_filter=none`, `lambda_mix=3.0`, `lambda_disc_feature_matching=0.0`, `g_freeze_steps=0`.
- Keep inherited Patch+DINO D, `dino_loss_weight=0.50`, `lr_d=2e-5`, and full low-lr 2D tokenizer+decoder training unchanged.
- Local 4-GPU setting uses `batch_size=4`, `accum_steps=6`, global batch 96.

Smoke result:
- Config parser resolved `lambda_gan=0.16`, `gan_input_filter=none`, `lambda_mix=3.0`, `max_steps=133000`, `batch_size=4`, `accum_steps=6`.
- 4-GPU smoke on GPUs 4,5,6,7 completed one G/D update from step 132000 to 132001.
- Run header confirmed `global_batch=96`, `mix:3.0`, `gan:0.16@72000+ramp0/none@64`, and `phase=joint`.
- Progress bar showed `mix_l1=0.130`, `mix_lp=0.299`, `psnr=19.75`, `base=16.57`, `mask=0.51`, `tok=130`, `gan=0.056`, `lf=0.000`, `d=0.652`.
- Temporary smoke output `/tmp/mot_smoke_gan016_mix3` was removed.

Eval result on 50k validation images, EMA:
- step 133000: FID 2.57684, PSNR 20.3993, LPIPS 0.19960, L1 0.13789, SSIM 0.50947, tokens 133.51.
- Compared with direct full-image `lambda_gan=0.16, lambda_mix=2.0` at step 133000 (FID 2.55916, PSNR 20.3476), `lambda_mix=3.0` preserves PSNR better but loses the FID gain.
- Conclusion: increasing reconstruction weight alone trades away the useful adversarial improvement; this branch is not a better FID direction.


## 2026-07-22 - Multi-scale Patch+DINO discriminator probe

Context:
- Goal: try a more FID-friendly discriminator while keeping 1D frozen and avoiding validation/test leakage.
- Previous best FID-oriented branch was direct full-image `lambda_gan=0.16`, but it only improved FID slightly and steadily degraded PSNR/L1/SSIM.
- This probe keeps full-image real/fake D inputs and changes only the discriminator structure.

Implementation:
- Added `discriminator_type=multiscale_patch_dino`.
- The patch branch is a `MultiScalePatchDiscriminator`; the DINO branch is the existing frozen DINOv2 feature discriminator.
- Added compatible loading from old `patch_dino` discriminator checkpoints: old single-scale patch weights are copied into each multi-scale patch discriminator, and the old DINO branch is loaded unchanged. This avoids resetting D when resuming from the clean 132000 checkpoint.
- Updated weighted-logit iteration and discriminator feature matching to flatten nested weighted outputs from `Patch+DINO(MultiScalePatch)`.

Probe setting:
- New config: `configs/titok_llamagen_mix_ae_unfreeze_encoder_gan_router_f2d_e2e_dynamic_freeze1d_multiscale_patchdinoD_gan014_from132000_4gpu_probe.yaml`.
- Resume checkpoint: `results/titok_llamagen_mix_ae_unfreeze_encoder_gan_router_f2d_e2e_dynamic_freeze1d_patchdinoD_gan012_from129360_to137360_local4gpu_bs4_accum6/step_00132000.pt`.
- Run from 132000 to 136000, save every 1000 steps.
- `lambda_gan=0.14`, `gan_input_filter=none`, `discriminator_type=multiscale_patch_dino`, `disc_scales=[1.0, 0.5]`, `disc_loss_weights=[1.0, 0.5]`, `dino_loss_weight=0.50`.
- Keep inherited D weights (`reset_discriminator=false`), EMA enabled, Router effectively frozen (`lr_router=0`), and 1D adapter frozen (`train_adapter=false`).
- Local 4-GPU setting uses `batch_size=4`, `accum_steps=6`, global batch 96.

Smoke result:
- `python3 -m py_compile train_titok_llamagen_recon.py train_titok_llamagen_decoder_adapt_router_f2d_e2e_dynamic.py` passed.
- Parser accepted `discriminator_type=multiscale_patch_dino`.
- First smoke attempts exposed only launch/config issues: `--wandb false` should be `--no-wandb`, and this machine needs local overrides for `data_path`, `adapter_init`, and `dino_repo`.
- 4-GPU smoke on GPUs 4,5,6,7 completed one full G/D update from step 132000 to 132001 with `batch_size=4`, `accum_steps=6`.
- Run header confirmed `disc:multiscale_patch_dino/scales=[1.0, 0.5]/weights=[1.0, 0.5]`, `gan:0.14@72000+ramp0/none@64`, and `loaded discriminator from resume (full)`.
- Progress bar showed `mix_l1=0.130`, `mix_lp=0.299`, `psnr=19.75`, `base=16.57`, `mask=0.51`, `tok=130`, `gan=0.038`, `d=0.731`, `phase=joint`.
- Temporary smoke output `/tmp/mot_smoke_multiscale_patchdino` was removed.


## 2026-07-23 - PatchDINO GAN 0.13 with mix reconstruction 2.5 boundary probe

Context:
- Goal: reduce FID while enforcing `PSNR >= 20.35` as a hard acceptance threshold.
- The previous `multiscale_patch_dino + lambda_gan=0.14` branch reached best FID 2.55063, but all measured checkpoints had PSNR below 20.35 and the generated checkpoint weights were deleted after preserving eval JSON/logs.
- This probe returns to the stable single-scale Patch+DINO discriminator and uses milder GAN pressure plus a moderate reconstruction-weight increase.

Probe setting:
- New config: `configs/titok_llamagen_mix_ae_unfreeze_encoder_gan_router_f2d_e2e_dynamic_freeze1d_patchdinoD_gan013_mix25_from132000_4gpu_probe.yaml`.
- Resume checkpoint: `results/titok_llamagen_mix_ae_unfreeze_encoder_gan_router_f2d_e2e_dynamic_freeze1d_patchdinoD_gan012_from129360_to137360_local4gpu_bs4_accum6/step_00132000.pt`.
- Run from 132000 to 134000, save every 500 steps for early-stop selection.
- `lambda_gan=0.13`, `lambda_mix=2.5`, `gan_input_filter=none`, `discriminator_type=patch_dino`, `dino_loss_weight=0.50`.
- Keep inherited D weights (`reset_discriminator=false`), EMA enabled, Router effectively frozen (`lr_router=0`), and 1D adapter frozen (`train_adapter=false`).
- Local 4-GPU setting uses `batch_size=4`, `accum_steps=6`, global batch 96.

Smoke result:
- 4-GPU smoke on GPUs 4,5,6,7 completed one full G/D update from step 132000 to 132001 with `batch_size=4`, `accum_steps=6`.
- Run header confirmed `mix:2.5`, `gan:0.13@72000+ramp0/none@64`, `disc:patch_dino/scales=[1.0]/weights=[1.0]`, and `loaded discriminator from resume (full)`.
- Progress bar showed `mix_l1=0.130`, `mix_lp=0.299`, `psnr=19.75`, `base=16.57`, `mask=0.51`, `tok=130`, `gan=0.046`, `d=0.652`, `phase=joint`.
- Temporary smoke output `/tmp/mot_smoke_gan013_mix25` was removed.

Eval result on 50k validation images, EMA:
- step 132500: FID 2.58550, PSNR 20.3920, LPIPS 0.19952, L1 0.13799, SSIM 0.50920, tokens 133.51.
- step 133000: FID 2.58550, PSNR 20.3944, LPIPS 0.19956, L1 0.13797, SSIM 0.50923, tokens 133.51.
- step 133500: FID 2.57874, PSNR 20.3966, LPIPS 0.19958, L1 0.13795, SSIM 0.50932, tokens 133.50.
- step 134000: FID 2.58428, PSNR 20.3981, LPIPS 0.19959, L1 0.13793, SSIM 0.50942, tokens 133.51.
- Conclusion: `lambda_mix=2.5` preserves PSNR safely above 20.35, but FID does not improve; checkpoint weights from this probe were deleted after preserving eval JSON/logs.


## 2026-07-23 - PatchDINO GAN 0.14 with mix reconstruction 2.0 probe

Context:
- `lambda_gan=0.13, lambda_mix=2.5` kept PSNR high but failed to reduce FID.
- The next probe returns mix reconstruction weight to 2.0 and uses moderate GAN pressure at 0.14 with the stable single-scale Patch+DINO discriminator.

Probe setting:
- New config: `configs/titok_llamagen_mix_ae_unfreeze_encoder_gan_router_f2d_e2e_dynamic_freeze1d_patchdinoD_gan014_mix2_from132000_4gpu_probe.yaml`.
- Resume checkpoint: `results/titok_llamagen_mix_ae_unfreeze_encoder_gan_router_f2d_e2e_dynamic_freeze1d_patchdinoD_gan012_from129360_to137360_local4gpu_bs4_accum6/step_00132000.pt`.
- Run from 132000 to 134000, save every 500 steps for early-stop selection.
- `lambda_gan=0.14`, `lambda_mix=2.0`, `gan_input_filter=none`, `discriminator_type=patch_dino`, `dino_loss_weight=0.50`.
- Keep inherited D weights (`reset_discriminator=false`), EMA enabled, Router effectively frozen (`lr_router=0`), and 1D adapter frozen (`train_adapter=false`).
- Local 4-GPU setting uses `batch_size=4`, `accum_steps=6`, global batch 96.

Smoke result:
- Initial smoke attempts exposed only local path issues in the portable MoT config: missing `../ImageNet/train`, `weights/step_00066000.pt`, and relative DINO hub repo. These were handled by CLI overrides on this machine.
- 4-GPU smoke on GPUs 4,5,6,7 completed one full G/D update from step 132000 to 132001.
- Run header confirmed `mix:2.0`, `gan:0.14@72000+ramp0/none@64`, `disc:patch_dino/scales=[1.0]/weights=[1.0]`, and `loaded discriminator from resume (full)`.
- Progress bar showed `mix_l1=0.130`, `mix_lp=0.299`, `psnr=19.75`, `base=16.57`, `mask=0.51`, `tok=130`, `gan=0.049`, `d=0.652`, `phase=joint`.
- Temporary smoke output `/tmp/mot_smoke_gan014_mix2` was removed.

## 2026-07-23 - StyleGAN discriminator probe

Context:
- Goal: try a more FID-friendly GAN recipe without using Inception/FID features as a training target.
- We intentionally avoid an Inception feature discriminator because FID itself is computed in Inception feature space and that would look like optimizing the metric proxy.
- This probe uses the StyleGAN/MaskGIT-style discriminator architecture already present in LlamaGen (`tokenizer/tokenizer_image/discriminator_stylegan.py`) and keeps the rest of the MoT training recipe comparable to the recent `patch_dino + lambda_gan=0.14, lambda_mix=2.0` probe.

Implementation:
- Added `discriminator_type=stylegan` to `train_titok_llamagen_recon.py` and `train_titok_llamagen_decoder_adapt_router_f2d_e2e_dynamic.py`.
- The implementation follows the LlamaGen StyleGAN discriminator structure; the blur layer is implemented locally with grouped `conv2d` because this environment does not have `kornia` installed.
- Existing discriminator choices are unchanged.

Probe setting:
- New config: `configs/titok_llamagen_mix_ae_unfreeze_encoder_gan_router_f2d_e2e_dynamic_freeze1d_styleganD_gan014_mix2_from132000_4gpu_probe.yaml`.
- Resume checkpoint: `results/titok_llamagen_mix_ae_unfreeze_encoder_gan_router_f2d_e2e_dynamic_freeze1d_patchdinoD_gan012_from129360_to137360_local4gpu_bs4_accum6/step_00132000.pt`.
- Run from 132000 to 134000, save every 500 steps.
- `discriminator_type=stylegan`, `reset_discriminator=true`, `d_warmup_steps=500`, `lambda_gan=0.14`, `lambda_mix=2.0`.
- Keep EMA enabled, Router effectively frozen (`lr_router=0`), and 1D adapter frozen (`train_adapter=false`).
- Local 4-GPU setting uses `batch_size=4`, `accum_steps=6`, global batch 96.

Smoke result:
- `python3 -m py_compile train_titok_llamagen_recon.py train_titok_llamagen_decoder_adapt_router_f2d_e2e_dynamic.py` passed.
- 4-GPU smoke on GPUs 4,5,6,7 completed one full step from 132000 to 132001 with local data/adapter path overrides.
- Run header confirmed `disc:stylegan/scales=[1.0]/weights=[1.0]`, `gan:0.14@72000+ramp0/none@64`, `d_warmup:500`, and resume from 132000.
- Progress bar showed `mix_l1=0.130`, `mix_lp=0.299`, `psnr=19.75`, `base=16.57`, `mask=0.51`, `tok=130`, `gan=0.000`, `d=1.001`; `gan=0.000` is expected during the discriminator warmup window.
- Temporary smoke output `/tmp/mot_smoke_styleganD_gan014_mix2` was removed.

## 2026-07-23 - DINOv1-S feature discriminator probe

Context:
- Goal: try a 2025-style VFM discriminator without using Inception/FID feature space as a training target.
- This branch follows the VFMTok-style idea more closely than `patch_dino`: the discriminator is a pure frozen DINOv1-S/16 feature discriminator with trainable spectral-norm heads on CLS and patch tokens, not PatchGAN plus DINOv2.
- StyleGAN probe checkpoint weights were deleted after preserving eval JSON/logs because its FID degraded from 2.80 to 3.27.

Implementation:
- Added DINOv1-S backbone support in `train_titok_llamagen_recon.py` via timm `vit_small_patch16_224` and local checkpoint `/home/heyefei/.cache/torch/hub/checkpoints/dino_deitsmall16_pretrain.pth`.
- Added `discriminator_type` choices `dino` and `dinov1s`. `dinov1s` forces the DINOv1-S/16 backbone and outputs one CLS logit plus patch-token logits.
- Added `--dino-ckpt` to both training scripts. Existing `patch_dino` and `multiscale_patch_dino` behavior is unchanged.

Probe setting:
- New config: `configs/titok_llamagen_mix_ae_unfreeze_encoder_gan_router_f2d_e2e_dynamic_freeze1d_dinov1sD_gan014_mix2_from132000_4gpu_probe.yaml`.
- Resume checkpoint: `results/titok_llamagen_mix_ae_unfreeze_encoder_gan_router_f2d_e2e_dynamic_freeze1d_patchdinoD_gan012_from129360_to137360_local4gpu_bs4_accum6/step_00132000.pt`.
- Run from 132000 to 134000, save every 500 steps.
- `discriminator_type=dinov1s`, `reset_discriminator=true`, `d_warmup_steps=1000`, `lambda_gan=0.14`, `lambda_mix=2.0`, `lr_d=2e-5`.
- Keep EMA enabled, Router effectively frozen (`lr_router=0`), and 1D adapter frozen (`train_adapter=false`).

Smoke result:
- `python3 -m py_compile train_titok_llamagen_recon.py train_titok_llamagen_decoder_adapt_router_f2d_e2e_dynamic.py` passed.
- Local dummy forward verified DINOv1-S loads without missing/unexpected keys and returns logits with shape `[B, 197]`.
- 4-GPU smoke on GPUs 4,5,6,7 completed one full step from 132000 to 132001 with local data/adapter path overrides.
- Run header confirmed `disc:dinov1s/scales=[1.0]/weights=[1.0]`, `dino:dinov1_vits16@0.5`, `d_warmup:1000`, and resume from 132000.
- Progress bar showed `gan=0.000`, `d=1.007`; `gan=0.000` is expected during the discriminator warmup window.
- Temporary smoke output `/tmp/mot_smoke_dinov1sD` was removed.

## 2026-07-23 - DINOv1-S GAN 1.5 probe and GPU watcher

Context:
- Goal: test whether the previous DINOv1-S GAN runs were too weak to affect the generator. Log analysis showed `lambda_gan=0.30` contributed only about 1.3% of total loss and 2.5% of mix reconstruction loss.
- This probe raises `lambda_gan` to 1.50 so the GAN term should be large enough to visibly affect training dynamics after warmup.

Probe setting:
- New config: `configs/titok_llamagen_mix_ae_unfreeze_encoder_gan_router_f2d_e2e_dynamic_freeze1d_dinov1sD_gan150_mix2_from132000_4gpu_probe.yaml`.
- Resume checkpoint: `results/titok_llamagen_mix_ae_unfreeze_encoder_gan_router_f2d_e2e_dynamic_freeze1d_patchdinoD_gan012_from129360_to137360_local4gpu_bs4_accum6/step_00132000.pt`.
- `discriminator_type=dinov1s`, `reset_discriminator=true`, `d_warmup_steps=1000`, `lambda_gan=1.50`, `lambda_mix=2.0`.
- Run from 132000 to 134000 and save every 500 steps.

Launch helper:
- Added `scripts/wait_and_train_dinov1s_gan150.sh`. It polls `nvidia-smi` until four candidate GPUs have memory usage below `MAX_USED_MB` and then launches the probe.
- Defaults: `NEEDED_GPUS=4`, `MAX_USED_MB=2000`, `CHECK_INTERVAL=60`, `GPU_CANDIDATES=0,1,2,3,4,5,6,7`.
- Syntax check and no-launch waiting branch test passed. A full smoke test was not run because all GPUs were occupied or had stale contexts at the time.

Eval result:
- 50k ImageNet validation eval used `eval_titok_llamagen_mix_metrics_router_f2d_e2e_dynamic.py`, mix-only path, absolute validation path `/home/heyefei/ImageNet/validation`, and absolute 66000 adapter init.
- `step_00132500.pt`: FID 4.35880, PSNR 21.15139, LPIPS 0.22606, L1 0.12712, tokens 133.64.
- `step_00133000.pt`: FID 4.80019, PSNR 21.22234, LPIPS 0.22917, L1 0.12623, tokens 133.66.
- `step_00133500.pt`: FID 5.16807, PSNR 20.58137, LPIPS 0.22665, L1 0.13745, tokens 133.57.
- `step_00134000.pt`: FID 3.66588, PSNR 20.67981, LPIPS 0.22202, L1 0.13534, tokens 133.66.
- Conclusion: `lambda_gan=1.50` with pure DINOv1-S discriminator is not useful. It can keep or raise PSNR on early checkpoints, but FID is far worse than the patch+DINO baseline and worse than the previous DINOv1-S low-weight probe.

## 2026-07-23 - DINOv1-S GAN 1.2 with stronger D and G-side ramp

Context:
- The previous `DINOv1-S D + lambda_gan=1.50` probe showed FID recovery at the end, but the D/G dynamics were unhealthy: after D warmup, `d_real - d_fake` became negative and the generator appeared to exploit a weak/reset D.
- Deleted the generated `gan150` checkpoint weights (`latest.pt`, `step_00132500.pt`, `step_00133000.pt`, `step_00133500.pt`, `step_00134000.pt`) after preserving eval JSON and log files.

Implementation:
- Added `--gan-g-ramp-after-d-warmup` to `train_titok_llamagen_decoder_adapt_router_f2d_e2e_dynamic.py`. D still trains according to `gan_factor`, while the generator GAN loss can ramp separately after the reset-D warmup window.
- New config: `configs/titok_llamagen_mix_ae_unfreeze_encoder_gan_router_f2d_e2e_dynamic_freeze1d_dinov1sD_gan120_lrd5e5_gramp500_from132000_4gpu_probe.yaml`.
- Setting: resume from clean `step_00132000.pt`, `lambda_gan=1.20`, `lr_d=5e-5`, `d_warmup_steps=1000`, `gan_ramp_steps=500`, `gan_g_ramp_after_d_warmup=true`, `batch_size=3`, `accum_steps=8`.
- Expected G-side factor: 0 through step 133000, 0.5 at step 133250, and 1.0 from step 133500 onward.

Smoke result:
- `python3 -m py_compile train_titok_llamagen_decoder_adapt_router_f2d_e2e_dynamic.py` passed.
- 4-GPU smoke on GPUs 4,5,6,7 completed one full step from 132000 to 132001 with local data/adapter path overrides.
- Run header confirmed `gan:1.2@72000+ramp500/g_after_dwarm:True`, `lr_d=5e-5`, `batch_size_per_gpu=3`, `accum_steps=8`.
- Progress bar showed `gan=0.000`, `d=1.013`, which is expected during D warmup.
- Temporary smoke output `/tmp/mot_smoke_dinov1sD_gan120_lrd5e5_gramp500` was removed.

## 2026-07-23 - Strong patch+DINO discriminator probe

Context:
- Pure DINOv1-S discriminator was too weak in practice: after D warmup the generator quickly drove `d_real - d_fake` toward zero and then negative, so increasing GAN weight alone produced unstable gradients and poor FID.
- This probe keeps the more FID-relevant local RGB PatchGAN branch and increases discriminator capacity instead of relying on pure DINO features.

Probe setting:
- New config: `configs/titok_llamagen_mix_ae_unfreeze_encoder_gan_router_f2d_e2e_dynamic_freeze1d_strongpatchdinoD_gan014_lrd5e5_gramp500_from132000_4gpu_probe.yaml`.
- Resume checkpoint: `results/titok_llamagen_mix_ae_unfreeze_encoder_gan_router_f2d_e2e_dynamic_freeze1d_patchdinoD_gan012_from129360_to137360_local4gpu_bs4_accum6/step_00132000.pt`.
- `discriminator_type=patch_dino`, `reset_discriminator=true`, `disc_hidden_channels=256`, `disc_num_stages=4`, `dino_head_hidden=512`, `lambda_gan=0.14`, `lr_d=5e-5`, `d_warmup_steps=1000`, `gan_ramp_steps=500`, `gan_g_ramp_after_d_warmup=true`.
- Local 4-GPU setting was updated to `batch_size=4`, `accum_steps=8`, global batch 128 after the initial smoke. Any already-running process that was launched before this edit still uses the old `batch_size=3`, `accum_steps=8` values from its startup config.

Smoke result:
- `python3 -m py_compile train_titok_llamagen_decoder_adapt_router_f2d_e2e_dynamic.py train_titok_llamagen_recon.py` passed.
- Initial smoke failed before training because the copied MoT config used relative `dino_repo: ../.cache/torch/hub/facebookresearch_dinov2_main`, which does not exist under `/home/heyefei/lichenge/MoT`. The config was corrected to absolute `/home/heyefei/.cache/torch/hub/facebookresearch_dinov2_main`.
- 4-GPU smoke on GPUs 4,5,6,7 completed one full step from 132000 to 132001 with local data/adapter path overrides.
- Run header confirmed `gan:0.14@72000+ramp500/g_after_dwarm:True`, `disc:patch_dino`, `dino:dinov2_vits14@0.5`, `d_warmup:1000`, `lr_d=5e-5`, `batch_size_per_gpu=3`, `accum_steps=8`.
- Progress bar showed `gan=0.000`, `d=1.001`, which is expected during D warmup.
- Temporary smoke output `/tmp/mot_smoke_strongpatchdinoD_gan014_lrd5e5_gramp500` was removed.
