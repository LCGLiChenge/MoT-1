# Experiment Notes

## Current H200 Handoff

Goal:
- Continue the current projected ConvNeXt discriminator branch on 8 H200 GPUs.
- Resume model/EMA from `weights/epoch_0005_step_00127360.pt`.
- Keep `weights/step_00066000.pt` as the adapter init reference.
- Do not update the 1D adapter.
- Keep Router effectively frozen with `lr_router=0`; Router still participates in forward selection.
- Train the 2D tokenizer/decoder at low LR and use EMA.
- Reset optimizer and discriminator when entering this projected ConvNeXt branch.

Main files:
- `train_titok_llamagen_decoder_adapt_router_f2d_e2e_dynamic.py`
- `eval_titok_llamagen_mix_metrics_router_f2d_e2e_dynamic.py`
- `configs/h200_projectedconvnext_from_epoch5.yaml`
- `configs/eval_projectedconvnext_50000.yaml`

Current H200 config:
- Output: `results/projectedconvnext_from_epoch5_h200_8gpu`
- `batch_size=32`, `accum_steps=1`
- `lambda_gan=0.10`, `lambda_mix=2.0`
- `discriminator_type=projected_convnext`
- `lr_d=5e-5`, `d_warmup_steps=200` after resetting the projected ConvNeXt discriminator
- `use_ema=true`, `ema_decay=0.999`
- `latest.pt` is updated every epoch.
- Train from step 127360 to `max_steps=152385`, about 5 more epochs on ImageNet.
- Extra step checkpoints are saved at explicit steps in the yaml, including final step 152385.

Known local result before H200 migration:
- Projected ConvNeXt branch around 136500-137000 gave the best recent FID/PSNR tradeoff.
- 50k val EMA metrics:
  - step 136500: FID 2.51763, PSNR 20.92721, LPIPS 0.20577, tokens 133.63
  - step 137000: FID 2.51945, PSNR 20.93426, LPIPS 0.20546, tokens 133.64
- Later local checkpoints 139000/141000/143000/145000 were evaluated after the CUDA driver recovered.
- 50k val EMA metrics, mix-only:
  - step 139000: FID 2.43991, PSNR 20.93324, LPIPS 0.20579, tokens 133.61
  - step 141000: FID 2.41792, PSNR 20.96448, LPIPS 0.20544, tokens 133.62
  - step 143000: FID 2.38564, PSNR 20.97675, LPIPS 0.20549, tokens 133.61
  - step 145000: FID 2.39550, PSNR 20.98483, LPIPS 0.20527, tokens 133.59
- Current best measured FID in this local continuation is step 143000; step 145000 has slightly better PSNR but a small FID rebound.

H200/Hugging Face eval result:
- Source checkpoint: `sophiaa/MoT-1-checkpoints`, file `weights/latest.pt`.
- Local eval file: `/tmp/mot_hf_latest_eval/weights/latest.pt`.
- Eval recognized `ckpt_step=152385`.
- Eval data path: `/home/heyefei/ImageNet/validation`.
- Eval setting: 50k validation images, mix-only, `mask_selection=router_e2e_dynamic`, `score_normalize_scope=per_image`, `use_model_ema=false`.
- Result: FID 2.23795, PSNR 21.01783, LPIPS 0.20518, L1 0.12797, SSIM 0.53135, tokens 133.49.
- This is the best measured FID so far in the projected ConvNeXt branch, and it also improves PSNR over the local 143000/145000 checkpoints.

Operational notes:
- Training should be launched by the user or collaborator, not automatically by Codex.
- Eval uses `configs/eval_projectedconvnext_50000.yaml` and should report mix metrics only.
- Do not train on validation/test statistics.
- Do not commit private checkpoint files to GitHub; keep them in `weights/` or pull from private Hugging Face storage.
