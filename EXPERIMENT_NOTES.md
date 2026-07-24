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
- `lr_d=5e-5`, `d_warmup_steps=500`
- `use_ema=true`, `ema_decay=0.999`
- `latest.pt` is updated every epoch.
- Extra step checkpoints are saved at explicit steps in the yaml.

Known local result before H200 migration:
- Projected ConvNeXt branch around 136500-137000 gave the best recent FID/PSNR tradeoff.
- 50k val EMA metrics:
  - step 136500: FID 2.51763, PSNR 20.92721, LPIPS 0.20577, tokens 133.63
  - step 137000: FID 2.51945, PSNR 20.93426, LPIPS 0.20546, tokens 133.64
- Later local checkpoints 139000/141000/143000/145000 were produced, but this machine's CUDA driver/NVML broke before eval could run.

Operational notes:
- Training should be launched by the user or collaborator, not automatically by Codex.
- Eval uses `configs/eval_projectedconvnext_50000.yaml` and should report mix metrics only.
- Do not train on validation/test statistics.
- Do not commit private checkpoint files to GitHub; keep them in `weights/` or pull from private Hugging Face storage.
