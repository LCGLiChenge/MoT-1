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
