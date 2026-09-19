# E83: E72 second-pass rate-drift compensation

## Question

Can a second, rate-only compensation place the independent E72 binary encoder
at mean K=96 on ImageNet-val while retaining FID below 1.2?

E82 calibrated the complete ImageNet-train score distribution to mean K=90,
then produced mean K=98.4832 on val50k.  Before running E83, the next train
target is frozen by the additive rate correction

```text
87.5168 = 90.0 - (98.4832 - 96.0).
```

This is validation-guided hyperparameter tuning.  It uses only E82's aggregate
token mean, not E82 FID, LPIPS, L1, MSE, PSNR, SSIM, per-image scores, or
reconstructions.  A paper must disclose this tuning and must not describe the
result as an untouched-test estimate.

## Frozen method

- Candidate budgets: exactly `{64,128}`.
- Per-image score: `float32((G128-G64)/64)`.
- Price source: the complete 1,281,167-image ImageNet-train score population.
- Train target: exactly `87.5168` mean refinement tokens.
- Decision: K128 iff `score > frozen_price`; ties select K64.
- Gain definition, component scales and weights, amplitude power, Router grid
  order, one K256 endpoint probe, model checkpoint, EMA state and precision:
  identical to E72/E82.
- After the scalar price is frozen, each image is encoded independently and no
  batch statistic is read.

The price must be written and hashed before E83 reads any validation image.
Only one small smoke may precede one formal val50k evaluation.

## Success gate

```text
95.5 <= actual val mean K <= 96.5
and val50k FID-2048 < 1.2
```

LPIPS-Alex, L1, MSE, PSNR and SSIM must all be reported.  Failure may not be
repaired by changing the E83 price after examining the formal result.

## Artifact contract

Only aggregate JSON may be written.  Do not save images, reconstructions,
features, score arrays, NPZ files or `stats.pt`.
