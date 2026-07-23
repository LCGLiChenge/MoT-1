#!/usr/bin/env bash
set -euo pipefail

# Wait until N GPUs are free, then launch the DINOv1-S GAN=1.5 probe.
# A GPU is considered free when its used memory is <= MAX_USED_MB.

NEEDED_GPUS=${NEEDED_GPUS:-4}
MAX_USED_MB=${MAX_USED_MB:-2000}
CHECK_INTERVAL=${CHECK_INTERVAL:-60}
GPU_CANDIDATES=${GPU_CANDIDATES:-0,1,2,3,4,5,6,7}

CONFIG=${CONFIG:-configs/titok_llamagen_mix_ae_unfreeze_encoder_gan_router_f2d_e2e_dynamic_freeze1d_dinov1sD_gan150_mix2_from132000_4gpu_probe.yaml}
DATA_PATH=${DATA_PATH:-/home/heyefei/ImageNet/train}
ADAPTER_INIT=${ADAPTER_INIT:-/home/heyefei/lichenge/Mixture-of-Tokenizer/version4/results/titok_llamagen_mix_ae_unfreeze_encoder_gan_from_latest_4gpu/step_00066000.pt}
OUTPUT_DIR=$(python3 - "$CONFIG" <<'PYCFG'
import sys
from omegaconf import OmegaConf
cfg = OmegaConf.load(sys.argv[1])
print(cfg.output_dir)
PYCFG
)
mkdir -p "$OUTPUT_DIR"

select_gpus() {
  python3 - "$GPU_CANDIDATES" "$MAX_USED_MB" "$NEEDED_GPUS" <<'PYSEL'
import subprocess
import sys

candidates = [int(x) for x in sys.argv[1].split(',') if x.strip()]
max_used = int(sys.argv[2])
needed = int(sys.argv[3])
raw = subprocess.check_output([
    'nvidia-smi',
    '--query-gpu=index,memory.used',
    '--format=csv,noheader,nounits',
], text=True)
used = {}
for line in raw.strip().splitlines():
    idx, mem = [part.strip() for part in line.split(',')]
    used[int(idx)] = int(mem)
free = [idx for idx in candidates if used.get(idx, 10**9) <= max_used]
print(','.join(map(str, free[:needed])))
PYSEL
}

while true; do
  SELECTED=$(select_gpus)
  COUNT=0
  if [[ -n "$SELECTED" ]]; then
    IFS=',' read -r -a GPU_ARRAY <<< "$SELECTED"
    COUNT=${#GPU_ARRAY[@]}
  fi
  NOW=$(date '+%F %T')
  if [[ "$COUNT" -ge "$NEEDED_GPUS" ]]; then
    echo "[$NOW] launching on GPUs: $SELECTED"
    LOG_FILE="$OUTPUT_DIR/launch_$(date '+%Y%m%d_%H%M%S').log"
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES="$SELECTED" \
    torchrun --standalone --nproc_per_node="$NEEDED_GPUS" train_titok_llamagen_decoder_adapt_router_f2d_e2e_dynamic.py \
      --config "$CONFIG" \
      --data-path "$DATA_PATH" \
      --adapter-init "$ADAPTER_INIT" \
      2>&1 | tee "$LOG_FILE"
    exit ${PIPESTATUS[0]}
  fi
  echo "[$NOW] waiting: need $NEEDED_GPUS GPUs with <= ${MAX_USED_MB}MiB used; candidates=$GPU_CANDIDATES; selected=$SELECTED"
  sleep "$CHECK_INTERVAL"
done
