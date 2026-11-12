#!/bin/bash
# Round 11 on one on-demand H200: two wide teachers in parallel, then the R9 student with 12 teachers.
# Run from the repo root on the VM after uploading this working tree (it has --allow-over-budget) and
# the ten teacher checkpoints into checkpoints/. Each stage is a tmux session; checkpoints are written
# after every epoch, so a stopped stage is continued by rerunning its command with --resume added.
#
#   bash .context/round11/run_h200.sh teachers   # stage 1, ~2 h: T_wide_s1 and T_wide_s2 side by side
#   bash .context/round11/run_h200.sh student    # stage 2, ~1.7 h: R11_wide, after both teachers finished
#   bash .context/round11/run_h200.sh finish     # evaluate, predict raw + balanced, ready to submit
set -euo pipefail

TEN=(
  checkpoints/S2_highres.pt checkpoints/KD_C_aug.pt
  checkpoints/C_ta.pt checkpoints/C_mix.pt checkpoints/C_ta_mix.pt
  checkpoints/C_ta_mix_noerase.pt checkpoints/A_nose_ta_mix.pt
  checkpoints/S4_deep.pt checkpoints/KD5_full.pt checkpoints/KD5_full_cutout.pt
)
WIDE=(checkpoints/T_wide_s1.pt checkpoints/T_wide_s2.pt)
# WideResNet-28-10 equivalent in our pre-activation ResNet: 36,688,170 params, our round 2 augmentation, no KD.
TEACHER=(--widths 160 320 640 --blocks 4 4 4 --no-se --autoaugment --mix 0.5 --erase-p 0
         --epochs 250 --workers 6 --allow-over-budget)
# Exact R9 recipe; only the teacher list changes.
STUDENT=(--widths 128 192 228 --blocks 4 3 2 --no-se --augmix --mix 0.5 --erase-p 0 --cutout --cutout-size 8
         --supcon 0.1 --supcon-temp 0.07 --teacher-flip --kd-alpha 0.7 --kd-temp 4 --label-smoothing 0
         --epochs 200 --seed 1 --workers 8)

case "${1:-}" in
  teachers)
    for t in "${TEN[@]}"; do test -s "$t" || { echo "missing $t" >&2; exit 1; }; done
    tmux new-session -d -s teacher1 "uv run python train.py ${TEACHER[*]} --seed 1 --out ${WIDE[0]} 2>&1 | tee logs/T_wide_s1.log"
    tmux new-session -d -s teacher2 "uv run python train.py ${TEACHER[*]} --seed 2 --out ${WIDE[1]} 2>&1 | tee logs/T_wide_s2.log"
    echo "started tmux sessions teacher1 and teacher2; watch with: tail -f logs/T_wide_s1.log"
    echo "timing rule: after 3 epochs, if an epoch takes more than 30 s, stop both and restart with --epochs 200" ;;
  student)
    for t in "${WIDE[@]}"; do
      test -s "$t" || { echo "missing $t" >&2; exit 1; }
      uv run python -c "import torch,sys; c=torch.load('$t',map_location='cpu'); assert c['epoch']==c['train_args']['epochs'], '$t is not finished'"
    done
    tmux new-session -d -s student "uv run python train.py ${STUDENT[*]} --teacher ${TEN[*]} ${WIDE[*]} --out checkpoints/R11_wide.pt 2>&1 | tee logs/R11_wide.log"
    echo "started tmux session student; watch with: tail -f logs/R11_wide.log" ;;
  finish)
    uv run python evaluate.py weights/R9_combined.pt checkpoints/R11_wide.pt "${WIDE[@]}"
    # Balanced decoding scored 0.893 vs 0.901 for R9 on 2026-11-06, so only the raw flip-TTA submission is produced.
    uv run python predict.py --checkpoint checkpoints/R11_wide.pt --out submission_R11_wide.csv
    sha256sum checkpoints/R11_wide.pt "${WIDE[@]}" submission_R11_wide.csv ;;
  *) echo "usage: $0 teachers|student|finish" >&2; exit 1 ;;
esac
