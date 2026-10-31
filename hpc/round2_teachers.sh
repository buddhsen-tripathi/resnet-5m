#!/bin/bash
# Rebuild the five round 2 teachers (see README). Needs ACCOUNT and PARTITION set; run from the repo root.
set -eu
submit() { sbatch --account="$ACCOUNT" --partition="$PARTITION" hpc/train.sbatch "$@"; }

C="--widths 80 160 320 --blocks 2 3 2"
submit $C --autoaugment --out checkpoints/C_ta.pt
submit $C --mix 0.5 --out checkpoints/C_mix.pt
submit $C --autoaugment --mix 0.5 --out checkpoints/C_ta_mix.pt
submit $C --autoaugment --mix 0.5 --erase-p 0 --out checkpoints/C_ta_mix_noerase.pt
submit --no-se --autoaugment --mix 0.5 --out checkpoints/A_nose_ta_mix.pt
