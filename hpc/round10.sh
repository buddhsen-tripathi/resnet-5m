#!/bin/bash
# Round 10: two R9-recipe students distilled from 12 teachers, one per GPU job.
#   R10_t12   : R9 recipe with 12 teachers (control)
#   R10_shift : R10_t12 plus resampling and brightness/contrast augmentation
# Each run is a chain of SEGMENTS 4-hour jobs in case one is not enough (e.g. on an L4);
# every segment resumes from the last saved epoch, and segments after completion exit at once.
# Needs ACCOUNT and PARTITION; run from the repo root on the burst /scratch after hpc/setup.sh,
# from a terminal on a burst compute node (e.g. a Jupyter terminal in Open OnDemand).
set -eu
unset ${!SLURM_@}  # don't leak the Jupyter session's allocation into the submitted jobs
SEGMENTS=${SEGMENTS:-2}
TEACHERS="teachers/S2_highres.pt teachers/KD_C_aug.pt teachers/C_ta.pt teachers/C_mix.pt
  teachers/C_ta_mix.pt teachers/C_ta_mix_noerase.pt teachers/A_nose_ta_mix.pt teachers/S4_deep.pt
  teachers/KD5_full.pt teachers/KD5_full_cutout.pt weights/R8_base_300.pt weights/R9_combined.pt"
for t in $TEACHERS; do test -s "$t" || { echo "missing teacher $t; run hpc/setup.sh" >&2; exit 1; }; done
RECIPE="--widths 128 192 228 --blocks 4 3 2 --no-se --augmix --mix 0.5 --erase-p 0 --cutout --cutout-size 8
  --supcon 0.1 --supcon-temp 0.07 --seed 1 --teacher-flip --kd-temp 4 --kd-alpha 0.7 --epochs 200
  --batch-size 256 --lr 0.2 --wd 0.0005 --warmup-epochs 5 --label-smoothing 0"

SHIFT="--resample 0.5 --resample-size 48 64 --jitter 0.2 --jitter-p 0.5"
if [ "${SMOKE:-0}" = 1 ]; then  # one epoch of the full R10_shift recipe, to check GPU, memory and speed
  rm -f checkpoints/smoke.pt
  sbatch --account="$ACCOUNT" --partition="$PARTITION" --job-name=smoke --time=00:30:00 \
    hpc/train.sbatch $RECIPE $SHIFT --epochs 1 --teacher $TEACHERS --out checkpoints/smoke.pt
  exit
fi

chain() {  # name, extra train.py args...
  local name=$1 dep="" id; shift
  for i in $(seq 1 "$SEGMENTS"); do
    id=$(sbatch --parsable --account="$ACCOUNT" --partition="$PARTITION" --job-name="$name" $dep \
      hpc/train.sbatch $RECIPE "$@" --teacher $TEACHERS --out "checkpoints/$name.pt")
    dep="--dependency=afterany:$id"
  done
  echo "$id"
}

last_t12=$(chain R10_t12)
last_shift=$(chain R10_shift $SHIFT)
sbatch --account="$ACCOUNT" --partition="$PARTITION" --dependency="afterany:$last_t12:$last_shift" \
  hpc/evaluate.sbatch weights/R9_combined.pt checkpoints/R10_t12.pt checkpoints/R10_shift.pt
squeue --me
