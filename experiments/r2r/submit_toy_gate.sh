#!/bin/bash
set -euo pipefail

R2R_ROOT=/project/6101829/draip/R2R
cd "${R2R_ROOT}"
CAMPAIGN=${R2R_CAMPAIGN:-$(date -u +%Y%m%dT%H%M%SZ)}
first=$(sbatch --parsable \
  --export=ALL,R2R_CAMPAIGN="${CAMPAIGN}",R2R_TOY_HORIZON=128 \
  experiments/r2r/slurm_toy.sh)
controller=$(sbatch --parsable \
  --dependency=afterok:${first} \
  --export=ALL,R2R_CAMPAIGN="${CAMPAIGN}",R2R_FIRST_JOB="${first}" \
  experiments/r2r/submit_toy_256_after_gate.sh)
mkdir -p "experiments/r2r/results/toy/${CAMPAIGN}"
printf '%s\n' "${first}" > "experiments/r2r/results/toy/${CAMPAIGN}/job128.txt"
printf '%s\n' "${controller}" > \
  "experiments/r2r/results/toy/${CAMPAIGN}/controller.txt"
printf '%s\n' "${CAMPAIGN} ${first} ${controller}"
