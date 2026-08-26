#!/bin/bash
set -euo pipefail

R2R_ROOT=/project/6101829/draip/R2R
cd "${R2R_ROOT}"
CAMPAIGN=${R2R_CAMPAIGN:-$(date -u +%Y%m%dT%H%M%SZ)}
DYN_SCALE=${R2R_DYN_SCALE:-0.005}
OUTPUT=experiments/r2r/results/toy_sparse_dyn_promotion/${CAMPAIGN}
mkdir -p "${OUTPUT}"
job=$(sbatch --parsable \
  --export=ALL,R2R_CAMPAIGN="${CAMPAIGN}",R2R_DYN_SCALE="${DYN_SCALE}" \
  experiments/r2r/slurm_toy_sparse_dyn_promotion.sh)
printf '%s\n' "${job}" > "${OUTPUT}/job-array.txt"
printf 'campaign=%s job=%s dyn_scale=%s arms=direct_bptt,full_r2r\n' \
  "${CAMPAIGN}" "${job}" "${DYN_SCALE}"
