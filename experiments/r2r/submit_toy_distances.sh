#!/bin/bash
set -euo pipefail

R2R_ROOT=/project/6101829/draip/R2R
cd "${R2R_ROOT}"
CAMPAIGN=${R2R_CAMPAIGN:-$(date -u +%Y%m%dT%H%M%SZ)}
CAMPAIGN_DIR=experiments/r2r/results/toy/${CAMPAIGN}
mkdir -p "${CAMPAIGN_DIR}"

for distance in 8 16 32 64; do
  job=$(sbatch --parsable \
    --export=ALL,R2R_CAMPAIGN="${CAMPAIGN}",R2R_TOY_DISTANCE="${distance}" \
    experiments/r2r/slurm_toy.sh)
  printf '%s\n' "${job}" > "${CAMPAIGN_DIR}/job-distance${distance}.txt"
  printf 'distance=%s job=%s\n' "${distance}" "${job}"
done
printf 'campaign=%s\n' "${CAMPAIGN}"
