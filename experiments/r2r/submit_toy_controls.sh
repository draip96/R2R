#!/bin/bash
set -euo pipefail

R2R_ROOT=/project/6101829/draip/R2R
cd "${R2R_ROOT}"
CAMPAIGN=${R2R_CAMPAIGN:-$(date -u +%Y%m%dT%H%M%SZ)}
CAMPAIGN_DIR=experiments/r2r/results/toy_controls/${CAMPAIGN}
mkdir -p "${CAMPAIGN_DIR}"

job=$(sbatch --parsable \
  --export=ALL,R2R_CAMPAIGN="${CAMPAIGN}" \
  experiments/r2r/slurm_toy_controls.sh)
printf '%s\n' "${job}" > "${CAMPAIGN_DIR}/job-array.txt"
printf 'campaign=%s job=%s arms=terminal_reward_falsifier,r2i_w1024\n' \
  "${CAMPAIGN}" "${job}"
