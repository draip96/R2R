#!/bin/bash
set -euo pipefail

R2R_ROOT=/project/6101829/draip/R2R
cd "${R2R_ROOT}"
CAMPAIGN=${R2R_CAMPAIGN:-$(date -u +%Y%m%dT%H%M%SZ)}
OUTPUT=experiments/r2r/results/toy_acquisition/${CAMPAIGN}
mkdir -p "${OUTPUT}"
job=$(sbatch --parsable \
  --export=ALL,R2R_CAMPAIGN="${CAMPAIGN}" \
  experiments/r2r/slurm_toy_terminal_weight_wm.sh)
printf '%s\n' "${job}" > "${OUTPUT}/job-terminal-weight-array.txt"
printf 'campaign=%s job=%s arms=terminal_weight10,terminal_weight100\n' \
  "${CAMPAIGN}" "${job}"
