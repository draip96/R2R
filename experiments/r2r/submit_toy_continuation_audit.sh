#!/bin/bash
set -euo pipefail

R2R_ROOT=/project/6101829/draip/R2R
cd "${R2R_ROOT}"
CAMPAIGN=${R2R_CAMPAIGN:-$(date -u +%Y%m%dT%H%M%SZ)}
OUTPUT=experiments/r2r/results/toy_continuation_audit/${CAMPAIGN}
mkdir -p "${OUTPUT}"
job=$(sbatch --parsable \
  --export=ALL,R2R_CAMPAIGN="${CAMPAIGN}" \
  experiments/r2r/slurm_toy_continuation_audit.sh)
printf '%s\n' "${job}" > "${OUTPUT}/job-array.txt"
printf 'campaign=%s job=%s arms=cont_shared,cont_detached\n' \
  "${CAMPAIGN}" "${job}"
