#!/bin/bash
set -euo pipefail

R2R_ROOT=/project/6101829/draip/R2R
cd "${R2R_ROOT}"
CAMPAIGN=${R2R_CAMPAIGN:-$(date -u +%Y%m%dT%H%M%SZ)}
REWARD_SCALE=${R2R_REWARD_SCALE:-10}
if [[ ${REWARD_SCALE} != 10 && ${REWARD_SCALE} != 32 ]]; then
  echo "R2R_REWARD_SCALE must be 10 or 32" >&2
  exit 2
fi
OUTPUT=experiments/r2r/results/toy_acquisition/${CAMPAIGN}
mkdir -p "${OUTPUT}"
job=$(sbatch --parsable \
  --export=ALL,R2R_CAMPAIGN="${CAMPAIGN}",R2R_REWARD_SCALE="${REWARD_SCALE}" \
  experiments/r2r/slurm_toy_reward10_wm.sh)
printf '%s\n' "${job}" > "${OUTPUT}/job-reward${REWARD_SCALE}.txt"
printf 'campaign=%s job=%s arm=distance8_reward%s_world_model\n' \
  "${CAMPAIGN}" "${job}" "${REWARD_SCALE}"
