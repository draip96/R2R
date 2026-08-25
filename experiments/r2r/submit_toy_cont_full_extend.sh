#!/bin/bash
set -euo pipefail

R2R_ROOT=/project/6101829/draip/R2R
cd "${R2R_ROOT}"
SOURCE_CAMPAIGN=${R2R_SOURCE_CAMPAIGN:?R2R_SOURCE_CAMPAIGN is required}
TARGET_STEPS=${R2R_TARGET_STEPS:-100000}
OUTPUT=experiments/r2r/results/toy_cont_promotion/${SOURCE_CAMPAIGN}/full_r2r
CONTINUATION=${OUTPUT}/continuations/${TARGET_STEPS}

if [[ ! -f ${OUTPUT}/checkpoint.ckpt || ! -f ${OUTPUT}/toy_summary.json ]]; then
  echo "missing full-R2R source checkpoint or summary under ${OUTPUT}" >&2
  exit 2
fi
mkdir -p "${CONTINUATION}"
job=$(sbatch --parsable \
  --export=ALL,R2R_SOURCE_CAMPAIGN="${SOURCE_CAMPAIGN}",R2R_TARGET_STEPS="${TARGET_STEPS}" \
  experiments/r2r/slurm_toy_cont_full_extend.sh)
printf '%s\n' "${job}" > "${CONTINUATION}/job-${job}.txt"
printf 'source_campaign=%s target_steps=%s job=%s arm=full_r2r\n' \
  "${SOURCE_CAMPAIGN}" "${TARGET_STEPS}" "${job}"
