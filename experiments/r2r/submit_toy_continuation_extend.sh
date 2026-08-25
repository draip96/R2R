#!/bin/bash
set -euo pipefail

R2R_ROOT=/project/6101829/draip/R2R
cd "${R2R_ROOT}"
SOURCE_CAMPAIGN=${R2R_SOURCE_CAMPAIGN:-20260825T215228Z}
TARGET_STEPS=${R2R_TARGET_STEPS:-50000}
OUTPUT=experiments/r2r/results/toy_continuation_audit/${SOURCE_CAMPAIGN}
CONTINUATION=${OUTPUT}/continuations/${TARGET_STEPS}

for arm in cont_shared cont_detached; do
  if [[ ! -f ${OUTPUT}/${arm}/checkpoint.ckpt || \
        ! -f ${OUTPUT}/${arm}/toy_summary.json ]]; then
    echo "missing ${arm} source checkpoint or summary under ${OUTPUT}" >&2
    exit 2
  fi
done
mkdir -p "${CONTINUATION}"
job=$(sbatch --parsable \
  --export=ALL,R2R_SOURCE_CAMPAIGN="${SOURCE_CAMPAIGN}",R2R_TARGET_STEPS="${TARGET_STEPS}" \
  experiments/r2r/slurm_toy_continuation_extend.sh)
printf '%s\n' "${job}" > "${CONTINUATION}/job-${job}.txt"
printf 'source_campaign=%s target_steps=%s job=%s arms=cont_shared,cont_detached\n' \
  "${SOURCE_CAMPAIGN}" "${TARGET_STEPS}" "${job}"
