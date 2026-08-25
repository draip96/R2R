#!/bin/bash
set -euo pipefail

R2R_ROOT=/project/6101829/draip/R2R
cd "${R2R_ROOT}"
SOURCE_CAMPAIGN=${R2R_SOURCE_CAMPAIGN:?R2R_SOURCE_CAMPAIGN is required}
SOURCE_METRICS=experiments/r2r/results/toy_cont_promotion/${SOURCE_CAMPAIGN}/full_r2r/metrics.jsonl

# Do not spend two more GPU runs on a transient solve. The source distance-8
# full-R2R arm must retain exact actor/model accuracy over five consecutive
# balanced panels and still be solved at 50k before distances 16 and 32 are
# eligible.
python experiments/r2r/check_toy_retention.py --metrics "${SOURCE_METRICS}"

CAMPAIGN=${R2R_CAMPAIGN:-$(date -u +%Y%m%dT%H%M%SZ)}
OUTPUT=experiments/r2r/results/toy_cont_distances/${CAMPAIGN}
mkdir -p "${OUTPUT}"
job=$(sbatch --parsable \
  --export=ALL,R2R_CAMPAIGN="${CAMPAIGN}",R2R_SEED="${R2R_SEED:-0}" \
  experiments/r2r/slurm_toy_cont_distances.sh)
printf '%s\n' "${job}" > "${OUTPUT}/job-array.txt"
printf 'campaign=%s job=%s distances=16,32 seed=%s source_distance8=%s\n' \
  "${CAMPAIGN}" "${job}" "${R2R_SEED:-0}" "${SOURCE_CAMPAIGN}"
