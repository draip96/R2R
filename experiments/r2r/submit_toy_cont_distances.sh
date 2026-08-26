#!/bin/bash
set -euo pipefail

CANONICAL_ROOT=/project/6101829/draip/R2R
cd "${CANONICAL_ROOT}"

if [[ -n $(git status --porcelain --untracked-files=all) ]]; then
  echo "commit tracked changes before creating the immutable run snapshot" >&2
  exit 2
fi

EXPECTED_COMMIT=$(git rev-parse HEAD)
SNAPSHOT_ROOT=${R2R_SNAPSHOT_ROOT:-/project/6101829/draip/.r2r_run_sources/${EXPECTED_COMMIT}}
if [[ -e ${SNAPSHOT_ROOT} ]]; then
  snapshot_commit=$(git -C "${SNAPSHOT_ROOT}" rev-parse HEAD)
  if [[ ${snapshot_commit} != "${EXPECTED_COMMIT}" ]]; then
    echo "existing snapshot ${SNAPSHOT_ROOT} is at ${snapshot_commit}" >&2
    exit 2
  fi
  if [[ -n $(git -C "${SNAPSHOT_ROOT}" status --porcelain --untracked-files=all) ]]; then
    echo "existing snapshot ${SNAPSHOT_ROOT} is not clean" >&2
    exit 2
  fi
else
  mkdir -p "$(dirname "${SNAPSHOT_ROOT}")"
  git worktree add --detach "${SNAPSHOT_ROOT}" "${EXPECTED_COMMIT}"
fi

SOURCE_RUN=${R2R_SOURCE_RUN:-experiments/r2r/results/toy_sparse_dyn_falsifier/20260826T013224Z/full_r2r_dyn005}
if [[ ${SOURCE_RUN} != /* ]]; then
  SOURCE_RUN=${CANONICAL_ROOT}/${SOURCE_RUN}
fi
SOURCE_FINAL_STEP=${R2R_SOURCE_FINAL_STEP:-60000}
SOURCE_DYN_SCALE=${R2R_SOURCE_DYN_SCALE:-0.05}
SOURCE_VALIDATION=$(python experiments/r2r/validate_toy_r2r_run.py \
  --source "${SOURCE_RUN}" \
  --distance 8 --seed 0 --final-step "${SOURCE_FINAL_STEP}" \
  --dyn-scale "${SOURCE_DYN_SCALE}")
printf '%s\n' "${SOURCE_VALIDATION}"
SOURCE_SUMMARY_SHA256=$(sha256sum "${SOURCE_RUN}/toy_summary.json" | cut -d ' ' -f 1)
SOURCE_CONFIG_SHA256=$(sha256sum "${SOURCE_RUN}/config.yaml" | cut -d ' ' -f 1)
SOURCE_METRICS_SHA256=$(sha256sum "${SOURCE_RUN}/metrics.jsonl" | cut -d ' ' -f 1)
SOURCE_TRAINING_COMMIT=$(tr -d '[:space:]' < "${SOURCE_RUN}/provenance/commit.txt")

CAMPAIGN=${R2R_CAMPAIGN:-$(date -u +%Y%m%dT%H%M%SZ)}
TARGET_STEPS=${R2R_TARGET_STEPS:-60000}
DYN_SCALE=${R2R_DYN_SCALE:-0.05}
SEED=${R2R_SEED:-0}
OUTPUT=experiments/r2r/results/toy_sparse_dyn_distances/${CAMPAIGN}
mkdir -p "${OUTPUT}"
printf '%s\n' "${SOURCE_VALIDATION}" \
  > "${OUTPUT}/source-distance8-validation.json"
job=$(sbatch --parsable \
  --export=ALL,R2R_CAMPAIGN="${CAMPAIGN}",R2R_SEED="${SEED}",R2R_TARGET_STEPS="${TARGET_STEPS}",R2R_DYN_SCALE="${DYN_SCALE}",R2R_SOURCE_ROOT="${SNAPSHOT_ROOT}",R2R_EXPECTED_COMMIT="${EXPECTED_COMMIT}" \
  experiments/r2r/slurm_toy_cont_distances.sh)
printf '%s\n' "${job}" > "${OUTPUT}/job-array.txt"
printf '%s\n' \
  "campaign=${CAMPAIGN}" \
  "job=${job}" \
  "distances=16,32" \
  "seed=${SEED}" \
  "target_steps=${TARGET_STEPS}" \
  "dynamics_loss_scale=${DYN_SCALE}" \
  "source_distance8=${SOURCE_RUN}" \
  "source_final_step=${SOURCE_FINAL_STEP}" \
  "source_dynamics_loss_scale=${SOURCE_DYN_SCALE}" \
  "source_training_commit=${SOURCE_TRAINING_COMMIT}" \
  "source_summary_sha256=${SOURCE_SUMMARY_SHA256}" \
  "source_config_sha256=${SOURCE_CONFIG_SHA256}" \
  "source_metrics_sha256=${SOURCE_METRICS_SHA256}" \
  "expected_commit=${EXPECTED_COMMIT}" \
  "source_snapshot=${SNAPSHOT_ROOT}" \
  > "${OUTPUT}/submission.txt"
printf 'campaign=%s job=%s distances=16,32 seed=%s target=%s dyn=%s commit=%s\n' \
  "${CAMPAIGN}" "${job}" "${SEED}" "${TARGET_STEPS}" "${DYN_SCALE}" \
  "${EXPECTED_COMMIT}"
