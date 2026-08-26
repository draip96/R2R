#!/bin/bash
set -euo pipefail

CANONICAL_ROOT=/project/6101829/draip/R2R
cd "${CANONICAL_ROOT}"
if [[ -n $(git status --porcelain --untracked-files=all) ]]; then
  echo "commit tracked and non-ignored files before submitting" >&2
  exit 2
fi

LAUNCHER_COMMIT=$(git rev-parse HEAD)
TRAINING_COMMIT=${R2R_TRAINING_COMMIT:-1b503a5fe3db27c2d8e59bd0695c46d9baf71e5d}
if ! git cat-file -e "${TRAINING_COMMIT}^{commit}"; then
  echo "unknown training commit ${TRAINING_COMMIT}" >&2
  exit 2
fi
SNAPSHOT_ROOT=${R2R_SNAPSHOT_ROOT:-/project/6101829/draip/.r2r_run_sources/${TRAINING_COMMIT}}
if [[ -e ${SNAPSHOT_ROOT} ]]; then
  snapshot_commit=$(git -C "${SNAPSHOT_ROOT}" rev-parse HEAD)
  if [[ ${snapshot_commit} != "${TRAINING_COMMIT}" ]]; then
    echo "existing snapshot ${SNAPSHOT_ROOT} is at ${snapshot_commit}" >&2
    exit 2
  fi
  if [[ -n $(git -C "${SNAPSHOT_ROOT}" status --porcelain --untracked-files=all) ]]; then
    echo "existing training snapshot is not clean" >&2
    exit 2
  fi
else
  mkdir -p "$(dirname "${SNAPSHOT_ROOT}")"
  git worktree add --detach "${SNAPSHOT_ROOT}" "${TRAINING_COMMIT}"
fi

SEED0_CAMPAIGN=${R2R_SEED0_CAMPAIGN:-20260826T160354Z}
DISTANCE8_SOURCE=${R2R_DISTANCE8_SOURCE:-experiments/r2r/results/toy_sparse_dyn_falsifier/20260826T013224Z/full_r2r_dyn005}
if [[ ${DISTANCE8_SOURCE} != /* ]]; then
  DISTANCE8_SOURCE=${CANONICAL_ROOT}/${DISTANCE8_SOURCE}
fi
CAMPAIGN=${R2R_CAMPAIGN:-$(date -u +%Y%m%dT%H%M%SZ)}
OUTPUT=experiments/r2r/results/toy_sparse_dyn_seed_audit/${CAMPAIGN}
mkdir -p "${OUTPUT}/seed0_gate"

python experiments/r2r/validate_toy_r2r_run.py \
  --source "${DISTANCE8_SOURCE}" \
  --distance 8 --seed 0 --final-step 60000 --dyn-scale 0.05 \
  > "${OUTPUT}/seed0_gate/distance8.json"
for distance in 16 32; do
  source_run=${CANONICAL_ROOT}/experiments/r2r/results/toy_sparse_dyn_distances/${SEED0_CAMPAIGN}/distance${distance}_seed0
  python experiments/r2r/validate_toy_r2r_run.py \
    --source "${source_run}" \
    --distance "${distance}" --seed 0 --final-step 60000 --dyn-scale 0.05 \
    --expected-commit "${TRAINING_COMMIT}" \
    > "${OUTPUT}/seed0_gate/distance${distance}.json"
done

job=$(sbatch --parsable --array=0-5 \
  --export=ALL,R2R_CAMPAIGN="${CAMPAIGN}",R2R_TARGET_STEPS=60000,R2R_DYN_SCALE=0.05,R2R_DISTANCES=8:16:32,R2R_SEEDS=1:2,R2R_RESULT_SERIES=toy_sparse_dyn_seed_audit,R2R_SOURCE_ROOT="${SNAPSHOT_ROOT}",R2R_EXPECTED_COMMIT="${TRAINING_COMMIT}" \
  experiments/r2r/slurm_toy_cont_distances.sh)
printf '%s\n' "${job}" > "${OUTPUT}/job-array.txt"
printf '%s\n' \
  "campaign=${CAMPAIGN}" \
  "job=${job}" \
  "distances=8,16,32" \
  "seeds=1,2" \
  "target_steps=60000" \
  "dynamics_loss_scale=0.05" \
  "seed0_campaign=${SEED0_CAMPAIGN}" \
  "launcher_commit=${LAUNCHER_COMMIT}" \
  "training_commit=${TRAINING_COMMIT}" \
  "source_snapshot=${SNAPSHOT_ROOT}" \
  > "${OUTPUT}/submission.txt"
printf 'campaign=%s job=%s distances=8,16,32 seeds=1,2 training_commit=%s\n' \
  "${CAMPAIGN}" "${job}" "${TRAINING_COMMIT}"
