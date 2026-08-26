#!/bin/bash
#SBATCH --account=aip-valenzan
#SBATCH --partition=gpubase_l40s_b3
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=1-00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@300
#SBATCH --array=0-1
#SBATCH --job-name=r2r-sparse-distance
#SBATCH --output=experiments/r2r/toy-sparse-distance-%A_%a.out

set -euo pipefail
module load apptainer/1.4.5

CANONICAL_ROOT=/project/6101829/draip/R2R
R2R_ROOT=${R2R_SOURCE_ROOT:-${CANONICAL_ROOT}}
R2R_EXPECTED_COMMIT=${R2R_EXPECTED_COMMIT:?R2R_EXPECTED_COMMIT is required}
R2R_IMAGE=${R2R_IMAGE:-${CANONICAL_ROOT}/.containers/r2i.sif}
RESULT_ROOT=${R2R_RESULT_ROOT:-${CANONICAL_ROOT}/experiments/r2r/results}
CAMPAIGN=${R2R_CAMPAIGN:?R2R_CAMPAIGN is required}
TARGET_STEPS=${R2R_TARGET_STEPS:-60000}
DYN_SCALE=${R2R_DYN_SCALE:-0.05}
RESULT_SERIES=${R2R_RESULT_SERIES:-toy_sparse_dyn_distances}

actual_commit=$(git -C "${R2R_ROOT}" rev-parse HEAD)
if [[ ${actual_commit} != "${R2R_EXPECTED_COMMIT}" ]]; then
  echo "source snapshot ${actual_commit} does not match ${R2R_EXPECTED_COMMIT}" >&2
  exit 2
fi
if [[ -n $(git -C "${R2R_ROOT}" status --porcelain --untracked-files=all) ]]; then
  echo "source snapshot has tracked or non-ignored untracked files" >&2
  exit 2
fi
if [[ ! ${TARGET_STEPS} =~ ^[0-9]+$ ]] || (( TARGET_STEPS < 50000 )); then
  echo "R2R_TARGET_STEPS must be an integer of at least 50000" >&2
  exit 2
fi
python - "${DYN_SCALE}" <<'PY'
import math
import sys

value = float(sys.argv[1])
if not math.isfinite(value) or not 0.0 < value <= 0.05:
  raise SystemExit('R2R_DYN_SCALE must be finite and in (0, 0.05]')
PY

IFS=: read -r -a DISTANCES <<< "${R2R_DISTANCES:-16:32}"
IFS=: read -r -a SEEDS <<< "${R2R_SEEDS:-${R2R_SEED:-0}}"
if (( ${#DISTANCES[@]} == 0 || ${#SEEDS[@]} == 0 )); then
  echo "distance and seed sets must not be empty" >&2
  exit 2
fi
index=${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}
total=$(( ${#DISTANCES[@]} * ${#SEEDS[@]} ))
if (( index < 0 || index >= total )); then
  echo "array index ${index} is outside the ${total}-cell grid" >&2
  exit 2
fi
DISTANCE=${DISTANCES[$(( index % ${#DISTANCES[@]} ))]}
SEED=${SEEDS[$(( index / ${#DISTANCES[@]} ))]}
if [[ ! ${DISTANCE} =~ ^(8|16|32)$ ]]; then
  echo "distance must be 8, 16, or 32" >&2
  exit 2
fi
if [[ ! ${SEED} =~ ^[0-9]+$ ]]; then
  echo "seed must be a nonnegative integer" >&2
  exit 2
fi
if [[ ! ${RESULT_SERIES} =~ ^[a-z0-9_]+$ ]]; then
  echo "R2R_RESULT_SERIES contains invalid characters" >&2
  exit 2
fi

DYN_TAG=${DYN_SCALE//./p}
OUTPUT=${RESULT_ROOT}/${RESULT_SERIES}/${CAMPAIGN}/distance${DISTANCE}_seed${SEED}
REPLAY_DIRECTORY=${SLURM_TMPDIR:?}/r2r-sparse-distance-${CAMPAIGN}-${DISTANCE}-${SEED}
mkdir -p "${OUTPUT}/provenance" "${OUTPUT}/lfs"
env R2R_ROOT="${R2R_ROOT}" "${R2R_ROOT}/experiments/r2r/record_provenance.sh" \
  "${OUTPUT}/provenance"
printf '%s\n' \
  "distance=${DISTANCE}" \
  "seed=${SEED}" \
  "result_series=${RESULT_SERIES}" \
  "target_steps=${TARGET_STEPS}" \
  "dynamics_loss_scale=${DYN_SCALE}" \
  "source_root=${R2R_ROOT}" \
  "expected_commit=${R2R_EXPECTED_COMMIT}" \
  > "${OUTPUT}/provenance/distance.txt"
export WANDB_MODE=offline
export PYTHONHASHSEED=0
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

child=''
forward_requeue() {
  if [[ -n "${child}" ]]; then
    kill -USR1 "${child}" 2>/dev/null || true
  fi
}
trap forward_requeue USR1

set +e
apptainer exec --cleanenv --nv \
  --env PYTHONPATH="${R2R_ROOT}" \
  --env WANDB_MODE=offline \
  --env PYTHONDONTWRITEBYTECODE=1 \
  --bind "${R2R_ROOT}:${R2R_ROOT}" \
  --bind "${CANONICAL_ROOT}:${CANONICAL_ROOT}" \
  --bind "${SLURM_TMPDIR}:${SLURM_TMPDIR}" \
  --pwd "${R2R_ROOT}" \
  "${R2R_IMAGE}" \
  python recall2imagine/train.py \
    --configs "toy_memory,toy_distance${DISTANCE},r2r_w64,toy_full_r2r,toy_balanced_sparse_dyn_cont_memory" \
    --seed "${SEED}" \
    --loss_scales.dyn "${DYN_SCALE}" \
    --run.steps "${TARGET_STEPS}" \
    --toy_arm "balanced_sparse_dyn${DYN_TAG}_full_r2r_distance${DISTANCE}" \
    --logdir "${OUTPUT}" \
    --replay_dir "${REPLAY_DIRECTORY}" \
    --lfs_dir "${OUTPUT}/lfs" \
    --use_lfs True \
    --wdb_name "R2R-toy-sparse-dyn${DYN_TAG}-distance${DISTANCE}-seed${SEED}" &
child=$!
wait "${child}"
status=$?
set -e
if [[ ${status} -ne 0 ]]; then
  if [[ ${status} -eq 75 || ${status} -eq 138 ]]; then
    scontrol requeue "${SLURM_JOB_ID}"
    exit 0
  fi
  exit ${status}
fi

python - "${OUTPUT}/toy_summary.json" "${DISTANCE}" "${TARGET_STEPS}" <<'PY'
import json
import pathlib
import sys

summary = json.loads(pathlib.Path(sys.argv[1]).read_text())
distance = int(sys.argv[2])
target_steps = int(sys.argv[3])
expected_updates = max(0, (target_steps - 4096 + 3) // 4)
expected = {
    'environment_steps': target_steps,
    'learner_updates': expected_updates,
    'cue_query_distance': distance,
    'window': 64,
    'batch_size': 64,
    'objective': 'balanced_sparse_reward_with_native_auxiliaries',
}
for key, value in expected.items():
  if summary.get(key) != value:
    raise SystemExit(
        f'distance {distance}: expected {key}={value!r}, '
        f'got {summary.get(key)!r}')
PY

set +e
python "${R2R_ROOT}/experiments/r2r/check_toy_retention.py" \
  --criterion model \
  --final-step "${TARGET_STEPS}" \
  --metrics "${OUTPUT}/metrics.jsonl" \
  > "${OUTPUT}/model_retention.json"
model_status=$?
python "${R2R_ROOT}/experiments/r2r/check_toy_retention.py" \
  --criterion joint \
  --final-step "${TARGET_STEPS}" \
  --metrics "${OUTPUT}/metrics.jsonl" \
  > "${OUTPUT}/retention.json"
joint_status=$?
set -e
if [[ ${model_status} -eq 0 ]]; then
  touch "${OUTPUT}/MODEL_ROBUST_SUCCESS"
elif [[ ${model_status} -ne 3 ]]; then
  exit ${model_status}
fi
if [[ ${joint_status} -eq 0 ]]; then
  touch "${OUTPUT}/ROBUST_SUCCESS"
elif [[ ${joint_status} -ne 3 ]]; then
  exit ${joint_status}
fi
