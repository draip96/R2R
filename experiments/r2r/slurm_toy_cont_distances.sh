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
#SBATCH --job-name=r2r-cont-distance
#SBATCH --output=experiments/r2r/toy-cont-distance-%A_%a.out

set -euo pipefail
module load apptainer/1.4.5

R2R_ROOT=/project/6101829/draip/R2R
R2R_IMAGE=${R2R_IMAGE:-${R2R_ROOT}/.containers/r2i.sif}
RESULT_ROOT=${R2R_RESULT_ROOT:-${R2R_ROOT}/experiments/r2r/results}
CAMPAIGN=${R2R_CAMPAIGN:?R2R_CAMPAIGN is required}
SEED=${R2R_SEED:-0}

case "${SLURM_ARRAY_TASK_ID}" in
  0) DISTANCE=16 ;;
  1) DISTANCE=32 ;;
  *)
    echo "array index must be 0 or 1" >&2
    exit 2
    ;;
esac

OUTPUT=${RESULT_ROOT}/toy_cont_distances/${CAMPAIGN}/distance${DISTANCE}_seed${SEED}
REPLAY_DIRECTORY=${SLURM_TMPDIR:?}/r2r-cont-distance-${CAMPAIGN}-${DISTANCE}-${SEED}
mkdir -p "${OUTPUT}/provenance" "${OUTPUT}/lfs"
env R2R_ROOT="${R2R_ROOT}" "${R2R_ROOT}/experiments/r2r/record_provenance.sh" \
  "${OUTPUT}/provenance"
export WANDB_MODE=offline
export PYTHONHASHSEED=0
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
  --bind "${R2R_ROOT}:${R2R_ROOT}" \
  --bind "${SLURM_TMPDIR}:${SLURM_TMPDIR}" \
  --pwd "${R2R_ROOT}" \
  "${R2R_IMAGE}" \
  python recall2imagine/train.py \
    --configs "toy_memory,toy_distance${DISTANCE},r2r_w64,toy_full_r2r,toy_balanced_terminal_cont_memory" \
    --seed "${SEED}" \
    --run.steps 50000 \
    --toy_arm "balanced_terminal_cont_full_r2r_distance${DISTANCE}" \
    --logdir "${OUTPUT}" \
    --replay_dir "${REPLAY_DIRECTORY}" \
    --lfs_dir "${OUTPUT}/lfs" \
    --use_lfs True \
    --wdb_name "R2R-toy-balanced-cont-distance${DISTANCE}-seed${SEED}" &
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

python - "${OUTPUT}/toy_summary.json" "${DISTANCE}" <<'PY'
import json
import pathlib
import sys

summary = json.loads(pathlib.Path(sys.argv[1]).read_text())
distance = int(sys.argv[2])
if summary['environment_steps'] != 50000:
  raise SystemExit(
      f"distance {distance} ended at {summary['environment_steps']} rather than 50000")
if summary['learner_updates'] != 11476:
  raise SystemExit(
      f"distance {distance} used {summary['learner_updates']} rather than 11476 updates")
if summary['cue_query_distance'] != distance:
  raise SystemExit(
      f"expected distance {distance}, got {summary['cue_query_distance']}")
if summary['objective'] != 'balanced_terminal_reward_with_native_auxiliaries':
  raise SystemExit(f"unexpected objective {summary['objective']!r}")
PY

python experiments/r2r/check_toy_retention.py \
  --metrics "${OUTPUT}/metrics.jsonl" \
  > "${OUTPUT}/retention.json"
touch "${OUTPUT}/ROBUST_SUCCESS"
