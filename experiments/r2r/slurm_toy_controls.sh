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
#SBATCH --job-name=r2r-toy-controls
#SBATCH --output=experiments/r2r/toy-controls-%A_%a.out

set -euo pipefail
module load apptainer/1.4.5

R2R_ROOT=/project/6101829/draip/R2R
R2R_IMAGE=${R2R_IMAGE:-${R2R_ROOT}/.containers/r2i.sif}
RESULT_ROOT=${R2R_RESULT_ROOT:-${R2R_ROOT}/experiments/r2r/results}
CAMPAIGN=${R2R_CAMPAIGN:?R2R_CAMPAIGN is required}

case "${SLURM_ARRAY_TASK_ID}" in
  0)
    ARM=terminal_reward_falsifier
    WINDOW_CONFIG=r2r_w64
    ARM_CONFIG=toy_terminal_reward_falsifier
    ;;
  1)
    ARM=r2i_w1024
    WINDOW_CONFIG=r2r_w1024
    ARM_CONFIG=toy_r2i_reference
    ;;
  *)
    echo "array index must be 0 or 1" >&2
    exit 2
    ;;
esac

OUTPUT=${RESULT_ROOT}/toy_controls/${CAMPAIGN}/${ARM}
REPLAY_DIRECTORY=${SLURM_TMPDIR:?}/r2r-toy-controls-${CAMPAIGN}-${ARM}
mkdir -p "${OUTPUT}/provenance" "${OUTPUT}/lfs"
R2R_ROOT=${R2R_ROOT} "${R2R_ROOT}/experiments/r2r/record_provenance.sh" \
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
    --configs "toy_memory,toy_distance8,${WINDOW_CONFIG},${ARM_CONFIG}" \
    --seed 0 \
    --logdir "${OUTPUT}" \
    --replay_dir "${REPLAY_DIRECTORY}" \
    --lfs_dir "${OUTPUT}/lfs" \
    --use_lfs True \
    --wdb_name "R2R-toy-controls-${ARM}" &
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

python - "${OUTPUT}/toy_summary.json" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
summary = json.loads(path.read_text())
if summary['environment_steps'] != 25000:
  raise SystemExit(
      f"control arm ended at {summary['environment_steps']} rather than 25000")
if summary['learner_updates'] != 5226:
  raise SystemExit(
      f"control arm used {summary['learner_updates']} rather than 5226 updates")
PY
