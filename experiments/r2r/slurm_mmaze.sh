#!/bin/bash
#SBATCH --account=aip-valenzan
#SBATCH --partition=gpubase_h100_b3
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=48
#SBATCH --mem=256G
#SBATCH --time=1-00:00
#SBATCH --array=0-3
#SBATCH --requeue
#SBATCH --signal=B:USR1@600
#SBATCH --job-name=r2r-mmaze
#SBATCH --output=experiments/r2r/mmaze-%A_%a.out

set -euo pipefail
module load apptainer/1.4.5

R2R_ROOT=/project/6101829/draip/R2R
R2R_IMAGE=${R2R_IMAGE:-${R2R_ROOT}/.containers/r2i.sif}
RESULT_ROOT=${R2R_RESULT_ROOT:-${R2R_ROOT}/experiments/r2r/results}
CAMPAIGN=${R2R_CAMPAIGN:?R2R_CAMPAIGN is required}
BSUITE_ROOT=${RESULT_ROOT}/bsuite/${CAMPAIGN}
test -f "${BSUITE_ROOT}/MEMORY_MAZE_UNLOCKED"
PHASE=${R2R_MMAZE_PHASE:-seed0}
TASK_ID=${SLURM_ARRAY_TASK_ID:?array task is required}
WINDOWS=(64 128 256 1024)
if [[ "${PHASE}" == seed0 ]]; then
  WINDOW=${WINDOWS[${TASK_ID}]}
  SEED=0
elif [[ "${PHASE}" == promoted ]]; then
  MANIFEST=${R2R_MMAZE_MANIFEST:?qualified window manifest is required}
  window_index=$((TASK_ID / 2 + 1))
  seed_index=$((TASK_ID % 2))
  WINDOW=$(sed -n "${window_index}p" "${MANIFEST}")
  SEED=$((seed_index + 1))
else
  echo "unknown Memory Maze phase ${PHASE}" >&2
  exit 2
fi

OUTPUT=${RESULT_ROOT}/mmaze/${CAMPAIGN}/${PHASE}/window${WINDOW}-seed${SEED}
REPLAY_DIRECTORY=${SLURM_TMPDIR:?}/r2r-mm-${CAMPAIGN}-${WINDOW}-${SEED}
mkdir -p "${OUTPUT}/provenance" "${OUTPUT}/lfs" "${OUTPUT}/checkpoints"
R2R_ROOT=${R2R_ROOT} "${R2R_ROOT}/experiments/r2r/record_provenance.sh" \
  "${OUTPUT}/provenance"
export WANDB_MODE=offline
export PYTHONHASHSEED=${SEED}
export OMP_NUM_THREADS=1
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
    --configs "mmaze,r2r_w${WINDOW}" \
    --seed "${SEED}" \
    --logdir "${OUTPUT}" \
    --checkpoint_dir "${OUTPUT}/checkpoints" \
    --replay_dir "${REPLAY_DIRECTORY}" \
    --lfs_dir "${OUTPUT}/lfs" \
    --use_lfs True \
    --wdb_name "R2R-mmaze-T${WINDOW}-s${SEED}" &
child=$!
wait "${child}"
status=$?
set -e
if [[ ${status} -eq 75 || ${status} -eq 138 ]]; then
  scontrol requeue "${SLURM_JOB_ID}"
  exit 0
fi
if [[ ${status} -ne 0 ]]; then
  exit ${status}
fi
test -f "${OUTPUT}/COMPLETE"
