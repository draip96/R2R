#!/bin/bash
#SBATCH --account=aip-valenzan
#SBATCH --partition=gpubase_l40s_b3
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=1-00:00
#SBATCH --array=0-19%6
#SBATCH --requeue
#SBATCH --signal=B:USR1@300
#SBATCH --job-name=r2r-bsuite
#SBATCH --output=experiments/r2r/bsuite-%A_%a.out

set -euo pipefail
module load apptainer/1.4.5

R2R_ROOT=/project/6101829/draip/R2R
R2R_IMAGE=${R2R_IMAGE:-${R2R_ROOT}/.containers/r2i.sif}
RESULT_ROOT=${R2R_RESULT_ROOT:-${R2R_ROOT}/experiments/r2r/results}
CAMPAIGN=${R2R_CAMPAIGN:?R2R_CAMPAIGN is required}
PHASE=${R2R_BSUITE_PHASE:-seed0}
TASK_ID=${SLURM_ARRAY_TASK_ID:?array task is required}
WINDOWS=(64 128 256 1024)
HORIZONS=(128 256 512 1024 2048)
if [[ "${PHASE}" == seed0 ]]; then
  window_index=$((TASK_ID / 5))
  horizon_index=$((TASK_ID % 5))
  WINDOW=${WINDOWS[${window_index}]}
  HORIZON=${HORIZONS[${horizon_index}]}
  SEED=0
elif [[ "${PHASE}" == promoted ]]; then
  MANIFEST=${R2R_BSUITE_MANIFEST:?promotion manifest is required}
  cell_index=$((TASK_ID / 2 + 1))
  seed_index=$((TASK_ID % 2))
  read -r WINDOW HORIZON < <(sed -n "${cell_index}p" "${MANIFEST}")
  SEED=$((seed_index + 1))
else
  echo "unknown BSuite phase ${PHASE}" >&2
  exit 2
fi

OUTPUT=${RESULT_ROOT}/bsuite/${CAMPAIGN}/${PHASE}/window${WINDOW}-horizon${HORIZON}-seed${SEED}
REPLAY_DIRECTORY=${SLURM_TMPDIR:?}/r2r-bs-${CAMPAIGN}-${WINDOW}-${HORIZON}-${SEED}
mkdir -p "${OUTPUT}/provenance" "${OUTPUT}/lfs"
R2R_ROOT=${R2R_ROOT} "${R2R_ROOT}/experiments/r2r/record_provenance.sh" \
  "${OUTPUT}/provenance"
export WANDB_MODE=offline
export PYTHONHASHSEED=${SEED}
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
    --configs "bsuite,r2r_w${WINDOW}" \
    --task "toymemory_${HORIZON}" \
    --seed "${SEED}" \
    --logdir "${OUTPUT}" \
    --replay_dir "${REPLAY_DIRECTORY}" \
    --lfs_dir "${OUTPUT}/lfs" \
    --use_lfs True \
    --wdb_name "R2R-bs-T${WINDOW}-H${HORIZON}-s${SEED}" &
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
