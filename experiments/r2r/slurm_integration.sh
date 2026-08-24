#!/bin/bash
#SBATCH --account=aip-valenzan
#SBATCH --partition=gpubase_l40s_b2
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=96G
#SBATCH --time=0-04:00
#SBATCH --array=0-1
#SBATCH --job-name=r2r-integration
#SBATCH --output=experiments/r2r/integration-%A_%a.out

set -euo pipefail
module load apptainer/1.4.5

R2R_ROOT=/project/6101829/draip/R2R
R2R_IMAGE=${R2R_IMAGE:-${R2R_ROOT}/.containers/r2i.sif}
RESULT_ROOT=${R2R_RESULT_ROOT:-${R2R_ROOT}/experiments/r2r/results}
TASK_ID=${SLURM_ARRAY_TASK_ID:?array task is required}
WINDOWS=(64 1024)
WINDOW=${WINDOWS[${TASK_ID}]}
OUTPUT=${RESULT_ROOT}/integration/${SLURM_ARRAY_JOB_ID}/window${WINDOW}
REPLAY_DIRECTORY=${SLURM_TMPDIR:?}/r2r-integration-window${WINDOW}
mkdir -p "${OUTPUT}/provenance"
R2R_ROOT=${R2R_ROOT} "${R2R_ROOT}/experiments/r2r/record_provenance.sh" \
  "${OUTPUT}/provenance"
export WANDB_MODE=offline
export PYTHONHASHSEED=0
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
apptainer exec --cleanenv --nv \
  --env PYTHONPATH="${R2R_ROOT}" \
  --env WANDB_MODE=offline \
  --bind "${R2R_ROOT}:${R2R_ROOT}" \
  --bind "${SLURM_TMPDIR}:${SLURM_TMPDIR}" \
  --pwd "${R2R_ROOT}" \
  "${R2R_IMAGE}" \
  python experiments/r2r/integration_smoke.py \
    --window "${WINDOW}" \
    --workers 8 \
    --output "${OUTPUT}" \
    --replay-directory "${REPLAY_DIRECTORY}"
