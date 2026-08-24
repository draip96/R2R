#!/bin/bash
#SBATCH --account=aip-valenzan
#SBATCH --partition=gpubase_l40s_b3
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=1-00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@300
#SBATCH --job-name=r2r-toy
#SBATCH --output=experiments/r2r/toy-%j.out

set -euo pipefail
module load apptainer/1.4.5

R2R_ROOT=/project/6101829/draip/R2R
R2R_IMAGE=${R2R_IMAGE:-${R2R_ROOT}/.containers/r2i.sif}
RESULT_ROOT=${R2R_RESULT_ROOT:-${R2R_ROOT}/experiments/r2r/results}
CAMPAIGN=${R2R_CAMPAIGN:?R2R_CAMPAIGN is required}
if [[ -n "${R2R_TOY_DISTANCE:-}" ]]; then
  DISTANCE=${R2R_TOY_DISTANCE}
  case "${DISTANCE}" in
    8|16|32|64) ;;
    *)
      echo "toy cue/query distance must be one of 8, 16, 32, or 64" >&2
      exit 2
      ;;
  esac
  TOY_CONFIG=toy_distance${DISTANCE}
  TOY_LABEL=distance${DISTANCE}
  TOY_DESCRIPTION="cue/query distance ${DISTANCE}"
else
  HORIZON=${R2R_TOY_HORIZON:?R2R_TOY_HORIZON or R2R_TOY_DISTANCE is required}
  if [[ "${HORIZON}" != 128 && "${HORIZON}" != 256 ]]; then
    echo "toy horizon must be 128 or 256" >&2
    exit 2
  fi
  TOY_CONFIG=toy_memory${HORIZON}
  TOY_LABEL=horizon${HORIZON}
  TOY_DESCRIPTION="horizon ${HORIZON}"
fi
OUTPUT=${RESULT_ROOT}/toy/${CAMPAIGN}/${TOY_LABEL}
REPLAY_DIRECTORY=${SLURM_TMPDIR:?}/r2r-toy-${CAMPAIGN}-${TOY_LABEL}
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
    --configs "toy_memory,${TOY_CONFIG},r2r_w64" \
    --seed 0 \
    --logdir "${OUTPUT}" \
    --replay_dir "${REPLAY_DIRECTORY}" \
    --lfs_dir "${OUTPUT}/lfs" \
    --use_lfs True \
    --wdb_name "R2R-toy-${TOY_LABEL}" &
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
if [[ ! -f "${OUTPUT}/SUCCESS" ]]; then
  echo "ToyMemory ${TOY_DESCRIPTION} missed the strict gate" >&2
  exit 3
fi
