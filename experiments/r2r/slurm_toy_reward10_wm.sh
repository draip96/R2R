#!/bin/bash
#SBATCH --account=aip-valenzan
#SBATCH --partition=gpubase_l40s_b3
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=1-00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@300
#SBATCH --job-name=r2r-reward-wm
#SBATCH --output=experiments/r2r/toy-reward-wm-%j.out

set -euo pipefail
module load apptainer/1.4.5

R2R_ROOT=/project/6101829/draip/R2R
R2R_IMAGE=${R2R_IMAGE:-${R2R_ROOT}/.containers/r2i.sif}
RESULT_ROOT=${R2R_RESULT_ROOT:-${R2R_ROOT}/experiments/r2r/results}
CAMPAIGN=${R2R_CAMPAIGN:?R2R_CAMPAIGN is required}
REWARD_SCALE=${R2R_REWARD_SCALE:-10}
if [[ ${REWARD_SCALE} != 10 && ${REWARD_SCALE} != 32 ]]; then
  echo "R2R_REWARD_SCALE must be 10 or 32" >&2
  exit 2
fi
OUTPUT=${RESULT_ROOT}/toy_acquisition/${CAMPAIGN}/distance8_reward${REWARD_SCALE}_world_model
REPLAY_DIRECTORY=${SLURM_TMPDIR:?}/r2r-reward${REWARD_SCALE}-wm-${CAMPAIGN}

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
    --configs "toy_memory,toy_distance8,r2r_w64,toy_world_model_only" \
    --seed 0 \
    --loss_scales.reward "${REWARD_SCALE}" \
    --run.steps 50000 \
    --toy_arm "reward${REWARD_SCALE}_world_model" \
    --logdir "${OUTPUT}" \
    --replay_dir "${REPLAY_DIRECTORY}" \
    --lfs_dir "${OUTPUT}/lfs" \
    --use_lfs True \
    --wdb_name "R2R-toy-reward${REWARD_SCALE}-world-model" &
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

summary = json.loads(pathlib.Path(sys.argv[1]).read_text())
if summary['environment_steps'] != 50000:
  raise SystemExit(
      f"reward-scale arm ended at {summary['environment_steps']} rather than 50000")
if summary['learner_updates'] != 11476:
  raise SystemExit(
      f"reward-scale arm used {summary['learner_updates']} rather than 11476 updates")
PY
