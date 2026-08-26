#!/bin/bash
#SBATCH --account=aip-valenzan
#SBATCH --partition=gpubase_l40s_b3
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=1-00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@300
#SBATCH --array=0-2
#SBATCH --job-name=r2r-sparse-dyn
#SBATCH --output=experiments/r2r/toy-sparse-dyn-%A_%a.out

set -euo pipefail
module load apptainer/1.4.5

R2R_ROOT=/project/6101829/draip/R2R
R2R_IMAGE=${R2R_IMAGE:-${R2R_ROOT}/.containers/r2i.sif}
RESULT_ROOT=${R2R_RESULT_ROOT:-${R2R_ROOT}/experiments/r2r/results}
CAMPAIGN=${R2R_CAMPAIGN:?R2R_CAMPAIGN is required}

# The zero-scale arm falsifies the three-class reward objective itself. The
# matched 0.05 arms test whether one tenth of the native dynamics scale trains
# the sampled prior without suppressing cue acquisition.
case "${SLURM_ARRAY_TASK_ID}" in
  0)
    ARM=direct_bptt_dyn0
    ARM_CONFIG=toy_direct_bptt
    DYN_SCALE=0.0
    ;;
  1)
    ARM=direct_bptt_dyn005
    ARM_CONFIG=toy_direct_bptt
    DYN_SCALE=0.05
    ;;
  2)
    ARM=full_r2r_dyn005
    ARM_CONFIG=toy_full_r2r
    DYN_SCALE=0.05
    ;;
  *)
    echo "array index must be 0, 1, or 2" >&2
    exit 2
    ;;
esac

OUTPUT=${RESULT_ROOT}/toy_sparse_dyn_falsifier/${CAMPAIGN}/${ARM}
REPLAY_DIRECTORY=${SLURM_TMPDIR:?}/r2r-sparse-dyn-${CAMPAIGN}-${ARM}
mkdir -p "${OUTPUT}/provenance" "${OUTPUT}/lfs"
env R2R_ROOT="${R2R_ROOT}" "${R2R_ROOT}/experiments/r2r/record_provenance.sh" \
  "${OUTPUT}/provenance"
printf 'arm=%s\ndynamics_loss_scale=%s\n' "${ARM}" "${DYN_SCALE}" \
  > "${OUTPUT}/provenance/falsifier.txt"
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
    --configs "toy_memory,toy_distance8,r2r_w64,${ARM_CONFIG},toy_balanced_sparse_dyn_cont_memory" \
    --seed 0 \
    --loss_scales.dyn "${DYN_SCALE}" \
    --run.steps 50000 \
    --toy_arm "balanced_sparse_${ARM}" \
    --logdir "${OUTPUT}" \
    --replay_dir "${REPLAY_DIRECTORY}" \
    --lfs_dir "${OUTPUT}/lfs" \
    --use_lfs True \
    --wdb_name "R2R-toy-balanced-sparse-${ARM}" &
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
      f"sparse-dynamics arm ended at {summary['environment_steps']} rather than 50000")
if summary['learner_updates'] != 11476:
  raise SystemExit(
      f"sparse-dynamics arm used {summary['learner_updates']} rather than 11476 updates")
if summary['objective'] != 'balanced_sparse_reward_with_native_auxiliaries':
  raise SystemExit(f"unexpected objective {summary['objective']!r}")
PY

set +e
python experiments/r2r/check_toy_retention.py \
  --criterion model \
  --metrics "${OUTPUT}/metrics.jsonl" \
  > "${OUTPUT}/model_retention.json"
model_status=$?
python experiments/r2r/check_toy_retention.py \
  --criterion joint \
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
