#!/bin/bash
#SBATCH --account=aip-valenzan
#SBATCH --partition=gpubase_l40s_b3
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=1-00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@300
#SBATCH --array=0-3
#SBATCH --job-name=r2r-aux-audit
#SBATCH --output=experiments/r2r/toy-auxiliary-ablation-%A_%a.out

set -euo pipefail
module load apptainer/1.4.5

R2R_ROOT=/project/6101829/draip/R2R
R2R_IMAGE=${R2R_IMAGE:-${R2R_ROOT}/.containers/r2i.sif}
RESULT_ROOT=${R2R_RESULT_ROOT:-${R2R_ROOT}/experiments/r2r/results}
CAMPAIGN=${R2R_CAMPAIGN:?R2R_CAMPAIGN is required}

# 2x2 factorial: KL losses and observation/continuation losses independently
# on or off. The balanced terminal reward term is identical in every arm.
case "${SLURM_ARRAY_TASK_ID}" in
  0)
    ARM=all_aux
    DYN=0.5; REP=0.1; VECTOR=1.0; CONT=1.0
    ;;
  1)
    ARM=kl_only
    DYN=0.5; REP=0.1; VECTOR=0.0; CONT=0.0
    ;;
  2)
    ARM=recon_cont_only
    DYN=0.0; REP=0.0; VECTOR=1.0; CONT=1.0
    ;;
  3)
    ARM=reward_only_oracle
    DYN=0.0; REP=0.0; VECTOR=0.0; CONT=0.0
    ;;
  *)
    echo "array index must be 0, 1, 2, or 3" >&2
    exit 2
    ;;
esac

OUTPUT=${RESULT_ROOT}/toy_auxiliary_audit/${CAMPAIGN}/${ARM}
REPLAY_DIRECTORY=${SLURM_TMPDIR:?}/r2r-aux-${CAMPAIGN}-${ARM}
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
    --configs "toy_memory,toy_distance8,r2r_w64,toy_balanced_terminal_aux_world_model" \
    --seed 0 \
    --loss_scales.dyn "${DYN}" \
    --loss_scales.rep "${REP}" \
    --loss_scales.vector "${VECTOR}" \
    --loss_scales.cont "${CONT}" \
    --run.steps 25000 \
    --toy_arm "balanced_terminal_${ARM}" \
    --logdir "${OUTPUT}" \
    --replay_dir "${REPLAY_DIRECTORY}" \
    --lfs_dir "${OUTPUT}/lfs" \
    --use_lfs True \
    --wdb_name "R2R-toy-aux-${ARM}" &
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
if summary['environment_steps'] != 25000:
  raise SystemExit(
      f"auxiliary arm ended at {summary['environment_steps']} rather than 25000")
if summary['learner_updates'] != 5226:
  raise SystemExit(
      f"auxiliary arm used {summary['learner_updates']} rather than 5226 updates")
if summary['objective'] != 'balanced_terminal_reward_with_native_auxiliaries':
  raise SystemExit(f"unexpected objective {summary['objective']!r}")
PY
