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
#SBATCH --job-name=r2r-cont-extend
#SBATCH --output=experiments/r2r/toy-continuation-extend-%A_%a.out

set -euo pipefail
module load apptainer/1.4.5

R2R_ROOT=/project/6101829/draip/R2R
R2R_IMAGE=${R2R_IMAGE:-${R2R_ROOT}/.containers/r2i.sif}
RESULT_ROOT=${R2R_RESULT_ROOT:-${R2R_ROOT}/experiments/r2r/results}
SOURCE_CAMPAIGN=${R2R_SOURCE_CAMPAIGN:?R2R_SOURCE_CAMPAIGN is required}
TARGET_STEPS=${R2R_TARGET_STEPS:-50000}

case "${SLURM_ARRAY_TASK_ID}" in
  0)
    ARM=cont_shared
    CONFIGS=toy_memory,toy_distance8,r2r_w64,toy_balanced_terminal_cont_world_model
    ;;
  1)
    ARM=cont_detached
    CONFIGS=toy_memory,toy_distance8,r2r_w64,toy_balanced_terminal_cont_world_model,toy_detached_aux_heads
    ;;
  *)
    echo "array index must be 0 or 1" >&2
    exit 2
    ;;
esac

OUTPUT=${RESULT_ROOT}/toy_continuation_audit/${SOURCE_CAMPAIGN}/${ARM}
REPLAY_DIRECTORY=${SLURM_TMPDIR:?}/r2r-cont-extend-${SOURCE_CAMPAIGN}-${ARM}
PROVENANCE=${OUTPUT}/continuations/${TARGET_STEPS}/attempts/${SLURM_JOB_ID}/provenance

if [[ ! ${TARGET_STEPS} =~ ^[0-9]+$ ]]; then
  echo "R2R_TARGET_STEPS must be an integer" >&2
  exit 2
fi
if [[ ! -f ${OUTPUT}/checkpoint.ckpt || ! -f ${OUTPUT}/toy_summary.json ]]; then
  echo "missing source checkpoint or summary under ${OUTPUT}" >&2
  exit 2
fi
current_steps=$(python -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["environment_steps"])' \
  "${OUTPUT}/toy_summary.json")
if (( current_steps >= TARGET_STEPS )); then
  echo "source is already at ${current_steps} steps (target ${TARGET_STEPS})" >&2
  exit 2
fi

mkdir -p "${PROVENANCE}" "${REPLAY_DIRECTORY}"
cp "${OUTPUT}/config.yaml" "${PROVENANCE}/config-before-resume.yaml"
R2R_ROOT=${R2R_ROOT} "${R2R_ROOT}/experiments/r2r/record_provenance.sh" \
  "${PROVENANCE}"
printf 'source_campaign=%s\narm=%s\nsource_steps=%s\ntarget_steps=%s\n' \
  "${SOURCE_CAMPAIGN}" "${ARM}" "${current_steps}" "${TARGET_STEPS}" \
  > "${PROVENANCE}/continuation.txt"
printf '%s\n' \
  'The checkpoint restores model, optimizer, counters, and RNG state at 25,000.' \
  'The LFS replay mirror contains only completed 1024-row chunks through 24,576,' \
  'so up to 424 partial-tail replay rows are replaced after resume. Sampling stays' \
  'uniform and unfiltered, but this continuation is not replay-bit-exact.' \
  > "${PROVENANCE}/resume-qualification.txt"
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
    --configs "${CONFIGS}" \
    --seed 0 \
    --run.steps "${TARGET_STEPS}" \
    --toy_arm "balanced_terminal_${ARM}" \
    --logdir "${OUTPUT}" \
    --replay_dir "${REPLAY_DIRECTORY}" \
    --lfs_dir "${OUTPUT}/lfs" \
    --use_lfs True \
    --wdb_name "R2R-toy-${ARM}-${TARGET_STEPS}" &
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

python - "${OUTPUT}/toy_summary.json" "${TARGET_STEPS}" <<'PY'
import json
import pathlib
import sys

summary = json.loads(pathlib.Path(sys.argv[1]).read_text())
target = int(sys.argv[2])
expected_updates = max(0, (target - 4096 + 3) // 4)
if summary['environment_steps'] != target:
  raise SystemExit(
      f"continuation ended at {summary['environment_steps']} rather than {target}")
if summary['learner_updates'] != expected_updates:
  raise SystemExit(
      f"continuation recorded {summary['learner_updates']} rather than "
      f"{expected_updates} cumulative updates")
PY
