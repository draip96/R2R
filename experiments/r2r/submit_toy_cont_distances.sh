#!/bin/bash
set -euo pipefail

CANONICAL_ROOT=/project/6101829/draip/R2R
cd "${CANONICAL_ROOT}"

if [[ -n $(git status --porcelain --untracked-files=all) ]]; then
  echo "commit tracked changes before creating the immutable run snapshot" >&2
  exit 2
fi

EXPECTED_COMMIT=$(git rev-parse HEAD)
SNAPSHOT_ROOT=${R2R_SNAPSHOT_ROOT:-/project/6101829/draip/.r2r_run_sources/${EXPECTED_COMMIT}}
if [[ -e ${SNAPSHOT_ROOT} ]]; then
  snapshot_commit=$(git -C "${SNAPSHOT_ROOT}" rev-parse HEAD)
  if [[ ${snapshot_commit} != "${EXPECTED_COMMIT}" ]]; then
    echo "existing snapshot ${SNAPSHOT_ROOT} is at ${snapshot_commit}" >&2
    exit 2
  fi
  if [[ -n $(git -C "${SNAPSHOT_ROOT}" status --porcelain --untracked-files=all) ]]; then
    echo "existing snapshot ${SNAPSHOT_ROOT} is not clean" >&2
    exit 2
  fi
else
  mkdir -p "$(dirname "${SNAPSHOT_ROOT}")"
  git worktree add --detach "${SNAPSHOT_ROOT}" "${EXPECTED_COMMIT}"
fi

SOURCE_RUN=${R2R_SOURCE_RUN:-experiments/r2r/results/toy_sparse_dyn_falsifier/20260826T013224Z/full_r2r_dyn005}
if [[ ${SOURCE_RUN} != /* ]]; then
  SOURCE_RUN=${CANONICAL_ROOT}/${SOURCE_RUN}
fi
SOURCE_FINAL_STEP=${R2R_SOURCE_FINAL_STEP:-60000}
python - "${SOURCE_RUN}" "${SOURCE_FINAL_STEP}" <<'PY'
import json
import math
import pathlib
import re
import sys

import yaml

source = pathlib.Path(sys.argv[1])
final_step = int(sys.argv[2])
required = [
    source / 'toy_summary.json',
    source / 'config.yaml',
    source / 'metrics.jsonl',
    source / 'provenance' / 'commit.txt',
    source / 'provenance' / 'git-status.txt',
    source / 'provenance' / 'working-tree.patch',
]
missing = [str(path) for path in required if not path.is_file()]
if missing:
  raise SystemExit(f'missing distance-8 source evidence: {missing}')

summary = json.loads((source / 'toy_summary.json').read_text())
config = yaml.safe_load((source / 'config.yaml').read_text())
expected_summary = {
    'environment_steps': final_step,
    'learner_updates': max(0, (final_step - 4096 + 3) // 4),
    'cue_query_distance': 8,
    'window': 64,
    'batch_size': 64,
    'objective': 'balanced_sparse_reward_with_native_auxiliaries',
    'success': True,
}
expected_config = {
    'task': 'toymemory_10',
    'batch_length': 64,
    'batch_size': 64,
    'replay': 'lfs',
    'replay_size': 1_000_000.0,
    'seed': 0,
    'toy_balanced_sparse_reward_with_aux': True,
    'toy_balanced_terminal_reward_with_aux': False,
}
for label, values, expected in (
    ('summary', summary, expected_summary),
    ('config', config, expected_config),
):
  for key, value in expected.items():
    if values.get(key) != value:
      raise SystemExit(
          f'distance-8 source {label}: expected {key}={value!r}, '
          f'got {values.get(key)!r}')

cache = config.get('state_gradient_cache', {})
if cache != {'enabled': True, 'storage_dtype': 'bfloat16'}:
  raise SystemExit(f'unexpected distance-8 cache config: {cache!r}')
scales = config.get('loss_scales', {})
for key, value in {'dyn': 0.05, 'rep': 0.0, 'vector': 0.0, 'cont': 1.0}.items():
  if not math.isclose(float(scales.get(key, math.nan)), value):
    raise SystemExit(
        f'distance-8 source: expected loss_scales.{key}={value}, '
        f'got {scales.get(key)!r}')
run = config.get('run', {})
for key, value in {
    'script': 'train_toy',
    'train_fill': 4096,
    'train_ratio': 1024.0,
    'toy_eval_every': 1000,
    'toy_eval_episodes': 128,
    'toy_stop_on_success': False,
}.items():
  if run.get(key) != value:
    raise SystemExit(
        f'distance-8 source: expected run.{key}={value!r}, '
        f'got {run.get(key)!r}')

commit = (source / 'provenance' / 'commit.txt').read_text().strip()
if not re.fullmatch(r'[0-9a-f]{40}', commit):
  raise SystemExit(f'invalid distance-8 provenance commit: {commit!r}')
if (source / 'provenance' / 'git-status.txt').read_text().strip():
  raise SystemExit('distance-8 source was launched from a dirty worktree')
if (source / 'provenance' / 'working-tree.patch').stat().st_size:
  raise SystemExit('distance-8 source contains a nonempty working-tree patch')
PY
python experiments/r2r/check_toy_retention.py \
  --criterion joint \
  --final-step "${SOURCE_FINAL_STEP}" \
  --metrics "${SOURCE_RUN}/metrics.jsonl"
SOURCE_SUMMARY_SHA256=$(sha256sum "${SOURCE_RUN}/toy_summary.json" | cut -d ' ' -f 1)
SOURCE_CONFIG_SHA256=$(sha256sum "${SOURCE_RUN}/config.yaml" | cut -d ' ' -f 1)
SOURCE_METRICS_SHA256=$(sha256sum "${SOURCE_RUN}/metrics.jsonl" | cut -d ' ' -f 1)
SOURCE_TRAINING_COMMIT=$(tr -d '[:space:]' < "${SOURCE_RUN}/provenance/commit.txt")

CAMPAIGN=${R2R_CAMPAIGN:-$(date -u +%Y%m%dT%H%M%SZ)}
TARGET_STEPS=${R2R_TARGET_STEPS:-60000}
DYN_SCALE=${R2R_DYN_SCALE:-0.05}
SEED=${R2R_SEED:-0}
OUTPUT=experiments/r2r/results/toy_sparse_dyn_distances/${CAMPAIGN}
mkdir -p "${OUTPUT}"
job=$(sbatch --parsable \
  --export=ALL,R2R_CAMPAIGN="${CAMPAIGN}",R2R_SEED="${SEED}",R2R_TARGET_STEPS="${TARGET_STEPS}",R2R_DYN_SCALE="${DYN_SCALE}",R2R_SOURCE_ROOT="${SNAPSHOT_ROOT}",R2R_EXPECTED_COMMIT="${EXPECTED_COMMIT}" \
  experiments/r2r/slurm_toy_cont_distances.sh)
printf '%s\n' "${job}" > "${OUTPUT}/job-array.txt"
printf '%s\n' \
  "campaign=${CAMPAIGN}" \
  "job=${job}" \
  "distances=16,32" \
  "seed=${SEED}" \
  "target_steps=${TARGET_STEPS}" \
  "dynamics_loss_scale=${DYN_SCALE}" \
  "source_distance8=${SOURCE_RUN}" \
  "source_final_step=${SOURCE_FINAL_STEP}" \
  "source_training_commit=${SOURCE_TRAINING_COMMIT}" \
  "source_summary_sha256=${SOURCE_SUMMARY_SHA256}" \
  "source_config_sha256=${SOURCE_CONFIG_SHA256}" \
  "source_metrics_sha256=${SOURCE_METRICS_SHA256}" \
  "expected_commit=${EXPECTED_COMMIT}" \
  "source_snapshot=${SNAPSHOT_ROOT}" \
  > "${OUTPUT}/submission.txt"
printf 'campaign=%s job=%s distances=16,32 seed=%s target=%s dyn=%s commit=%s\n' \
  "${CAMPAIGN}" "${job}" "${SEED}" "${TARGET_STEPS}" "${DYN_SCALE}" \
  "${EXPECTED_COMMIT}"
