#!/usr/bin/env python3
"""Validate a completed retained ToyMemory full-R2R run."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import sys

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
  sys.path.insert(0, str(SCRIPT_DIR))
from check_toy_retention import load_evaluations, retained  # noqa: E402


class ValidationError(ValueError):
  pass


def _sha256(path):
  digest = hashlib.sha256()
  with path.open('rb') as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b''):
      digest.update(chunk)
  return digest.hexdigest()


def _require_equal(scope, values, expected):
  for key, value in expected.items():
    if values.get(key) != value:
      raise ValidationError(
          f'{scope}: expected {key}={value!r}, got {values.get(key)!r}')


def validate_run(source, distance, seed, final_step, dyn_scale,
                 expected_commit=None):
  source = Path(source).resolve()
  paths = {
      'summary': source / 'toy_summary.json',
      'config': source / 'config.yaml',
      'metrics': source / 'metrics.jsonl',
      'commit': source / 'provenance' / 'commit.txt',
      'git_status': source / 'provenance' / 'git-status.txt',
      'patch': source / 'provenance' / 'working-tree.patch',
      'source_hashes': source / 'provenance' / 'source-sha256.txt',
  }
  missing = [str(path) for path in paths.values() if not path.is_file()]
  if missing:
    raise ValidationError(f'missing source evidence: {missing}')

  summary = json.loads(paths['summary'].read_text())
  config = yaml.safe_load(paths['config'].read_text())
  expected_updates = max(0, (final_step - 4096 + 3) // 4)
  _require_equal('summary', summary, {
      'environment_steps': final_step,
      'learner_updates': expected_updates,
      'cue_query_distance': distance,
      'episode_steps': distance + 2,
      'window': 64,
      'batch_size': 64,
      'objective': 'balanced_sparse_reward_with_native_auxiliaries',
      'success': True,
  })
  _require_equal('config', config, {
      'task': f'toymemory_{distance + 2}',
      'batch_length': 64,
      'batch_size': 64,
      'replay': 'lfs',
      'replay_size': 1_000_000.0,
      'replay_online': False,
      'seed': seed,
      'task_behavior': 'Greedy',
      'toy_balanced_sparse_reward_with_aux': True,
      'toy_balanced_terminal_reward_with_aux': False,
      'unlocked_sampling': False,
  })

  cache = config.get('state_gradient_cache', {})
  if cache != {'enabled': True, 'storage_dtype': 'bfloat16'}:
    raise ValidationError(f'unexpected cache config: {cache!r}')
  scales = config.get('loss_scales', {})
  for key, value in {
      'reward': 1.0,
      'cont': 1.0,
      'dyn': dyn_scale,
      'rep': 0.0,
      'vector': 0.0,
  }.items():
    if not math.isclose(float(scales.get(key, math.nan)), value):
      raise ValidationError(
          f'expected loss_scales.{key}={value}, got {scales.get(key)!r}')
  run = config.get('run', {})
  _require_equal('run config', run, {
      'script': 'train_toy',
      'steps': float(final_step),
      'train_fill': 4096,
      'train_ratio': 1024.0,
      'toy_eval_every': 1000,
      'toy_eval_episodes': 128,
      'toy_stop_on_success': False,
  })

  commit = paths['commit'].read_text().strip()
  if not re.fullmatch(r'[0-9a-f]{40}', commit):
    raise ValidationError(f'invalid provenance commit: {commit!r}')
  if expected_commit and commit != expected_commit:
    raise ValidationError(
        f'expected training commit {expected_commit}, got {commit}')
  if paths['git_status'].read_text().strip():
    raise ValidationError('run was launched from a dirty worktree')
  if paths['patch'].stat().st_size:
    raise ValidationError('run contains a nonempty working-tree patch')

  evaluations = load_evaluations(paths['metrics'])
  passed, streak, final = retained(
      evaluations, panels=5, eval_every=1000, final_step=final_step,
      min_model_margin=0.1, criterion='joint')
  if not passed:
    raise ValidationError('run does not pass final five-panel joint retention')

  return {
      'passed': True,
      'source': str(source),
      'distance': distance,
      'seed': seed,
      'final_step': final_step,
      'learner_updates': expected_updates,
      'dynamics_loss_scale': dyn_scale,
      'training_commit': commit,
      'retained_steps': [int(record['step']) for record in streak],
      'final_actor_accuracy': float(final['toy_eval/actor_accuracy']),
      'final_model_reward_choice_accuracy': float(
          final['toy_eval/model_reward_choice_accuracy']),
      'final_model_reward_margin': float(
          final['toy_eval/model_reward_margin']),
      'sha256': {
          name: _sha256(paths[name])
          for name in ('summary', 'config', 'metrics', 'source_hashes')
      },
  }


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--source', type=Path, required=True)
  parser.add_argument('--distance', type=int, required=True)
  parser.add_argument('--seed', type=int, required=True)
  parser.add_argument('--final-step', type=int, required=True)
  parser.add_argument('--dyn-scale', type=float, required=True)
  parser.add_argument('--expected-commit')
  return parser.parse_args()


def main():
  args = parse_args()
  try:
    result = validate_run(
        args.source, args.distance, args.seed, args.final_step,
        args.dyn_scale, args.expected_commit)
  except (ValidationError, json.JSONDecodeError, yaml.YAMLError) as exc:
    print(f'validation failed: {exc}', file=sys.stderr)
    raise SystemExit(3) from exc
  print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
  main()
