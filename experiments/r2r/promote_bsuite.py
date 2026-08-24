#!/usr/bin/env python3
"""Analyze BSuite seed-0 cells and three-seed cross-window qualification."""

import argparse
import json
import os
from pathlib import Path


WINDOWS = (64, 128, 256, 1024)
HORIZONS = (128, 256, 512, 1024, 2048)


def _read(path):
  return json.loads(path.read_text())


def _write(path, value):
  temporary = path.with_suffix(path.suffix + '.tmp')
  temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
  os.replace(str(temporary), str(path))


def seed0(root):
  records = []
  promoted = []
  for window in WINDOWS:
    for horizon in HORIZONS:
      directory = root / 'seed0' / 'window{}-horizon{}-seed0'.format(
          window, horizon)
      path = directory / 'bsuite_summary.json'
      if not path.exists():
        records.append({
            'window': window, 'horizon': horizon, 'seed': 0,
            'status': 'missing'})
        continue
      summary = _read(path)
      reward = float(summary['final_1000_episode_mean_reward'])
      record = {
          'window': window, 'horizon': horizon, 'seed': 0,
          'status': 'complete', 'mean_reward': reward,
          'promoted': reward >= 0.5}
      records.append(record)
      if record['promoted']:
        promoted.append((window, horizon))
  value = {
      'protocol': 'r2r-bsuite-seed0-promotion-v1',
      'complete': all(row['status'] == 'complete' for row in records),
      'threshold': 0.5,
      'records': records,
      'promoted_cells': [list(cell) for cell in promoted],
  }
  _write(root / 'seed0_promotion.json', value)
  manifest = root / 'promoted_cells.txt'
  manifest.write_text(''.join('{} {}\n'.format(*cell) for cell in promoted))
  if promoted:
    (root / 'SEED0_PROMOTIONS').write_text('seed 1 and 2 cells are eligible\n')
  return value


def qualify(root):
  promotion = _read(root / 'seed0_promotion.json')
  records = []
  qualified = []
  for window, horizon in promotion['promoted_cells']:
    rewards = []
    complete = True
    for seed in (0, 1, 2):
      phase = 'seed0' if seed == 0 else 'promoted'
      path = root / phase / 'window{}-horizon{}-seed{}'.format(
          window, horizon, seed) / 'bsuite_summary.json'
      if not path.exists():
        complete = False
        rewards.append(None)
      else:
        rewards.append(float(_read(path)['final_1000_episode_mean_reward']))
    passed = (
        complete and horizon >= 2 * window and
        all(value >= 0.5 for value in rewards))
    records.append({
        'window': window, 'horizon': horizon, 'rewards': rewards,
        'dependency_in_windows': horizon / window,
        'complete': complete, 'qualified': passed})
    if passed:
      qualified.append((window, horizon))
  value = {
      'protocol': 'r2r-bsuite-three-seed-qualification-v1',
      'threshold': 0.5,
      'minimum_dependency': '2T',
      'records': records,
      'qualified_cells': [list(cell) for cell in qualified],
      'qualified_windows': sorted(set(cell[0] for cell in qualified)),
      'memory_maze_unlocked': bool(qualified),
  }
  _write(root / 'three_seed_qualification.json', value)
  (root / 'qualified_windows.txt').write_text(''.join(
      '{}\n'.format(window) for window in value['qualified_windows']))
  if qualified:
    (root / 'MEMORY_MAZE_UNLOCKED').write_text(
        'at least one three-seed dependency reached 2T\n')
  return value


def main():
  parser = argparse.ArgumentParser()
  location = parser.add_mutually_exclusive_group(required=True)
  location.add_argument('--root', type=Path)
  location.add_argument('--campaign')
  parser.add_argument('--stage', choices=('seed0', 'qualify'), required=True)
  args = parser.parse_args()
  root = args.root or (
      Path('/project/6101829/draip/R2R/experiments/r2r/results/bsuite') /
      args.campaign)
  root = root.resolve()
  value = seed0(root) if args.stage == 'seed0' else qualify(root)
  print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == '__main__':
  main()
