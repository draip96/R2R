#!/usr/bin/env python3
"""Submit gated BSuite and Memory Maze campaign stages."""

import argparse
import json
import os
import subprocess
from pathlib import Path


def submit(command):
  return subprocess.check_output(command, text=True).strip().split(';')[0]


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--campaign', required=True)
  parser.add_argument(
      '--stage', choices=('bsuite-seed0', 'bsuite-promoted', 'mmaze-seed0',
                         'mmaze-promoted'), required=True)
  parser.add_argument('--dependency', default='')
  args = parser.parse_args()
  root = Path('/project/6101829/draip/R2R')
  os.chdir(root)
  results = root / 'experiments/r2r/results'
  toy = results / 'toy' / args.campaign
  bsuite = results / 'bsuite' / args.campaign
  common = ['sbatch', '--parsable']
  if args.dependency:
    common.append('--dependency=afterok:{}'.format(args.dependency))
  exports = ['ALL', 'R2R_CAMPAIGN={}'.format(args.campaign)]
  if args.stage == 'bsuite-seed0':
    if not (toy / 'horizon128/SUCCESS').exists() or not (
        toy / 'horizon256/SUCCESS').exists():
      raise RuntimeError('both ToyMemory gates must pass first')
    exports.append('R2R_BSUITE_PHASE=seed0')
    script = 'experiments/r2r/slurm_bsuite.sh'
  elif args.stage == 'bsuite-promoted':
    promotion = json.loads((bsuite / 'seed0_promotion.json').read_text())
    if not promotion['complete']:
      raise RuntimeError('the full 20-cell seed-0 grid is not complete')
    manifest = bsuite / 'promoted_cells.txt'
    count = len(manifest.read_text().splitlines())
    if not count:
      raise RuntimeError('no seed-0 cells were promoted')
    common.append('--array=0-{}%6'.format(2 * count - 1))
    exports += [
        'R2R_BSUITE_PHASE=promoted',
        'R2R_BSUITE_MANIFEST={}'.format(manifest)]
    script = 'experiments/r2r/slurm_bsuite.sh'
  elif args.stage == 'mmaze-seed0':
    if not (bsuite / 'MEMORY_MAZE_UNLOCKED').exists():
      raise RuntimeError('three-seed 2T BSuite gate has not passed')
    exports.append('R2R_MMAZE_PHASE=seed0')
    script = 'experiments/r2r/slurm_mmaze.sh'
  else:
    manifest = bsuite / 'qualified_windows.txt'
    count = len(manifest.read_text().splitlines())
    if not count:
      raise RuntimeError('no qualified Memory Maze windows')
    common.append('--array=0-{}'.format(2 * count - 1))
    exports += [
        'R2R_MMAZE_PHASE=promoted',
        'R2R_MMAZE_MANIFEST={}'.format(manifest)]
    script = 'experiments/r2r/slurm_mmaze.sh'
  command = common + ['--export={}'.format(','.join(exports)), script]
  job = submit(command)
  record = results / 'submissions' / args.campaign
  record.mkdir(parents=True, exist_ok=True)
  (record / '{}.json'.format(args.stage)).write_text(json.dumps({
      'stage': args.stage, 'job': job, 'command': command,
  }, indent=2, sort_keys=True) + '\n')
  print(job)


if __name__ == '__main__':
  main()
