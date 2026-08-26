#!/usr/bin/env python3
"""Check that a ToyMemory solution is retained across final evaluations."""

import argparse
import json
import math
from pathlib import Path


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--metrics', type=Path, required=True)
  parser.add_argument('--panels', type=int, default=5)
  parser.add_argument('--eval-every', type=int, default=1000)
  parser.add_argument('--final-step', type=int, default=50000)
  parser.add_argument('--min-model-margin', type=float, default=0.1)
  parser.add_argument('--criterion', choices=('joint', 'model'), default='joint')
  return parser.parse_args()


def load_evaluations(path):
  evaluations = {}
  with path.open() as stream:
    for line_number, line in enumerate(stream, 1):
      try:
        record = json.loads(line)
      except json.JSONDecodeError as exc:
        raise ValueError(f'{path}:{line_number}: invalid JSON: {exc}') from exc
      if 'toy_eval/actor_accuracy' not in record:
        continue
      step = int(record['step'])
      evaluations[step] = record
  return [evaluations[step] for step in sorted(evaluations)]


def retained(
    evaluations, panels, eval_every, final_step, min_model_margin,
    criterion='joint'):
  if panels < 1:
    raise ValueError('--panels must be positive')
  if criterion not in ('joint', 'model'):
    raise ValueError(f'unknown retention criterion {criterion!r}')

  def solved(record):
    margin = float(record['toy_eval/model_reward_margin'])
    model_solved = (
        float(record['toy_eval/model_reward_choice_accuracy']) == 1.0 and
        float(record.get('toy_eval/finite', 0.0)) == 1.0 and
        math.isfinite(margin) and margin >= min_model_margin)
    return (
        model_solved and
        (criterion == 'model' or
         float(record['toy_eval/actor_accuracy']) == 1.0))

  best = []
  current = []
  for record in evaluations:
    step = int(record['step'])
    gap = None if not current else step - int(current[-1]['step'])
    contiguous = (
        not current or gap == eval_every or
        (step == final_step and 0 < gap <= eval_every))
    if solved(record) and contiguous:
      current.append(record)
    elif solved(record):
      current = [record]
    else:
      current = []
    if len(current) >= len(best):
      best = list(current)

  final = evaluations[-1] if evaluations else None
  passed = bool(
      len(current) >= panels and final and
      int(final['step']) == final_step and solved(final))
  # A successful report must show the exact final contiguous streak that
  # justified promotion. For a miss, retain the best historical streak as a
  # diagnostic without allowing it to satisfy the gate.
  selected = current[-panels:] if passed else best[-panels:]
  return passed, selected, final


def main():
  args = parse_args()
  evaluations = load_evaluations(args.metrics)
  passed, selected, final = retained(
      evaluations, args.panels, args.eval_every, args.final_step,
      args.min_model_margin, args.criterion)
  result = {
      'passed': passed,
      'criterion': args.criterion,
      'required_panels': args.panels,
      'retained_streak': [{
          'step': int(record['step']),
          'actor_accuracy': float(record['toy_eval/actor_accuracy']),
          'model_reward_choice_accuracy': float(
              record['toy_eval/model_reward_choice_accuracy']),
          'model_reward_margin': float(record['toy_eval/model_reward_margin']),
      } for record in selected],
      'final_panel': None if final is None else {
          'step': int(final['step']),
          'actor_accuracy': float(final['toy_eval/actor_accuracy']),
          'model_reward_choice_accuracy': float(
              final['toy_eval/model_reward_choice_accuracy']),
          'model_reward_margin': float(final['toy_eval/model_reward_margin']),
      },
  }
  print(json.dumps(result, indent=2, sort_keys=True))
  raise SystemExit(0 if passed else 3)


if __name__ == '__main__':
  main()
