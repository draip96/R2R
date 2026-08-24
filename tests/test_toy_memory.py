import importlib.util
import sys
import types
import unittest
from pathlib import Path

import numpy as np


class _Env:
  pass


class _Space:

  def __init__(self, dtype, shape=(), low=None, high=None):
    self.dtype = dtype
    self.shape = shape
    self.low = low
    self.high = high


STUB = types.ModuleType('embodied')
STUB.Env = _Env
STUB.Space = _Space
previous = sys.modules.get('embodied')
sys.modules['embodied'] = STUB
try:
  path = (
      Path(__file__).resolve().parents[1] / 'recall2imagine/embodied/envs' /
      'toy_memory.py')
  spec = importlib.util.spec_from_file_location('r2r_toy_memory', path)
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
finally:
  if previous is None:
    del sys.modules['embodied']
  else:
    sys.modules['embodied'] = previous

ToyMemory = module.ToyMemory


def _episode(env):
  rows = [env.step({'reset': True, 'action': 0})]
  while not rows[-1]['is_last']:
    action = env.context if rows[-1]['log_is_query'] else 0
    rows.append(env.step({'reset': False, 'action': action}))
  return rows


class ToyMemoryTest(unittest.TestCase):

  def test_exact_replay_lengths_and_terminal_answer(self):
    for length in (10, 18, 34, 66, 128, 256):
      env = ToyMemory(str(length), seed=3, balanced=True)
      rows = _episode(env)
      self.assertEqual(len(rows), length)
      self.assertEqual(sum(bool(row['is_first']) for row in rows), 1)
      self.assertEqual(sum(bool(row['is_last']) for row in rows), 1)
      self.assertEqual(sum(bool(row['log_is_query']) for row in rows), 1)
      self.assertTrue(rows[0]['is_first'])
      self.assertTrue(rows[-1]['is_last'])
      self.assertEqual(float(rows[-1]['reward']), 1.0)
      self.assertTrue(all(float(row['reward']) == 0.0 for row in rows[:-1]))

  def test_requested_cue_query_distances_are_literal(self):
    for distance in (8, 16, 32, 64):
      env = ToyMemory(str(distance + 2), seed=3, balanced=True)
      rows = _episode(env)
      cue_positions = np.flatnonzero(
          np.asarray([row['observation'][2] != 0.0 for row in rows]))
      query_positions = np.flatnonzero(
          np.asarray([row['log_is_query'] for row in rows]))
      np.testing.assert_array_equal(cue_positions, [0])
      np.testing.assert_array_equal(query_positions, [distance])
      self.assertEqual(query_positions[0] - cue_positions[0], distance)

  def test_cue_only_first_and_opposite_cues_have_identical_suffixes(self):
    env = ToyMemory('128', seed=5, balanced=True)
    zero = _episode(env)
    one = _episode(env)
    self.assertEqual(int(zero[0]['log_context']), 0)
    self.assertEqual(int(one[0]['log_context']), 1)
    self.assertEqual(float(zero[0]['observation'][2]), -1.0)
    self.assertEqual(float(one[0]['observation'][2]), 1.0)
    zero_obs = np.stack([row['observation'] for row in zero])
    one_obs = np.stack([row['observation'] for row in one])
    self.assertTrue(np.all(zero_obs[1:, 2] == 0.0))
    self.assertTrue(np.all(one_obs[1:, 2] == 0.0))
    np.testing.assert_array_equal(zero_obs[1:], one_obs[1:])

  def test_wrong_final_answer_is_negative(self):
    env = ToyMemory('128', seed=7, balanced=True)
    row = env.step({'reset': True, 'action': 0})
    while not row['log_is_query']:
      row = env.step({'reset': False, 'action': 0})
    terminal = env.step({'reset': False, 'action': 1 - env.context})
    self.assertTrue(terminal['is_terminal'])
    self.assertEqual(float(terminal['reward']), -1.0)


if __name__ == '__main__':
  unittest.main()
