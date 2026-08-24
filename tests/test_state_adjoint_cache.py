import tempfile
import unittest
import importlib.util
from pathlib import Path

import numpy as np

MODULE_PATH = (
    Path(__file__).resolve().parents[1] / 'recall2imagine' / 'embodied' /
    'replay' / 'state_adjoint_cache.py')
SPEC = importlib.util.spec_from_file_location('r2r_state_adjoint_cache', MODULE_PATH)
CACHE_MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CACHE_MODULE)
DenseStateAdjointCache = CACHE_MODULE.DenseStateAdjointCache
aligned_capacity = CACHE_MODULE.aligned_capacity
bf16_bits_to_complex = CACHE_MODULE.bf16_bits_to_complex
bf16_bits_to_float32 = CACHE_MODULE.bf16_bits_to_float32
cache_bytes = CACHE_MODULE.cache_bytes
complex_to_bf16_bits = CACHE_MODULE.complex_to_bf16_bits
float32_to_bf16_bits = CACHE_MODULE.float32_to_bf16_bits
sample_slot_layout = CACHE_MODULE.sample_slot_layout


class BFloat16Test(unittest.TestCase):

  def test_float_round_trip_matches_bfloat16_precision(self):
    values = np.array(
        [-100.25, -1.003, -0.0, 0.0, 0.3333, 1.003, 100.25],
        np.float32)
    restored = bf16_bits_to_float32(float32_to_bf16_bits(values))
    np.testing.assert_allclose(restored, values, rtol=8e-3, atol=1e-3)

  def test_complex_round_trip(self):
    values = np.array([1.25 + 2.5j, -3.75 - 0.125j], np.complex64)
    restored = bf16_bits_to_complex(complex_to_bf16_bits(values))
    np.testing.assert_allclose(restored, values, rtol=8e-3, atol=1e-3)

  def test_memory_maze_payload_size(self):
    value = cache_bytes(1_000_000, 5, 512)
    self.assertEqual(value, 20_480_000_000)
    self.assertAlmostEqual(value / 1024 ** 3, 19.073486328125)

  def test_capacity_alignment_preserves_configured_replay_size(self):
    self.assertEqual(aligned_capacity(1_000_000, 64), 1_000_000)
    self.assertEqual(aligned_capacity(1_000_000, 128), 1_000_064)
    self.assertEqual(aligned_capacity(1_000_000, 256), 1_000_192)
    self.assertEqual(aligned_capacity(1_000_000, 1024), 1_000_448)

  def test_physical_slot_layout_spliced_and_full_chunk(self):
    spliced = sample_slot_layout(4, 2, 7, 3, 6, 2)
    np.testing.assert_array_equal(spliced['slots'], [26, 27, 28, 29])
    np.testing.assert_array_equal(spliced['generations'], [2, 2, 3, 3])
    self.assertEqual(spliced['anchor_slot'], 25)
    self.assertEqual(spliced['anchor_generation'], 2)
    whole = sample_slot_layout(4, 4, 7, 3, 6, 2)
    np.testing.assert_array_equal(whole['slots'], [28, 29, 30, 31])
    np.testing.assert_array_equal(whole['generations'], [3, 3, 3, 3])
    self.assertEqual(whole['anchor_slot'], 27)
    missing = sample_slot_layout(4, 4, 7, 3, -1, -1)
    np.testing.assert_array_equal(missing['slots'], [28, 29, 30, 31])
    self.assertEqual(missing['anchor_slot'], -1)
    self.assertEqual(missing['anchor_generation'], -1)


class DenseCacheTest(unittest.TestCase):

  def setUp(self):
    self.temporary = tempfile.TemporaryDirectory()
    self.root = Path(self.temporary.name)
    self.cache = DenseStateAdjointCache(
        self.root / 'local', capacity=8, layers=2, width=3,
        stoch=2, classes=4, action_dim=3,
        mirror_directory=self.root / 'mirror')
    self.cache.invalidate_chunk(0, generation=0, chunk_length=4)
    self.cache.invalidate_chunk(1, generation=0, chunk_length=4)

  def tearDown(self):
    self.cache.close()
    self.temporary.cleanup()

  def _payload(self, slots, generations, values):
    values = np.asarray(values, np.float32)
    state = np.empty((len(slots), 2, 3), np.complex64)
    adjoint = np.empty_like(state)
    for index, value in enumerate(values):
      state[index] = value + 1j * (value + 0.5)
      adjoint[index] = (value + 10) + 1j * (value + 20)
    return {
        'state_slots': np.asarray(slots, np.int32),
        'state_generations': np.asarray(generations, np.int32),
        'state_bits': complex_to_bf16_bits(state),
        'stoch': np.tile(np.array([[1, 2]], np.uint8), (len(slots), 1)),
        'action': np.arange(len(slots), dtype=np.uint16) % 3,
        'adjoint_slots': np.asarray(slots, np.int32),
        'adjoint_generations': np.asarray(generations, np.int32),
        'adjoint_bits': complex_to_bf16_bits(adjoint),
        'terminal_slots': np.empty((0,), np.int32),
        'terminal_generations': np.empty((0,), np.int32),
    }

  def test_missing_rows_gather_as_zero_and_invalid(self):
    result = self.cache.gather_boundaries(
        np.array([1]), np.array([0]), np.array([2]), np.array([0]))
    self.assertFalse(result['initial_valid'][0])
    self.assertFalse(result['future_adjoint_valid'][0])
    self.assertEqual(float(np.abs(result['initial_state_real']).max()), 0.0)

  def test_last_duplicate_wins(self):
    self.cache.commit(self._payload([2, 2], [0, 0], [1, 7]))
    result = self.cache.gather_boundaries(
        np.array([2]), np.array([0]), np.array([2]), np.array([0]))
    self.assertTrue(result['initial_valid'][0])
    self.assertTrue(result['future_adjoint_valid'][0])
    np.testing.assert_allclose(result['initial_state_real'][0], 7.0)
    np.testing.assert_allclose(result['future_adjoint_real'][0], 17.0)

  def test_overwrite_generation_rejects_stale_commit(self):
    stale = self._payload([1], [0], [3])
    self.cache.invalidate_chunk(0, generation=1, chunk_length=4)
    result = self.cache.commit(stale)
    self.assertEqual(result['state_rows_written'], 0)
    gathered = self.cache.gather_boundaries(
        np.array([1]), np.array([1]), np.array([1]), np.array([1]))
    self.assertFalse(gathered['initial_valid'][0])

  def test_generation_reconcile_preserves_match_and_invalidates_mismatch(self):
    self.cache.commit(self._payload([1], [0], [3]))
    self.assertFalse(self.cache.reconcile_chunk(0, 0, 4))
    gathered = self.cache.gather_boundaries(
        np.array([1]), np.array([0]), np.array([1]), np.array([0]))
    self.assertTrue(gathered['initial_valid'][0])
    self.assertTrue(self.cache.reconcile_chunk(0, 1, 4))
    gathered = self.cache.gather_boundaries(
        np.array([1]), np.array([1]), np.array([1]), np.array([1]))
    self.assertFalse(gathered['initial_valid'][0])

  def test_nonfinite_update_aborts(self):
    payload = self._payload([1], [0], [3])
    bad = np.full((1, 2, 3), np.nan + 0j, np.complex64)
    payload['adjoint_bits'] = complex_to_bf16_bits(bad)
    with self.assertRaises(FloatingPointError):
      self.cache.commit(payload)

  def test_terminal_boundary_forces_zero_adjoint(self):
    payload = self._payload([3], [0], [4])
    payload['terminal_slots'] = np.array([3], np.int32)
    payload['terminal_generations'] = np.array([0], np.int32)
    self.cache.commit(payload)
    result = self.cache.gather_boundaries(
        np.array([3]), np.array([0]), np.array([3]), np.array([0]))
    self.assertTrue(result['future_adjoint_valid'][0])
    self.assertEqual(float(np.abs(result['future_adjoint_real']).max()), 0.0)
    self.assertEqual(float(np.abs(result['future_adjoint_imag']).max()), 0.0)

  def test_mirror_restores_cache(self):
    self.cache.commit(self._payload([5], [0], [9]))
    self.cache.flush(mirror=True)
    self.cache.close()
    for path in (self.root / 'local').iterdir():
      path.unlink()
    self.cache = DenseStateAdjointCache(
        self.root / 'local', capacity=8, layers=2, width=3,
        stoch=2, classes=4, action_dim=3,
        mirror_directory=self.root / 'mirror')
    result = self.cache.gather_boundaries(
        np.array([5]), np.array([0]), np.array([5]), np.array([0]))
    self.assertTrue(result['initial_valid'][0])
    np.testing.assert_allclose(result['initial_state_real'][0], 9.0)


if __name__ == '__main__':
  unittest.main()
