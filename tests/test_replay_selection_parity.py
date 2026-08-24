import types
import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
  from recall2imagine.embodied.replay import generic_lfs, selectors
except (ImportError, ModuleNotFoundError):
  generic_lfs = None
  selectors = None


class _Limiter:
  def want_sample(self):
    return True, ''


class _Manager:

  def __init__(self, length):
    self.length = length

  def read_chunk(self, flip, offset, worker, half):
    del flip, worker, half
    values = offset * 100 + np.arange(self.length, dtype=np.int32)
    data = {
        'value': values,
        'is_first': np.zeros(self.length, bool),
    }
    return 0, types.SimpleNamespace(data=data)


class _Cache:

  def __init__(self):
    self.layers = 1
    self.width = 1
    self.stoch = 1
    self.boundary = None

  def gather_boundaries(self, anchor_slots, anchor_generations, future_slots,
                        future_generations):
    self.boundary = (
        np.asarray(anchor_slots).copy(),
        np.asarray(anchor_generations).copy(),
        np.asarray(future_slots).copy(),
        np.asarray(future_generations).copy())
    return {
        'initial_state_real': np.zeros((1, 1, 1), np.float32),
        'initial_state_imag': np.zeros((1, 1, 1), np.float32),
        'initial_stoch': np.zeros((1, 1), np.uint8),
        'initial_action': np.zeros((1,), np.uint16),
        'initial_valid': np.zeros((1,), bool),
        'future_adjoint_real': np.zeros((1, 1, 1), np.float32),
        'future_adjoint_imag': np.zeros((1, 1, 1), np.float32),
        'future_adjoint_valid': np.zeros((1,), bool),
    }


@unittest.skipIf(generic_lfs is None, 'R2I runtime dependencies are unavailable')
class ReplaySelectionParityTest(unittest.TestCase):

  def _replay(self, seed, length=8, chunks=7):
    replay = generic_lfs.FIFO_LFS.__new__(generic_lfs.FIFO_LFS)
    replay.length = length
    replay.limiter = _Limiter()
    replay.metrics = {
        'samples': 0, 'sample_wait_dur': 0, 'sample_wait_count': 0}
    replay.sampler = selectors.Uniform(seed)
    replay.rng = np.random.default_rng(seed)
    replay.table = {key: key for key in range(chunks)}
    replay.bwd_links = {key: (key - 1) % chunks for key in range(chunks)}
    replay.chunk_generations = {key: 0 for key in range(chunks)}
    for key in range(chunks):
      replay.sampler[key] = key
    replay.manager = _Manager(length)
    replay.cache = _Cache()
    return replay

  @staticmethod
  def _buffer(length):
    prefix = (1, 1)
    return {
        'value': np.empty(prefix + (length,), np.int32),
        'is_first': np.empty(prefix + (length,), bool),
        '_r2r_slot': np.empty(prefix + (length,), np.int32),
        '_r2r_generation': np.empty(prefix + (length,), np.int32),
        '_r2r_anchor_slot': np.empty(prefix, np.int32),
        '_r2r_anchor_generation': np.empty(prefix, np.int32),
        '_r2r_initial_state_real': np.empty(prefix + (1, 1), np.float32),
        '_r2r_initial_state_imag': np.empty(prefix + (1, 1), np.float32),
        '_r2r_initial_stoch': np.empty(prefix + (1,), np.uint8),
        '_r2r_initial_action': np.empty(prefix, np.uint16),
        '_r2r_initial_valid': np.empty(prefix, bool),
        '_r2r_future_adjoint_real': np.empty(prefix + (1, 1), np.float32),
        '_r2r_future_adjoint_imag': np.empty(prefix + (1, 1), np.float32),
        '_r2r_future_adjoint_valid': np.empty(prefix, bool),
    }

  def test_fixed_seed_choices_and_offsets_match_unmodified_r2i(self):
    seed, length, chunks = 19, 8, 7
    replay = self._replay(seed, length, chunks)
    buffer = self._buffer(length)
    reference_sampler = selectors.Uniform(seed)
    for key in range(chunks):
      reference_sampler[key] = key
    reference_rng = np.random.default_rng(seed)
    expected = [
        (reference_sampler(), reference_rng.integers(1, length + 1).item())
        for _ in range(256)]
    observed = []
    for _ in range(256):
      replay.sample(0, 0, buffer)
      values = buffer['value'][0, 0]
      current = int(values[-1] // 100)
      end = int(values[-1] % 100) + 1
      observed.append((current, end))
    self.assertEqual(observed, expected)

  def test_full_chunk_missing_cache_predecessor_does_not_resample(self):
    class Sampler:
      def __init__(self):
        self.calls = 0
      def __call__(self):
        self.calls += 1
        return 1

    class FullLengthRng:
      def integers(self, low, high):
        del low
        return np.asarray(high - 1, np.int64)

    replay = self._replay(3, length=4, chunks=2)
    replay.sampler = Sampler()
    replay.rng = FullLengthRng()
    replay.bwd_links = {1: 0}

    class RacingManager(_Manager):
      def read_chunk(self, flip, offset, worker, half):
        result = super().read_chunk(flip, offset, worker, half)
        replay.bwd_links.clear()
        return result

    replay.manager = RacingManager(4)
    buffer = self._buffer(4)
    replay.sample(0, 0, buffer)
    self.assertEqual(replay.sampler.calls, 1)
    self.assertEqual(int(buffer['_r2r_anchor_slot'][0, 0]), -1)
    self.assertEqual(int(buffer['_r2r_anchor_generation'][0, 0]), -1)

  def test_replay_specific_batch_allocator_is_not_shadowed(self):
    replay = self._replay(3, length=4, chunks=2)
    replay.batch_buffer = None  # Retained upstream instance field.
    replay.serializer = types.SimpleNamespace(
        batch_buffer=lambda count, batch, length: {
            'value': np.empty((count, batch, length), np.int32)})
    batch = replay.make_batch_buffer(2, 3, 4)
    self.assertEqual(batch['value'].shape, (2, 3, 4))
    self.assertEqual(batch['_r2r_slot'].shape, (2, 3, 4))

  def test_lustre_mirror_resume_restores_replay_and_cache(self):
    class Step:
      def __init__(self):
        self.value = -1
      def load(self, value):
        self.value = int(value)

    class Agent:
      state_adjoint_cache_spec = {
          'enabled': True, 'layers': 1, 'width': 2, 'stoch': 1,
          'classes': 2, 'action_dim': 2}
      def __init__(self):
        self.step = Step()
        self.loaded = None
      def save(self):
        return {'weight': np.asarray([17], np.int32)}
      def load(self, value):
        self.loaded = value

    with tempfile.TemporaryDirectory() as temporary:
      root = Path(temporary)
      mirror = root / 'mirror'
      first = generic_lfs.FIFO_LFS(
          root / 'local1', length=4, capacity=16,
          lfs_directory=mirror, use_lfs=True, batch_size=1, num_buffers=1,
          lfs_kwargs={'prefix_size_mb': 1}, seed=7)
      first_agent = Agent()
      first.set_agent(first_agent)
      for index in range(12):
        first.add({
            'value': np.int32(index),
            'is_first': np.bool_(index in (0, 8)),
            'is_last': np.bool_(index in (7,)),
        })
      payload = {
          'state_slots': np.asarray([5], np.int32),
          'state_generations': np.asarray([0], np.int32),
          'state_bits': np.zeros((1, 1, 2, 2), np.uint16),
          'stoch': np.zeros((1, 1), np.uint8),
          'action': np.zeros((1,), np.uint16),
          'adjoint_slots': np.asarray([5], np.int32),
          'adjoint_generations': np.asarray([0], np.int32),
          'adjoint_bits': np.zeros((1, 1, 2, 2), np.uint16),
          'terminal_slots': np.empty((0,), np.int32),
          'terminal_generations': np.empty((0,), np.int32),
      }
      first.cache.commit(payload)
      first.save()
      first.cache.close()
      first.manager.tmp_file.close()
      first.manager.lfs_file.close()

      restored = generic_lfs.FIFO_LFS(
          root / 'local2', length=4, capacity=16,
          lfs_directory=mirror, use_lfs=True, batch_size=1, num_buffers=1,
          lfs_kwargs={'prefix_size_mb': 1}, seed=7)
      restored_agent = Agent()
      restored.set_agent(restored_agent)
      self.assertIsNone(restored_agent.loaded)
      self.assertIsNone(restored.maybe_restore())
      self.assertEqual(len(restored), 12)
      self.assertEqual(restored_agent.step.value, 12)
      self.assertEqual(int(restored_agent.loaded['weight'][0]), 17)
      gathered = restored.cache.gather_boundaries(
          np.asarray([5]), np.asarray([0]),
          np.asarray([5]), np.asarray([0]))
      self.assertTrue(gathered['initial_valid'][0])
      self.assertTrue(gathered['future_adjoint_valid'][0])
      restored.cache.close()
      restored.manager.tmp_file.close()
      restored.manager.lfs_file.close()

  def test_replay_ring_wraparound_invalidates_old_generation(self):
    class Step:
      def load(self, value):
        del value

    class Agent:
      state_adjoint_cache_spec = {
          'enabled': True, 'layers': 1, 'width': 2, 'stoch': 1,
          'classes': 2, 'action_dim': 2}
      def __init__(self):
        self.step = Step()
      def save(self):
        return {}
      def load(self, value):
        del value

    with tempfile.TemporaryDirectory() as temporary:
      replay = generic_lfs.FIFO_LFS(
          Path(temporary) / 'local', length=4, capacity=8,
          use_lfs=False, batch_size=1, num_buffers=1,
          lfs_kwargs={'prefix_size_mb': 1}, seed=11)
      replay.set_agent(Agent())
      for index in range(8):
        replay.add({
            'value': np.int32(index),
            'is_first': np.bool_(index == 0),
            'is_last': np.bool_(False),
        })
      replay.cache.commit({
          'state_slots': np.asarray([1], np.int32),
          'state_generations': np.asarray([0], np.int32),
          'state_bits': np.zeros((1, 1, 2, 2), np.uint16),
          'stoch': np.zeros((1, 1), np.uint8),
          'action': np.zeros((1,), np.uint16),
          'adjoint_slots': np.asarray([1], np.int32),
          'adjoint_generations': np.asarray([0], np.int32),
          'adjoint_bits': np.zeros((1, 1, 2, 2), np.uint16),
          'terminal_slots': np.empty((0,), np.int32),
          'terminal_generations': np.empty((0,), np.int32),
      })
      for index in range(8, 12):
        replay.add({
            'value': np.int32(index),
            'is_first': np.bool_(False),
            'is_last': np.bool_(False),
        })
      self.assertEqual(replay.manager.overwrite_layers, 1)
      self.assertEqual(len(replay), 8)
      np.testing.assert_array_equal(
          replay.cache.generations(np.arange(4)), np.ones(4, np.int32))
      gathered = replay.cache.gather_boundaries(
          np.asarray([1]), np.asarray([0]),
          np.asarray([1]), np.asarray([0]))
      self.assertFalse(gathered['initial_valid'][0])
      self.assertFalse(gathered['future_adjoint_valid'][0])
      replay.cache.close()
      replay.manager.tmp_file.close()


if __name__ == '__main__':
  unittest.main()
