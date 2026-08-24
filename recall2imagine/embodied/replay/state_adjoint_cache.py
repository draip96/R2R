"""Dense replay-aligned state and boundary-adjoint storage for R2R.

The cache is deliberately a passive sidecar.  It never participates in replay
selection and therefore cannot change which sequence is sampled.  Replay owns
physical slots and overwrite generations; this module only validates those
identities before reading or replacing values.

Complex values are stored as two raw bfloat16 components.  NumPy versions used
by the original R2I environment do not expose a native bfloat16 dtype, so the
on-disk representation is uint16 and conversion is implemented explicitly.
"""

import json
import os
import shutil
import threading
import math
from pathlib import Path

import numpy as np


CACHE_VERSION = 1


def aligned_capacity(capacity, chunk_length):
  """Round transition capacity up to a whole physical replay chunk."""
  capacity = int(capacity)
  chunk_length = int(chunk_length)
  if capacity <= 0 or chunk_length <= 0:
    raise ValueError((capacity, chunk_length))
  return int(math.ceil(capacity / chunk_length) * chunk_length)


def float32_to_bf16_bits(values):
  """Round float32 values to bfloat16 and return their raw uint16 bits."""
  values = np.asarray(values, dtype=np.float32)
  bits = values.view(np.uint32)
  rounding = np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))
  return ((bits + rounding) >> np.uint32(16)).astype(np.uint16)


def bf16_bits_to_float32(values):
  """Decode raw bfloat16 uint16 bits into float32 values."""
  values = np.asarray(values, dtype=np.uint16)
  return (values.astype(np.uint32) << np.uint32(16)).view(np.float32)


def complex_to_bf16_bits(values):
  values = np.asarray(values, dtype=np.complex64)
  return np.stack(
      (float32_to_bf16_bits(values.real),
       float32_to_bf16_bits(values.imag)), axis=-1)


def bf16_bits_to_complex(values):
  values = np.asarray(values, dtype=np.uint16)
  if values.shape[-1] != 2:
    raise ValueError('complex bfloat16 storage requires a final dimension of 2')
  real = bf16_bits_to_float32(values[..., 0])
  imag = bf16_bits_to_float32(values[..., 1])
  return (real + 1j * imag).astype(np.complex64)


def cache_bytes(capacity, layers, width):
  """Bytes occupied by the dense x/G payload, excluding small metadata."""
  # Two caches, two bfloat16 components, two bytes per component.
  return int(capacity) * int(layers) * int(width) * 2 * 2 * 2


def sample_slot_layout(
    chunk_length, end_position, current_offset, current_generation,
    previous_offset, previous_generation):
  """Map an unchanged R2I chunk/offset sample to stable physical slots."""
  length = int(chunk_length)
  end = int(end_position)
  current_offset = int(current_offset)
  previous_offset = int(previous_offset)
  current_generation = int(current_generation)
  previous_generation = int(previous_generation)
  if length <= 0 or not 1 <= end <= length:
    raise ValueError((length, end))
  if current_offset < 0:
    raise ValueError(current_offset)
  if previous_offset < 0:
    if end != length:
      raise ValueError(
          'a spliced replay sample requires a physical predecessor')
    return {
        'slots': current_offset * length + np.arange(length, dtype=np.int32),
        'generations': np.full(
            length, current_generation, dtype=np.int32),
        'anchor_slot': np.int32(-1),
        'anchor_generation': np.int32(-1),
    }
  if end < length:
    slots = np.concatenate((
        previous_offset * length + np.arange(end, length, dtype=np.int32),
        current_offset * length + np.arange(end, dtype=np.int32)))
    generations = np.concatenate((
        np.full(length - end, previous_generation, dtype=np.int32),
        np.full(end, current_generation, dtype=np.int32)))
    anchor_slot = previous_offset * length + end - 1
  else:
    slots = current_offset * length + np.arange(length, dtype=np.int32)
    generations = np.full(length, current_generation, dtype=np.int32)
    anchor_slot = previous_offset * length + length - 1
  return {
      'slots': slots,
      'generations': generations,
      'anchor_slot': np.int32(anchor_slot),
      'anchor_generation': np.int32(previous_generation),
  }


def _atomic_json(path, value):
  temporary = path.with_suffix(path.suffix + '.tmp')
  temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
  os.replace(str(temporary), str(path))


def _runs(indexes):
  """Yield half-open contiguous runs from sorted unique integer indexes."""
  indexes = np.unique(np.asarray(indexes, dtype=np.int64))
  if not len(indexes):
    return
  start = previous = int(indexes[0])
  for value in indexes[1:]:
    value = int(value)
    if value != previous + 1:
      yield start, previous + 1
      start = value
    previous = value
  yield start, previous + 1


class DenseStateAdjointCache:
  """Memory-mapped x/G cache aligned with a chunked FIFO replay ring."""

  ARRAY_SPECS = {
      'state': (np.uint16, ('capacity', 'layers', 'width', 2)),
      'adjoint': (np.uint16, ('capacity', 'layers', 'width', 2)),
      'stoch': (np.uint8, ('capacity', 'stoch')),
      'action': (np.uint16, ('capacity',)),
      'state_valid': (np.bool_, ('capacity',)),
      'adjoint_valid': (np.bool_, ('capacity',)),
      'generation': (np.int32, ('capacity',)),
  }

  def __init__(
      self, directory, capacity, layers, width, stoch, classes, action_dim,
      mirror_directory=None):
    self.directory = Path(directory)
    self.directory.mkdir(parents=True, exist_ok=True)
    self.mirror_directory = (
        None if mirror_directory is None else Path(mirror_directory))
    if self.mirror_directory is not None:
      self.mirror_directory.mkdir(parents=True, exist_ok=True)
    self.capacity = int(capacity)
    self.layers = int(layers)
    self.width = int(width)
    self.stoch = int(stoch)
    self.classes = int(classes)
    self.action_dim = int(action_dim)
    if min(self.capacity, self.layers, self.width, self.stoch,
           self.classes, self.action_dim) <= 0:
      raise ValueError('all cache dimensions must be positive')
    if self.classes > 256:
      raise ValueError('uint8 posterior indexes require at most 256 classes')
    if self.action_dim > 65536:
      raise ValueError('uint16 action indexes require at most 65536 actions')
    self.spec = {
        'version': CACHE_VERSION,
        'capacity': self.capacity,
        'layers': self.layers,
        'width': self.width,
        'stoch': self.stoch,
        'classes': self.classes,
        'action_dim': self.action_dim,
        'state_adjoint_bytes': cache_bytes(
            self.capacity, self.layers, self.width),
    }
    self._lock = threading.RLock()
    self._dirty = np.zeros(self.capacity, dtype=np.bool_)
    self._prepare_metadata()
    self.arrays = {}
    self.mirror_arrays = {}
    for name, (dtype, symbolic_shape) in self.ARRAY_SPECS.items():
      shape = tuple(self.spec[item] if isinstance(item, str) else item
                    for item in symbolic_shape)
      self.arrays[name], created = self._open_array(
          self.directory, name, dtype, shape, copy_from=self.mirror_directory)
      if created and name in ('state_valid', 'adjoint_valid'):
        self.arrays[name][:] = False
        self.arrays[name].flush()
      if created and name == 'generation':
        self.arrays[name][:] = -1
        self.arrays[name].flush()
      if self.mirror_directory is not None:
        self.mirror_arrays[name], mirror_created = self._open_array(
            self.mirror_directory, name, dtype, shape)
        if mirror_created and name in ('state_valid', 'adjoint_valid'):
          self.mirror_arrays[name][:] = False
          self.mirror_arrays[name].flush()
        if mirror_created and name == 'generation':
          self.mirror_arrays[name][:] = -1
          self.mirror_arrays[name].flush()

  def _prepare_metadata(self):
    local = self.directory / 'metadata.json'
    mirror = (
        None if self.mirror_directory is None
        else self.mirror_directory / 'metadata.json')
    if not local.exists() and mirror is not None and mirror.exists():
      shutil.copyfile(str(mirror), str(local))
    if local.exists():
      found = json.loads(local.read_text())
      if found != self.spec:
        raise ValueError(
            'state-adjoint cache metadata mismatch:\n'
            'expected {}\nfound {}'.format(self.spec, found))
    else:
      _atomic_json(local, self.spec)
    if mirror is not None:
      if mirror.exists():
        found = json.loads(mirror.read_text())
        if found != self.spec:
          raise ValueError('persistent cache metadata mismatch')
      else:
        _atomic_json(mirror, self.spec)

  @staticmethod
  def _open_array(directory, name, dtype, shape, copy_from=None):
    path = directory / (name + '.bin')
    source = None if copy_from is None else copy_from / (name + '.bin')
    created = False
    if not path.exists() and source is not None and source.exists():
      shutil.copyfile(str(source), str(path))
    expected = int(np.prod(shape)) * np.dtype(dtype).itemsize
    if not path.exists():
      descriptor = os.open(str(path), os.O_RDWR | os.O_CREAT, 0o600)
      try:
        os.ftruncate(descriptor, expected)
      finally:
        os.close(descriptor)
      created = True
    actual = path.stat().st_size
    if actual != expected:
      raise ValueError(
          '{} has {} bytes but {} are required'.format(path, actual, expected))
    return np.memmap(str(path), mode='r+', dtype=dtype, shape=shape), created

  @property
  def stats(self):
    with self._lock:
      state_valid = int(np.count_nonzero(self.arrays['state_valid']))
      adjoint_valid = int(np.count_nonzero(self.arrays['adjoint_valid']))
      return {
          'capacity': self.capacity,
          'state_valid': state_valid,
          'adjoint_valid': adjoint_valid,
          'dirty': int(np.count_nonzero(self._dirty)),
          'state_adjoint_bytes': self.spec['state_adjoint_bytes'],
      }

  def invalidate_chunk(self, chunk_offset, generation, chunk_length):
    start = int(chunk_offset) * int(chunk_length)
    stop = start + int(chunk_length)
    if start < 0 or stop > self.capacity:
      raise IndexError((start, stop, self.capacity))
    with self._lock:
      self.arrays['generation'][start:stop] = int(generation)
      self.arrays['state_valid'][start:stop] = False
      self.arrays['adjoint_valid'][start:stop] = False
      self._dirty[start:stop] = True

  def reconcile_chunk(self, chunk_offset, generation, chunk_length):
    """Invalidate a restored physical chunk only when its identity differs."""
    start = int(chunk_offset) * int(chunk_length)
    stop = start + int(chunk_length)
    if start < 0 or stop > self.capacity:
      raise IndexError((start, stop, self.capacity))
    with self._lock:
      if np.all(self.arrays['generation'][start:stop] == int(generation)):
        return False
    self.invalidate_chunk(chunk_offset, generation, chunk_length)
    return True

  def generations(self, slots):
    slots = np.asarray(slots, dtype=np.int32)
    result = np.full(slots.shape, -1, dtype=np.int32)
    valid = (slots >= 0) & (slots < self.capacity)
    with self._lock:
      result[valid] = self.arrays['generation'][slots[valid]]
    return result

  def gather_boundaries(
      self, anchor_slots, anchor_generations, future_slots,
      future_generations):
    anchor_slots = np.asarray(anchor_slots, dtype=np.int32)
    anchor_generations = np.asarray(anchor_generations, dtype=np.int32)
    future_slots = np.asarray(future_slots, dtype=np.int32)
    future_generations = np.asarray(future_generations, dtype=np.int32)
    if anchor_slots.shape != future_slots.shape:
      raise ValueError('anchor and future slot batches must have equal shape')
    batch = anchor_slots.size
    state = np.zeros(
        (batch, self.layers, self.width), dtype=np.complex64)
    adjoint = np.zeros_like(state)
    stoch = np.zeros((batch, self.stoch), dtype=np.uint8)
    action = np.zeros((batch,), dtype=np.uint16)
    state_valid = np.zeros((batch,), dtype=np.bool_)
    adjoint_valid = np.zeros((batch,), dtype=np.bool_)
    with self._lock:
      anchor_range = (
          (anchor_slots >= 0) & (anchor_slots < self.capacity))
      if np.any(anchor_range):
        rows = np.nonzero(anchor_range)[0]
        slots = anchor_slots[rows]
        matching = (
            self.arrays['generation'][slots] == anchor_generations[rows])
        rows = rows[matching]
        slots = anchor_slots[rows]
        matching = self.arrays['state_valid'][slots]
        rows = rows[matching]
        slots = anchor_slots[rows]
        if len(rows):
          state[rows] = bf16_bits_to_complex(self.arrays['state'][slots])
          stoch[rows] = self.arrays['stoch'][slots]
          action[rows] = self.arrays['action'][slots]
          state_valid[rows] = True
      future_range = (
          (future_slots >= 0) & (future_slots < self.capacity))
      if np.any(future_range):
        rows = np.nonzero(future_range)[0]
        slots = future_slots[rows]
        matching = (
            self.arrays['generation'][slots] == future_generations[rows])
        rows = rows[matching]
        slots = future_slots[rows]
        matching = self.arrays['adjoint_valid'][slots]
        rows = rows[matching]
        slots = future_slots[rows]
        if len(rows):
          adjoint[rows] = bf16_bits_to_complex(
              self.arrays['adjoint'][slots])
          adjoint_valid[rows] = True
    return {
        'initial_state_real': state.real.astype(np.float32),
        'initial_state_imag': state.imag.astype(np.float32),
        'initial_stoch': stoch,
        'initial_action': action,
        'initial_valid': state_valid,
        'future_adjoint_real': adjoint.real.astype(np.float32),
        'future_adjoint_imag': adjoint.imag.astype(np.float32),
        'future_adjoint_valid': adjoint_valid,
    }

  def _last_valid_rows(self, slots, generations):
    slots = np.asarray(slots, dtype=np.int32).reshape(-1)
    generations = np.asarray(generations, dtype=np.int32).reshape(-1)
    if slots.shape != generations.shape:
      raise ValueError('slot and generation arrays must have equal shape')
    chosen = {}
    for index, (slot, generation) in enumerate(zip(slots, generations)):
      slot = int(slot)
      if slot < 0 or slot >= self.capacity:
        continue
      if int(self.arrays['generation'][slot]) != int(generation):
        continue
      chosen[slot] = index
    if not chosen:
      return np.empty((0,), np.int64), np.empty((0,), np.int32)
    indexes = np.asarray(sorted(chosen.values()), dtype=np.int64)
    return indexes, slots[indexes]

  def commit(self, payload):
    """Directly replace sampled rows; stable last occurrence wins."""
    required = (
        'state_slots', 'state_generations', 'state_bits', 'stoch', 'action',
        'adjoint_slots', 'adjoint_generations', 'adjoint_bits',
        'terminal_slots', 'terminal_generations')
    missing = [key for key in required if key not in payload]
    if missing:
      raise KeyError('cache payload missing {}'.format(missing))
    state_bits = np.asarray(payload['state_bits'], dtype=np.uint16)
    adjoint_bits = np.asarray(payload['adjoint_bits'], dtype=np.uint16)
    expected_tail = (self.layers, self.width, 2)
    if state_bits.shape[-3:] != expected_tail:
      raise ValueError(('state', state_bits.shape, expected_tail))
    if adjoint_bits.shape[-3:] != expected_tail:
      raise ValueError(('adjoint', adjoint_bits.shape, expected_tail))
    # A bfloat16 is non-finite exactly when all exponent bits are set. Check
    # the compact representation directly to avoid materializing another
    # state-sized complex array on every learner update.
    nonfinite = lambda bits: np.any(
        (bits & np.uint16(0x7F80)) == np.uint16(0x7F80))
    if nonfinite(state_bits):
      raise FloatingPointError('non-finite recurrent state cache update')
    if nonfinite(adjoint_bits):
      raise FloatingPointError('non-finite boundary adjoint cache update')
    with self._lock:
      state_rows, state_slots = self._last_valid_rows(
          payload['state_slots'], payload['state_generations'])
      flat_state = state_bits.reshape((-1,) + expected_tail)
      flat_stoch = np.asarray(payload['stoch'], dtype=np.uint8).reshape(
          (-1, self.stoch))
      flat_action = np.asarray(payload['action'], dtype=np.uint16).reshape(-1)
      if len(flat_state) != len(flat_stoch) or len(flat_state) != len(flat_action):
        raise ValueError('state, posterior, and action update lengths differ')
      if np.any(flat_stoch >= self.classes):
        raise ValueError('categorical posterior cache index is out of range')
      if np.any(flat_action >= self.action_dim):
        raise ValueError('cached action index is out of range')
      if len(state_rows):
        self.arrays['state'][state_slots] = flat_state[state_rows]
        self.arrays['stoch'][state_slots] = flat_stoch[state_rows]
        self.arrays['action'][state_slots] = flat_action[state_rows]
        self.arrays['state_valid'][state_slots] = True
        self._dirty[state_slots] = True

      adjoint_rows, adjoint_slots = self._last_valid_rows(
          payload['adjoint_slots'], payload['adjoint_generations'])
      flat_adjoint = adjoint_bits.reshape((-1,) + expected_tail)
      if len(adjoint_rows):
        self.arrays['adjoint'][adjoint_slots] = flat_adjoint[adjoint_rows]
        self.arrays['adjoint_valid'][adjoint_slots] = True
        self._dirty[adjoint_slots] = True

      # True terminal boundaries have no future loss, regardless of write order.
      terminal_rows, terminal_slots = self._last_valid_rows(
          payload['terminal_slots'], payload['terminal_generations'])
      del terminal_rows
      if len(terminal_slots):
        self.arrays['adjoint'][terminal_slots] = 0
        self.arrays['adjoint_valid'][terminal_slots] = True
        self._dirty[terminal_slots] = True
    return {
        'state_rows_written': int(len(state_slots)),
        'adjoint_rows_written': int(len(adjoint_slots)),
        'terminal_rows_zeroed': int(len(terminal_slots)),
    }

  def flush(self, mirror=False):
    with self._lock:
      for array in self.arrays.values():
        array.flush()
      if mirror and self.mirror_arrays:
        dirty = np.nonzero(self._dirty)[0]
        ranges = tuple(_runs(dirty))
        # Two-phase dirty-range publication. A crash can leave an old cache hit
        # or a conservative miss, but never a valid row whose payload and
        # overwrite generation came from different replay chunks.
        for start, stop in ranges:
          self.mirror_arrays['state_valid'][start:stop] = False
          self.mirror_arrays['adjoint_valid'][start:stop] = False
        self.mirror_arrays['state_valid'].flush()
        self.mirror_arrays['adjoint_valid'].flush()
        payload_names = ('state', 'adjoint', 'stoch', 'action')
        for start, stop in ranges:
          for name in payload_names:
            self.mirror_arrays[name][start:stop] = self.arrays[name][start:stop]
        for name in payload_names:
          self.mirror_arrays[name].flush()
        for start, stop in ranges:
          self.mirror_arrays['generation'][start:stop] = self.arrays[
              'generation'][start:stop]
        self.mirror_arrays['generation'].flush()
        for start, stop in _runs(dirty):
          self.mirror_arrays['state_valid'][start:stop] = self.arrays[
              'state_valid'][start:stop]
          self.mirror_arrays['adjoint_valid'][start:stop] = self.arrays[
              'adjoint_valid'][start:stop]
        self.mirror_arrays['state_valid'].flush()
        self.mirror_arrays['adjoint_valid'].flush()
        self._dirty[dirty] = False

  def close(self):
    self.flush(mirror=bool(self.mirror_arrays))
    self.arrays.clear()
    self.mirror_arrays.clear()
