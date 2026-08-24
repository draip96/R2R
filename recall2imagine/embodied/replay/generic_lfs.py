import time
import pickle
import bz2
import os
from collections import defaultdict, deque
from functools import partial as bind
import pickle

import embodied
import bz2
import numpy as np
import io

from . import saver
from . import selectors, limiters
from .lfs_manager import LFSManager
from . import selectors
from .chunk import Chunk, ChunkSerializer
from .state_adjoint_cache import (
    DenseStateAdjointCache, aligned_capacity, sample_slot_layout)

class FIFO_LFS:
  """
  This class represents a standard FIFO replay buffer where the data is stored on disk.
  After the first chuck of experience arrives, the buffer obtains the data format
  and initializes the replay buffer file. This file contains chunks 
  The main idea behind this implementation is to make the training 
  continuable after interruptions, since we not only need a saved model but also
  the replay buffer.
  
  Therefore, the following (optional) trick is implemented. There are two versions of the buffer. 
  The first is stored in some fast and easily accessible storage (e.g. SSD disk).
  The second is stored in some potentially slower but more persistent storage.
  These versions of the buffer are synced in amortized O(1) time via copy-on-write mechanism.
  If you do not need this dual buffer system, turn it off via `use_lfs=False`.

  For implementation details, see the code comments below.

  Typical buffer size for image envs: 134GB for the 10M buffer; 
  for vector envs: 8GB for the 10M buffer.
  """
  def __init__(
      self, directory, length, capacity, #sampler, limiter,
      overlap=None, online=False, lfs_directory=None, 
      lfs_kwargs=None, samples_per_insert=None,
      use_lfs=False, unlocked_sampling=False,
      tolerance=1e4, min_size=1, batch_size=1,
      num_buffers=2, seed=0):
    assert capacity is None or 1 <= capacity
    if lfs_kwargs is None:
      lfs_kwargs = {}
    # we have to be VERY careful with batch size and
    # num buffers here. if they get corrupted 
    # and diverge from actual batch size and the 
    # bufferization factor, the data agent loads
    # will be a garbage without any notice 
    self.manager = LFSManager(
      tmp_path=directory, lfs_path=lfs_directory,
      readers=batch_size, num_buffers=num_buffers,
      replay_buffer_size=capacity,
      use_lfs=use_lfs, **lfs_kwargs,
      saver_method=self.save, loader_method=self.load
    )
    self.serializer = None
    self.batch_buffer = None
    self.num_buffers = num_buffers
    self.length = length
    self.capacity = capacity
    self.chunks = length
    self.remover = selectors.Fifo()
    self.sampler = selectors.Uniform(seed)
    if samples_per_insert:
      self.limiter = limiters.SamplesPerInsert(
        samples_per_insert, tolerance, min_size, unlocked_sampling)
    else:
      self.limiter = limiters.MinSize(min_size)
    self.stride = 1 if overlap is None else length - overlap
    self.chunk_buffers = defaultdict(bind(Chunk, self.chunks))
    self.streams = defaultdict(bind(deque, maxlen=length))
    self.counters = defaultdict(int)
    self.rng = np.random.default_rng(seed)
    self.was = defaultdict(bool)
    self.table = {}
    self.serializer_pattern = None
    self.bwd_links = {}
    self.fwd_links = {}
    self.inv_table = {}
    self.chunk_generations = {}
    self.online = online
    self.cache = None
    self.cache_write_metrics = {
        'state_rows_written': 0,
        'adjoint_rows_written': 0,
        'terminal_rows_zeroed': 0,
    }
    if self.online:
      self.online_queue = deque()
      self.online_stride = length
      self.online_counters = defaultdict(int)
    self.metrics = {
        'samples': 0,
        'sample_wait_dur': 0,
        'sample_wait_count': 0,
        'inserts': 0,
        'insert_wait_dur': 0,
        'insert_wait_count': 0,
    }


  def set_agent(self, agent):
    self._agent = agent
    spec = getattr(agent, 'state_adjoint_cache_spec', None)
    if not spec or not spec.get('enabled', False) or self.cache is not None:
      return
    padded_capacity = aligned_capacity(self.capacity, self.length)
    local = os.path.join(str(self.manager.tmp_path), 'state_adjoint_cache')
    mirror = None
    if self.manager.use_lfs:
      mirror = os.path.join(
          str(self.manager.lfs_path), 'state_adjoint_cache')
    self.cache = DenseStateAdjointCache(
        directory=local,
        mirror_directory=mirror,
        capacity=padded_capacity,
        layers=spec['layers'],
        width=spec['width'],
        stoch=spec['stoch'],
        classes=spec['classes'],
        action_dim=spec['action_dim'])
    print('R2R state-adjoint cache:', self.cache.spec)

  def __len__(self):
    return len(self.table) * self.length
  
  @property
  def initialized(self):
    return self.manager.initialized

  @property
  def stats(self):
    ratio = lambda x, y: x / y if y else np.nan
    m = self.metrics
    stats = {
        'size': len(self),
        'inserts': m['inserts'],
        'samples': m['samples'],
        'insert_wait_avg': ratio(m['insert_wait_dur'], m['inserts']),
        'insert_wait_frac': ratio(m['insert_wait_count'], m['inserts']),
        'sample_wait_avg': ratio(m['sample_wait_dur'], m['samples']),
        'sample_wait_frac': ratio(m['sample_wait_count'], m['samples']),
    }
    if self.cache is not None:
      stats.update({
          'cache_' + key: value
          for key, value in self.cache_write_metrics.items()})
      stats['cache_state_adjoint_bytes'] = self.cache.spec[
          'state_adjoint_bytes']
    for key in self.metrics:
      self.metrics[key] = 0
    return stats

  def add(self, step, worker=0, load=False):
    step = {k: v for k, v in step.items() if not k.startswith('log_')}
    step['id'] = np.asarray(embodied.uuid(step.get('id')))
    self.chunk_buffers[worker].append(step)
    self.counters[worker] += 1
    if self.serializer is None:
      self.serializer = ChunkSerializer(self.chunk_buffers[worker])
      self.manager.serializer = self.serializer
      self.manager.initialize(environment_step=step, length=self.length)
    if self.was[worker] >= 2:
      if load:
        assert self.limiter.want_load()[0]
      else:
        dur = wait(self.limiter.want_insert, 'Replay insert is waiting')
        self.metrics['inserts'] += 1                             
        self.metrics['insert_wait_dur'] += dur
        self.metrics['insert_wait_count'] += int(dur > 0)
    if self.counters[worker] < self.length:
      return
    # to think about.
    # when the buffer is restrored from the saved state
    # we get new worker ids here (in the self.was object). Without much pondering, 
    # it seems to me that this should not have any negative effects 
    # on the result but it better to keep that in mind
    self.was[worker] += 1
    self.counters[worker] = 0
    old_buffer = self.chunk_buffers[worker]
    self.chunk_buffers[worker] = Chunk(self.chunks)
    old_buffer.successor = self.chunk_buffers[worker].uuid_b
    expected_offset = self.manager.offset
    write_generation = self.manager.overwrite_layers
    if self.cache is not None:
      # Invalidate before the replay write because write_chunk() may trigger a
      # persistent checkpoint through the LFS manager.
      self.cache.invalidate_chunk(
          expected_offset, write_generation, self.length)
    _, offset = self.manager.write_chunk(old_buffer)
    if self.cache is not None and offset != expected_offset:
      raise RuntimeError('replay and state cache physical offsets diverged')
    if offset in self.inv_table:
      # we are overriding the table entry
      try: # handling race condition
        key = self.inv_table[offset]
        del self.table[key]
        del self.sampler[key]
        self.chunk_generations.pop(key, None)
      except KeyError:
        pass
      if key in self.bwd_links:
        try: # handling race condition
          del self.bwd_links[key]
        except KeyError:
          pass
      else:
        try: # handling race condition
          nxt = self.fwd_links[key]
          del self.bwd_links[nxt]
        except KeyError:
            pass
        try: # handling race condition
          del self.fwd_links[key]
        except KeyError:
            pass
    self.inv_table[offset] = old_buffer.uuid_b
    self.table[old_buffer.uuid_b] = offset
    self.chunk_generations[old_buffer.uuid_b] = write_generation
    self.bwd_links[old_buffer.successor] = old_buffer.uuid_b
    self.fwd_links[old_buffer.uuid_b] = old_buffer.successor 
    self.sampler[old_buffer.uuid_b] = offset

  @property
  def ready(self):
    return self.serializer is not None

  def make_batch_buffer(self, num_buffers, batch_size, sequence_length):
    batch = self.serializer.batch_buffer(
        num_buffers, batch_size, sequence_length)
    if self.cache is None:
      return batch
    prefix = (num_buffers, batch_size)
    batch.update({
        '_r2r_slot': np.empty(
            prefix + (sequence_length,), dtype=np.int32),
        '_r2r_generation': np.empty(
            prefix + (sequence_length,), dtype=np.int32),
        '_r2r_anchor_slot': np.empty(prefix, dtype=np.int32),
        '_r2r_anchor_generation': np.empty(prefix, dtype=np.int32),
        '_r2r_initial_state_real': np.empty(
            prefix + (self.cache.layers, self.cache.width), np.float32),
        '_r2r_initial_state_imag': np.empty(
            prefix + (self.cache.layers, self.cache.width), np.float32),
        '_r2r_initial_stoch': np.empty(
            prefix + (self.cache.stoch,), np.uint8),
        '_r2r_initial_action': np.empty(prefix, np.uint16),
        '_r2r_initial_valid': np.empty(prefix, np.bool_),
        '_r2r_future_adjoint_real': np.empty(
            prefix + (self.cache.layers, self.cache.width), np.float32),
        '_r2r_future_adjoint_imag': np.empty(
            prefix + (self.cache.layers, self.cache.width), np.float32),
        '_r2r_future_adjoint_valid': np.empty(prefix, np.bool_),
    })
    return batch

  def serialize(self, ):
    data = {
      'fwd_links': self.fwd_links,
      'bwd_links': self.bwd_links,
      'table': self.table,
      'inv_table': self.inv_table,
      'sampler': (self.sampler.indices, self.sampler.keys),
      'was': dict(self.was),
      'offset': self.manager.offset,
      'layers': self.manager.overwrite_layers,
      'chunk_generations': self.chunk_generations,
    }
    if self.serializer is not None:
      data['serializer'] = self.serializer.pattern
    else:
      data['serializer'] = self.serializer_pattern
    data = pickle.dumps(data)
    return np.frombuffer(bz2.compress(data), dtype=np.uint8)
  
  def deserialize(self, data):
    data = pickle.loads(bz2.decompress(data.tobytes()))
    self.serializer_pattern = data['serializer']
    if self.manager.serializer is None:
      serializer = ChunkSerializer(pattern=self.serializer_pattern, pattern_obj=None)
      env_step = {k: v[0] for k, v in serializer.dummy_chunk().items()}
      self.serializer = serializer
      self.manager.serializer = serializer
      # We are already deserializing the prefix. Calling the loader again from
      # initialize() would recursively re-enter this method on resume.
      self.manager.initialize(
          environment_step=env_step, length=self.length, load_prefix=False)
    self.fwd_links = data['fwd_links']
    self.bwd_links = data['bwd_links']
    self.table = data['table']
    self.inv_table = data['inv_table']
    self.was = defaultdict(bool)
    self.was.update(data['was'])
    self.sampler.indices, self.sampler.keys = data['sampler']
    self.manager.offset = data['offset']
    self.manager.lfs_offset = data['offset']
    self.manager.overwrite_layers = data['layers']
    if 'chunk_generations' in data:
      self.chunk_generations = data['chunk_generations']
    else:
      # Safe migration for an upstream replay created before R2R tracked
      # physical generations explicitly. Entries behind the write pointer are
      # from the current ring pass; entries ahead are from the previous pass.
      layer = int(self.manager.overwrite_layers)
      pointer = int(self.manager.offset)
      self.chunk_generations = {
          key: (layer if layer == 0 or offset < pointer else layer - 1)
          for key, offset in self.table.items()}
    if self.cache is not None:
      for key, offset in self.table.items():
        self.cache.reconcile_chunk(
            offset, self.chunk_generations[key], self.length)

  def sample(self, flip: int, worker_id: int, batch_buffer):
    dur = wait(self.limiter.want_sample, 'Replay sample is waiting')
    self.metrics['samples'] += 1
    self.metrics['sample_wait_dur'] += dur
    self.metrics['sample_wait_count'] += int(dur > 0)
    trying = True
    while trying:
      trying = False
      key = self.sampler()
      while key not in self.bwd_links or self.bwd_links[key] not in self.table:
        key = self.sampler()
      try: # handling race condition
        offset = self.table[key]
      except KeyError:
        trying = True
        continue
      _, chunk = self.manager.read_chunk(flip, offset, worker_id, 0)
      end_pos = self.rng.integers(1, self.length + 1).item()
      prev_key = None
      prev_offset = -1
      if end_pos < self.length:
        try:
          prev_key = self.bwd_links[key]
          prev_offset = self.table[prev_key]
        except KeyError:
          # This is the same replay-integrity retry as unmodified R2I. Cache
          # metadata itself must never introduce another sampling attempt.
          trying = True
          continue
        _, prev_chunk = self.manager.read_chunk(flip, prev_offset, worker_id, 1)
        chunk = {k: np.concatenate([prev_chunk.data[k][end_pos:], chunk.data[k][:end_pos]]) 
                for k in chunk.data.keys()}
        if not (key in self.bwd_links and prev_key in self.table):
          # this is another extremely rare race condition 
          # this checks in the data remains integral
          trying = True
      else:
        chunk = chunk.data
      if self.cache is not None:
        current_generation = self.chunk_generations.get(key, -1)
        if prev_key is None:
          # A predecessor disappearing after the upstream topology check is a
          # valid full-chunk sample. Represent its cache boundary as missing
          # instead of resampling it.
          try:
            prev_key = self.bwd_links[key]
            prev_offset = self.table[prev_key]
          except KeyError:
            prev_key = None
            prev_offset = -1
        previous_generation = (
            -1 if prev_key is None
            else self.chunk_generations.get(prev_key, -1))
    # Unmodified R2I fabricated a reset at every sampled boundary. R2R keeps
    # that legacy behavior only when the dense cache is disabled.
    if 'is_first' in chunk and self.cache is None:
      chunk['is_first'][0] = True
    for k in chunk.keys():
      batch_buffer[k][flip, worker_id] = chunk[k]
    if self.cache is not None:
      layout = sample_slot_layout(
          self.length, end_pos, offset, current_generation,
          prev_offset, previous_generation)
      slots = layout['slots']
      generations = layout['generations']
      anchor_slot = layout['anchor_slot']
      anchor_generation = layout['anchor_generation']
      gathered = self.cache.gather_boundaries(
          np.asarray([anchor_slot], np.int32),
          np.asarray([anchor_generation], np.int32),
          np.asarray([slots[-1]], np.int32),
          np.asarray([generations[-1]], np.int32))
      batch_buffer['_r2r_slot'][flip, worker_id] = slots
      batch_buffer['_r2r_generation'][flip, worker_id] = generations
      batch_buffer['_r2r_anchor_slot'][flip, worker_id] = anchor_slot
      batch_buffer['_r2r_anchor_generation'][
          flip, worker_id] = anchor_generation
      for name, value in gathered.items():
        batch_buffer['_r2r_' + name][flip, worker_id] = value[0]
    return True

  def update_cache(self, outputs):
    if self.cache is None:
      return None
    names = {
        '_r2r_state_slots': 'state_slots',
        '_r2r_state_generations': 'state_generations',
        '_r2r_state_bits': 'state_bits',
        '_r2r_stoch': 'stoch',
        '_r2r_action': 'action',
        '_r2r_adjoint_slots': 'adjoint_slots',
        '_r2r_adjoint_generations': 'adjoint_generations',
        '_r2r_adjoint_bits': 'adjoint_bits',
        '_r2r_terminal_slots': 'terminal_slots',
        '_r2r_terminal_generations': 'terminal_generations',
    }
    missing = [key for key in names if key not in outputs]
    if missing:
      raise KeyError('learner omitted cache updates {}'.format(missing))
    result = self.cache.commit({
        target: outputs[source] for source, target in names.items()})
    for key, value in result.items():
      self.cache_write_metrics[key] += int(value)
    return result

  def _remove(self, key):
    wait(self.limiter.want_remove, 'Replay remove is waiting')
    del self.table[key]
    del self.remover[key]
    del self.sampler[key]

  def dataset(self, flip: int, worker_id: int, batch_buffer):
    while True:
      yield self._sample(flip, worker_id, batch_buffer)

  def prioritize(self, ids, prios):
    if hasattr(self.sampler, 'prioritize'):
      self.sampler.prioritize(ids, prios)

  def save(self, wait=True, lfs_only=False):
    del wait
    if self.cache is not None:
      self.cache.flush(mirror=False)
    if not lfs_only:
      # Publish ordinary replay payloads before the sidecar and publish the
      # replay table last. After a crash, a restored table can therefore never
      # refer to a newer cache generation than the mirrored replay chunk.
      self.manager.maybe_flush(force=True, trigger_saving=False)
    if self.cache is not None and self.manager.use_lfs:
      self.cache.flush(mirror=True)
    table_bytes = self.serialize()
    assert self._agent is not None, 'Please call .set_agent(agent)!!!'
    with io.BytesIO() as stream:
      # the only requirement for the agent saving api 
      # is that it should be able to output a dict of numpy arrays 
      # with parameters
      np.savez(stream, self._agent.save())
      stream.seek(0)
      agent_bytes = np.frombuffer(stream.read(), dtype=np.uint8)
    # we save the weight of the agent and the replay buffer table 
    # into the training state file.
    self.manager.write_prefix(agent_bytes, table_bytes, lfs_only=lfs_only)
    # to keep the training state file synchronized with the 
    # actual training state, we need to flush accumulated 
    # chunks in the long-term storage
    # note that the line below has an effect only if
    # we use a long-term storage
    return table_bytes

  def maybe_restore(self):
    self.load()

  def load(self, data=None):
    ret = self.manager.read_prefix()
    if len(ret) != 2:
      return False
    agent_bytes, table_bytes = ret
    with io.BytesIO() as stream:
      stream.write(agent_bytes)
      stream.seek(0)
      agent_weights = {k:v for k,v in np.load(stream, allow_pickle=True).items()}['arr_0'].item()
    self._agent.load(agent_weights)
    print('agent loaded!')
    self.deserialize(table_bytes)
    print(f'replay deserialized! The current offset is {self.manager.offset}')
    new_step = (self.manager.offset 
      + self.manager.overwrite_layers 
      * (self.manager.total_chunks - self.manager.prefix_size_stripes)
    ) * self.length
    self._agent.step.load(new_step)
    print(f'Continuing from step {new_step}')

def wait(predicate, message, sleep=0.001, notify=1.0):
  start = time.time()
  notified = False
  while True:
    allowed, detail = predicate()
    duration = time.time() - start
    if allowed:
      return duration
    if not notified and duration >= notify:
      print(f'{message} ({detail})')
      notified = True
    time.sleep(sleep)
