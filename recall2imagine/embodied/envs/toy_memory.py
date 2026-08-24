"""One-bit cue/query environment for the staged R2R acquisition gate."""

import embodied
import numpy as np


class ToyMemory(embodied.Env):
  """BSuite-style one-bit memory with an exactly sized replay episode.

  The cue is visible only in the first observation. The answer action is taken
  at the penultimate observation and its reward arrives on the terminal row,
  matching Embodied/Dreamer transition alignment. Thus ``episode_steps`` is
  exactly the number of rows added to replay.
  """

  def __init__(self, task, seed=0, balanced=False):
    self.episode_steps = int(task)
    if self.episode_steps < 3:
      raise ValueError('ToyMemory episodes need at least three transitions')
    self._rng = np.random.RandomState(int(seed) % (2 ** 32 - 1))
    self._balanced = bool(balanced)
    self._episode = 0
    self._position = 0
    self._context = 0
    self._done = True

  @property
  def obs_space(self):
    return {
        'observation': embodied.Space(np.float32, (3,)),
        'reward': embodied.Space(np.float32),
        'is_first': embodied.Space(bool),
        'is_last': embodied.Space(bool),
        'is_terminal': embodied.Space(bool),
        'log_is_query': embodied.Space(bool),
        'log_context': embodied.Space(np.int32, (), 0, 2),
    }

  @property
  def act_space(self):
    return {
        'action': embodied.Space(np.int32, (), 0, 2),
        'reset': embodied.Space(bool),
    }

  @property
  def context(self):
    return self._context

  def step(self, action):
    if bool(action['reset']) or self._done:
      return self._reset()
    choice = int(action['action'])
    if choice not in (0, 1):
      raise ValueError('ToyMemory action must be 0 or 1')
    if self._position == self.episode_steps - 2:
      self._position += 1
      self._done = True
      reward = 1.0 if choice == self._context else -1.0
      return self._observation(reward, is_last=True, is_terminal=True)
    self._position += 1
    return self._observation(0.0)

  def _reset(self):
    self._position = 0
    if self._balanced:
      self._context = self._episode % 2
    else:
      self._context = int(self._rng.binomial(1, 0.5))
    self._episode += 1
    self._done = False
    return self._observation(0.0, is_first=True)

  def _observation(
      self, reward, is_first=False, is_last=False, is_terminal=False):
    query_position = self.episode_steps - 2
    progress = max(0.0, 1.0 - self._position / query_position)
    observation = np.zeros(3, np.float32)
    observation[0] = progress
    if self._position == 0:
      observation[2] = 2.0 * self._context - 1.0
    return {
        'observation': observation,
        'reward': np.float32(reward),
        'is_first': bool(is_first),
        'is_last': bool(is_last),
        'is_terminal': bool(is_terminal),
        'log_is_query': bool(self._position == query_position),
        'log_context': np.int32(self._context),
    }
