#!/usr/bin/env python3
"""Run two real learner updates through the production replay/cache path."""

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import jax
import numpy as np

import recall2imagine.agent as agent_module
import recall2imagine.embodied as embodied
from recall2imagine import train as train_module


WINDOW_BATCH = {64: 64, 128: 32, 256: 16, 1024: 4}


def _write_json(path, value):
  temporary = path.with_suffix(path.suffix + '.tmp')
  temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
  os.replace(str(temporary), str(path))


def _finite(tree):
  leaves = jax.tree_util.tree_leaves(jax.device_get(tree))
  return all(np.isfinite(np.asarray(value)).all() for value in leaves)


def run(window, output, replay_directory, workers):
  if window not in WINDOW_BATCH:
    raise ValueError(window)
  if (output / 'summary.json').exists() or (output / 'SUCCESS').exists():
    raise FileExistsError(output)
  output.mkdir(parents=True, exist_ok=True)
  replay_directory.mkdir(parents=True)
  config = embodied.Config(agent_module.Agent.configs['defaults'])
  for name in ('toy_memory', 'r2r_w{}'.format(window)):
    config = config.update(agent_module.Agent.configs[name])
  config = config.update({
      'jax.platform': 'gpu',
      'jax.prealloc': False,
      'jax.precision': 'float32',
      'data_loaders': int(workers),
      'num_buffers': 1,
      'use_lfs': False,
      'logdir': str(output),
      'replay_dir': str(replay_directory),
  })
  config.save(output / 'config.yaml')
  started = time.time()
  env = train_module.make_envs(config)
  replay = train_module.make_replay(config, embodied.Path(replay_directory))
  step = embodied.Counter()
  agent = agent_module.Agent(env.obs_space, env.act_space, step, config)
  replay.set_agent(agent)
  driver = embodied.Driver(env)
  driver.on_step(replay.add)
  random_agent = embodied.RandomAgent(env.act_space)
  driver(random_agent.policy, steps=4096)
  if len(replay) != 4096:
    raise RuntimeError('integration prefill did not produce exactly 4096 rows')
  dataset = agent.dataset(replay, shared_memory=True)
  learner_state = None
  records = []
  for update in range(2):
    batch = next(dataset)
    initial_hits = float(np.asarray(jax.device_get(
        batch['_r2r_initial_valid'])).mean())
    future_hits = float(np.asarray(jax.device_get(
        batch['_r2r_future_adjoint_valid'])).mean())
    outputs, learner_state, metrics = agent.train(batch, learner_state)
    if not _finite(outputs) or not _finite(metrics):
      raise FloatingPointError('non-finite learner smoke output')
    writes = replay.update_cache(outputs)
    records.append({
        'update': update + 1,
        'initial_hit_rate': initial_hits,
        'future_hit_rate': future_hits,
        'writes': writes,
    })
  replay.cache.flush()
  summary = {
      'protocol': 'r2r-gpu-integration-v1',
      'commit': subprocess.check_output(
          ['git', 'rev-parse', 'HEAD'], text=True).strip(),
      'window': int(window),
      'batch_size': WINDOW_BATCH[window],
      'transitions_per_update': window * WINDOW_BATCH[window],
      'updates': records,
      'cache': replay.cache.stats,
      'elapsed_seconds': time.time() - started,
      'device': str(jax.devices('gpu')[0]),
      'success': True,
  }
  _write_json(output / 'summary.json', summary)
  (output / 'SUCCESS').write_text('two production learner updates passed\n')
  replay.cache.close()
  replay.manager.tmp_file.close()
  env.close()
  return summary


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--window', type=int, choices=tuple(WINDOW_BATCH), required=True)
  parser.add_argument('--output', type=Path, required=True)
  parser.add_argument('--replay-directory', type=Path, required=True)
  parser.add_argument('--workers', type=int, default=8)
  args = parser.parse_args()
  run(args.window, args.output.resolve(), args.replay_directory.resolve(), args.workers)


if __name__ == '__main__':
  main()
