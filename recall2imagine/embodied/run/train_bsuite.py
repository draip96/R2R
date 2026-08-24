"""Million-interaction BSuite-style memory training and final panel."""

import json
import os
from pathlib import Path

from .train import train
from .train_toy import _balanced_evaluation


def _write(path, value):
  temporary = path.with_suffix(path.suffix + '.tmp')
  temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
  os.replace(str(temporary), str(path))


def train_bsuite(agent, env, eval_env, replay, logger, args, config):
  """Preserve native R2I training, then run 1,000 fresh greedy episodes."""
  train(agent, env, replay, logger, args, config)
  episodes = int(args.bsuite_eval_episodes)
  evaluation = _balanced_evaluation(agent, eval_env, episodes)
  summary = {
      'protocol': 'r2r-bsuite-memory-v1',
      'task': config.task,
      'seed': int(config.seed),
      'window': int(config.batch_length),
      'batch_size': int(config.batch_size),
      'environment_steps': int(logger.step),
      'evaluation': evaluation,
      'final_1000_episode_mean_reward': evaluation['mean_reward'],
      'promotion_pass': evaluation['mean_reward'] >= 0.5,
  }
  root = Path(str(args.logdir))
  _write(root / 'bsuite_summary.json', summary)
  replay.save()
  (root / 'COMPLETE').write_text('final balanced evaluation complete\n')
  if summary['promotion_pass']:
    (root / 'PROMOTED').write_text(
        'final 1000 episode mean reward is at least 0.5\n')
  return summary
