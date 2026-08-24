"""Online ToyMemory training with a strict actor-plus-model early gate."""

import json
import re
import time
from pathlib import Path

import embodied
import jax
import numpy as np


def _write_summary(logdir, value):
  path = Path(str(logdir)) / 'toy_summary.json'
  temporary = path.with_suffix('.json.tmp')
  temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
  temporary.replace(path)


def _balanced_evaluation(agent, env, episodes):
  actor_actions = []
  model_choices = []
  reset_actions = []
  contexts = []
  query_states = []
  rewards = []

  def policy(obs, state):
    outputs, next_state = agent.policy(obs, state, mode='eval')
    if bool(obs['log_is_query'][0]):
      context = int(obs['log_context'][0])
      action = int(np.argmax(outputs['action'][0]))
      model_choice, _ = agent.model_reward_choice(next_state)
      reset_outputs, _ = agent.policy(obs, None, mode='eval')
      contexts.append(context)
      actor_actions.append(action)
      model_choices.append(int(model_choice[0]))
      reset_actions.append(int(np.argmax(reset_outputs['action'][0])))
      query_states.append(next_state)
    return outputs, next_state

  driver = embodied.Driver(env)
  driver.on_episode(
      lambda episode, worker: rewards.append(float(episode['reward'].sum())))
  driver(policy, episodes=int(episodes))
  contexts = np.asarray(contexts, np.int32)
  actor_actions = np.asarray(actor_actions, np.int32)
  model_choices = np.asarray(model_choices, np.int32)
  reset_actions = np.asarray(reset_actions, np.int32)
  if len(contexts) != episodes or not np.array_equal(
      contexts, np.arange(episodes, dtype=np.int32) % 2):
    raise RuntimeError('ToyMemory evaluation did not produce balanced contexts')
  merged = jax.tree_util.tree_map(
      lambda *values: np.concatenate(values, axis=0), *query_states)
  order = np.arange(episodes, dtype=np.int32).reshape(-1, 2)[:, ::-1].reshape(-1)
  swapped = jax.tree_util.tree_map(lambda value: value[order], merged)
  opposite_actions = np.argmax(agent.actor_from_state(swapped), axis=-1)
  reward_values = np.asarray(rewards, np.float64)
  return {
      'episodes': int(episodes),
      'actor_accuracy': float(np.mean(actor_actions == contexts)),
      'model_reward_choice_accuracy': float(
          np.mean(model_choices == contexts)),
      'reset_state_accuracy': float(np.mean(reset_actions == contexts)),
      'opposite_cue_state_accuracy': float(
          np.mean(opposite_actions == contexts)),
      'mean_reward': float(reward_values.mean()),
      'finite': bool(np.isfinite(reward_values).all()),
  }


def train_toy(agent, env, eval_env, replay, logger, args, config):
  logdir = embodied.Path(args.logdir)
  logdir.mkdirs()
  episode_steps = int(config.task.split('_', 1)[1])
  cue_query_distance = episode_steps - 2
  should_expl = embodied.when.Until(args.expl_until)
  should_train = embodied.when.Ratio(args.train_ratio / args.batch_steps)
  should_log = embodied.when.Clock(args.log_every)
  should_save = embodied.when.Clock(args.save_every)
  should_sync = embodied.when.Every(args.sync_every)
  step = logger.step
  updates = embodied.Counter()
  metrics = embodied.Metrics()
  timer = embodied.Timer()
  timer.wrap('agent', agent, ['policy', 'train', 'save'])
  timer.wrap('env', env, ['step'])
  timer.wrap('replay', replay, ['add', 'save'])
  timer.wrap('logger', logger, ['write'])

  nonzeros = set()

  def per_episode(episode):
    length = len(episode['reward']) - 1
    score = float(episode['reward'].astype(np.float64).sum())
    logger.add({'length': length, 'score': score}, prefix='episode')
    stats = {}
    for key, value in episode.items():
      if not args.log_zeros and key not in nonzeros and (value == 0).all():
        continue
      nonzeros.add(key)
      if re.match(args.log_keys_sum, key):
        stats['sum_' + key] = value.sum()
      if re.match(args.log_keys_mean, key):
        stats['mean_' + key] = value.mean()
    metrics.add(stats, prefix='stats')

  driver = embodied.Driver(env)
  driver.on_episode(lambda episode, worker: per_episode(episode))
  driver.on_step(lambda transition, worker: step.increment())
  driver.on_step(replay.add)
  replay.maybe_restore()

  random_agent = embodied.RandomAgent(env.act_space)
  fill = max(args.batch_steps * config.envs.amount, args.train_fill)
  prefill_start = int(step)
  while len(replay) < fill:
    # `fill` is divisible by every R2R chunk length. Limit the final driver
    # call so the smoke protocol contains exactly 4096 random transitions,
    # rather than the 4100 produced by an unconditional 100-step loop.
    remaining = max(1, int(fill) - int(step))
    driver(random_agent.policy, steps=min(100, remaining))
  if (prefill_start < int(fill) == 4096 and
      (len(replay) != 4096 or int(step) != 4096)):
    raise RuntimeError(
        'ToyMemory prefill must contain exactly 4096 replay transitions')
  logger.add(metrics.result())
  logger.write()

  dataset = agent.dataset(replay, shared_memory=True)
  learner_state = [None]
  latest_batch = [None]

  def train_step(transition, worker):
    del transition, worker
    for _ in range(should_train(step)):
      with timer.scope('dataset'):
        latest_batch[0] = next(dataset)
      outputs, learner_state[0], train_metrics = agent.train(
          latest_batch[0], learner_state[0])
      replay.update_cache(outputs)
      metrics.add(train_metrics, prefix='train')
      updates.increment()
    if should_sync(updates):
      agent.sync()
    if should_log(step):
      logger.add(metrics.result())
      logger.add(replay.stats, prefix='replay')
      logger.add(timer.stats(), prefix='timer')
      logger.write(fps=True)

  driver.on_step(train_step)
  checkpoint = embodied.Checkpoint(logdir / 'checkpoint.ckpt', parallel=False)
  checkpoint.step = step
  checkpoint.agent = agent
  checkpoint.replay = replay
  if args.from_checkpoint:
    checkpoint.load(args.from_checkpoint)
  checkpoint.load_or_save()
  should_save(step)

  policy = lambda *values: agent.policy(
      *values, mode='explore' if should_expl(step) else 'train')
  evaluation_every = int(args.toy_eval_every)
  # Anchor the panel schedule to the exact prefill boundary. On resume, the
  # durable summary advances the grid instead of shifting it by another 1000.
  summary_path = Path(str(logdir)) / 'toy_summary.json'
  if summary_path.exists():
    last_evaluation = int(json.loads(
        summary_path.read_text())['environment_steps'])
    next_evaluation = last_evaluation + evaluation_every
  else:
    next_evaluation = int(fill) + evaluation_every
  started = time.time()
  final_evaluation = None
  success = False
  while step < args.steps:
    remaining = min(100, int(args.steps - step), next_evaluation - int(step))
    driver(policy, steps=max(1, remaining))
    if embodied.run.requeue_requested():
      checkpoint.save()
      (Path(str(logdir)) / 'REQUEUE_READY').write_text(
          'checkpoint and replay cache flushed after SIGUSR1\n')
      raise SystemExit(75)
    if int(step) >= next_evaluation:
      final_evaluation = _balanced_evaluation(
          agent, eval_env, int(args.toy_eval_episodes))
      logger.add(final_evaluation, prefix='toy_eval')
      logger.write(fps=True)
      success = (
          final_evaluation['actor_accuracy'] == 1.0 and
          final_evaluation['model_reward_choice_accuracy'] == 1.0)
      summary = {
          'protocol': 'r2r-toy-memory-v1',
          'task': config.task,
          'episode_steps': episode_steps,
          'cue_query_distance': cue_query_distance,
          'window': int(config.batch_length),
          'batch_size': int(config.batch_size),
          'environment_steps': int(step),
          'learner_updates': int(updates),
          'evaluation': final_evaluation,
          'success': bool(success),
          'elapsed_seconds': time.time() - started,
      }
      _write_summary(logdir, summary)
      if success:
        checkpoint.save()
        (Path(str(logdir)) / 'SUCCESS').write_text(
            'actor and model reward-choice accuracy reached 100%\n')
        return summary
      next_evaluation += evaluation_every
    if should_save(step):
      checkpoint.save()

  if final_evaluation is None:
    final_evaluation = _balanced_evaluation(
        agent, eval_env, int(args.toy_eval_episodes))
  summary = {
      'protocol': 'r2r-toy-memory-v1',
      'task': config.task,
      'episode_steps': episode_steps,
      'cue_query_distance': cue_query_distance,
      'window': int(config.batch_length),
      'batch_size': int(config.batch_size),
      'environment_steps': int(step),
      'learner_updates': int(updates),
      'evaluation': final_evaluation,
      'success': False,
      'elapsed_seconds': time.time() - started,
  }
  _write_summary(logdir, summary)
  checkpoint.save()
  (Path(str(logdir)) / 'FAILED').write_text(
      '25k-step actor-plus-model accuracy gate was not reached\n')
  return summary
