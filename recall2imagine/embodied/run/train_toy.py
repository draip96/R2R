"""Online ToyMemory training and matched world-model diagnostics."""

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
  model_values = []
  reset_actions = []
  contexts = []
  query_states = []
  terminal_states = []
  terminal_rewards = []
  rewards = []

  def policy(obs, state):
    outputs, next_state = agent.policy(obs, state, mode='eval')
    if bool(obs['log_is_query'][0]):
      context = int(obs['log_context'][0])
      action = int(np.argmax(outputs['action'][0]))
      model_choice, values = agent.model_reward_choice(next_state)
      reset_outputs, _ = agent.policy(obs, None, mode='eval')
      contexts.append(context)
      actor_actions.append(action)
      model_choices.append(int(model_choice[0]))
      model_values.append(np.asarray(values[0], np.float64))
      reset_actions.append(int(np.argmax(reset_outputs['action'][0])))
      query_states.append(next_state)
    if bool(obs['is_last'][0]):
      terminal_states.append(next_state)
      terminal_rewards.append(float(obs['reward'][0]))
    return outputs, next_state

  driver = embodied.Driver(env)
  driver.on_episode(
      lambda episode, worker: rewards.append(float(episode['reward'].sum())))
  driver(policy, episodes=int(episodes))
  contexts = np.asarray(contexts, np.int32)
  actor_actions = np.asarray(actor_actions, np.int32)
  model_choices = np.asarray(model_choices, np.int32)
  model_values = np.asarray(model_values, np.float64)
  reset_actions = np.asarray(reset_actions, np.int32)
  if len(contexts) != episodes or not np.array_equal(
      contexts, np.arange(episodes, dtype=np.int32) % 2):
    raise RuntimeError('ToyMemory evaluation did not produce balanced contexts')
  if len(terminal_states) != episodes:
    raise RuntimeError('ToyMemory evaluation did not produce one terminal state')
  merged = jax.tree_util.tree_map(
      lambda *values: np.concatenate(values, axis=0), *query_states)
  order = np.arange(episodes, dtype=np.int32).reshape(-1, 2)[:, ::-1].reshape(-1)
  swapped = jax.tree_util.tree_map(lambda value: value[order], merged)
  opposite_actions = np.argmax(agent.actor_from_state(swapped), axis=-1)
  opposite_contexts = contexts[order]
  terminal_merged = jax.tree_util.tree_map(
      lambda *values: np.concatenate(values, axis=0), *terminal_states)
  terminal_values = np.asarray(
      agent.model_reward_from_state(terminal_merged), np.float64)
  terminal_rewards = np.asarray(terminal_rewards, np.float64)
  terminal_positive = terminal_rewards > 0.0
  terminal_negative = terminal_rewards < 0.0
  terminal_sign_correct = (
      (terminal_values > 0.0) == terminal_positive)
  reward_values = np.asarray(rewards, np.float64)
  correct_values = model_values[np.arange(episodes), contexts]
  incorrect_values = model_values[np.arange(episodes), 1 - contexts]
  return {
      'episodes': int(episodes),
      'actor_accuracy': float(np.mean(actor_actions == contexts)),
      'model_reward_choice_accuracy': float(
          np.mean(model_choices == contexts)),
      'model_reward_correct_value': float(np.mean(correct_values)),
      'model_reward_incorrect_value': float(np.mean(incorrect_values)),
      'model_reward_margin': float(np.mean(
          correct_values - incorrect_values)),
      'model_reward_margin_std': float(np.std(
          correct_values - incorrect_values)),
      'teacher_terminal_sign_accuracy': float(np.mean(
          terminal_sign_correct)),
      'teacher_terminal_positive_accuracy': float(np.mean(
          terminal_values[terminal_positive] > 0.0)),
      'teacher_terminal_negative_accuracy': float(np.mean(
          terminal_values[terminal_negative] < 0.0)),
      'teacher_terminal_value_margin': float(
          terminal_values[terminal_positive].mean() -
          terminal_values[terminal_negative].mean()),
      'teacher_terminal_mae': float(np.mean(np.abs(
          terminal_values - terminal_rewards))),
      'reset_state_accuracy': float(np.mean(reset_actions == contexts)),
      'opposite_cue_state_accuracy': float(
          np.mean(opposite_actions == opposite_contexts)),
      'mean_reward': float(reward_values.mean()),
      'finite': bool(np.isfinite(reward_values).all()),
  }


def train_toy(
    agent, env, eval_env, replay, logger, args, config,
    world_model_only=False):
  logdir = embodied.Path(args.logdir)
  logdir.mkdirs()
  summary_path = Path(str(logdir)) / 'toy_summary.json'
  previous_summary = (
      json.loads(summary_path.read_text()) if summary_path.exists() else {})
  resumed_from_step = int(previous_summary.get('environment_steps', 0))
  episode_steps = int(config.task.split('_', 1)[1])
  cue_query_distance = episode_steps - 2
  arm = str(config.toy_arm)
  if arm == 'auto':
    arm = ('world_model_only' if world_model_only else
           'full_r2r' if config.state_gradient_cache.enabled else
           'direct_bptt')
  objective = (
      'balanced_terminal_reward_only'
      if config.toy_terminal_reward_only else
      'normalized_terminal_weighted_reward'
      if float(config.toy_terminal_reward_weight) != 1.0 else
      'native')

  def passed(evaluation):
    if config.toy_terminal_reward_only:
      return evaluation['teacher_terminal_sign_accuracy'] == 1.0
    if world_model_only:
      return evaluation['model_reward_choice_accuracy'] == 1.0
    return (
        evaluation['actor_accuracy'] == 1.0 and
        evaluation['model_reward_choice_accuracy'] == 1.0)

  success_message = (
      'teacher-forced terminal reward accuracy reached 100%\n'
      if config.toy_terminal_reward_only else
      'model reward-choice accuracy reached 100%\n'
      if world_model_only else
      'actor and model reward-choice accuracy reached 100%\n')
  failure_message = (
      f'{int(args.steps)}-step teacher-forced terminal reward gate was not '
      'reached\n'
      if config.toy_terminal_reward_only else
      f'{int(args.steps)}-step model reward-choice accuracy gate was not '
      'reached\n'
      if world_model_only else
      f'{int(args.steps)}-step actor-plus-model accuracy gate was not '
      'reached\n')

  should_expl = embodied.when.Until(args.expl_until)
  should_train = embodied.when.Ratio(args.train_ratio / args.batch_steps)
  should_log = embodied.when.Clock(args.log_every)
  should_save = embodied.when.Clock(args.save_every)
  should_sync = embodied.when.Every(args.sync_every)
  step = logger.step
  # Older ToyMemory checkpoints did not include this counter. Seed it from the
  # durable evaluation summary; newer checkpoints override it on load below.
  updates = embodied.Counter(previous_summary.get('learner_updates', 0))
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
  checkpoint.updates = updates
  if args.from_checkpoint:
    checkpoint.load(args.from_checkpoint)
  checkpoint.load_or_save()
  if resumed_from_step and resumed_from_step < int(args.steps):
    # A previous target was missed, but the extended run is not yet a failure.
    failed_marker = Path(str(logdir)) / 'FAILED'
    if failed_marker.exists():
      failed_marker.unlink()
  should_save(step)

  policy = lambda *values: agent.policy(
      *values, mode='explore' if should_expl(step) else 'train')
  evaluation_every = int(args.toy_eval_every)
  # Anchor the panel schedule to the exact prefill boundary. On resume, the
  # durable summary advances the grid instead of shifting it by another 1000.
  if previous_summary:
    last_evaluation_step = int(previous_summary['environment_steps'])
    next_evaluation = last_evaluation_step + evaluation_every
  else:
    previous_summary = {}
    last_evaluation_step = None
    next_evaluation = int(fill) + evaluation_every
  started = time.time()
  final_evaluation = previous_summary.get('evaluation')
  success = bool(final_evaluation and passed(final_evaluation))
  ever_success = bool(previous_summary.get('ever_success', False))
  first_success_step = previous_summary.get('first_success_step')
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
      last_evaluation_step = int(step)
      logger.add(final_evaluation, prefix='toy_eval')
      logger.write(fps=True)
      success = passed(final_evaluation)
      if success and not ever_success:
        ever_success = True
        first_success_step = int(step)
      summary = {
          'protocol': 'r2r-toy-memory-v3',
          'arm': arm,
          'objective': objective,
          'terminal_reward_weight': float(config.toy_terminal_reward_weight),
          'task': config.task,
          'episode_steps': episode_steps,
          'cue_query_distance': cue_query_distance,
          'window': int(config.batch_length),
          'batch_size': int(config.batch_size),
          'resumed_from_step': resumed_from_step,
          'environment_steps': int(step),
          'learner_updates': int(updates),
          'evaluation': final_evaluation,
          'success': bool(success),
          'ever_success': bool(ever_success),
          'first_success_step': first_success_step,
          'elapsed_seconds': time.time() - started,
      }
      _write_summary(logdir, summary)
      if success:
        checkpoint.save()
        (Path(str(logdir)) / 'SUCCESS').write_text(success_message)
        if args.toy_stop_on_success:
          return summary
      next_evaluation += evaluation_every
    if should_save(step):
      checkpoint.save()

  if last_evaluation_step != int(step):
    final_evaluation = _balanced_evaluation(
        agent, eval_env, int(args.toy_eval_episodes))
    logger.add(final_evaluation, prefix='toy_eval')
    logger.write(fps=True)
    success = passed(final_evaluation)
    if success and not ever_success:
      ever_success = True
      first_success_step = int(step)
      (Path(str(logdir)) / 'SUCCESS').write_text(success_message)
  summary = {
      'protocol': 'r2r-toy-memory-v3',
      'arm': arm,
      'objective': objective,
      'terminal_reward_weight': float(config.toy_terminal_reward_weight),
      'task': config.task,
      'episode_steps': episode_steps,
      'cue_query_distance': cue_query_distance,
      'window': int(config.batch_length),
      'batch_size': int(config.batch_size),
      'resumed_from_step': resumed_from_step,
      'environment_steps': int(step),
      'learner_updates': int(updates),
      'evaluation': final_evaluation,
      'success': bool(success),
      'ever_success': bool(ever_success),
      'first_success_step': first_success_step,
      'elapsed_seconds': time.time() - started,
  }
  _write_summary(logdir, summary)
  checkpoint.save()
  if not ever_success:
    (Path(str(logdir)) / 'FAILED').write_text(failure_message)
  return summary


def train_toy_world_model(agent, env, eval_env, replay, logger, args, config):
  """Train only the world model while collecting uniformly random actions."""
  return train_toy(
      agent, env, eval_env, replay, logger, args, config,
      world_model_only=True)
