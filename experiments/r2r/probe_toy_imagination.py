#!/usr/bin/env python3
"""Probe phase-conditioned actor imagination from a ToyMemory checkpoint."""

import argparse
import json
from pathlib import Path

import jax
import numpy as np

import recall2imagine.embodied as embodied
from recall2imagine import agent as agt
from recall2imagine import train


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--checkpoint', type=Path, required=True)
  parser.add_argument('--output', type=Path, required=True)
  parser.add_argument('--episodes', type=int, default=128)
  parser.add_argument('--horizon', type=int, default=15)
  return parser.parse_args()


def merge_states(states):
  return jax.tree_util.tree_map(
      lambda *values: np.concatenate(values, axis=0), *states)


def statistics(value):
  value = np.asarray(value, np.float64)
  return {
      'mean': float(value.mean()),
      'min': float(value.min()),
      'max': float(value.max()),
  }


def main():
  args = parse_args()
  config = embodied.Config(agt.Agent.configs['defaults'])
  for name in (
      'toy_memory', 'toy_distance8', 'r2r_w64', 'toy_full_r2r',
      'toy_balanced_terminal_cont_memory'):
    config = config.update(agt.Agent.configs[name])
  step = embodied.Counter()
  env = train.make_envs(config, seed=10_000_019, balanced=True)
  agent = agt.Agent(env.obs_space, env.act_space, step, config)
  checkpoint = embodied.Checkpoint()
  checkpoint.agent = agent
  checkpoint.load(str(args.checkpoint), keys=['agent'])

  query_position = int(config.task.split('_', 1)[1]) - 2
  captures = {phase: [] for phase in range(query_position + 2)}
  contexts = {phase: [] for phase in captures}

  def policy(obs, state):
    outputs, next_state = agent.policy(obs, state, mode='eval')
    if bool(obs['is_last'][0]):
      phase = query_position + 1
    elif bool(obs['log_is_query'][0]):
      phase = query_position
    else:
      progress = float(obs['observation'][0, 0])
      phase = int(round((1.0 - progress) * query_position))
    captures[phase].append(next_state)
    contexts[phase].append(int(obs['log_context'][0]))
    return outputs, next_state

  embodied.Driver(env)(policy, episodes=args.episodes)
  result = {
      'checkpoint': str(args.checkpoint),
      'episodes': args.episodes,
      'horizon': args.horizon,
      'phases': {},
  }
  for phase in sorted(captures):
    if len(captures[phase]) != args.episodes:
      raise RuntimeError(
          f'phase {phase} captured {len(captures[phase])} states, '
          f'expected {args.episodes}')
    state = merge_states(captures[phase])
    is_terminal = np.full(
        args.episodes, phase == query_position + 1, np.float32)
    diagnostics = jax.tree_util.tree_map(
        np.asarray, agent.imagination_diagnostics(
            state, is_terminal, args.horizon))
    context = np.asarray(contexts[phase], np.int32)
    action = np.argmax(diagnostics['action'], axis=-1)
    phase_result = {
        'actor_action_1_rate': float(np.mean(action[0] == 1)),
        'actor_context_accuracy': float(np.mean(action[0] == context)),
        'first_reward': statistics(diagnostics['reward'][0]),
        'first_return': statistics(diagnostics['return'][0]),
        'first_successor_cont': statistics(diagnostics['cont'][1]),
        'reward': statistics(diagnostics['reward']),
        'return': statistics(diagnostics['return']),
        'weight': statistics(diagnostics['weight']),
    }
    for cue in (0, 1):
      selected = context == cue
      phase_result[f'cue{cue}'] = {
          'actor_action_1_rate': float(np.mean(action[0, selected] == 1)),
          'first_reward': statistics(
              diagnostics['reward'][0, selected]),
          'first_return': statistics(
              diagnostics['return'][0, selected]),
          'first_successor_cont': statistics(
              diagnostics['cont'][1, selected]),
      }
    result['phases'][str(phase)] = phase_result

  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
  print(json.dumps(result, indent=2, sort_keys=True))
  env.close()


if __name__ == '__main__':
  main()
