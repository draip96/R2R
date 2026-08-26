import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import yaml


MODULE_PATH = (
    Path(__file__).parents[1] / 'experiments' / 'r2r' /
    'validate_toy_r2r_run.py')
SPEC = importlib.util.spec_from_file_location('validate_toy_r2r_run', MODULE_PATH)
VALIDATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VALIDATOR)


class ToyR2RRunValidationTest(unittest.TestCase):

  def make_run(self, root):
    source = Path(root) / 'run'
    provenance = source / 'provenance'
    provenance.mkdir(parents=True, exist_ok=True)
    summary = {
        'environment_steps': 60000,
        'learner_updates': 13976,
        'cue_query_distance': 16,
        'episode_steps': 18,
        'window': 64,
        'batch_size': 64,
        'objective': 'balanced_sparse_reward_with_native_auxiliaries',
        'success': True,
    }
    config = {
        'task': 'toymemory_18',
        'batch_length': 64,
        'batch_size': 64,
        'replay': 'lfs',
        'replay_size': 1_000_000.0,
        'replay_online': False,
        'seed': 0,
        'task_behavior': 'Greedy',
        'toy_balanced_sparse_reward_with_aux': True,
        'toy_balanced_terminal_reward_with_aux': False,
        'unlocked_sampling': False,
        'state_gradient_cache': {
            'enabled': True,
            'storage_dtype': 'bfloat16',
        },
        'loss_scales': {
            'reward': 1.0,
            'cont': 1.0,
            'dyn': 0.05,
            'rep': 0.0,
            'vector': 0.0,
        },
        'run': {
            'script': 'train_toy',
            'steps': 60000.0,
            'train_fill': 4096,
            'train_ratio': 1024.0,
            'toy_eval_every': 1000,
            'toy_eval_episodes': 128,
            'toy_stop_on_success': False,
        },
    }
    (source / 'toy_summary.json').write_text(json.dumps(summary))
    (source / 'config.yaml').write_text(yaml.safe_dump(config))
    with (source / 'metrics.jsonl').open('w') as stream:
      for step in (56000, 57000, 58000, 59000, 60000):
        stream.write(json.dumps({
            'step': step,
            'toy_eval/actor_accuracy': 1.0,
            'toy_eval/model_reward_choice_accuracy': 1.0,
            'toy_eval/model_reward_margin': 1.9,
            'toy_eval/finite': 1.0,
        }) + '\n')
    (provenance / 'commit.txt').write_text('a' * 40 + '\n')
    (provenance / 'git-status.txt').write_text('')
    (provenance / 'working-tree.patch').write_text('')
    (provenance / 'source-sha256.txt').write_text('evidence\n')
    return source

  def test_accepts_exact_retained_run(self):
    with tempfile.TemporaryDirectory() as root:
      source = self.make_run(root)
      result = VALIDATOR.validate_run(
          source, distance=16, seed=0, final_step=60000,
          dyn_scale=0.05, expected_commit='a' * 40)
      self.assertTrue(result['passed'])
      self.assertEqual(result['retained_steps'], [56000, 57000, 58000, 59000, 60000])

  def test_rejects_cache_or_retention_mismatch(self):
    with tempfile.TemporaryDirectory() as root:
      source = self.make_run(root)
      config = yaml.safe_load((source / 'config.yaml').read_text())
      config['state_gradient_cache']['enabled'] = False
      (source / 'config.yaml').write_text(yaml.safe_dump(config))
      with self.assertRaisesRegex(VALIDATOR.ValidationError, 'cache config'):
        VALIDATOR.validate_run(
            source, distance=16, seed=0, final_step=60000,
            dyn_scale=0.05)

      source = self.make_run(root)
      records = [
          json.loads(line)
          for line in (source / 'metrics.jsonl').read_text().splitlines()
      ]
      records[-2]['toy_eval/actor_accuracy'] = 0.5
      with (source / 'metrics.jsonl').open('w') as stream:
        for record in records:
          stream.write(json.dumps(record) + '\n')
      with self.assertRaisesRegex(VALIDATOR.ValidationError, 'retention'):
        VALIDATOR.validate_run(
            source, distance=16, seed=0, final_step=60000,
            dyn_scale=0.05)


if __name__ == '__main__':
  unittest.main()
