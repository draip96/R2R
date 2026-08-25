import unittest
from pathlib import Path

try:
  import ruamel.yaml as yaml
except ImportError:
  yaml = None


@unittest.skipIf(yaml is None, 'ruamel.yaml is not installed')
class R2RConfigTest(unittest.TestCase):

  @classmethod
  def setUpClass(cls):
    path = Path(__file__).resolve().parents[1] / 'recall2imagine/configs.yaml'
    cls.config = yaml.YAML(typ='safe').load(path.read_text())

  def test_exact_window_batch_map_and_update_cadence(self):
    expected = {
        'r2r_w64': (64, 64),
        'r2r_w128': (128, 32),
        'r2r_w256': (256, 16),
        'r2r_w1024': (1024, 4),
    }
    cadences = []
    for name, (length, batch) in expected.items():
      preset = self.config[name]
      self.assertEqual(set(preset), {'batch_length', 'batch_size'})
      self.assertEqual((preset['batch_length'], preset['batch_size']), (length, batch))
      self.assertEqual(length * batch, 4096)
      cadences.append(1024 / (length * batch))
    self.assertEqual(cadences, [0.25] * 4)

  def test_all_production_profiles_use_one_million_replay(self):
    self.assertEqual(int(self.config['defaults']['replay_size']), 1_000_000)
    for name in ('mmaze', 'bsuite', 'toy_memory'):
      self.assertEqual(int(self.config[name]['replay_size']), 1_000_000)

  def test_dense_bfloat16_cache_is_enabled(self):
    cache = self.config['defaults']['state_gradient_cache']
    self.assertTrue(cache['enabled'])
    self.assertEqual(cache['storage_dtype'], 'bfloat16')

  def test_toy_distance_presets_use_exact_episode_geometry(self):
    expected = {
        'toy_distance8': 'toymemory_10',
        'toy_distance16': 'toymemory_18',
        'toy_distance32': 'toymemory_34',
        'toy_distance64': 'toymemory_66',
    }
    for preset, task in expected.items():
      self.assertEqual(self.config[preset], {'task': task})

  def test_matched_toy_arms_have_one_mechanism_difference(self):
    defaults = self.config['defaults']
    self.assertEqual(int(defaults['run']['steps']), 10_000_000_000)
    self.assertTrue(defaults['run']['toy_stop_on_success'])

    world_model = self.config['toy_world_model_only']
    self.assertEqual(world_model['task_behavior'], 'Random')
    self.assertEqual(world_model['run']['script'], 'train_toy_world_model')
    self.assertFalse(world_model['run']['toy_stop_on_success'])
    self.assertFalse(world_model['state_gradient_cache.enabled'])

    direct = self.config['toy_direct_bptt']
    self.assertFalse(direct['run.toy_stop_on_success'])
    self.assertFalse(direct['state_gradient_cache.enabled'])

    cached = self.config['toy_full_r2r']
    self.assertFalse(cached['run.toy_stop_on_success'])
    self.assertNotIn('state_gradient_cache.enabled', cached)

  def test_falsifier_and_r2i_window_controls_are_explicit(self):
    defaults = self.config['defaults']
    self.assertFalse(defaults['toy_terminal_reward_only'])
    self.assertFalse(defaults['toy_balanced_terminal_reward_with_aux'])
    self.assertEqual(defaults['toy_terminal_reward_weight'], 1.0)
    self.assertEqual(defaults['toy_arm'], 'auto')

    falsifier = self.config['toy_terminal_reward_falsifier']
    self.assertEqual(falsifier['task_behavior'], 'Random')
    self.assertEqual(falsifier['run']['script'], 'train_toy_world_model')
    self.assertFalse(falsifier['run']['toy_stop_on_success'])
    self.assertFalse(falsifier['state_gradient_cache.enabled'])
    self.assertTrue(falsifier['toy_terminal_reward_only'])
    self.assertEqual(falsifier['toy_arm'], 'terminal_reward_falsifier')

    weighted = self.config['toy_terminal_weighted_world_model']
    self.assertEqual(weighted['task_behavior'], 'Random')
    self.assertEqual(weighted['run']['script'], 'train_toy_world_model')
    self.assertFalse(weighted['run']['toy_stop_on_success'])
    self.assertFalse(weighted['state_gradient_cache.enabled'])
    self.assertEqual(weighted['toy_arm'], 'terminal_weighted_world_model')

    balanced_aux = self.config['toy_balanced_terminal_aux_world_model']
    self.assertEqual(balanced_aux['task_behavior'], 'Random')
    self.assertEqual(balanced_aux['run']['script'], 'train_toy_world_model')
    self.assertFalse(balanced_aux['run']['toy_stop_on_success'])
    self.assertFalse(balanced_aux['state_gradient_cache.enabled'])
    self.assertTrue(balanced_aux['toy_balanced_terminal_reward_with_aux'])

    memory = self.config['toy_balanced_terminal_memory']
    self.assertTrue(memory['toy_balanced_terminal_reward_with_aux'])
    for key in ('dyn', 'rep', 'vector', 'cont'):
      self.assertEqual(memory[f'loss_scales.{key}'], 0.0)

    cont = self.config['toy_balanced_terminal_cont_world_model']
    self.assertEqual(cont['task_behavior'], 'Random')
    self.assertEqual(cont['run']['script'], 'train_toy_world_model')
    self.assertFalse(cont['state_gradient_cache.enabled'])
    self.assertTrue(cont['toy_balanced_terminal_reward_with_aux'])
    self.assertEqual(cont['loss_scales.cont'], 1.0)
    for key in ('dyn', 'rep', 'vector'):
      self.assertEqual(cont[f'loss_scales.{key}'], 0.0)
    self.assertEqual(self.config['toy_detached_aux_heads']['grad_heads'],
                     ['reward'])

    reference = self.config['toy_r2i_reference']
    self.assertFalse(reference['run.toy_stop_on_success'])
    self.assertFalse(reference['state_gradient_cache.enabled'])
    self.assertEqual(reference['toy_arm'], 'r2i_w1024')


if __name__ == '__main__':
  unittest.main()
