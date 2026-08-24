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


if __name__ == '__main__':
  unittest.main()
