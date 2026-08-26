import unittest

import numpy as np

from recall2imagine.embodied.run.train_toy import (
    _mean_or_zero,
    _opposite_cue_diagnostics,
)


class ToyEvaluationDiagnosticsTest(unittest.TestCase):

  def test_empty_class_mean_is_finite(self):
    self.assertEqual(_mean_or_zero(np.asarray([], np.float32)), 0.0)
    self.assertEqual(_mean_or_zero(np.asarray([1.0, 3.0])), 2.0)

  def test_opposite_cue_uses_original_counterfactual_label(self):
    contexts = np.asarray([0, 1, 0, 1])
    actor_actions = contexts.copy()
    opposite_actions = np.asarray([1, 0, 1, 0])
    metrics = _opposite_cue_diagnostics(
        actor_actions, opposite_actions, contexts)
    self.assertEqual(metrics['opposite_cue_state_accuracy'], 0.0)
    self.assertEqual(metrics['opposite_cue_state_action_flip_rate'], 1.0)


if __name__ == '__main__':
  unittest.main()
