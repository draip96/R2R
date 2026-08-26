import runpy
import unittest


_MODULE = runpy.run_path('experiments/r2r/check_toy_retention.py')
retained = _MODULE['retained']


def _record(step, actor, model, margin=1.0):
  return {
      'step': step,
      'toy_eval/actor_accuracy': actor,
      'toy_eval/model_reward_choice_accuracy': model,
      'toy_eval/model_reward_margin': margin,
      'toy_eval/finite': 1.0,
  }


class ToyRetentionTest(unittest.TestCase):

  def test_model_criterion_does_not_require_actor_solution(self):
    evaluations = [
        _record(step, 0.5, 1.0) for step in range(46000, 50001, 1000)]
    passed, selected, final = retained(
        evaluations, 5, 1000, 50000, 0.1, criterion='model')
    self.assertTrue(passed)
    self.assertEqual(len(selected), 5)
    self.assertEqual(final['step'], 50000)

  def test_joint_criterion_requires_actor_solution(self):
    evaluations = [
        _record(step, 0.5, 1.0) for step in range(46000, 50001, 1000)]
    passed, selected, final = retained(
        evaluations, 5, 1000, 50000, 0.1, criterion='joint')
    self.assertFalse(passed)
    self.assertEqual(selected, [])
    self.assertEqual(final['step'], 50000)


if __name__ == '__main__':
  unittest.main()
