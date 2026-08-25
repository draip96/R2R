"""Tests for the balanced terminal-reward diagnostic objective."""

import unittest

import numpy as np

try:
  import jax
  import jax.numpy as jnp
  from recall2imagine import jaxutils
except ImportError:
  jax = None
  jnp = None
  jaxutils = None


@unittest.skipIf(jax is None, 'JAX dependencies are not installed')
class RewardFalsifierTest(unittest.TestCase):

  def test_equal_class_mean_excludes_nonterminal_rows(self):
    losses = jnp.asarray([[100.0, 2.0, 4.0], [6.0, 8.0, 100.0]])
    rewards = jnp.asarray([[0.0, 1.0, 1.0], [-1.0, -1.0, 0.0]])
    is_last = jnp.asarray([[0, 1, 1], [1, 1, 0]], bool)
    value, metrics = jaxutils.balanced_terminal_reward_loss(
        losses, rewards, is_last)
    # Positive mean is 3, negative mean is 7, and each class gets half weight.
    self.assertEqual(float(value), 5.0)
    self.assertEqual(float(metrics['terminal_reward_positive_count']), 2.0)
    self.assertEqual(float(metrics['terminal_reward_negative_count']), 2.0)

    gradient = jax.grad(lambda x: (
        jaxutils.balanced_terminal_reward_loss(
            x, rewards, is_last)[0]))(losses)
    np.testing.assert_array_equal(
        np.asarray(gradient),
        np.asarray([[0.0, 0.25, 0.25], [0.25, 0.25, 0.0]]))

  def test_single_present_class_remains_well_defined(self):
    losses = jnp.asarray([[9.0, 2.0, 4.0]])
    rewards = jnp.asarray([[0.0, 1.0, 1.0]])
    is_last = jnp.asarray([[0, 1, 1]], bool)
    value, metrics = jaxutils.balanced_terminal_reward_loss(
        losses, rewards, is_last)
    self.assertEqual(float(value), 3.0)
    self.assertEqual(float(metrics['terminal_reward_negative_count']), 0.0)


if __name__ == '__main__':
  unittest.main()
