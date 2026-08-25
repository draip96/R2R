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

  def test_terminal_weighting_preserves_mean_weight(self):
    losses = jnp.asarray([[1.0, 2.0, 3.0, 4.0]])
    is_last = jnp.asarray([[0, 0, 0, 1]], bool)
    value, metrics = jaxutils.normalized_terminal_reward_loss(
        losses, is_last, 3.0)
    # Raw weights [1, 1, 1, 3] are divided by their mean of 1.5.
    np.testing.assert_allclose(
        np.asarray(value), np.asarray([[2 / 3, 4 / 3, 2, 8]]))
    self.assertAlmostEqual(float(metrics['terminal_reward_mean_weight']), 1.5)
    self.assertAlmostEqual(float(metrics['terminal_reward_row_rate']), 0.25)

  def test_terminal_weight_one_is_exact_identity(self):
    losses = jnp.asarray([[1.0, 2.0], [3.0, 4.0]])
    is_last = jnp.asarray([[0, 1], [1, 0]], bool)
    value, _ = jaxutils.normalized_terminal_reward_loss(
        losses, is_last, 1.0)
    np.testing.assert_array_equal(np.asarray(value), np.asarray(losses))

  def test_balanced_reward_replaces_native_reward_and_keeps_auxiliaries(self):
    reward_losses = jnp.asarray([[2.0, 4.0]])
    rewards = jnp.asarray([[1.0, -1.0]])
    is_last = jnp.asarray([[1, 1]], bool)
    scaled = {
        'reward': reward_losses * 999.0,
        'observation': jnp.asarray([[2.0, 4.0]]),
        'dyn': jnp.asarray([[1.0, 3.0]]),
    }
    value, metrics = (
        jaxutils.balanced_terminal_reward_with_auxiliary_loss(
            scaled, reward_losses, rewards, is_last, 2.0))
    # Auxiliary means are 3 + 2, balanced reward is 3, and reward scale is 2.
    self.assertEqual(float(value), 11.0)
    self.assertEqual(float(metrics['terminal_reward_auxiliary_loss']), 5.0)
    gradient = jax.grad(lambda loss: (
        jaxutils.balanced_terminal_reward_with_auxiliary_loss(
            scaled, loss, rewards, is_last, 2.0)[0]))(reward_losses)
    np.testing.assert_array_equal(np.asarray(gradient), [[1.0, 1.0]])


if __name__ == '__main__':
  unittest.main()
