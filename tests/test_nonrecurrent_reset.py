"""Regression tests for posterior teacher forcing at episode resets."""

import unittest

import numpy as np

try:
  import jax.numpy as jnp
  from recall2imagine import nets
except ImportError:
  jnp = None
  nets = None


@unittest.skipIf(jnp is None, 'JAX dependencies are not installed')
class NonrecurrentResetTest(unittest.TestCase):

  def test_internal_first_uses_initial_not_previous_episode_posterior(self):
    initial = jnp.asarray([[[10.0, 11.0], [12.0, 13.0]]])
    posterior = jnp.arange(16, dtype=jnp.float32).reshape(1, 4, 2, 2)
    is_first = jnp.asarray([[False, False, True, False]])
    shifted = nets.reset_shifted_stoch(
        initial, initial, posterior, is_first)
    expected = np.stack([
        np.asarray(initial[0]),
        np.asarray(posterior[0, 0]),
        np.asarray(initial[0]),
        np.asarray(posterior[0, 2]),
    ], axis=0)[None]
    np.testing.assert_array_equal(np.asarray(shifted), expected)

  def test_first_row_resets_a_cached_predecessor(self):
    initial = jnp.zeros((2, 1, 2), jnp.float32)
    cached = jnp.full((2, 1, 2), 9.0, jnp.float32)
    posterior = jnp.arange(12, dtype=jnp.float32).reshape(2, 3, 1, 2)
    is_first = jnp.asarray([
        [True, False, False],
        [False, True, False],
    ])
    shifted = nets.reset_shifted_stoch(
        cached, initial, posterior, is_first)
    expected = np.asarray([
        [initial[0], posterior[0, 0], posterior[0, 1]],
        [cached[1], initial[1], posterior[1, 1]],
    ])
    np.testing.assert_array_equal(np.asarray(shifted), expected)


if __name__ == '__main__':
  unittest.main()
