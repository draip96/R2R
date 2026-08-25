"""Regression tests for episode resets in the parallel SSM scan."""

import importlib.util
import unittest
from pathlib import Path

import numpy as np

try:
  import jax
  import jax.numpy as jnp
except ImportError:  # Allows storage-only environments to collect the suite.
  jax = None
  jnp = None


ROOT = Path(__file__).resolve().parents[1]


def _load_common():
  spec = importlib.util.spec_from_file_location(
      'r2r_reset_common', ROOT / 'recall2imagine/ssm/common.py')
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


if jax is not None:
  COMMON = _load_common()


@unittest.skipIf(jax is None, 'JAX is not installed')
class SSMResetTest(unittest.TestCase):

  def setUp(self):
    self.a = jnp.asarray(
        [0.71 + 0.13j, 0.52 - 0.19j, 0.63 + 0.07j], jnp.complex64)
    self.b = jnp.asarray(
        [[0.4 + 0.2j, -0.1 + 0.3j],
         [0.2 - 0.4j, 0.3 + 0.1j],
         [-0.2 + 0.1j, 0.5 - 0.3j]], jnp.complex64)
    self.c = jnp.asarray(
        [[0.7 - 0.2j, 0.1 + 0.5j, -0.3 + 0.1j],
         [-0.4 + 0.1j, 0.6 - 0.3j, 0.2 + 0.4j]], jnp.complex64)
    self.inputs = jnp.asarray(
        [[0.2, -0.1], [0.7, 0.3], [-0.2, 0.4], [0.5, -0.6],
         [0.1, 0.8], [-0.7, 0.2], [0.3, -0.5]], jnp.float32)
    self.x0 = jnp.asarray(
        [0.2 + 0.4j, -0.1 + 0.3j, 0.5 - 0.2j], jnp.complex64)
    self.init = jnp.asarray(
        [-0.3 + 0.1j, 0.25 - 0.4j, 0.1 + 0.2j], jnp.complex64)
    self.taps = jnp.asarray(
        [[0.02 + 0.01j, -0.01 + 0.03j, 0.04 - 0.02j],
         [0.03 - 0.01j, 0.02 + 0.01j, -0.02 + 0.03j],
         [-0.01 + 0.02j, 0.04 - 0.03j, 0.01 + 0.01j],
         [0.05 - 0.04j, -0.02 + 0.02j, 0.03 + 0.01j],
         [0.01 + 0.03j, 0.02 - 0.02j, -0.04 + 0.01j],
         [-0.03 + 0.01j, 0.01 + 0.04j, 0.02 - 0.01j],
         [0.02 - 0.03j, -0.04 + 0.01j, 0.01 + 0.02j]],
        jnp.complex64)

  def _sequential(self, dones, taps):
    state = self.x0
    states = []
    for index in range(len(self.inputs)):
      if dones[index]:
        state = self.init
      elif taps is not None:
        state = state + taps[index]
      state = self.a * state + self.b @ self.inputs[index]
      states.append(state)
    states = jnp.stack(states)
    outputs = jax.vmap(lambda value: (self.c @ value).real)(states)
    return outputs, states

  def test_tapped_and_untapped_scans_match_sequential_resets(self):
    patterns = (
        [1, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1],
        [0, 1, 0, 0, 1, 0, 0],
    )
    for pattern in patterns:
      dones = jnp.asarray(pattern, jnp.float32)
      for taps in (None, self.taps):
        with self.subTest(pattern=pattern, tapped=taps is not None):
          outputs, states = COMMON.fast_scan(
              self.a, self.b, self.c, self.inputs, self.x0, self.init,
              dones=dones, mode='init', state_taps=taps)
          expected_outputs, expected_states = self._sequential(pattern, taps)
          # Parallel and left-to-right complex reductions have different
          # floating-point order (and GPU matmuls may use reduced precision).
          # The pre-fix reset defects are O(1e-1), far above this tolerance.
          np.testing.assert_allclose(
              states, expected_states, rtol=3e-3, atol=2e-4)
          np.testing.assert_allclose(
              outputs, expected_outputs, rtol=3e-3, atol=2e-4)

  def test_internal_reset_severs_primal_and_adjoint_history(self):
    reset_index = 3
    dones = jnp.asarray([0, 0, 0, 1, 0, 0, 0], jnp.float32)

    def post_reset_loss(initial, taps):
      outputs, _ = COMMON.fast_scan(
          self.a, self.b, self.c, self.inputs, initial, self.init,
          dones=dones, mode='init', state_taps=taps)
      return jnp.sum(jnp.square(outputs[reset_index:]))

    grad_initial, grad_taps = jax.grad(
        post_reset_loss, argnums=(0, 1))(self.x0, self.taps)
    np.testing.assert_array_equal(
        np.asarray(grad_initial), np.zeros_like(np.asarray(self.x0)))
    np.testing.assert_array_equal(
        np.asarray(grad_taps[:reset_index + 1]),
        np.zeros_like(np.asarray(self.taps[:reset_index + 1])))
    self.assertGreater(
        float(np.linalg.norm(np.asarray(grad_taps[reset_index + 1:]))), 0.0)


if __name__ == '__main__':
  unittest.main()
