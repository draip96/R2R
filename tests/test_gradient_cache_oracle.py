"""Frozen-parameter numerical oracle for R2R boundary adjoints."""

import importlib.util
import unittest
from pathlib import Path

import numpy as np

try:
  import jax
  import jax.numpy as jnp
except ImportError:  # Allows the storage-only suite to run without JAX.
  jax = None
  jnp = None


ROOT = Path(__file__).resolve().parents[1]


def _load(name, path):
  spec = importlib.util.spec_from_file_location(name, path)
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


if jax is not None:
  COMMON = _load('r2r_ssm_common', ROOT / 'recall2imagine/ssm/common.py')
  CACHE = _load(
      'r2r_cache_bits',
      ROOT / 'recall2imagine/embodied/replay/state_adjoint_cache.py')


@unittest.skipIf(jax is None, 'JAX is not installed')
class GradientCacheOracleTest(unittest.TestCase):

  def setUp(self):
    self.a = jnp.asarray([0.71 + 0.13j, 0.52 - 0.19j], jnp.complex64)
    self.b = jnp.asarray(
        [[0.4 + 0.2j, -0.1 + 0.3j],
         [0.2 - 0.4j, 0.3 + 0.1j]], jnp.complex64)
    self.c = jnp.asarray(
        [[0.7 - 0.2j, 0.1 + 0.5j],
         [-0.4 + 0.1j, 0.6 - 0.3j]], jnp.complex64)
    self.init = jnp.zeros((2,), jnp.complex64)
    self.x0 = jnp.asarray([0.2 + 0.4j, -0.1 + 0.3j], jnp.complex64)
    self.u1 = jnp.asarray(
        [[0.2, -0.1], [0.7, 0.3], [-0.2, 0.4], [0.5, -0.6]],
        jnp.float32)
    self.u2 = jnp.asarray(
        [[-0.3, 0.2], [0.1, 0.8], [0.4, -0.5], [-0.7, 0.2]],
        jnp.float32)
    self.target1 = jnp.asarray(
        [[0.1, -0.2], [0.3, 0.2], [-0.1, 0.4], [0.2, -0.3]],
        jnp.float32)
    self.target2 = -self.target1

  def _segment(self, params, inputs, target, initial, future=None):
    a, b, c = params
    outputs, states = COMMON.fast_scan(
        a, b, c, inputs, initial, self.init, dones=None,
        state_taps=jnp.zeros((len(inputs), len(a)), jnp.complex64))
    loss = jnp.mean(jnp.square(outputs - target))
    if future is not None:
      loss += jnp.real(jnp.sum(future * states[-1]))
    return loss, states[-1]

  @staticmethod
  def _quality(reference, candidate):
    ref = np.concatenate([
        np.asarray(x).view(np.float32).reshape(-1)
        for x in jax.tree_util.tree_leaves(reference)])
    got = np.concatenate([
        np.asarray(x).view(np.float32).reshape(-1)
        for x in jax.tree_util.tree_leaves(candidate)])
    relative = np.linalg.norm(got - ref) / max(np.linalg.norm(ref), 1e-30)
    cosine = float(np.dot(ref, got) / (
        max(np.linalg.norm(ref), 1e-30) * max(np.linalg.norm(got), 1e-30)))
    return relative, cosine

  def test_fp32_and_bf16_cached_adjoint_match_full_bptt(self):
    first_params = (self.a, self.b, self.c)
    future_params = tuple(x * (0.93 + 0.02j) for x in first_params)

    def future_loss(boundary):
      return self._segment(
          future_params, self.u2, self.target2, boundary)[0]

    _, boundary = self._segment(
        first_params, self.u1, self.target1, self.x0)
    exact_adjoint = jax.grad(future_loss)(boundary)

    def full_loss(params):
      local, middle = self._segment(
          params, self.u1, self.target1, self.x0)
      return local + future_loss(middle)

    def cached_loss(params, adjoint):
      return self._segment(
          params, self.u1, self.target1, self.x0, adjoint)[0]

    reference = jax.grad(full_loss)(first_params)
    fp32 = jax.grad(lambda p: cached_loss(p, exact_adjoint))(first_params)
    relative, cosine = self._quality(reference, fp32)
    self.assertLessEqual(relative, 1e-5)
    self.assertGreaterEqual(cosine, 0.99999)

    quantized = CACHE.bf16_bits_to_complex(
        CACHE.complex_to_bf16_bits(np.asarray(exact_adjoint)))
    bf16 = jax.grad(
        lambda p: cached_loss(p, jnp.asarray(quantized)))(first_params)
    relative, cosine = self._quality(reference, bf16)
    self.assertLessEqual(relative, 0.05)
    self.assertGreaterEqual(cosine, 0.99)
    self.assertTrue(all(
        np.isfinite(np.asarray(x)).all()
        for x in jax.tree_util.tree_leaves(bf16)))

  def test_state_taps_equal_sequential_boundary_cotangents(self):
    taps = jnp.zeros((len(self.u1), len(self.a)), jnp.complex64)

    def parallel_loss(value):
      outputs, states = COMMON.fast_scan(
          self.a, self.b, self.c, self.u1, self.x0, self.init,
          dones=jnp.zeros((len(self.u1),), jnp.float32),
          state_taps=value)
      return jnp.mean(jnp.square(outputs - self.target1)), states

    def sequential_loss(value):
      state = self.x0
      outputs = []
      states = []
      for index in range(len(self.u1)):
        state = self.a * (state + value[index]) + self.b @ self.u1[index]
        outputs.append((self.c @ state).real)
        states.append(state)
      outputs = jnp.stack(outputs)
      return jnp.mean(jnp.square(outputs - self.target1)), jnp.stack(states)

    (parallel_value, parallel_states), parallel_grad = jax.value_and_grad(
        parallel_loss, has_aux=True)(taps)
    (sequential_value, sequential_states), sequential_grad = jax.value_and_grad(
        sequential_loss, has_aux=True)(taps)
    # Associative and left-to-right complex reductions have different floating
    # point order; this check is functional rather than bitwise.
    np.testing.assert_allclose(parallel_value, sequential_value, rtol=2e-4)
    np.testing.assert_allclose(
        parallel_states, sequential_states, rtol=3e-4, atol=1e-4)
    relative, cosine = self._quality(sequential_grad, parallel_grad)
    self.assertLessEqual(relative, 3e-4)
    self.assertGreaterEqual(cosine, 0.99999)

  def test_true_reset_zeros_predecessor_credit(self):
    dones = jnp.asarray([1, 0, 0, 0], jnp.float32)
    taps = jnp.zeros((len(self.u1), len(self.a)), jnp.complex64)

    def loss(initial):
      outputs, _ = COMMON.fast_scan(
          self.a, self.b, self.c, self.u1, initial, self.init,
          dones=dones, state_taps=taps)
      return jnp.mean(jnp.square(outputs - self.target1))

    gradient = jax.grad(loss)(self.x0)
    np.testing.assert_array_equal(np.asarray(gradient), np.zeros((2,), np.complex64))

  def test_one_primal_evaluation_for_joint_reverse_pass(self):
    calls = {'count': 0}

    def objective(a, taps):
      calls['count'] += 1
      outputs, _ = COMMON.fast_scan(
          a, self.b, self.c, self.u1, self.x0, self.init,
          state_taps=taps)
      return jnp.mean(jnp.square(outputs - self.target1))

    taps = jnp.zeros((len(self.u1), len(self.a)), jnp.complex64)
    _, gradients = jax.value_and_grad(objective, argnums=(0, 1))(
        self.a, taps)
    self.assertEqual(calls['count'], 1)
    self.assertEqual(len(gradients), 2)


if __name__ == '__main__':
  unittest.main()
