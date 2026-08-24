"""Two-device check for R2R's local-adjoint/global-parameter scaling."""

import os
import subprocess
import sys
import textwrap
import unittest


class MultiDeviceAdjointScalingTest(unittest.TestCase):

  def test_local_adjoint_and_pmean_parameter_gradient_match_global_objective(self):
    program = textwrap.dedent(r'''
        import jax
        import jax.numpy as jnp
        import numpy as np

        if jax.local_device_count() != 2:
          raise RuntimeError(jax.devices())
        theta = jnp.asarray(1.7, jnp.float32)
        values = jnp.asarray([[0.2, -0.4, 0.7], [0.3, 0.9, -0.5]])
        future = jnp.asarray([[0.6, -0.1, 0.2], [-0.3, 0.4, 0.8]])

        def shard(theta, value, adjoint):
          def objective(parameter, taps):
            state = parameter * taps
            return jnp.mean(jnp.square(state)) + jnp.sum(adjoint * state)
          _, (parameter_gradient, tap_gradient) = jax.value_and_grad(
              objective, argnums=(0, 1))(theta, value)
          parameter_gradient = jax.lax.pmean(parameter_gradient, 'i')
          return parameter_gradient, tap_gradient

        mapped = jax.pmap(shard, in_axes=(None, 0, 0), axis_name='i')
        parameter_gradient, tap_gradient = mapped(theta, values, future)

        def global_objective(parameter):
          state = parameter * values
          native = jnp.mean(jnp.square(state))
          # pmean of each device's local future-adjoint sum.
          surrogate = jnp.sum(future * state) / 2
          return native + surrogate

        reference_parameter = jax.grad(global_objective)(theta)
        np.testing.assert_allclose(
            parameter_gradient, jnp.repeat(reference_parameter[None], 2),
            rtol=1e-6, atol=1e-6)

        def local_tap_gradient(value, adjoint):
          return jax.grad(lambda tap: (
              jnp.mean(jnp.square(theta * tap)) +
              jnp.sum(adjoint * theta * tap)))(value)

        reference_taps = jax.vmap(local_tap_gradient)(values, future)
        # Input cotangents stay local and are neither pmean'ed nor divided by
        # the device count; their later consumption is pmean'ed exactly once.
        np.testing.assert_allclose(
            tap_gradient, reference_taps, rtol=1e-6, atol=1e-6)
    ''')
    environment = os.environ.copy()
    flags = environment.get('XLA_FLAGS', '')
    environment['XLA_FLAGS'] = (
        flags + ' --xla_force_host_platform_device_count=2').strip()
    result = subprocess.run(
        [sys.executable, '-c', program], env=environment,
        text=True, capture_output=True, timeout=90)
    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == '__main__':
  unittest.main()
