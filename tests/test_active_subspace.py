"""
Tests for the double_pass_randomized_eigh algorithm.

Assumptions:
    - The input matrix A is symmetric. For correct behavior, A should be
      real symmetric (or Hermitian if complex) because jnp.linalg.eigh is used.
    - The algorithm performs QR re-orthonormalization during power iterations,
      ensuring the subspace Q remains (approximately) orthonormal.
    - The projected matrix T = Qᵀ A Q is symmetric, so jnp.linalg.eigh returns
      its eigen-decomposition correctly, with eigenvalues in ascending order.
      These are then sorted in descending order.
    - The tests compare the computed eigenpairs to those from np.linalg.eigh
      on the full matrix A.
"""

import random

import jax
import jax.numpy as jnp

from bayesflux import estimate_input_active_subspace


def test_ridge_likelihood():
    dim = 20
    num_samples = 50000
    subspace_rank = 2

    for noise_variance in [0.35, 1.1, 3.0]:
        # Generate a random orthonormal basis A = [a1, a2]
        main_key = jax.random.PRNGKey(random.randint(0, 10))
        key, subkey1, subkey2 = jax.random.split(main_key, 3)
        A_random = jax.random.normal(subkey1, (dim, dim))
        Q, _ = jnp.linalg.qr(A_random)
        a1 = Q[:, 0]
        a2 = Q[:, 1]

        def grad_f(x):
            z1 = jnp.dot(a1, x)
            z2 = jnp.dot(a2, x)
            df_dz2 = jnp.cos(z2)
            return 2 * z1 * a1 + df_dz2 * a2

        # Sample from standard normal and compute Jacobians
        x_samples = jax.random.normal(subkey2, (num_samples, dim))
        J_samples = jax.vmap(lambda x: grad_f(x).reshape(1, -1))(
            x_samples
        )  # shape (N, 1, d)

        # Prior precision = Identity (standard normal)
        prior_precision = jnp.eye(dim)

        # Estimate subspace
        input_encoder_decoder_results = estimate_input_active_subspace(
            key, J_samples, noise_variance, prior_precision, subspace_rank
        )
        est_eigvals = input_encoder_decoder_results["eigenvalues"]
        est_eigvecs = input_encoder_decoder_results["decoder"]
        # Analytical eigenvalues:
        lambda_1 = 4 / noise_variance
        lambda_2 = 0.5676675 / noise_variance

        assert jnp.isclose(est_eigvals[0], lambda_1, rtol=1e-1)
        assert jnp.isclose(est_eigvals[1], lambda_2, rtol=1e-1)

        # Check that recovered eigenvectors are aligned with a1 and a2
        dot1 = jnp.abs(jnp.dot(est_eigvecs[:, 0], a1))
        dot2 = jnp.abs(jnp.dot(est_eigvecs[:, 1], a2))
        assert jnp.isclose(dot1, 1.0, atol=1e-2)
        assert jnp.isclose(dot2, 1.0, atol=1e-2)


if __name__ == "__main__":
    test_ridge_likelihood()
    print("All tests passed!")
