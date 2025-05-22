import random

import jax
import jax.numpy as jnp
import numpy as np

# Import the encoder functions.
from bayesflux.encoding import encode_input_output_Jacobian_data
from bayesflux.subspace_detection import (
    estimate_input_active_subspace,
    information_theoretic_dimension_reduction,
    moment_based_dimension_reduction,
)


def test_ridge_likelihood_active_subspace():
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
        J_samples = jax.vmap(lambda x: grad_f(x).reshape(1, -1))(x_samples)  # shape (N, 1, d)

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


def test_information_theoretic_dimension_reduction_bayesian_inference():
    """
    Simulates a Bayesian inverse problem.

    - Inputs are sampled from a standard normal.
    - The forward model is defined as:
          f(x) = (a1·x)^2 + sin(a2·x)
      with gradient:
          grad_f(x) = 2*(a1·x)*a1 + cos(a2·x)*a2
    - Jacobians are computed via vmap.
    - A prior precision (and covariance) are identity.

    The information theoretic dimension reduction is performed with given max
    input/output dimensions. The test then calls
    encode_input_output_Jacobian_data using the reduction matrices returned by
    information_theoretic_dimension_reduction and verifies the shapes.
    """
    dim = 20
    num_samples = 1000  # Use a smaller sample for testing speed.
    max_input_dimension = 2  # Desired input subspace dimension.
    max_output_dimension = 1  # Desired output subspace dimension (f is scalar).

    noise_variance = 1.1

    # Set up a random orthonormal basis.
    main_key = jax.random.PRNGKey(random.randint(0, 10))
    key, subkey1, subkey2, subkey3 = jax.random.split(main_key, 4)
    A_random = jax.random.normal(subkey1, (dim, dim))
    Q, _ = jnp.linalg.qr(A_random)
    a1 = Q[:, 0]
    a2 = Q[:, 1]

    def f(x):
        # Forward model: scalar output.
        z1 = jnp.dot(a1, x)
        z2 = jnp.dot(a2, x)
        return (z1**2) + jnp.sin(z2)

    def grad_f(x):
        # Gradient of f.
        z1 = jnp.dot(a1, x)
        z2 = jnp.dot(a2, x)
        return 2 * z1 * a1 + jnp.cos(z2) * a2

    # Generate samples.
    x_samples = jax.random.normal(subkey2, (num_samples, dim))
    outputs = jax.vmap(f)(x_samples).reshape(num_samples, 1)
    # Compute Jacobians; since f is scalar, each Jacobian is (1, dim).
    J_samples = jax.vmap(lambda x: grad_f(x).reshape(1, -1))(x_samples)

    prior_precision = jnp.eye(dim)
    prior_covariance = jnp.eye(dim)

    # Perform information theoretic dimension reduction.
    reduction_dict = information_theoretic_dimension_reduction(
        key=key,
        J_samples=J_samples,
        noise_variance=noise_variance,
        max_input_dimension=max_input_dimension,
        max_output_dimension=max_output_dimension,
        prior_precision=prior_precision,
        prior_covariance=prior_covariance,
    )
    # reduction_dict is expected to have keys "input" and "output", each
    # containing an "encoder" and "decoder".

    # Use the reduction matrices for encoding.
    encoded_data = encode_input_output_Jacobian_data(
        inputs=x_samples,
        outputs=outputs,
        jacobians=J_samples,
        input_encoder=reduction_dict["input"]["encoder"],
        output_encoder=reduction_dict["output"]["encoder"],
        input_decoder=reduction_dict["input"]["decoder"],
        batched=True,
        batch_size=50,
    )

    # Verify that the shapes match the specified subspace dimensions.
    assert encoded_data["encoded_inputs"].shape == (num_samples, max_input_dimension)
    assert encoded_data["encoded_outputs"].shape == (num_samples, max_output_dimension)
    # The encoded Jacobians should have shape
    # (num_samples, max_output_dimension, max_input_dimension)
    assert encoded_data["encoded_Jacobians"].shape == (
        num_samples,
        max_output_dimension,
        max_input_dimension,
    )


def test_moment_based_dimension_reduction():
    """
    Test that moment_based_dimension_reduction correctly recovers the dominant eigenvalues
    for input and output covariance matrices constructed with known eigenvalue decays.
    """
    key = jax.random.PRNGKey(0)

    # Dimensions and target subspace ranks
    d_x, d_y = 10, 8
    max_input_dimension = 5
    max_output_dimension = 4
    N = 1000

    # Create input covariance with known decay: eigenvalues ~ 1/(i+1)^2
    lambda_in = np.array([1.0 / (i + 1) ** 2 for i in range(d_x)])
    Q_x, _ = np.linalg.qr(np.random.randn(d_x, d_x))
    input_covariance_matrix = Q_x @ np.diag(lambda_in) @ Q_x.T

    # Use identity for the L2 inner product matrix
    L2_inner_product_matrix = np.eye(d_x)

    # Create output covariance with known decay: eigenvalues ~ 1/(i+1)
    lambda_out = np.array([1.0 / (i + 1) for i in range(d_y)])
    Q_y, _ = np.linalg.qr(np.random.randn(d_y, d_y))
    output_cov = Q_y @ np.diag(lambda_out) @ Q_y.T

    # Generate synthetic output samples from the Gaussian with covariance output_cov
    chol_y = np.linalg.cholesky(output_cov)
    output_samples = np.random.randn(N, d_y) @ chol_y.T

    # Convert to JAX arrays
    input_cov = jnp.array(input_covariance_matrix)
    l2_mat = jnp.array(L2_inner_product_matrix)
    out_samps = jnp.array(output_samples)

    # Run the dimension reduction
    result = moment_based_dimension_reduction(
        key, max_input_dimension, max_output_dimension, input_cov, l2_mat, out_samps
    )

    # Extract computed eigenvalues
    computed_in = np.array(result["input"]["eigenvalues"])
    computed_out = np.array(result["output"]["eigenvalues"])

    # Expected top eigenvalues in descending order
    expected_in = np.sort(lambda_in)[::-1][:max_input_dimension]
    expected_out = np.sort(lambda_out)[::-1][:max_output_dimension]

    # Assert within a reasonable tolerance
    assert np.allclose(
        computed_in, expected_in, rtol=1e-1
    ), f"Input eigenvalues {computed_in} differ from expected {expected_in}"
    assert np.allclose(
        computed_out, expected_out, rtol=1e-1
    ), f"Output eigenvalues {computed_out} differ from expected {expected_out}"


if __name__ == "__main__":
    test_ridge_likelihood_active_subspace()
    test_information_theoretic_dimension_reduction_bayesian_inference()
    test_moment_based_dimension_reduction()
    print("All tests passed!")
