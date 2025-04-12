import random

import jax
import jax.numpy as jnp

# Import the encoder functions.
from bayesflux.encoding import encode_input_output_Jacobian_data
from bayesflux.subspace_detection import information_theoretic_dimension_reduction


def test_no_encoders():
    """No encoders provided: outputs should be unchanged."""
    num_samples = 100
    input_dim = 20
    output_dim = 10

    inputs = jax.random.normal(jax.random.PRNGKey(0), (num_samples, input_dim))
    outputs = jax.random.normal(jax.random.PRNGKey(1), (num_samples, output_dim))
    jacobians = jax.random.normal(jax.random.PRNGKey(2), (num_samples, output_dim, input_dim))

    results = encode_input_output_Jacobian_data(inputs=inputs, outputs=outputs, jacobians=jacobians)

    assert results["encoded_inputs"].shape == (num_samples, input_dim)
    assert results["encoded_outputs"].shape == (num_samples, output_dim)
    # Jacobians remain unchanged.
    assert results["encoded_Jacobians"].shape == (num_samples, output_dim, input_dim)


def test_input_encoder_only():
    """Only an input encoder is provided: inputs are reduced; outputs and
    jacobians remain unchanged.
    """
    num_samples = 100
    input_dim = 20
    output_dim = 10
    reduced_in = 5

    inputs = jax.random.normal(jax.random.PRNGKey(0), (num_samples, input_dim))
    outputs = jax.random.normal(jax.random.PRNGKey(1), (num_samples, output_dim))
    jacobians = jax.random.normal(jax.random.PRNGKey(2), (num_samples, output_dim, input_dim))

    input_encoder = jax.random.normal(jax.random.PRNGKey(3), (input_dim, reduced_in))

    results = encode_input_output_Jacobian_data(
        inputs=inputs,
        outputs=outputs,
        jacobians=jacobians,
        input_encoder=input_encoder,
    )

    assert results["encoded_inputs"].shape == (num_samples, reduced_in)
    assert results["encoded_outputs"].shape == (num_samples, output_dim)
    assert results["encoded_Jacobians"].shape == (num_samples, output_dim, input_dim)


def test_output_encoder_only():
    """Only an output encoder is provided: outputs are reduced; for
    jacobians the output dimension is reduced.
    """
    num_samples = 100
    input_dim = 20
    output_dim = 10
    reduced_out = 4

    inputs = jax.random.normal(jax.random.PRNGKey(0), (num_samples, input_dim))
    outputs = jax.random.normal(jax.random.PRNGKey(1), (num_samples, output_dim))
    jacobians = jax.random.normal(jax.random.PRNGKey(2), (num_samples, output_dim, input_dim))

    output_encoder = jax.random.normal(jax.random.PRNGKey(4), (output_dim, reduced_out))

    results = encode_input_output_Jacobian_data(
        inputs=inputs,
        outputs=outputs,
        jacobians=jacobians,
        output_encoder=output_encoder,
    )

    assert results["encoded_outputs"].shape == (num_samples, reduced_out)
    assert results["encoded_inputs"].shape == (num_samples, input_dim)
    # Jacobians are reduced only along the output dimension.
    assert results["encoded_Jacobians"].shape == (num_samples, reduced_out, input_dim)


def test_both_encoders_with_decoder():
    """
    Provide both input and output encoders (for encoding inputs/outputs)
    and an input decoder (for reducing the input dimension of jacobians).
    """
    num_samples = 100
    input_dim = 20
    output_dim = 10
    reduced_in = 5
    reduced_out = 4

    inputs = jax.random.normal(jax.random.PRNGKey(0), (num_samples, input_dim))
    outputs = jax.random.normal(jax.random.PRNGKey(1), (num_samples, output_dim))
    jacobians = jax.random.normal(jax.random.PRNGKey(2), (num_samples, output_dim, input_dim))

    input_encoder = jax.random.normal(jax.random.PRNGKey(3), (input_dim, reduced_in))
    input_decoder = jax.random.normal(jax.random.PRNGKey(4), (input_dim, reduced_in))
    output_encoder = jax.random.normal(jax.random.PRNGKey(5), (output_dim, reduced_out))

    results = encode_input_output_Jacobian_data(
        inputs=inputs,
        outputs=outputs,
        jacobians=jacobians,
        input_encoder=input_encoder,
        output_encoder=output_encoder,
        input_decoder=input_decoder,
        batched=True,
        batch_size=10,
    )

    assert results["encoded_inputs"].shape == (num_samples, reduced_in)
    assert results["encoded_outputs"].shape == (num_samples, reduced_out)
    # With both reductions, jacobians become
    # (num_samples, reduced_out, reduced_in)
    assert results["encoded_Jacobians"].shape == (num_samples, reduced_out, reduced_in)


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


if __name__ == "__main__":
    test_no_encoders()
    test_input_encoder_only()
    test_output_encoder_only()
    test_both_encoders_with_decoder()
    test_information_theoretic_dimension_reduction_bayesian_inference()
    print("All tests passed!")

    # generate_reduced_training_data
    # generate_full_Jacobian_data_for_computing_dimension_reduction
    # test GaussianInputOuputAndDerivativesSampler
