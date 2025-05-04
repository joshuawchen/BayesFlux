import random

import jax
import jax.numpy as jnp

# Import the encoder functions.
from bayesflux.encoding import encode_input_output_Jacobian_data


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


if __name__ == "__main__":
    test_no_encoders()
    test_input_encoder_only()
    test_output_encoder_only()
    test_both_encoders_with_decoder()
    print("All tests passed!")
