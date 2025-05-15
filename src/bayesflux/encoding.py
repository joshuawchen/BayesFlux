import time
from typing import Dict, Optional

import jax.numpy as jnp
from jax import device_put


def encode_inputs(*, inputs: jnp.ndarray, encoder: jnp.ndarray) -> jnp.ndarray:
    """
    Encodes input data using the provided encoder matrix.

    Parameters:
      inputs: jax array, shape (n_samples, input_dim).
      encoder: jax array, shape (input_dim, reduced_dim).

    Returns:
      jax array, shape (n_samples, reduced_dim), with encoded inputs.
    """
    return jnp.einsum("nx,xr->nr", inputs, encoder)


def encode_outputs(*, outputs: jnp.ndarray, encoder: jnp.ndarray) -> jnp.ndarray:
    """
    Encodes output data using the provided encoder matrix.

    Parameters:
      outputs: jax array, shape (n_samples, output_dim).
      encoder: jax array, shape (output_dim, reduced_dim).

    Returns:
      jax array, shape (n_samples, reduced_dim), with encoded outputs.
    """
    return jnp.einsum("no,or->nr", outputs, encoder)


def encode_Jacobians(
    *,
    jacobians: jnp.ndarray,
    input_decoder: Optional[jnp.ndarray] = None,
    output_encoder: Optional[jnp.ndarray] = None,
    batched: bool = False,
    batch_size: int = 50,
) -> jnp.ndarray:
    """
    Reduces Jacobians using input decoder and/or output encoder matrices.

    Parameters:
      jacobians: jax array, shape (n_samples, output_dim, input_dim).
      input_decoder: Optional; jax array of shape
        (input_dim, reduced_in_dim).
      output_encoder: Optional; jax array of shape
        (output_dim, reduced_out_dim).
      batched: Process reduction in batches (default: False).
        If batched, the jacobians may reside on the cpu a priori.
        They will be moved to gpu to be reduced.
      batch_size: Batch size when batched is True (default: 50).

    Returns:
      jax array with reduced dimensions:
        - If both input_decoder and output_encoder provided:
            (n_samples, reduced_out_dim, reduced_in_dim)
        - If only output_encoder provided:
            (n_samples, reduced_out_dim, input_dim)
        - If only input_decoder provided:
            (n_samples, output_dim, reduced_in_dim)
        - If neither provided, returns original jacobians.
    """

    def reduce_batch(jacs: jnp.ndarray) -> jnp.ndarray:
        if output_encoder is not None and input_decoder is not None:
            return jnp.einsum("ol,nox,xr->nlr", output_encoder, jacs, input_decoder)
        elif output_encoder is not None:
            return jnp.einsum("ol,nox->nlx", output_encoder, jacs)
        elif input_decoder is not None:
            return jnp.einsum("nox,xr->nor", jacs, input_decoder)
        else:
            return jacs

    if batched:
        total_len = jacobians.shape[0]
        reduced_batches = []
        for start in range(0, total_len, batch_size):
            end = min(start + batch_size, total_len)
            batch = device_put(jacobians[start:end])
            reduced_batches.append(reduce_batch(batch))
        return jnp.concatenate(reduced_batches, axis=0)
    else:
        return reduce_batch(jacobians)


def encode_input_output_Jacobian_data(
    *,
    inputs: jnp.ndarray,
    outputs: jnp.ndarray,
    jacobians: Optional[jnp.ndarray] = None,
    input_encoder: Optional[jnp.ndarray] = None,
    output_encoder: Optional[jnp.ndarray] = None,
    input_decoder: Optional[jnp.ndarray] = None,
    batched: bool = False,
    batch_size: int = 50,
) -> Dict[str, jnp.ndarray]:
    """
    Encodes inputs, outputs, and reduces Jacobians using the provided matrices.

    Parameters:
      inputs: jax array, shape (n_samples, input_dim).
      outputs: jax array, shape (n_samples, output_dim).
      jacobians:  Optional; jax array, shape (n_samples, output_dim, input_dim).
      input_encoder: Optional; for encoding inputs, shape
        (input_dim, reduced_dim).
      output_encoder: Optional; for encoding outputs, shape
        (output_dim, reduced_dim).
      input_decoder: Optional; for reducing jacobian input dim, shape
        (input_dim, reduced_in_dim).
      batched: Process jacobians in batches (default: False).
      batch_size: Batch size if batched is True.

    Returns:
      Dict with keys:
        "encoded_inputs": encoded inputs (or original if not provided).
        "encoded_outputs": encoded outputs (or original if not provided).
        "reduced_Jacobians": reduced jacobians (or original if not reduced).
    """
    print("Placing all data on GPU if not already")
    inputs = device_put(inputs) if inputs is not None else None
    outputs = device_put(outputs) if outputs is not None else None
    input_decoder = device_put(input_decoder) if input_decoder is not None else None
    input_encoder = device_put(input_encoder) if input_encoder is not None else None
    output_encoder = device_put(output_encoder) if output_encoder is not None else None

    start = time.time()
    encoded_inputs = encode_inputs(inputs=inputs, encoder=input_encoder) if input_encoder is not None else inputs
    input_encoding_time = time.time() - start

    start = time.time()
    encoded_outputs = encode_outputs(outputs=outputs, encoder=output_encoder) if output_encoder is not None else outputs
    output_encoding_time = time.time() - start
    encoded_dict = {
        "encoded_inputs": encoded_inputs,
        "encoded_outputs": encoded_outputs,
        "input_encoding_time": input_encoding_time,
        "output_encoding_time": output_encoding_time,
    }
    if jacobians is not None:
        start = time.time()
        encoded_Jacobians = encode_Jacobians(
            jacobians=jacobians,
            input_decoder=input_decoder,
            output_encoder=output_encoder,
            batched=batched,
            batch_size=batch_size,
        )
        jacobian_encoding_time = time.time() - start

        encoded_dict["encoded_Jacobians"] = encoded_Jacobians
        encoded_dict["Jacobian_encoding_time"] = jacobian_encoding_time
    return encoded_dict
