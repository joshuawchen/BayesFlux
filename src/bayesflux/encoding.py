import time
from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np


class FunctionAndDerivatives(ABC):
    """
    Interface for functions and their derivatives for convenient training data
    sampling. This class is meant for objects that work solely with NumPy arrays.
    The input and output dimensions are fixed (provided in the constructor) and
    used for sampling training data.

    Attributes:
        input_dimension: The dimension of the input vector.
        output_dimension: The dimension of the output vector.
        output_computation_time: Optional time to compute outputs.
        jacobian_product_computation_time: Optional time to evaluate jacobian products.
    """

    @property
    def input_dimension(self) -> int:
        """
        The dimension of the input vector.

        Returns:
            An integer representing the input dimension.
        """
        return self._input_dimension

    @property
    def output_dimension(self) -> int:
        """
        The dimension of the output vector.

        Returns:
            An integer representing the output dimension.
        """
        return self._output_dimension

    @abstractmethod
    def sample_input(self) -> np.ndarray:
        """
        Sample an input vector from the function's domain. This method should
        return a NumPy array of shape (input_dimension,).

        Returns:
            A NumPy array representing a single input sample.
        """
        ...

    def set_matrix_jacobian_prod(self, *, matrix: Optional[np.ndarray] = None) -> None:
        """
        Set the matrix used for the matrix–Jacobian product computation.

        Parameters:
            matrix: A NumPy array used to modify the Jacobian product, or None.
        """
        self._matrix_jacobian_prod_matrix = matrix

    @abstractmethod
    def value_and_matrix_jacobian_prod(self, input_sample: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the function value and the matrix–Jacobian product at a given
        input sample. The computation uses the matrix set by
        set_matrix_jacobian_prod.

        Parameters:
            input_sample: A NumPy array of shape (input_dimension,).

        Returns:
            A tuple containing:
              - A NumPy array of shape (output_dimension,) representing the
                function value.
              - A NumPy array representing the matrix–Jacobian product. Its shape
                depends on the applied matrix.
        """
        ...

    def extract_and_clear_computation_times(self) -> Dict[str, Optional[float]]:
        """
        Extract computation times (if they exist) and reset them to zero.

        This method looks for the attributes:
          - 'output_computation_time'
          - 'jacobian_product_computation_time'

        These attributes should have been used to count the total computation times
        for evaluating the function and its matrix jacobian product respectively

        Returns:
            A dictionary with keys:
              'output_computation_time' and
              'jacobian_product_computation_time' containing their values.
              If an attribute is not set, its value will be None.
        """
        times: Dict[str, Optional[float]] = {}
        if hasattr(self, "output_computation_time"):
            times["output_computation_time"] = self.output_computation_time
            self.output_computation_time = 0.0
        else:
            times["output_computation_time"] = None

        if hasattr(self, "jacobian_product_computation_time"):
            times["jacobian_product_computation_time"] = self.jacobian_product_computation_time
            self.jacobian_product_computation_time = 0.0
        else:
            times["jacobian_product_computation_time"] = None

        return times


def generate_full_Jacobian_data_for_computing_dimension_reduction(
    func_wrapper: FunctionAndDerivatives, N_samples: int
) -> Dict[str, np.ndarray]:
    """
    Generate full Jacobian data for dimension reduction.

    This is a thin wrapper around generate_reduced_training_data, which only
    requires an object that accepts NumPy arrays as input and returns NumPy arrays.

    Parameters:
        func_wrapper: An object implementing FunctionAndDerivatives.
        N_samples: The number of samples to generate.

    Returns:
        A dictionary containing training data and optional computation times.
    """
    return generate_reduced_training_data(func_wrapper, N_samples)


def generate_reduced_training_data(
    func_wrapper: FunctionAndDerivatives,
    N_samples: int,
    output_encoder: Optional[np.ndarray] = None,
    input_decoder: Optional[np.ndarray] = None,
    input_encoder: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """
    Generate reduced training data using a function and its derivatives.

    This function repeatedly samples inputs and computes the corresponding
    function outputs and matrix–Jacobian products. It then applies optional
    encoding/decoding to reduce the dimensionality of the inputs, outputs, and
    Jacobian products. This function requires that the provided func_wrapper works
    entirely with NumPy arrays.

    Parameters:
        func_wrapper: An object implementing FunctionAndDerivatives that works with
            NumPy arrays.
        N_samples: The number of samples to generate.
        output_encoder: Optional; a NumPy array of shape
            (func_wrapper.output_dimension, reduced_out_dim) to encode outputs.
        input_decoder: Optional; a NumPy array of shape
            (func_wrapper.input_dimension, reduced_in_dim) to reduce the input
            dimension of the Jacobian product.
        input_encoder: Optional; a NumPy array of shape
            (func_wrapper.input_dimension, reduced_in_dim) to encode inputs.

    Returns:
        A dictionary with keys:
          'inputs': A NumPy array of shape (N_samples, D_in_encoded), where
              D_in_encoded equals reduced_in_dim if input_encoder is provided,
              otherwise func_wrapper.input_dimension.
          'outputs': A NumPy array of shape (N_samples, D_out_encoded), where
              D_out_encoded equals reduced_out_dim if output_encoder is provided,
              otherwise func_wrapper.output_dimension.
          'Jacobians': A NumPy array whose shape depends on the provided
              encoders/decoders:
                - (N_samples, reduced_out_dim, reduced_in_dim) if both
                  output_encoder and input_decoder are provided.
                - (N_samples, reduced_out_dim, func_wrapper.input_dimension) if only
                  output_encoder is provided.
                - (N_samples, func_wrapper.output_dimension, reduced_in_dim) if only
                  input_decoder is provided.
                - (N_samples, func_wrapper.output_dimension,
                  func_wrapper.input_dimension) if neither is provided.
          'output_computation_time': The time for computing outputs (if set).
          'Jacobian_computation_time': The time for computing the Jacobian product
              (if set).
    """
    encoded_input_dimension = input_encoder.shape[1] if input_encoder is not None else func_wrapper.input_dimension
    encoded_output_dimension = output_encoder.shape[1] if output_encoder is not None else func_wrapper.output_dimension
    print("Preparing for sampling...")
    encoded_inputs = np.empty((N_samples, encoded_input_dimension))
    encoded_outputs = np.empty((N_samples, encoded_output_dimension))
    encoded_jacobian_prod = np.empty((N_samples, encoded_output_dimension, encoded_input_dimension))

    func_wrapper.set_matrix_jacobian_prod(matrix=output_encoder)
    print("Sampling...")
    input_encoding_time = 0.0
    output_encoding_time = 0.0
    jacobian_encoding_time = 0.0
    for i in range(N_samples):
        input_sample = func_wrapper.sample_input()
        if input_encoder is not None:
            start = time.time()
            encoded_inputs[i] = input_sample @ input_encoder
            input_encoding_time += time.time() - start

        else:
            encoded_inputs[i] = input_sample
        output_i, matrix_jacobian_prod_i = func_wrapper.value_and_matrix_jacobian_prod(input_sample)
        if output_encoder is not None:
            start = time.time()
            encoded_outputs[i] = output_i @ output_encoder
            output_encoding_time += time.time() - start
        else:
            encoded_outputs[i] = output_i

        if input_decoder is not None:
            start = time.time()
            encoded_jacobian_prod[i] = matrix_jacobian_prod_i @ input_decoder
            jacobian_encoding_time += time.time() - start
        else:
            encoded_jacobian_prod[i] = matrix_jacobian_prod_i

    computation_times = func_wrapper.extract_and_clear_computation_times()
    input_key = "encoded_inputs" if input_encoder is not None else "inputs"
    output_key = "encoded_output" if output_encoder is not None else "outputs"
    jacobian_key = "encoded_Jacobians" if (input_decoder is not None or output_encoder is not None) else "Jacobians"

    results = {
        input_key: encoded_inputs,
        output_key: encoded_outputs,
        jacobian_key: encoded_jacobian_prod,
        "output_computation_time": computation_times.get("output_computation_time"),
        "Jacobian_computation_time": computation_times.get("jacobian_product_computation_time"),
    }
    if input_encoder is not None:
        results["input_encoding_time"] = input_encoding_time
    if output_encoder is not None:
        results["output_encoding_time"] = output_encoding_time
    if input_decoder is not None:
        results["Jacobian_encoding_time"] = jacobian_encoding_time
    return results


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
            batch = jax.device_put(jacobians[start:end])
            reduced_batches.append(reduce_batch(batch))
        return jnp.concatenate(reduced_batches, axis=0)
    else:
        return reduce_batch(jacobians)


def encode_input_output_Jacobian_data(
    *,
    inputs: jnp.ndarray,
    outputs: jnp.ndarray,
    jacobians: jnp.ndarray,
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
      jacobians: jax array, shape (n_samples, output_dim, input_dim).
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
    inputs = jax.device_put(inputs) if inputs is not None else None
    outputs = jax.device_put(outputs) if outputs is not None else None
    input_decoder = jax.device_put(input_decoder) if input_decoder is not None else None
    input_encoder = jax.device_put(input_encoder) if input_encoder is not None else None
    output_encoder = jax.device_put(output_encoder) if output_encoder is not None else None

    start = time.time()
    encoded_inputs = encode_inputs(inputs=inputs, encoder=input_encoder) if input_encoder is not None else inputs
    input_encoding_time = time.time() - start

    start = time.time()
    encoded_outputs = encode_outputs(outputs=outputs, encoder=output_encoder) if output_encoder is not None else outputs
    output_encoding_time = time.time() - start

    start = time.time()
    encoded_Jacobians = encode_Jacobians(
        jacobians=jacobians,
        input_decoder=input_decoder,
        output_encoder=output_encoder,
        batched=batched,
        batch_size=batch_size,
    )
    jacobian_encoding_time = time.time() - start
    return {
        "encoded_inputs": encoded_inputs,
        "encoded_outputs": encoded_outputs,
        "encoded_Jacobians": encoded_Jacobians,
        "input_encoding_time": input_encoding_time,
        "output_encoding_time": output_encoding_time,
        "Jacobian_encoding_time": jacobian_encoding_time,
    }
