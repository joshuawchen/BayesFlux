import time
from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple

import numpy as np
from numpy.typing import ArrayLike


class InputOuputAndDerivativesSampler(ABC):
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

    def init_matrix_jacobian_prod(self, *, matrix: Optional[np.ndarray] = None) -> None:
        """
        Set the matrix used for the matrix–Jacobian product computation.

        Parameters:
            matrix: A NumPy array used to modify the Jacobian product, or None.
        """
        self._matrix_jacobian_prod_matrix = matrix
        self.jacobian_product_computation_time = 0
        self._init_matrix_jacobian_prod(matrix=self._matrix_jacobian_prod_matrix)

    def init_value(self) -> None:
        self.output_computation_time = 0
        self._init_value()

    @abstractmethod
    def _init_value(self) -> None: ...

    @abstractmethod
    def _init_matrix_jacobian_prod(self, *, matrix: Optional[np.ndarray] = None) -> None: ...

    def value(self, input_sample: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the function value at a given
        input sample.

        Parameters:
            input_sample: A NumPy array of shape (input_dimension,).

        Returns:
            - A NumPy array of shape (output_dimension,) representing the
                function value.
        """
        start = time.time()
        value = self._value(input_sample)
        self.output_computation_time += time.time() - start
        return value

    @abstractmethod
    def _value(self, input_sample: np.ndarray) -> Tuple[np.ndarray, np.ndarray]: ...

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


class GaussianInputOuputAndDerivativesSampler(InputOuputAndDerivativesSampler):

    @property
    def precision(self) -> ArrayLike:
        return self._precision

    @property
    def L2_inner_product_matrix(self) -> ArrayLike:
        return self._L2_inner_product_matrix


def generate_full_Jacobian_data(
    *,
    sampler_wrapper: InputOuputAndDerivativesSampler,
    N_samples: int,
) -> Dict[str, np.ndarray]:
    """
    Generate full Jacobian data for dimension reduction.

    This is a thin wrapper around generate_reduced_training_data, which only
    requires an object that accepts NumPy arrays as input and returns NumPy arrays.

    Parameters:
        sampler_wrapper: An object implementing InputOuputAndDerivativesSampler.
        N_samples: The number of samples to generate.

    Returns:
        A dictionary containing training data and optional computation times.
    """
    return generate_reduced_training_data(sampler_wrapper=sampler_wrapper, N_samples=N_samples, generate_Jacobians=True)


def generate_output_data(*, sampler_wrapper: InputOuputAndDerivativesSampler, N_samples: int) -> Dict[str, np.ndarray]:
    """
    Generate full Jacobian data for dimension reduction.

    This is a thin wrapper around generate_reduced_training_data, which only
    requires an object that accepts NumPy arrays as input and returns NumPy arrays.

    Parameters:
        sampler_wrapper: An object implementing InputOuputAndDerivativesSampler.
        N_samples: The number of samples to generate.

    Returns:
        A dictionary containing training data and optional computation times.
    """
    return generate_reduced_training_data(
        sampler_wrapper=sampler_wrapper,
        N_samples=N_samples,
        generate_Jacobians=False,
    )


def generate_reduced_training_data(
    *,
    sampler_wrapper: InputOuputAndDerivativesSampler,
    N_samples: int,
    output_encoder: Optional[np.ndarray] = None,
    input_decoder: Optional[np.ndarray] = None,
    input_encoder: Optional[np.ndarray] = None,
    generate_Jacobians: bool = True,
    reduce_input_before: bool = False,
) -> Dict[str, np.ndarray]:
    """
    Generate reduced training data using a function and its derivatives.

    This function repeatedly samples inputs and computes the corresponding
    function outputs and matrix–Jacobian products. It then applies optional
    encoding/decoding to reduce the dimensionality of the inputs, outputs, and
    Jacobian products. This function requires that the provided sampler_wrapper works
    entirely with NumPy arrays.

    Parameters:
        sampler_wrapper: An object implementing InputOuputAndDerivativesSampler that works
            with NumPy arrays.
        N_samples: The number of samples to generate.
        output_encoder: Optional; a NumPy array of shape
            (sampler_wrapper.output_dimension, reduced_out_dim) to encode outputs.
        input_decoder: Optional; a NumPy array of shape
            (sampler_wrapper.input_dimension, reduced_in_dim) to reduce the input
            dimension of the Jacobian product.
        input_encoder: Optional; a NumPy array of shape
            (sampler_wrapper.input_dimension, reduced_in_dim) to encode inputs.

    Returns:
        A dictionary with keys:
          'inputs': A NumPy array of shape (N_samples, D_in_encoded), where
              D_in_encoded equals reduced_in_dim if input_encoder is provided,
              otherwise sampler_wrapper.input_dimension.
          'outputs': A NumPy array of shape (N_samples, D_out_encoded), where
              D_out_encoded equals reduced_out_dim if output_encoder is provided,
              otherwise sampler_wrapper.output_dimension.
          'Jacobians': A NumPy array whose shape depends on the provided
              encoders/decoders:
                - (N_samples, reduced_out_dim, reduced_in_dim) if both
                  output_encoder and input_decoder are provided.
                - (N_samples, reduced_out_dim, sampler_wrapper.input_dimension) if only
                  output_encoder is provided.
                - (N_samples, sampler_wrapper.output_dimension, reduced_in_dim) if only
                  input_decoder is provided.
                - (N_samples, sampler_wrapper.output_dimension,
                  sampler_wrapper.input_dimension) if neither is provided.
          'output_computation_time': The time for computing outputs (if set).
          'Jacobian_computation_time': The time for computing the Jacobian product
              (if set).
    """
    encoded_input_dimension = input_encoder.shape[1] if input_encoder is not None else sampler_wrapper.input_dimension
    encoded_output_dimension = (
        output_encoder.shape[1] if output_encoder is not None else sampler_wrapper.output_dimension
    )
    encoded_inputs = np.empty((N_samples, encoded_input_dimension))
    encoded_outputs = np.empty((N_samples, encoded_output_dimension))
    sampler_wrapper.init_value()
    if generate_Jacobians:
        encoded_jacobian_prod = np.empty((N_samples, encoded_output_dimension, encoded_input_dimension))
        sampler_wrapper.init_matrix_jacobian_prod(
            matrix=output_encoder if output_encoder is not None else np.identity(encoded_output_dimension)
        )

    print("Sampling...")
    input_encoding_time = 0.0
    output_encoding_time = 0.0
    jacobian_decoding_time = 0.0
    for i in range(N_samples):
        input_sample = sampler_wrapper.sample_input()
        if input_encoder is not None:
            start = time.time()
            encoded_inputs[i] = input_sample @ input_encoder
            input_encoding_time += time.time() - start

        else:
            if reduce_input_before:
                encoded_inputs[i] = input_sample @ input_encoder
                input_sample = encoded_inputs[i] @ input_decoder.T
            else:
                encoded_inputs[i] = input_sample
        if generate_Jacobians:
            output_i, matrix_jacobian_prod_i = sampler_wrapper.value_and_matrix_jacobian_prod(input_sample)
        else:
            output_i = sampler_wrapper.value(input_sample)
        if output_encoder is not None:
            start = time.time()
            encoded_outputs[i] = output_i @ output_encoder
            output_encoding_time += time.time() - start
        else:
            encoded_outputs[i] = output_i
        if generate_Jacobians:
            if input_decoder is not None:
                start = time.time()
                encoded_jacobian_prod[i] = matrix_jacobian_prod_i @ input_decoder
                jacobian_decoding_time += time.time() - start
            else:
                encoded_jacobian_prod[i] = matrix_jacobian_prod_i

    computation_times = sampler_wrapper.extract_and_clear_computation_times()
    input_key = "encoded_inputs" if input_encoder is not None else "inputs"
    output_key = "encoded_outputs" if output_encoder is not None else "outputs"
    jacobian_key = "encoded_Jacobians" if (input_decoder is not None or output_encoder is not None) else "Jacobians"

    results = {
        input_key: encoded_inputs,
        output_key: encoded_outputs,
        "output_computation_time": computation_times.get("output_computation_time"),
    }
    if generate_Jacobians:
        results[jacobian_key] = encoded_jacobian_prod
        results["Jacobian_encoding_time"] = computation_times.get("jacobian_product_computation_time")

    if input_encoder is not None:
        results["input_encoding_time"] = input_encoding_time
    if output_encoder is not None:
        results["output_encoding_time"] = output_encoding_time
    if input_decoder is not None:
        results["Jacobian_decoding_time"] = jacobian_decoding_time
    return results
