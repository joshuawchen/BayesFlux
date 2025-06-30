import random
import time
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np

# Import the encoder functions.
from bayesflux.generation import (GaussianInputOuputAndDerivativesSampler,
                                  generate_full_Jacobian_data,
                                  generate_output_data,
                                  generate_reduced_training_data)


class ExampleSampler(GaussianInputOuputAndDerivativesSampler):
    def __init__(self):
        self._noise_precision = 1
        self._input_dimension = 5
        self._output_dimension = 4
        self._precision, self._L2_inner_product_matrix = np.identity(self._input_dimension), np.identity(
            self._input_dimension
        )

    def sample_input(self) -> np.ndarray:
        return np.random.normal(0, 1, (5,))  # N(0, Cov) Cov = I_5

    def _init_value(self): ...

    def _init_matrix_jacobian_prod(self, *, matrix: np.ndarray = None): ...

    def _value(self, input_sample: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return input_sample[0:-1] ** 3.0  # f(x) = x*3, pointwise cube, except for the last dimension of x

    def value_and_matrix_jacobian_prod(self, input_sample: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        output_sample = self.value(input_sample)
        start = time.perf_counter()
        jacobian_sample = np.zeros((self._input_dimension, self._output_dimension))
        jacobian_sample[0:-1] = (3 * input_sample[0:-1] ** 2.0)[:, None]
        jacobian_sample = jacobian_sample @ self._matrix_jacobian_prod_matrix
        self.jacobian_product_computation_time += time.perf_counter() - start
        return output_sample, jacobian_sample.T  # grad f(x) = diag(3x^2) except the last dimension of x


def test_generate_full_Jacobian_data():
    sampler = ExampleSampler()
    results = generate_full_Jacobian_data(sampler_wrapper=sampler, N_samples=10)
    assert results["Jacobians"].shape == (10, 4, 5)
    assert results["inputs"].shape == (10, 5)
    assert results["outputs"].shape == (10, 4)


def test_generate_output_data():
    sampler = ExampleSampler()
    results = generate_output_data(sampler_wrapper=sampler, N_samples=10)
    assert results["inputs"].shape == (10, 5)
    assert results["outputs"].shape == (10, 4)
    assert results.get("Jacobians") == None


def test_generate_reduced_training_data():
    sampler = ExampleSampler()
    results = generate_reduced_training_data(
        sampler_wrapper=sampler,
        N_samples=12,
        output_encoder=np.identity(4)[:, :3],
        input_decoder=np.identity(5)[:, :2],
        input_encoder=np.identity(5)[:, :2],
    )
    assert results["encoded_inputs"].shape == (12, 2)
    assert results["encoded_outputs"].shape == (12, 3)
    assert results["encoded_Jacobians"].shape == (12, 3, 2)


if __name__ == "__main__":
    test_generate_full_Jacobian_data()
    test_generate_output_data()
    test_generate_reduced_training_data()
    print("All tests passed!")
