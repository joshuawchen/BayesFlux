import time
from typing import Any, Dict

import jax
import jax.numpy as jnp
from jax.scipy.linalg import solve_triangular
from randlax import double_pass_randomized_eigh, double_pass_randomized_gen_eigh


@jax.jit
def __JCJtransp(J: jnp.ndarray, L: jnp.ndarray) -> jnp.ndarray:
    # Solve L Y = J.T for Y.
    Y = solve_triangular(L, J.T, lower=True)
    # Then, J C^-1 J.T = Y.T @ Y
    return Y.T @ Y


def average_JCJtranspose(J_samples: jnp.ndarray, prior_precision: jnp.ndarray) -> jnp.ndarray:
    L = jnp.linalg.cholesky(prior_precision)
    # Average over the batch dimension (axis=0)
    return jnp.mean(jax.vmap(lambda J: __JCJtransp(J, L))(J_samples), axis=0)


def information_theoretic_dimension_reduction(
    key: Any,
    J_samples: jnp.ndarray,
    noise_variance: float,
    prior_precision: jnp.ndarray,
    max_input_dimension: int = None,
    max_output_dimension: int = None,
    prior_covariance: jnp.ndarray = None,
):
    """Document that this is for both input/output dimension reduction"""
    if max_input_dimension is not None:
        start = time.time()
        input_encodec_dict = estimate_input_active_subspace(
            key=key,
            J_samples=J_samples,
            noise_variance=noise_variance,
            prior_precision=prior_precision,
            subspace_rank=max_input_dimension,
        )
        input_encodec_dict["computation_time"] = time.time() - start
    else:
        input_encodec_dict = dict()
    if max_output_dimension is not None:
        start = time.time()
        output_encodec_dict = estimate_output_informative_subspace(
            key=key,
            J_samples=J_samples,
            noise_variance=noise_variance,
            subspace_rank=max_output_dimension,
            prior_precision=prior_precision,
            prior_covariance=prior_covariance,
        )
        output_encodec_dict["computation_time"] = time.time() - start
    else:
        output_encodec_dict = dict()
    return {"input": input_encodec_dict, "output": output_encodec_dict}


def moment_based_dimension_reduction(
    key: Any,
    input_covariance_matrix: jnp.ndarray = None,
    L2_inner_product_matrix: jnp.ndarray = None,
    output_samples: jnp.ndarray = None,
    max_input_dimension: int = None,
    max_output_dimension: int = None,
):
    if max_input_dimension is not None:
        start = time.time()
        input_encodec_dict = estimate_input_Karhunen_Loeve_subspace(
            key=key,
            input_covariance_matrix=input_covariance_matrix,
            L2_inner_product_matrix=L2_inner_product_matrix,
            subspace_rank=max_input_dimension,
        )
        input_encodec_dict["computation_time"] = time.time() - start
    else:
        input_encodec_dict = dict()
    if max_output_dimension is not None:
        start = time.time()
        output_encodec_dict = estimate_output_Proper_Orthogonal_Decomposition_subspace(
            key=key,
            output_samples=output_samples,
            subspace_rank=max_output_dimension,
        )
        output_encodec_dict["computation_time"] = time.time() - start
    else:
        output_encodec_dict = dict()
    return {"input": input_encodec_dict, "output": output_encodec_dict}


def estimate_output_Proper_Orthogonal_Decomposition_subspace(
    key: Any,
    output_samples,
    subspace_rank: int,
) -> Dict[str, jnp.ndarray]:
    """
    Compute the rank-r Proper Orthogonal Decomposition of output random variables
    and produce subspace encoder and decoder matrices.

    This is equivalent to producing the eigenpairs of a truncated SVD of output samples.

    """
    # SVD of output samples
    # return output_encodec_dict
    N = output_samples.shape[0]
    C = jnp.einsum("nd,ne->de", output_samples, output_samples) / (N - 1)
    computed_eigvals, computed_evecs = double_pass_randomized_eigh(
        key, C, subspace_rank, subspace_rank + 15, power_iters=2
    )
    return {
        "eigenvalues": computed_eigvals,
        "decoder": computed_evecs,
        "encoder": computed_evecs,
    }


def estimate_input_Karhunen_Loeve_subspace(
    key: Any,
    input_covariance_matrix: jnp.ndarray,
    L2_inner_product_matrix: jnp.ndarray,
    subspace_rank: int,
) -> Dict[str, jnp.ndarray]:
    """ "
    Compute the rank-r Karhunen Loeve (KL) expansion truncation of a Gaussian random variable
    and produce subspace encoder and decoder matrices.

    This function is applicable for finding the dominant eigenfunctions of a random variable
    distributed according to a Gaussian prior N(0, Cov), where Cov = input_covariance_matrix.
    It assumes the usage of a matrix (often a  Mass Matrix) used to define the L2 inner product,
    abbr as M. Thus, truncation of the KL expansion amounts to solving for the dominant eigenvectors
    x^i/values v^i such that Cov x^i = lambda^i M x^i. We do this via a double-pass randomized
    generalized eigendecomposition. The decoder is the eigenvectors, while the encoder is given by
    M @ eigenvectors.

    Parameters:
      key: Random key for the randomized eigendecomposition.
      input_covariance_matrix: An 2D dense covariance matrix array.
      L2_inner_product_matrix: A 2D matrix representing the L2 inner product.
      r: Target rank for the eigendecomposition.

    Returns:
      Dict with:
        "eigenvalues": 1D array of computed eigenvalues.
        "decoder": 2D array of computed eigenvectors.
        "encoder": 2D array computed as prior_precision @ eigenvectors.
    """
    computed_eigvals, computed_evecs = double_pass_randomized_gen_eigh(
        key,
        input_covariance_matrix,
        L2_inner_product_matrix,
        subspace_rank,
        subspace_rank + 15,
        power_iters=2,
        reorthog_iter=5,
    )
    return {
        "eigenvalues": computed_eigvals,
        "decoder": computed_evecs,
        "encoder": L2_inner_product_matrix @ computed_evecs,
    }


def estimate_input_active_subspace(
    key: Any,
    J_samples: jnp.ndarray,
    noise_variance: float,
    prior_precision: jnp.ndarray,
    subspace_rank: int,
) -> Dict[str, jnp.ndarray]:
    """
    Compute the rank-r active subspace encoder and decoder matrices.

    This function is applicable for finding the active subspace of the
    parameter under a Gaussian prior and additive Gaussian noise.
    It computes the matrix A as the mean of:
      J_samples[i]^T * (1/noise_variance) * J_samples[i]
    over all samples, and then uses a double-pass randomized generalized
    eigendecomposition to obtain the eigenvalues and eigenvectors. The
    decoder is the eigenvectors, while the encoder is given by
    prior_precision @ eigenvectors.

    Parameters:
      key: Random key for the randomized eigendecomposition.
      J_samples: 3D array of Jacobian samples with shape (N, a, b), where N
                is the number of samples.
      noise_variance: a positive scalar; noise precision is
                      1/noise_variance.
      prior_precision: 2D prior precision matrix.
      r: Target rank for the eigendecomposition.

    Returns:
      Dict with:
        "eigenvalues": 1D array of computed eigenvalues.
        "decoder": 2D array of computed eigenvectors.
        "encoder": 2D array computed as prior_precision @ eigenvectors.
    """
    A = jnp.einsum("iab,iac->bc", J_samples, J_samples / J_samples.shape[0]) / noise_variance
    computed_eigvals, computed_evecs = double_pass_randomized_gen_eigh(
        key,
        A,
        prior_precision,
        subspace_rank,
        subspace_rank + 15,
        power_iters=2,
        reorthog_iter=5,
    )
    return {
        "eigenvalues": computed_eigvals,
        "decoder": computed_evecs,
        "encoder": prior_precision @ computed_evecs,
    }


def estimate_output_informative_subspace(
    key: Any,
    J_samples: jnp.ndarray,
    noise_variance: float,
    subspace_rank: int,
    prior_covariance: jnp.ndarray = None,
    prior_precision: jnp.ndarray = None,
) -> Dict[str, jnp.ndarray]:
    """
    Compute the rank-r encoder and decoder for informative data.

    This function is used for computing the informative data subspace
    under a Gaussian prior and additive Gaussian noise. It forms the
    matrix A as the mean over samples:
      A = (1/N) * sum(J_samples[i] * (prior_covariance/noise_variance) *
                      J_samples[i]^T)
    A double-pass randomized eigendecomposition is applied to obtain the
    eigenvalues and eigenvectors. The decoder scales the eigenvectors by
    sqrt(noise_variance), while the encoder scales them by the inverse of
    sqrt(noise_variance).

    Parameters:
      key: Random key for the eigendecomposition.
      J_samples: 3D array of Jacobian samples with shape (N, a, b), where N
                is the number of samples.
      noise_variance: positive scalar.
      r: Target rank for the eigendecomposition.
      prior_covariance: 2D prior covariance matrix.
      prior_precision: 2D prior precision matrix. If prior covariance is not
        provided, the prior precision will be used to perform a solve

    Returns:
      Dict with:
        "eigenvalues": 1D array of computed eigenvalues.
        "decoder": 2D array (scaled eigenvectors).
        "encoder": 2D array (inversely scaled eigenvectors).
    """
    if prior_covariance is None:
        A = average_JCJtranspose(J_samples, prior_precision)
    else:
        A = jnp.einsum(
            "iab,bc,idc->ad",
            J_samples,
            prior_covariance / noise_variance,
            J_samples / J_samples.shape[0],
        )
    computed_eigvals, computed_evecs = double_pass_randomized_eigh(
        key, A, subspace_rank, p=subspace_rank + 15, power_iters=2
    )
    noise_sigma = jnp.sqrt(noise_variance)
    return {
        "eigenvalues": computed_eigvals,
        "decoder": noise_sigma * computed_evecs,
        "encoder": computed_evecs / noise_sigma,
    }
