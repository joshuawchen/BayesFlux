from typing import Any, Dict

import jax.numpy as jnp

from randlax import double_pass_randomized_gen_eigh, double_pass_randomized_eigh

def input_active_subspace_encoder_decoder(
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
      J_samples: 3D array of Jacobian samples with shape (N, a, b), where N is
                 the number of samples.
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

    #TODO: expose power iterations, reorthog_iter to the interface
    A = jnp.einsum(
        'iab,iac->bc',
        J_samples / noise_variance,
        J_samples / J_samples.shape[0]
    )
    computed_eigvals, computed_evecs = \
        double_pass_randomized_gen_eigh(
            key, A, prior_precision, subspace_rank, subspace_rank + 10, power_iters=2, 
            reorthog_iter=5
        )
    return {
        "eigenvalues": computed_eigvals,
        "decoder": computed_evecs,
        "encoder": prior_precision @ computed_evecs,
    }


def output_informative_subspace_encoder_decoder(
    key: Any,
    J_samples: jnp.ndarray,
    prior_covariance: jnp.ndarray,
    noise_variance: float,
    r: int,
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
      J_samples: 3D array of Jacobian samples with shape (N, a, b), where N is
                 the number of samples.
      prior_covariance: 2D prior covariance matrix.
      noise_variance: positive scalar.
      r: Target rank for the eigendecomposition.

    Returns:
      Dict with:
        "eigenvalues": 1D array of computed eigenvalues.
        "decoder": 2D array (scaled eigenvectors).
        "encoder": 2D array (inversely scaled eigenvectors).
    """
    #TODO: expose p, power iterations to the interface, allow passing of prior_precision and solving in-line via batch linear algebra
    A = jnp.einsum(
        'iab,bc,idc->ad',
        J_samples,
        prior_covariance / noise_variance,
        J_samples / J_samples.shape[0]
    )
    computed_eigvals, computed_evecs = \
        double_pass_randomized_eigh(
            key, A, r, p=r + 10, power_iters=1
        )
    noise_sigma = jnp.sqrt(noise_variance)
    return {
        "eigenvalues": computed_eigvals,
        "decoder": noise_sigma * computed_evecs,
        "encoder": computed_evecs / noise_sigma,
    }