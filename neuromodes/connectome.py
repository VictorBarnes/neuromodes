"""
Module for generating models of cortical structural connectomes.
"""

import numpy as np
from numpy.typing import NDArray, ArrayLike

def model_connectome(
    emodes: ArrayLike,
    evals: ArrayLike,
    r: float = 9.53,
    k: int = 108
) -> NDArray:
    """
    Generate a vertex-wise structural connectivity matrix using the Green's function approach
    described in Normand et al., 2025.

    Parameters
    ----------
    emodes : array-like
        The eigenmodes array of shape (n_verts, n_modes), where n_verts is the number of vertices 
        and n_modes is the number of eigenmodes.
    evals : array-like
        The eigenvalues array of shape (n_modes,).
    r : float, optional
        Spatial scale parameter for the Green's function. Default is 9.53.
    k : int, optional
        Number of eigenmodes to use. Default is 108.

    Returns
    -------
    np.ndarray
        The generated vertex-wise structural connectivity matrix.

    Raises
    ------
    ValueError
        If any input parameter is invalid, such as negative or non-numeric values for  
        `r`, or if `k` is not a positive integer within the valid range.

    Notes
    -----
    If comparing this model to empirical connectomes, consider thresholding the generated connectome
    to match the density of the empirical data.
    """
    emodes = np.asarray(emodes)
    evals = np.asarray(evals)
    r = float(r)

    if r <= 0:
        raise ValueError("Parameter `r` must be positive.")
    if emodes.ndim != 2:
        raise ValueError("`emodes` must be a 2D array (vertices x modes).")
    if len(evals) != emodes.shape[1]:
        raise ValueError("Length of `evals` must match the number of modes (columns) in `emodes`.")
    if k <= 0 or k > len(evals) or not isinstance(k, int):
        raise ValueError(f"Parameter `k` must be an integer in the range [1, {len(evals)}].")

    # Select the first k eigenmodes and eigenvalues
    k_evals = evals[:k]
    k_emodes = emodes[:, :k]

    # Compute structural connectivity
    denom = 1/(1 + k_evals * r**2)

    connectome = k_emodes @ np.diag(denom) @ np.linalg.pinv(k_emodes)

    # Replace diagonal and negative values with zero
    np.fill_diagonal(connectome, 0)
    connectome[connectome < 0] = 0
    
    # Symmetrize
    connectome = (connectome + connectome.T) / 2

    # Normalize
    connectome /= np.max(connectome)

    return connectome