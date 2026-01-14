"""
Module for generating models of cortical structural connectomes.
"""

from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
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
        Spatial scale parameter for the Green's function, in millimeters. Default is `9.53`.
    k : int, optional
        Number of eigenmodes to use. Default is `108`.

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
    # Format / validate arguments
    emodes = np.asarray_chkfinite(emodes)
    evals = np.asarray_chkfinite(evals)
    r = float(r)

    if r <= 0:
        raise ValueError("Parameter `r` must be positive.")
    if emodes.ndim != 2:
        raise ValueError("`emodes` must be a 2D array of shape (n_verts, n_modes).")
    if evals.shape != (emodes.shape[1],):
        raise ValueError("`evals` must be a 1-dimensional array with length matching the number of "
                         "modes (columns) in `emodes`.")
    if k <= 0 or k > len(evals) or not isinstance(k, int):
        raise ValueError(f"Parameter `k` must be an integer in the range [1, {len(evals)}].")

    # Compute the Geometric Eigenmode Model
    denom = 1/(1 + evals[:k] * r**2)
    gem = emodes[:, :k] @ np.diag(denom) @ np.linalg.pinv(emodes[:, :k])

    # Replace diagonal and negative values with zero
    np.fill_diagonal(gem, 0)
    gem = np.maximum(gem, 0)

    # Symmetrise
    gem = (gem + gem.T) / 2

    # Normalise
    return gem / np.max(gem)