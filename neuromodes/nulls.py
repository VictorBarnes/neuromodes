"""
Spatial null models via eigenmode rotation.

This module provides functions for generating null brain maps that preserve spatial
autocorrelation structure through random rotation of geometric eigenmodes.
"""

import warnings
import numpy as np
from numpy.typing import NDArray, ArrayLike
from scipy.stats import special_ortho_group
from joblib import Parallel, delayed
from typing import Union

from neuromodes.basis import decompose
from neuromodes.eigen import get_eigengroup_inds


def eigenstrap(
    data: ArrayLike,
    emodes: ArrayLike,
    evals: ArrayLike,
    n_nulls: int = 1000,
    method: str = 'project',
    mass: Union[ArrayLike, None] = None,
    resample: bool = True,
    randomize: bool = False,
    residual: Union[str, None] = None,
    n_jobs: int = -1,
    seed: Union[int, None] = None,
    check_ortho: bool = True,
) -> NDArray:
    """
    Generate surface-based null maps via eigenstrapping [1].
    
    This function generates spatial null models that preserve the spatial autocorrelation
    structure of brain maps through random rotation of geometric eigenmodes. The method
    works by rotating eigenmodes within eigengroups (sets of modes with similar eigenvalues),
    then reconstructing null maps using the original decomposition coefficients.
    
    Parameters
    ----------
    data : array-like
        Empirical brain map of shape (n_verts,) to generate nulls from. If working on a 
        masked surface, `data` must already be masked (see Notes).
    emodes : array-like of shape (n_verts, n_modes)
        The eigenmodes array of shape (n_verts, n_modes). If working on a masked surface,
        `emodes` must already be masked (see Notes). The first column is assumed to be the 
        constant mode and will be removed from the computation. Note that this function 
        rotates modes within eigengroups. If `n_modes` is not a perfect square (i.e. 
        number of modes doesn't allow for complete eigengroups), then the last incomplete 
        eigengroup will be excluded.
    evals : array-like
        The eigenvalues array of shape (n_modes,).
    n_nulls : int, optional
        Number of null maps to generate. Default is 1000.
    method : str, optional
        The method used for the decomposition, either `'project'` to project data into a
        mass-orthonormal space or `'regress'` for least-squares fitting. Default is `'project'`.
    mass : array-like, optional
        The mass matrix of shape (n_verts, n_verts) used for the decomposition when method is
        `'project'`. If working on a masked surface, `mass` must already be masked (see Notes). 
        Default is None.
    resample : bool, optional
        Whether to resample null values from the empirical map to preserve the
        empirical distribution. Default is True.
    randomize : bool, optional
        Whether to shuffle decomposition coefficients within eigengroups. This increases
        randomization but reduces spatial autocorrelation similarity to empirical data. 
        Default is False.
    residual : str, optional
        How to handle reconstruction residuals. Either None to exclude residuals, `'add'` 
        to add original residuals, or `'permute'` to adds shuffled residuals. Default is 
        None.
    n_jobs : int, optional
        Number of parallel workers. -1 uses all available cores. Default is -1.
    seed : int, optional
        Random seed for reproducibility. If provided, generates deterministic nulls (see 
        Notes). Default is None.
    check_ortho : bool, optional
        Whether to check if `emodes` are mass-orthonormal before using the `'project'` method
        in `neuromodes.basis.decompose`. Default is `True`.
    
    Returns
    -------
    ndarray of shape (n_verts, n_nulls)
        Generated null maps.
    
    Raises
    ------
    ValueError
        If `emodes` is not 2D or has more columns than rows.
    ValueError
        If `evals` length doesn't match number of columns in `emodes`.
    ValueError
        If `residual` is not one of None, 'add', or 'permute'.

    Notes
    -----
    This function does not apply any vertex masking. If working on a masked surface (e.g., 
    excluding the medial wall), the user must pass `data`, `emodes`, and `mass` (if used) 
    already restricted to the desired masked vertices.

    Seeding is handled in two stages to ensure reproducibility when parallelising. First 
    `seed` is used to initialize a master random number generator (RNG) that generates an 
    independent integer seed for each null map. Then each null uses its allocated integer 
    to generate its own RNG to use for all rotations/permutations of that null. This 
    ensures that each null is independent of the number of parallel jobs used.
    
    References
    ----------
    ..  [1] Koussis, N. C., et al. (2024). Generation of surrogate brain maps preserving 
        spatial autocorrelation through random rotation of geometric eigenmodes. 
        Imaging Neuroscience. https://doi.org/10.1162/IMAG.a.71
    """
    # Validate inputs
    emodes = np.asarray(emodes)  # chkfinite in decompose
    evals = np.asarray_chkfinite(evals) 
    data = np.asarray_chkfinite(data)

    if emodes.ndim != 2 or emodes.shape[0] < emodes.shape[1]:
        raise ValueError("`emodes` must have shape (n_verts, n_modes), where n_verts ≥ n_modes.")
    _, n_modes = emodes.shape
    if evals.shape != (n_modes,):
        raise ValueError("`evals` must have shape (n_modes,), matching the number of columns in "
                         f"`emodes` ({n_modes}).")
    
    if residual is not None and residual not in ('add', 'permute'):
        raise ValueError(f"Invalid residual method '{residual}'; must be 'add', 'permute', or None.")
    
    # Compute decomposition coefficients and residuals
    coeffs = decompose(data, emodes[:, 1:], method=method, mass=mass, check_ortho=check_ortho)
    coeffs = coeffs.squeeze()
    
    reconstructed = emodes[:, 1:] @ coeffs
    residuals = data - reconstructed
    
    # Simplified approach for non-eigenstrapping mode
    rng = np.random.RandomState(seed)
    null_seeds = rng.randint(np.iinfo(np.int32).max, size=n_nulls)

    # Identify eigengroups for the modes that will be rotated
    groups = get_eigengroup_inds(n_modes)[1:]    # Exclude constant mode group
    # If `n_modes` is not a perfect square then exclude the last group
    if int(np.sqrt(n_modes))**2 != n_modes:
        warnings.warn(f"Number of modes ({n_modes}) is not a perfect square. Last "
                      f"eigengroup containing {len(groups[-1])} modes will be excluded.")
        groups = groups[:-1]
    
    # Generate nulls in parallel
    nulls = Parallel(n_jobs=n_jobs)(
        delayed(_eigenstrap_single)(
            data=data,
            emodes=emodes,
            evals=evals,
            groups=groups,
            coeffs=coeffs,
            residual_data=residuals,
            resample=resample,
            randomize=randomize,
            residual=residual,
            seed=seed_i
        )
        for seed_i in null_seeds
    )
    
    result = np.asarray(nulls).T

    return result

def _eigenstrap_single(
    data: NDArray,
    emodes: NDArray,
    evals: NDArray,
    groups: list[NDArray],
    coeffs: NDArray,
    residual_data: NDArray,
    resample: bool,
    randomize: bool,
    residual: Union[str, None],
    seed: Union[int, None],
) -> NDArray:
    """
    Generate a single null map.
    
    Parameters
    ----------
    data : array-like
        Empirical brain map of shape (n_verts,) to generate nulls from. 
    emodes : array-like
        The eigenmodes array of shape (n_verts, n_modes), where n_verts is the number of 
        vertices and n_modes is the number of eigenmodes. 
    evals : array-like
        The eigenvalues array of shape (n_modes,).
    groups : list of array-like
        Eigengroup indices.
    coeffs : array-like
        Decomposition coefficients of shape (n_modes-1,) excluding the constant mode.
    residual_data : array-like
        Residuals from reconstruction of shape (n_verts,).
    resample : bool
        Whether to resample values from original data.
    randomize : bool
        Whether to shuffle coefficients within eigengroups.
    residual : str
        How to handle residuals.
    seed : int
        Random seed for this null to generate its own RNG. 
    
    Returns
    -------
    ndarray of shape (n_verts,)
        Generated null map with full vertex indexing.
    """
    # Initialize RNG for this null using the provided seed to ensure reproducibility
    rng = np.random.RandomState(seed)
    
    # Initialize rotated modes
    rotated_emodes = np.zeros_like(emodes)
    
    # Rotate each eigengroup
    for group_idx in groups:
        group_modes = emodes[:, group_idx]
        group_evals = evals[group_idx]
        
        # Transform to spheroid, rotate, transform back to ellipsoid
        normalised_modes = group_modes / np.sqrt(group_evals)
        rotated_modes = _rotate_emodes(normalised_modes, random_state=rng)
        rotated_emodes[:, group_idx] = rotated_modes * np.sqrt(group_evals)
    rotated_emodes = rotated_emodes[:, 1:]  # Exclude constant mode from reconstruction
    
    # Optionally shuffle coefficients within eigengroups
    null_coeffs = coeffs.copy()
    if randomize:
        for group_idx in groups:
            adjusted_idx = np.array(group_idx) - 1  # Adjust for excluded constant mode
            null_coeffs[adjusted_idx] = rng.permutation(null_coeffs[adjusted_idx])
    
    # Reconstruct null
    null_map = rotated_emodes @ null_coeffs
    
    # Handle residuals
    if residual == 'add':
        null_map += residual_data
    elif residual == 'permute':
        null_map += rng.permutation(residual_data)
    
    # Resample values from original data
    if resample:
        sorted_data = np.sort(data)
        sorted_indices = np.argsort(null_map)
        null_map[sorted_indices] = sorted_data
    else:
        # Force match the minimum and maximum to original data range
        scale_factor = (np.max(data) - np.min(data)) / (np.max(null_map) - np.min(null_map))
        offset = np.min(data) - scale_factor * np.min(null_map)
        null_map = null_map * scale_factor + offset
    
    return null_map

def _rotate_emodes(
    emodes: NDArray,
    random_state: Union[int, np.random.RandomState, None]
    ) -> NDArray:
    """
    Apply random orthogonal rotation to eigenmodes.
    
    Uses scipy's special_ortho_group to generate random rotation matrices from the
    special orthogonal group SO(n), which preserves orthonormality.
    
    Parameters
    ----------
    emodes : ndarray
        Eigenmodes to rotate of shape (n_vertices, n_modes).
    random_state : int or numpy.random.RandomState or None
        Random seed or RNG for reproducibility. If int, will be used to create a new RNG.
        If None, will use the global RNG.
    
    Returns
    -------
    ndarray of shape (n_vertices, n_modes)
        Rotated eigenmodes.
    """
    n_modes = emodes.shape[1]

    # Generate a random rotation matrix from the special orthogonal group SO(n_modes)
    rotation_matrix = special_ortho_group.rvs(dim=n_modes, random_state=random_state)
    
    return emodes @ rotation_matrix

