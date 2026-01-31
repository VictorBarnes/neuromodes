"""
Spatial null models via eigenmode rotation.

This module provides functions for generating null brain maps that preserve spatial
autocorrelation structure through random rotation of geometric eigenmodes.
"""

import numpy as np
from numpy.typing import NDArray, ArrayLike
from scipy.stats import special_ortho_group
from joblib import Parallel, delayed
from typing import Union, Optional

from neuromodes.basis import decompose
from neuromodes.eigen import get_eigengroups

match_eigenstrapping = True

def generate_nulls(
    data: ArrayLike,
    emodes: ArrayLike,
    evals: ArrayLike,
    n_nulls: int = 1000,
    mask: Optional[ArrayLike] = None,
    method: str = 'project',
    mass: Union[ArrayLike, None] = None,
    resample: bool = True,
    randomize: bool = False,
    residual: Optional[str] = None,
    normalize: bool = True,
    n_jobs: int = -1,
    seed: Optional[int] = None,
) -> NDArray:
    """
    Generate surface-based null maps via eigenmode rotation.
    
    This function generates spatial null models that preserve the spatial autocorrelation
    structure of brain maps through random rotation of geometric eigenmodes. The method
    works by rotating eigenmodes within eigengroups (sets of modes with similar eigenvalues),
    then reconstructing null maps using the original decomposition coefficients.
    
    Parameters
    ----------
    data : array-like of shape (n_verts,)
        Empirical brain map to generate nulls from.
    emodes : array-like of shape (n_verts, n_modes)
        Eigenmodes from surface eigendecomposition (excluding the constant mode).
    evals : array-like of shape (n_modes,)
        Eigenvalues corresponding to eigenmodes (excluding the eval corresponding to the 
        constant mode).
    n : int, optional
        Number of null maps to generate. Default is 1000.
    mask : array-like of shape (n_verts,) or None, optional
        Boolean mask indicating valid vertices (True) vs medial wall or invalid regions (False).
        If None, all vertices are used. Default is None.
    mass : array-like or None, optional
        Mass matrix for mass-orthonormal eigenmodes. If eigenmodes are orthonormal in
        Euclidean space, leave as None. Default is None.
    method : {'project', 'regress'}, optional
        Method for eigenmode decomposition. Use 'project' for orthonormal bases,
        'regress' for non-orthonormal. Default is 'project'.
    resample : bool, optional
        Whether to resample null values from the empirical map to preserve the
        empirical distribution. Default is True.
    randomize : bool, optional
        Whether to shuffle decomposition coefficients within eigengroups. This increases
        randomization but reduces spatial autocorrelation similarity to empirical data. 
        Default is False.
    residual : {'add', 'permute', None}, optional
        How to handle reconstruction residuals. 'add' preserves original residuals,
        'permute' adds shuffled residuals, None excludes residuals. Default is None.
    normalize : bool, optional
        Whether to apply spheroid/ellipsoid transforms before/after rotation. This
        normalizes eigenvalue scaling for uniform rotation. Default is True.
    n_jobs : int, optional
        Number of parallel workers. -1 uses all available cores. Default is -1.
    seed : int or None, optional
        Random seed for reproducibility. If provided, generates deterministic nulls
        regardless of n_jobs. Default is None.
    
    Returns
    -------
    ndarray of shape (n, n_verts)
        Generated null maps. If mask is provided, returns full vertex arrays with
        NaN at masked locations.
    
    Raises
    ------
    ValueError
        If residual is not None, 'add', or 'permute'.
    ValueError
        If number of modes is less than 3.
    ValueError
        If emodes columns are not orthonormal when method='project' and check_ortho=True.
        
    References
    ----------
    ..  [1] Koussis, N. C., et al. (2024). Generation of surrogate brain maps preserving 
        spatial autocorrelation through random rotation of geometric eigenmodes. 
        Imaging Neuroscience. https://doi.org/10.1162/IMAG.a.71
    """
    # Validate inputs
    emodes = np.asarray(emodes) # chkfinite in decompose
    evals = np.asarray_chkfinite(evals)
    data = np.asarray_chkfinite(data)

    # Check whether emodes have been truncated at first non-constant mode
    if np.std(emodes[:, 0]) < 1e-7 or evals[0] < 1e-7:
        raise ValueError("Eigenmodes appear to include the constant mode; however, null "
                         "generation requires modes starting from the first non-constant "
                         "mode. Please exclude the constant mode and corresponding eval.")
    
    if residual is not None and residual not in ('add', 'permute'):
        raise ValueError(f"Invalid residual method '{residual}'; must be 'add', 'permute', or None.")
    
    # Apply mask if provided
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        masked_data = data[mask]
        masked_emodes = emodes[mask, :]
    else:
        masked_data = data
        masked_emodes = emodes
    
    # Compute decomposition coefficients and residuals
    if match_eigenstrapping:
        coeffs = np.linalg.solve(masked_emodes.T @ masked_emodes, masked_emodes.T @ masked_data)
    else:
        coeffs = decompose(masked_data, masked_emodes, method=method, mass=mass)
    coeffs = coeffs.squeeze()
    
    reconstructed = coeffs @ masked_emodes.T
    residuals = masked_data - reconstructed
    
    # Identify eigengroups
    groups = get_eigengroups(masked_emodes)
    
    if match_eigenstrapping:
        # Match eigenstrapping exactly: set global seed AND create RandomState
        if seed is not None:
            np.random.seed(seed)
        rs = _check_random_state(seed)
        null_seeds = rs.randint(np.iinfo(np.int32).max, size=n_nulls)
    else:
        # Simplified approach for non-eigenstrapping mode
        if seed is not None:
            rs = np.random.RandomState(seed)
            null_seeds = rs.randint(np.iinfo(np.int32).max, size=n_nulls)
        else:
            null_seeds = np.random.randint(np.iinfo(np.int32).max, size=n_nulls)
    
    # Generate nulls in parallel
    nulls = Parallel(n_jobs=n_jobs)(
        delayed(_generate_single_null)(
            masked_data, masked_emodes, evals, groups, coeffs, residuals,
            mask, resample, randomize, residual, normalize, seed_i
        )
        for seed_i in null_seeds
    )
    
    if match_eigenstrapping:
        result = np.row_stack(nulls)
        result = np.asarray(result.squeeze()).T
    else:
        result = np.asarray(nulls).T
        if n_nulls == 1:
            result = result.squeeze()
    return result

def _generate_single_null(
    data: NDArray,
    emodes: NDArray,
    evals: NDArray,
    groups: list[NDArray],
    coeffs: NDArray,
    residuals: NDArray,
    mask: Optional[NDArray],
    resample: bool,
    randomize: bool,
    residual: Optional[str],
    normalize: bool,
    seed: Optional[int],
) -> NDArray:
    """
    Generate a single null map.
    
    Parameters
    ----------
    data : ndarray of shape (n_verts_masked,)
        Original data (masked).
    emodes : ndarray of shape (n_verts_masked, n_modes)
        Eigenmodes (masked).
    evals : ndarray of shape (n_modes,)
        Eigenvalues.
    groups : list of ndarray
        Eigengroup indices.
    coeffs : ndarray of shape (n_modes,)
        Decomposition coefficients.
    residuals : ndarray of shape (n_verts_masked,)
        Residuals from reconstruction.
    mask : ndarray of shape (n_verts,) or None
        Boolean mask indicating valid vertices.
    resample : bool
        Whether to resample values from original data.
    randomize : bool
        Whether to shuffle coefficients within eigengroups.
    residual : {'add', 'permute', None}
        How to handle residuals.
    normalize : bool
        Whether to apply spheroid/ellipsoid transforms.
    seed : int or None
        Random seed for this null. 
    
    Returns
    -------
    ndarray of shape (n_verts,)
        Generated null map with full vertex indexing.
    """
    # Create RandomState for non-rotation operations
    if match_eigenstrapping:
        rng = _check_random_state(seed)
    else:
        rng = np.random.RandomState(seed)
    
    # Initialize rotated modes
    rotated_emodes = np.zeros_like(emodes)
    
    # Rotate each eigengroup
    for group_idx in groups:
        group_emodes = emodes[:, group_idx]
        group_evals = evals[group_idx]
        
        # Transform to spheroid, rotate, transform back to ellipsoid
        if normalize:
            spheroid_modes = group_emodes / np.sqrt(group_evals)
        else:
            spheroid_modes = group_emodes
        
        if match_eigenstrapping:
            rotated_spheroid = _rotate_emodes(spheroid_modes, seed=None)
        else:
            rotated_spheroid = _rotate_emodes(spheroid_modes, seed=seed)
        
        if normalize:
            rotated_spheroid = rotated_spheroid * np.sqrt(group_evals)
        
        rotated_emodes[:, group_idx] = rotated_spheroid
    
    # Optionally shuffle coefficients within eigengroups
    null_coeffs = coeffs.copy()
    if randomize:
        for group_idx in groups:
            null_coeffs[group_idx] = rng.permutation(null_coeffs[group_idx])
    
    # Reconstruct null
    null_map = null_coeffs @ rotated_emodes.T
    
    # Handle residuals
    if residual == 'permute':
        null_map += rng.permutation(residuals)
    elif residual == 'add':
        null_map += residuals
    
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
    
    # Return with proper shape
    if mask is None:
        return null_map
    else:
        # Create full array with NaN where mask is False
        n_verts = mask.shape[0]
        output_null = np.full(n_verts, np.nan)
        output_null[mask] = null_map
        return output_null

def _rotate_emodes(emodes: NDArray, seed: Optional[int] = None) -> NDArray:
    """
    Apply random orthogonal rotation to eigenmodes.
    
    Uses scipy's special_ortho_group to generate random rotation matrices from the
    special orthogonal group SO(n), which preserves orthonormality.
    
    Parameters
    ----------
    emodes : ndarray of shape (n_vertices, n_modes)
        Eigenmodes to rotate.
    seed : int or None, optional
        Random seed for reproducibility. If None, uses global numpy random state.
        Default is None.
    
    Returns
    -------
    ndarray of shape (n_vertices, n_modes)
        Rotated eigenmodes.
    """
    n_modes = emodes.shape[1]

    if match_eigenstrapping:
        rs = _check_random_state(seed)
        rotation_matrix = special_ortho_group.rvs(dim=n_modes, random_state=rs)
    else:
        rotation_matrix = special_ortho_group.rvs(dim=n_modes, random_state=seed)
    
    return emodes @ rotation_matrix

def _check_random_state(seed):
    """
    Turn seed into a np.random.RandomState instance.

    Parameters
    ----------
    seed : None | int | np.random.RandomState

    Returns
    -------
    np.random.RandomState
    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, (int, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError(f"{seed} cannot be used to seed a numpy.random.RandomState instance")
