"""
Spatial null models via eigenmode rotation.

This module provides functions for generating null brain maps that preserve spatial
autocorrelation structure through random rotation of geometric eigenmodes.
"""
from __future__ import annotations
from typing import Union, TYPE_CHECKING
from warnings import warn
from joblib import Parallel, delayed
import numpy as np
from scipy.stats import special_ortho_group
from neuromodes.basis import decompose
from neuromodes.eigen import get_eigengroup_inds

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray

def eigenstrap_old(
    data: ArrayLike,
    emodes: ArrayLike,
    evals: ArrayLike,
    n_nulls: int = 1000,
    method: str = 'project',
    mass: Union[ArrayLike, None] = None,
    resample: Union[str, None] = None,
    randomize: bool = False,
    residual: Union[str, None] = None,
    n_jobs: int = -1,
    seed: Union[int, None] = None,
    check_ortho: bool = True,
) -> NDArray:
    """
    Generate null maps via eigenstrapping [1].
    
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
        `emodes` must already be masked (see Notes). This function rotates modes 
        within eigengroups. If the number of eigenmodes is not a perfect square (i.e. 
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
        How to resample values from original data. Options are 'match' to match the sorted 
        distribution of the original data, 'zscore' to z-score and rescale to original mean
        and std, 'mean' to preserve the mean, and 'range' to preserve the minimum and 
        maximum. Default is None for no resampling.
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
    ValueError
        If `resample` is not one of None, 'match', 'zscore', 'mean', or 'range'.

    Notes
    -----
    This function does not apply any vertex masking. If working on a masked surface (e.g., 
    excluding the medial wall), the user must pass `data`, `emodes`, and `mass` (if used) 
    already restricted to the desired masked vertices.

    This function uses the constant mode (first column of `emodes`) and correpsonding 
    eigenvalue to generates mean-preserving nulls. The constant mode is not rotated 
    and the corresponding eigenvalue is set to 1 to avoid division by zero when 
    normalizing the modes.

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
    # Format / validate arguments
    emodes = np.asarray(emodes)  # chkfinite in decompose
    evals = np.asarray_chkfinite(evals) 
    data = np.asarray(data)

    if emodes.ndim != 2 or emodes.shape[0] < emodes.shape[1]:
        raise ValueError("`emodes` must have shape (n_verts, n_modes), where n_verts ≥ n_modes.")
    n_modes = emodes.shape[1]
    if evals.shape != (n_modes,):
        raise ValueError("`evals` must have shape (n_modes,), matching the number of columns in "
                         f"`emodes` ({n_modes}).")
    
    if residual not in (None, 'add', 'permute'):
        raise ValueError(f"Invalid residual method '{residual}'; must be 'add', 'permute', or None.")
    
    if resample not in (None, 'match', 'zscore', 'mean', 'range'):
        raise ValueError(f"Invalid resampling method '{resample}'; must be 'match', 'zscore', "
                         "'mean', 'range', or None.")
    
    # Compute decomposition coefficients and residuals
    coeffs = decompose(data, emodes, method=method, mass=mass, check_ortho=check_ortho).squeeze()
    residual_data = data - emodes @ coeffs if residual is not None else None
    
    # Simplified approach for non-eigenstrapping mode
    rng = np.random.RandomState(seed)
    null_seeds = rng.randint(np.iinfo(np.int32).max, size=n_nulls)

    # Identify eigengroups for the modes that will be rotated
    groups = get_eigengroup_inds(n_modes)
    # If `n_modes` is not a perfect square then exclude the last group
    if int(np.sqrt(n_modes))**2 != n_modes:
        warn(f"Number of modes ({n_modes}) is not a perfect square. Last eigengroup containing "
                      f"{len(groups[-1])} modes will be excluded.")
        groups = groups[:-1]

    # Transform each eigengroup from ellipsoid to spheroid
    norm_emodes = emodes.copy()
    norm_emodes[:, 1:] /= np.sqrt(evals[1:])  # Don't transform constant mode to preserve mean
    
    # Generate nulls in parallel
    nulls = Parallel(n_jobs=n_jobs)(
        delayed(_eigenstrap_single)(
            data=data,
            emodes=norm_emodes,
            evals=evals,
            groups=groups,
            coeffs=coeffs,
            residual_data=residual_data,
            resample=resample,
            randomize=randomize,
            residual=residual,
            seed=seed
        )
        for seed in null_seeds
    )

    return np.stack(nulls, axis=1)

def _eigenstrap_single(
    data: NDArray,
    emodes: NDArray,
    evals: NDArray,
    groups: list[NDArray],
    coeffs: NDArray,
    residual_data: Union[NDArray, None],
    resample: Union[str, None],
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
        Decomposition coefficients of shape (n_modes,).
    residual_data : array-like
        Residuals from reconstruction of shape (n_verts,).
    resample : bool
        How to resample values from original data.
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

    # Keep constant mode to preserve mean
    rotated_emodes[:, 0] = emodes[:, 0]
    
    # Rotate each eigengroup
    for mode_inds in groups[1:]:        
        rotated_emodes[:, mode_inds] = _rotate_emodes(emodes[:, mode_inds], random_state=rng)
        # TODO: profile group-wise multiplication vs. generating a large sparse blockdiagonal matrix 
        # and then multiplying/reconstructing all at once (perhaps including the coefficients too)
    
    # Transform from spheroid back to ellipsoid
    rotated_emodes[:, 1:] *= np.sqrt(evals[1:])

    # Optionally shuffle coefficients within eigengroups
    if randomize:
        coeffs = coeffs.copy()
        for group_idx in groups:
            coeffs[group_idx] = rng.permutation(coeffs[group_idx])
    
    # Reconstruct null
    null_map = rotated_emodes @ coeffs
    
    # Handle residuals
    if residual == 'add':
        null_map += residual_data
    elif residual == 'permute':
        null_map += rng.permutation(residual_data)

    # Resample values from original data
    if resample == 'match':
        sorted_data = np.sort(data)
        sorted_indices = np.argsort(null_map)
        null_map[sorted_indices] = sorted_data
    elif resample == 'zscore':
        null_map = (null_map - null_map.mean()) / null_map.std() * data.std() + data.mean()
    elif resample == 'mean':
        null_map = (null_map - null_map.mean()) + data.mean()
    elif resample == 'range':
        # Force match the minimum and maximum to original data range
        scale_factor = (data.max() - data.min()) / (null_map.max() - null_map.min())
        offset = data.min() - scale_factor * null_map.min()
        null_map = null_map * scale_factor + offset
    
    return null_map

def _rotate_emodes(
    emodes: NDArray,
    random_state: Union[np.random.RandomState, None]
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
    
    # Rotate modes
    return emodes @ rotation_matrix

def _eigenstrap_single_new(
    norm_emodes: NDArray,
    sqrt_evals: NDArray,
    groups: list[NDArray],
    coeffs: NDArray,
    randomize: bool,
    seed: Union[int, None],
):
    # Initialize RNG for this null using the provided seed to ensure reproducibility
    rng = np.random.RandomState(seed)

    # Initialize rotated modes with constant mode to preserve mean
    rotated_emodes = np.empty_like(norm_emodes)
    rotated_emodes[:, 0] = norm_emodes[:, 0]
    
    # Rotate each eigengroup
    for mode_inds in groups[1:]:
        rotation_matrix = special_ortho_group.rvs(dim=len(mode_inds), random_state=rng)
        rotated_emodes[:, mode_inds] = norm_emodes[:, mode_inds] @ rotation_matrix

    # Transform from spheroid back to ellipsoid
    rotated_emodes *= sqrt_evals

    # Optionally shuffle coefficients within eigengroups
    if randomize:
        coeffs = coeffs.copy()
        for group_idx in groups:
            coeffs[group_idx] = rng.permutation(coeffs[group_idx])
    
    # Reconstruct null
    null_map = rotated_emodes @ coeffs

    # Reconstruct null
    return null_map

def eigenstrap(
    data: ArrayLike,
    emodes: ArrayLike,
    evals: ArrayLike,
    n_nulls: int = 1000,
    method: str = 'project',
    mass: Union[ArrayLike, None] = None,
    resample: Union[str, None] = None,
    randomize: bool = False,
    residual: Union[str, None] = None,
    n_jobs: int = -1,
    seed: Union[int, None] = None,
    check_ortho: bool = True,
) -> NDArray:

    # Format / validate arguments
    emodes = np.asarray(emodes)  # chkfinite in decompose
    evals = np.asarray_chkfinite(evals) 
    data = np.asarray(data)

    if emodes.ndim != 2 or emodes.shape[0] < emodes.shape[1]:
        raise ValueError("`emodes` must have shape (n_verts, n_modes), where n_verts ≥ n_modes.")
    n_modes = emodes.shape[1]
    if evals.shape != (n_modes,):
        raise ValueError("`evals` must have shape (n_modes,), matching the number of columns in "
                         f"`emodes` ({n_modes}).")
    
    if residual not in (None, 'add', 'permute'):
        raise ValueError(f"Invalid residual method '{residual}'; must be 'add', 'permute', or None.")
    
    if resample not in (None, 'match', 'zscore', 'mean', 'range'):
        raise ValueError(f"Invalid resampling method '{resample}'; must be 'match', 'zscore', "
                         "'mean', 'range', or None.")
    
    # Compute decomposition coefficients and residuals
    coeffs = decompose(data, emodes, method=method, mass=mass, check_ortho=check_ortho).squeeze()
    residual_data = (data - emodes @ coeffs)[:, np.newaxis] if residual is not None else None
    
    # Simplified approach for non-eigenstrapping mode
    rng = np.random.RandomState(seed)
    null_seeds = rng.randint(np.iinfo(np.int32).max, size=n_nulls)

    # Identify eigengroups for the modes that will be rotated
    groups = get_eigengroup_inds(n_modes)
    # If `n_modes` is not a perfect square then exclude the last group
    if int(np.sqrt(n_modes))**2 != n_modes:
        warn(f"Number of modes ({n_modes}) is not a perfect square. Last eigengroup containing "
                      f"{len(groups[-1])} modes will be excluded.")
        groups = groups[:-1]

    # Transform each eigengroup from ellipsoid to spheroid
    sqrt_evals = np.sqrt(evals)
    sqrt_evals[0] = 1  # Set constant mode eigenvalue to 1 to avoid division by zero when normalizing
    norm_emodes = emodes / sqrt_evals
    
    # Generate nulls in parallel
    nulls = Parallel(n_jobs=n_jobs)(
        delayed(_eigenstrap_single_new)(
            norm_emodes=norm_emodes,
            sqrt_evals=sqrt_evals,
            groups=groups,
            coeffs=coeffs,
            randomize=randomize,
            seed=seed
        )
        for seed in null_seeds
    )
    nulls = np.stack(nulls, axis=1)

    # Handle residuals
    if residual == 'add':
        nulls += residual_data
    elif residual == 'permute':
        nulls += rng.permutation(residual_data)

    # Resample values from original data
    if resample == 'match':
        sorted_data = np.sort(data)[:, np.newaxis]
        sorted_indices = np.argsort(nulls, axis=0)
        nulls = np.take_along_axis(sorted_data, sorted_indices, axis=0)
    elif resample == 'zscore':
        nulls_z = (nulls - nulls.mean(axis=0)) / nulls.std(axis=0)
        nulls = nulls_z * data.std(axis=0) + data.mean(axis=0)
    elif resample == 'mean':
        nulls = (nulls - nulls.mean(axis=0)) + data.mean(axis=0)
    elif resample == 'range':
        # Force match the minimum and maximum to original data range
        data_mins = data.min(axis=0)
        null_mins = nulls.min(axis=0)
        scale_factor = (data.max(axis=0) - data.min(axis=0)) / (nulls.max(axis=0) - null_mins)
        offset = data_mins - scale_factor * null_mins
        nulls = nulls * scale_factor + offset

    return nulls