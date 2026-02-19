"""
Spatial null models via eigenmode rotation.

This module provides functions for generating null brain maps that preserve spatial
autocorrelation structure through random rotation of geometric eigenmodes.
"""
from __future__ import annotations
from typing import cast, Union, TYPE_CHECKING
from warnings import warn
from joblib import Parallel, delayed
import numpy as np
from scipy.stats import special_ortho_group
from neuromodes.basis import decompose
from neuromodes.eigen import get_eigengroup_inds

if TYPE_CHECKING:
    from numpy import floating, integer
    from numpy.typing import ArrayLike, NDArray
    from scipy.sparse import spmatrix

def eigenstrap_gmh(
    data: ArrayLike,
    emodes: ArrayLike,
    evals: ArrayLike,
    n_nulls: int = 1000,
    method: str = 'project',
    mass: Union[spmatrix, ArrayLike, None] = None,
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
        How to resample values from original data. Options are 'exact' to match the sorted 
        distribution of the original data, 'affine' to match the original mean and standard
        deviation, 'mean' to match the mean, and 'range' to match the minimum and 
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
        If `resample` is not one of None, 'exact', 'affine', 'mean', or 'range'.

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
    
    if resample is not None and resample not in ('exact', 'affine', 'mean', 'range'):
        raise ValueError(f"Invalid resampling method '{resample}'; must be 'exact', 'affine', 'mean', "
                         f"'range', or None.")
    
    # Compute decomposition coefficients and residuals
    coeffs = decompose(data, emodes, method=method, mass=mass, check_ortho=check_ortho).squeeze()
    residual_data = data - emodes @ coeffs if residual is not None else None
    
    # Get individual seeds to be used for individual null generations
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
        delayed(_eigenstrap_single_gmh)(
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
    nulls = cast(list[NDArray], nulls)  # placate Pyright
    return np.stack(nulls, axis=1)

def _eigenstrap_single_gmh(
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

    # Square root the eigenvalues and return as list (one array for each group)
    get_sqrt_evals = lambda evals: np.sqrt(evals) if evals.size != 1 else np.array([1])
    sqrt_evals = [get_sqrt_evals(evals[group]) for group in groups]

    # Get rotation matrices for each group (need special case for groups with exactly one mode)
    get_rotation_matrix = lambda sz: special_ortho_group.rvs(dim=sz, random_state=rng) if sz != 1 else np.array([[1]])
    rots = [get_rotation_matrix(group.size) for group in groups]

    # Optionally shuffle coefficients within eigengroups
    get_coeff_perm = lambda coeffs: rng.permutation(coeffs) if randomize else coeffs
    null_coeffs = [get_coeff_perm(coeffs[group]) for group in groups]

    # Generate transform vectors by combining the above matrices at this point
    tforms = [np.diag(1/sqrt_evals[i]) @ rots[i] @ np.diag(sqrt_evals[i]) @ null_coeffs[i] for i in np.arange(len(groups))]
    
    # Reconstruct nulls
    # - making sure we use only the eigenmodes up to the group number previously determined
    # - not necessarily all the modes, especially if that is not a square number
    null_map = emodes[:,np.concatenate(groups)] @ np.concatenate(tforms)

    if residual is not None:
        if residual_data is None: 
            raise ValueError(f"residual_data must be supplied when residual ('{residual}') is not "
                             f"None.")
        
        if residual == 'add':
            null_map += residual_data
        elif residual == 'permute':
            null_map += rng.permutation(residual_data)
        else:
            raise ValueError(f"Invalid residual method '{residual}'; must be 'add' or "
                             f"'permute'.")

    # Resample values from original data
    if resample is not None:
        if resample == 'range':
            # Force match the minimum and maximum to original data range
            scale_factor = (np.max(data) - np.min(data)) / (np.max(null_map) - np.min(null_map))
            null_map = (null_map - np.min(null_map)) * scale_factor + np.min(data)
        elif resample == 'mean':
            null_map = (null_map - np.mean(null_map)) + np.mean(data)
        elif resample == 'affine':
            null_map = (null_map - np.mean(null_map)) / np.std(null_map) * np.std(data) + np.mean(data)
        elif resample == 'exact':
            sorted_data = np.sort(data)
            sorted_indices = np.argsort(null_map)
            null_map[sorted_indices] = sorted_data
        else:
            raise ValueError(f"Invalid resampling method '{resample}'; must be 'exact', "
                             f"'affine', 'mean', or 'range'.")
    
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

def _eigenstrap_single(
    norm_emodes: NDArray[floating],
    sqrt_evals: NDArray[floating],
    groups: list[NDArray[integer]],
    coeffs: NDArray[floating],
    randomize: bool,
    seed: Union[int, None],
) -> NDArray[floating]:
    # Initialize RNG for this null using the provided seed to ensure reproducibility
    rng = np.random.RandomState(seed)

    # Get rotation matrices for each group (need special case for groups with exactly one mode)
    rots = [special_ortho_group.rvs(dim=len(group), random_state=rng) for group in groups[1:]]
    rots.insert(0, np.array([[1]]))  # Don't rotate constant mode to preserve mean
    
    # Optionally shuffle coefficients within eigengroups
    if randomize:
        for group in groups:
            coeffs[group] = rng.permutation(coeffs[group])

    # Generate transform vectors by combining the above matrices at this point
    # TODO: precompute right two matrices in parent function, if randomize can be done
    tforms = [rots[i] @ np.diag(sqrt_evals[group]) @ coeffs[group] for i, group in enumerate(groups)]

    # Reconstruct null
    return norm_emodes[:,np.concatenate(groups)] @ np.concatenate(tforms)

def eigenstrap(
    data: ArrayLike,
    emodes: ArrayLike,
    evals: ArrayLike,
    n_nulls: int = 1000,
    resample: Union[str, None] = None,
    randomize: bool = False,
    residual: Union[str, None] = None,
    decomp_method: str = 'project',
    mass: Union[ArrayLike, None] = None,
    n_jobs: int = -1,
    verbose: int = 0,
    seed: Union[int, None] = None,
    check_ortho: bool = True,
) -> NDArray[floating]:
    # Format / validate arguments
    emodes = np.asarray(emodes)  # chkfinite in decompose
    evals = np.asarray_chkfinite(evals) 
    data = np.asarray(data)
    if data.ndim == 1:
        data = data[:, np.newaxis]  # Ensure data is 2D for consistent processing
    if residual not in (None, 'add', 'permute'):
        raise ValueError(f"Invalid residual method '{residual}'; must be 'add', 'permute', or "
                         "None.")
    if resample not in (None, 'match', 'zscore', 'mean', 'range'):
        raise ValueError(f"Invalid resampling method '{resample}'; must be 'match', 'zscore', "
                         "'mean', 'range', or None.")
    
    # Eigendecompose maps
    coeffs = decompose(data, emodes, method=decomp_method, mass=mass, check_ortho=check_ortho)

    n_modes = emodes.shape[1]
    if evals.shape != (n_modes,):
        raise ValueError("`evals` must have shape (n_modes,), matching the number of columns in "
                         f"`emodes` ({n_modes}).")

    # Identify eigengroups for the modes that will be rotated
    groups = get_eigengroup_inds(n_modes)
    # If `n_modes` is not a perfect square then exclude the last group
    if int(np.sqrt(n_modes))**2 != n_modes:
        warn("`emodes` contains an incomplete eigengroup (i.e, number of columns (modes) is not a "
             f"perfect square). These last {len(groups[-1])} modes will be excluded.")
        groups = groups[:-1]

    # Transform each eigengroup from ellipsoid to spheroid
    sqrt_evals = np.sqrt(evals)
    sqrt_evals[0] = 1  # Avoids division by zero when normalizing
    norm_emodes = emodes / sqrt_evals

    # Initialise RNG, with seed for each null
    rng = np.random.RandomState(seed)
    null_seeds = rng.randint(np.iinfo(np.int32).max, size=n_nulls)
    
    # Generate nulls in parallel
    nulls = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(_eigenstrap_single)(
            norm_emodes=norm_emodes,
            sqrt_evals=sqrt_evals,
            groups=groups,
            coeffs=coeffs,
            randomize=randomize,
            seed=seed
        )
        for seed in null_seeds
    )
    nulls = np.stack(nulls, axis=1)  # Pyright HATES this one trick

    # Handle residuals
    if residual is not None:
        residual_data = (data - emodes @ coeffs)[:, np.newaxis, :]
        if residual == 'add':
            nulls += residual_data
        elif residual == 'permute':
            nulls += rng.permutation(residual_data)

    # Resample values from original data
    if resample == 'match':
        sorted_data = np.sort(data, axis=0)[:, np.newaxis, :]
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

    return nulls.squeeze()

