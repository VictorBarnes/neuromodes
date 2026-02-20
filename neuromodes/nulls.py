"""
Spatial null models via eigenmode rotation.

This module provides functions for generating null brain maps that preserve spatial
autocorrelation structure through random rotation of geometric eigenmodes.
"""
from __future__ import annotations
from typing import Union, TYPE_CHECKING
from warnings import warn
import numpy as np
from scipy.stats import special_ortho_group
from neuromodes.basis import decompose
from neuromodes.eigen import get_eigengroup_inds

if TYPE_CHECKING:
    from numpy import floating, integer
    from numpy.typing import ArrayLike, NDArray

def eigenstrap(
    data: ArrayLike,
    emodes: ArrayLike,
    evals: ArrayLike,
    n_modes: Union[int, None] = None, 
    n_nulls: int = 1000,
    resample: Union[str, None] = None,
    randomize: bool = False,
    residual: Union[str, None] = None,
    decomp_method: str = 'project',
    mass: Union[ArrayLike, None] = None,
    seed: Union[int, None] = None,
    check_ortho: bool = True,
) -> NDArray[floating]:
    """
    Generate null maps via eigenstrapping [1].
    
    This function generates spatial null models that preserve the spatial autocorrelation
    structure of brain maps through random rotation of geometric eigenmodes. The method
    works by rotating eigenmodes within eigengroups (sets of modes with similar eigenvalues),
    then reconstructing null maps using the original decomposition coefficients.
    
    Parameters
    ----------
    data : array-like
        Empirical brain map(s) of shape (n_verts,) or (n_verts, n_maps) to generate nulls from.
        If working on a masked surface, `data` must already be masked (see Notes).
    emodes : array-like of shape (n_verts, n_modes)
        The eigenmodes array of shape (n_verts, n_modes). This function rotates modes within
        eigengroups. If the number of eigenmodes is not a perfect square (i.e., number of modes
        doesn't allow for complete eigengroups), then the last incomplete eigengroup will be
        excluded.
    evals : array-like
        The eigenvalues array of shape (n_modes,). 
    n_nulls : int, optional
        Number of null maps to generate per input map. Default is 1000.
    resample : bool, optional
        How to resample values from original data. Options are `'exact'` to match the sorted 
        distribution of the original data, `'affine'` to match the original mean and standard
        deviation, `'mean'` to match the mean, and `'range'` to match the minimum and 
        maximum. Default is `None` for no resampling.
    randomize : bool, optional
        Whether to shuffle decomposition coefficients within eigengroups. This increases
        randomization but reduces spatial autocorrelation similarity to empirical data. 
        Default is `False`.
    residual : str, optional
        How to handle reconstruction residuals after generating null maps. Either `None` to exclude
        residuals, `'add'` to add original residuals, or `'permute'` to adds shuffled residuals.
        Default is `None`.
    decomp_method : str, optional
        The method used for eigendecomposition, either `'project'` to project data into a
        mass-orthonormal space or `'regress'` for least-squares fitting. Default is `'project'`.
    mass : array-like, optional
        The mass matrix of shape (n_verts, n_verts) used for the decomposition when `decomp_method`
        is `'project'`. Default is `None`.
    seed : int, optional
        Random seed for reproducibility. If provided, generates deterministic nulls (see 
        Notes). Default is `None`.
    check_ortho : bool, optional
        Whether to check if `emodes` are mass-orthonormal before using the `'project'` method
        in `neuromodes.basis.decompose`. Default is `True`.
    
    Returns
    -------
    ndarray of shape (n_verts, n_nulls) or (n_verts, n_nulls, n_maps)
        Generated null maps.
    
    Raises
    ------
    ValueError
        If `evals` length doesn't match number of columns in `emodes`.
    ValueError
        If `residual` is not one of `None`, `'add'`, or `'permute'`.
    ValueError
        If `resample` is not one of `None`, `'exact'`, `'affine'`, `'mean'`, or `'range'`.

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
    if data.ndim == 1:
        data = data[:, np.newaxis]
    if residual not in (None, 'add', 'permute'):
        raise ValueError(f"Invalid residual method '{residual}'; must be 'add', 'permute', or "
                         "None.")
    if resample not in (None, 'exact', 'affine', 'mean', 'range'):
        raise ValueError(f"Invalid resampling method '{resample}'; must be 'exact', 'affine', "
                         "'mean', 'range', or None.")    

    if n_modes is None:
        n_modes = emodes.shape[1] # decompose (below) also checks data shape
    n_groups = int(np.sqrt(n_modes))
    if n_groups**2 != n_modes:
        warn("`emodes` contains an incomplete eigengroup (i.e, number of modes is not a "
             f"perfect square). These last {n_modes - n_groups**2} modes will be excluded.")
        n_modes = n_groups**2
    groups = get_eigengroup_inds(n_modes)
    emodes = emodes[:, :n_modes].copy()
    if evals.shape < (n_modes,):
        raise ValueError("`evals` must have shape (>=n_modes,), matching `n_modes` ({n_modes}).")
    evals = evals[:n_modes].copy()
    
    # Eigendecompose maps
    coeffs = decompose(data, emodes, method=decomp_method, mass=mass, check_ortho=check_ortho)

    # Precompute transformed modes (ellipsoid -> spheroid for each eigengroup)
    sqrt_evals = np.sqrt(evals)
    sqrt_evals[0] = 1  # No transform for constant mode (preserves mean and avoids division by zero)
    norm_emodes = emodes / sqrt_evals

    sqrt_evals = sqrt_evals[:, np.newaxis]

    # Initialise RNG, with seed for each null
    random_state = np.random.default_rng(seed)
    
    if randomize:
        perm_seeds = random_state.integers(np.iinfo(np.int32).max, size=n_nulls)

        # Permute coefficients within eigengroups for each null
        coeffs = np.stack([
            np.concatenate([
                np.random.default_rng(seed).permutation(coeffs[group, :], axis=0)
                for group in groups
            ], axis=0)
            for seed in perm_seeds
        # Place each null's coeffs along a third axis
        ], axis=2)

        sqrt_evals = sqrt_evals[:, :, np.newaxis]

    # Precompute inverse-transformed coefficients (spheroid -> ellipsoid for each eigengroup)
    inv_coeffs = sqrt_evals * coeffs

    # Compute random rotation matrices for each eigengroup across all nulls
    rotations = [
        special_ortho_group.rvs(
            dim=len(group), 
            size=n_nulls, 
            random_state=random_state
        ) 
        if len(group) != 1 
        else np.ones([n_nulls,1,1])
        for group in groups
    ]

    # Generate nulls
    nulls = np.stack([
        _eigenstrap_single(
            norm_emodes=norm_emodes,
            groups=groups,
            inv_coeffs=inv_coeffs[:, :, i] if randomize else inv_coeffs,
            rotations=rotations,
            null_idx=i
            )
            for i in range(n_nulls)
            ], axis=1)

    # Handle residuals
    if residual is not None:
        residual_data = (data - emodes @ coeffs)[:, np.newaxis, :]
        if residual == 'add':
            nulls += residual_data
        elif residual == 'permute':
            nulls += random_state.permutation(residual_data)

    # Resample values from original data
    if resample == 'exact':
        sorted_data = np.sort(data, axis=0)[:, np.newaxis, :]
        sorted_indices = np.argsort(nulls, axis=0)
        nulls = np.take_along_axis(sorted_data, sorted_indices, axis=0)
    elif resample == 'mean':
        nulls = (nulls - nulls.mean(axis=0)) + data.mean(axis=0)
    elif resample == 'affine':
        nullz = (nulls - nulls.mean(axis=0)) / nulls.std(axis=0)
        nulls = nullz * data.std(axis=0) + data.mean(axis=0)
    elif resample == 'range':
        # Force match the minimum and maximum to original data range
        scale_factor = (data.max(axis=0) - data.min(axis=0)) / (nulls.max(axis=0) - nulls.min(axis=0))
        nulls = (nulls - nulls.min(axis=0)) * scale_factor + data.min(axis=0)

    if data.shape[1] == 1: # number of maps
        nulls = nulls.squeeze(axis=2)

    return nulls

def _eigenstrap_single(
    norm_emodes: NDArray[floating],
    groups: list[NDArray[integer]],
    inv_coeffs: NDArray[floating],
    rotations: list[NDArray[floating]],
    null_idx: int
) -> NDArray[floating]:
    """
    Generate a single null for each input map by applying random rotations to each eigengroup and
    reconstructing the map using the eigendecomposition coefficients. For computational efficiency,
    each eigengroup's ellipsoid-to-spheroid transformation and its inverse are already incorporated
    into `norm_emodes` and `inv_coeffs`, respectively.

    Parameters
    ----------
    norm_emodes : ndarray of shape (n_verts, n_modes)
        The group-normalized eigenmodes of shape (n_verts, n_modes).
    groups : list of ndarrays of integers
        List of arrays, where each array contains the column indices of `norm_emodes` that belong to
        the same eigengroup (i.e., [[0], [1,2,3], [4,5,6,7,8], ..., [..., n_modes-1]]).
    inv_coeffs : ndarray of shape (n_modes, n_maps)
        The inverse-transformed decomposition coefficients of shape (n_modes, n_maps). If
        `randomize` is False, this is the same for all nulls.
    rotations : list of ndarrays of shape (n_nulls, len(group), len(group))
        The random rotation matrices to apply to each eigengroup (except the first).
    null_idx : int
        The index of the null being generated, used to select the corresponding rotation matrices
        from `rotations` list.

    Returns
    -------
    ndarray of shape (n_verts, n_maps)
        The generated null map(s) of shape (n_verts, n_maps).
    """
    # Construct matrix encoding all operations to apply to the transformed modes
    tforms = np.concatenate(
        # Multiply each group's random rotation matrix by its inverse-transformed coefficients
        [rotations[i][null_idx, :, :] @ inv_coeffs[group, :] for i, group in enumerate(groups)],
         axis=0
         )

    # Generate null
    return norm_emodes @ tforms