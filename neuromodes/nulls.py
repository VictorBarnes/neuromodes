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
    from scipy.sparse import spmatrix
    from numpy import floating, integer
    from numpy.typing import ArrayLike, NDArray

def eigenstrap(
    data: ArrayLike,
    emodes: ArrayLike,
    evals: ArrayLike, 
    n_nulls: int = 1000,
    n_groups: Union[int, None] = None,
    rotation_method: str = 'qr',
    randomize: bool = False,
    residual: Union[str, None] = None,
    resample: Union[str, None] = None,
    decomp_method: str = 'project',
    mass: Union[spmatrix, ArrayLike, None] = None,
    seed: Union[int, ArrayLike, None] = None,
    check_ortho: bool = True,
) -> NDArray[floating]:
    """
    Generate spatial null maps via eigenstrapping [1].
    
    This function generates spatial null maps that preserve the spatial autocorrelation structure of
    brain maps through random rotation of geometric eigenmodes. The method works by rotating
    eigenmodes within eigengroups (sets of modes with similar eigenvalues), then reconstructing null
    maps using the original decomposition coefficients.
    
    Parameters
    ----------
    data : array-like
        Empirical brain map(s) of shape (n_verts,) or (n_verts, n_maps) to generate nulls from. If
        n_maps > 1, the same set of randomized rotations is applied to all maps for each null (see
        Notes). 
    emodes : array-like of shape (n_verts, n_modes)
        The eigenmodes array of shape (n_verts, n_modes). This function rotates modes within
        eigengroups. Note that, unlike the original implementation as shown in [1], this requires
        the constant mode (the first column) to be input too. If the number of eigenmodes is not a
        perfect square (i.e., number of modes doesn't allow for complete eigengroups), then the last
        incomplete eigengroup will be excluded.
    evals : array-like of shape (n_modes,)
        The eigenvalues array of shape (n_modes,). Note that, unlike the original implementation as
        shown in [1], this requires the zero eigenvalue (the first eigenvalue) to be input too. 
    n_nulls : int, optional
        Number of null maps to generate per input map. Default is 1000.
    n_groups : int or None, optional
        Number of eigengroups to use for generating nulls. If `None`, uses all complete eigengroups
        contained in `emodes` (⌊√(n_modes)⌋). Default is `None`.
    rotation_method : str, optional
        The method used to generate random rotations for each eigengroup. Either `'qr'` to generate
        random orthogonal matrices using QR decomposition of random normal matrices, or `'scipy'` to
        sample random orthogonal matrices from SO(N) using `scipy.stats.special_ortho_group.rvs`.
        Default is `'qr'`. See Notes for details on which option to choose.
    randomize : bool, optional
        Whether to shuffle decomposition coefficients within eigengroups. This increases
        randomization but reduces spatial autocorrelation similarity to empirical data. Default is
        `False`.
    residual : str, optional
        How to handle reconstruction residuals after generating null maps. Either `None` to exclude
        residuals, `'add'` to add original residuals, or `'permute'` to adds shuffled residuals.
        Default is `None`. See Notes for details on which option to choose.
    resample : bool, optional
        How to resample values from original data. Options are `'exact'` to match the sorted
        distribution of the original data, `'affine'` to match the original mean and standard
        deviation, `'mean'` to match the mean, and `'range'` to match the minimum and maximum.
        Default is `None` for no resampling.
    decomp_method : str, optional
        The method used for eigendecomposition, either `'project'` to project data into a
        mass-orthonormal space or `'regress'` for least-squares fitting. Default is `'project'`.
    mass : array-like, optional
        The mass matrix of shape (n_verts, n_verts) used for the decomposition when `decomp_method`
        is `'project'`. Default is `None`.
    seed : array-like of shape (n_nulls,) or int, optional
        Random seed for reproducibility. If an array of shape (n_nulls,) is provided, it is used
        directly as the seed for each null. Otherwise, if a single integer is provided, it is used
        to generate a master seed that is then used to generate a different seed for each null. If
        `None`, the global state is used/progressed (in accordance with the original implementation
        of eigenstrapping). Default is `None`.
    check_ortho : bool, optional
        Whether to check if `emodes` are mass-orthonormal before using the `'project'` method in
        `neuromodes.basis.decompose`. Default is `True`.
    
    Returns
    -------
    ndarray of shape (n_verts, n_nulls) or (n_verts, n_nulls, n_maps)
        Generated null maps of shape (n_verts, n_nulls) if data has shape (n_verts,), or (n_verts,
        n_nulls, n_maps) if data has shape (n_verts, n_maps).
    
    Raises
    ------
    ValueError
        If `emodes` is not a 2D array or has n_verts ≥ n_modes.
    ValueError
        If `evals` length doesn't match number of columns in `emodes`.
    ValueError
        If `residual` is not one of `None`, `'add'`, or `'permute'`.
    ValueError
        If `resample` is not one of `None`, `'exact'`, `'affine'`, `'mean'`, or `'range'`.
    ValueError
        If `rotation_method` is not one of `'qr'` or `'scipy'`.
    ValueError
        If `n_groups` is greater than the number of eigengroups than can be formed from the number
        of modes in `emodes`.

    Notes
    -----
    1. When `data` contains multiple maps (n_maps > 1), the same set of randomized rotations 
    is applied to all maps for each null. This means that null i for map A and null i for map B use
    identical eigenmode rotations.

    2. This function uses the constant mode (first column of `emodes`) and its correpsonding 
    eigenvalue to generate mean-preserving nulls. The constant mode is neither rotated nor
    transformed as no non-trivial rotations exist for this eigengroup consisting of a single mode
    (SO(1) = {1}).

    3. The choice of `n_groups` and `residual` will affect the spatial autocorrelation 
    similarity between the nulls and empirical data. See [1] for a heuristic for choosing `n_groups`
    and to see how the choice of `residual` affects the spatial autocorrelation of the nulls. 

    4. The choice of `resample` will affect the distribution of values in the nulls. Linear
    transformations (`"mean"` and `"affine"`) preserve the shape of the distribution, while
    non-linear transformations (`"range"` and `"exact"`) alter the shape of the distribution to
    match the empirical distribution of the original data. The choice of `resample` should be guided
    by the importance of matching the original distribution of values and ultimately by whichever
    option produces the lowest false discovery rate (FDR). See [1] for an example of how to compute
    the FDR.

    5. Seeding is handled in one or two stages. First, if `seed` is not of shape (n_nulls,) or
    similar, `seed` is used to initialize a master random number generator (RNG) for each null map
    (this will start at a random integer given by `seed` and will then increment). Then, each null
    uses its allocated integer to generate its own RNG to use for all rotations/permutations of that
    null.

    6. To exactly match the default version of the original implementation of eigenstrapping in [1],
    users must do the following: 
        - Ensure `data` has a mean of zero. 
        - Set the global seed before running this function (e.g., `np.random.seed(seed)`). 
        - Set `resample="range"`
        - Set `decomp_method="regress"` 
        - Set `rotation_method="scipy"` 
        - Set `seed=None` 
    Note that the original implementation (`eigenstrapping.SurfaceEigenstrapping`) must also be run
    with a particular configuration to ensure reproducibility/compatibility: 
        - Set the global seed before running this function (e.g., `np.random.seed(seed)`). 
        - Additionally, pass this seed into the function call: `SurfaceEigenstrapping(...,
          seed=seed)` 
        - Remember to remove the first eigenmode/eigenvalue from the call to SurfaceEigenstrapping 
    For an example of how to do this, see:
    https://neuromodes.readthedocs.io/en/latest/validation/eigenstrapping_match_orig.html

    7. If both resampling and adding residuals is requested, the original implementation adds
    residuals after resampling. Here, the order of these steps is swapped (ie, add residuals and
    then resample). This ensures that the resampling remains intact (eg, that the surrogates and
    original actually have the same values). If (instead) the resampling is done before the
    residuals are added, then neither step will remain intact. This difference is only relevant if
    both `resample` and `residual` are used. 
    
    References
    ----------
    ..  [1] Koussis, N. C., et al. (2024). Generation of surrogate brain maps preserving 
        spatial autocorrelation through random rotation of geometric eigenmodes. 
        Imaging Neuroscience. https://doi.org/10.1162/IMAG.a.71
    """
    # Format / validate arguments
    # data
    data = np.asarray(data)  # chkfinite in decompose
    if (is_vector_data := data.ndim == 1):
        data = data[:, np.newaxis]
    n_maps = data.shape[1]

    # emodes and evals
    emodes = np.asarray(emodes)  # chkfinite in decompose
    evals = np.asarray_chkfinite(evals) 
    n_cols = emodes.shape[1]
    if emodes.ndim != 2 or emodes.shape[0] < n_cols:
        raise ValueError("`emodes` must have shape (n_verts, n_modes), where n_verts ≥ n_modes.")
    if evals.shape != (n_cols,):
        raise ValueError(f"`evals` must have shape (n_modes,) = ({n_cols},), matching the number "
                         "of columns in `emodes`.")
    
    # n_groups : Determine number of eigengroups, and trim emodes and evals to match
    if n_groups is None:
        n_groups = int(np.sqrt(n_cols))  # floor of root
        if n_groups**2 != n_cols:
            warn("`emodes` contains an incomplete eigengroup (i.e, number of modes is not a "
                 f"perfect square). Last {n_cols - n_groups**2} modes will be excluded.")
    elif n_groups**2 > n_cols:
        raise ValueError(f"`n_groups`={n_groups} implies n_modes={n_groups**2}, which exceeds the "
                         f"number of columns in `emodes` ({n_cols}).")
    n_modes = n_groups**2
    groups = get_eigengroup_inds(n_modes)
    emodes = emodes[:, :n_modes].copy()
    evals = evals[:n_modes].copy()
    
    # rotation_method : Set which helper function to use
    if rotation_method == 'qr':
        _rotate_coeffs = _rotate_coeffs_qr
    elif rotation_method == 'scipy':
        _rotate_coeffs = _rotate_coeffs_scipy
    else: 
        raise ValueError(f"Invalid rotation method '{rotation_method}'; must be 'qr' or 'scipy'.")
    
    # residual and resample
    if residual not in (None, 'add', 'permute'):
        raise ValueError(f"Invalid residual method '{residual}'; must be 'add', 'permute', or "
                         "None.")
    if resample not in (None, 'exact', 'affine', 'mean', 'range'):
        raise ValueError(f"Invalid resampling method '{resample}'; must be 'exact', 'affine', "
                         "'mean', 'range', or None.")
    
    # seed : Initialise RNG with seed for each null
    if seed is None: 
        null_seeds = np.full((n_nulls,), None) # to match original
    else: 
        seed_array = np.ravel(seed).astype(int) # ravel handles ints, lists, and arrays
        if seed_array.size == n_nulls:
            # If user has provided a seed for each null (including where n_nulls == 1), use those
            # directly. 
            null_seeds = seed_array
        elif seed_array.size == 1:
            # Turn the seed into a random start point, and then use sequential integers after that.
            # This is the easiest way to get random, different integers:
            #   - `.integers` does not offer sampling without replacement. 
            #   - `.choice` is not reproducible if the number of nulls is changed due to hidden
            # floating point round off in its implementation. Compare
            # `np.random.default_rng(1).choice(2**31-1, size=4, shuffle=False, replace=False)` and
            # `np.random.default_rng(1).choice(2**31-1, size=5, shuffle=False, replace=False)`.
            #   - Just adding the index to the seed can lead to nulls being repeated across
            # different seeds, when they would be expected to be different e.g. for `seed=0,
            # n_nulls=1000` and `seed=42, n_nulls=1000` (or `seed=314`, `seed=365` etc). 
            null_seeds = (np.arange(n_nulls, dtype=np.int64) 
                          + np.random.default_rng(int(seed_array[0])).integers(np.iinfo(np.int32).max))
        else:
            raise ValueError(
                f"If `seed` is a single value, it must be an integer. "
                f"If `seed` is an array, it must have exactly {n_nulls} elements. "
                f"Got an input with {seed_array.size} elements."
            )

    # Main calculations
    # Precompute transformed modes (ellipsoid -> spheroid for each eigengroup)
    sqrt_evals = np.sqrt(evals)
    sqrt_evals[0] = 1 # No transform for constant mode (preserves mean and avoids division by zero)
    norm_emodes = emodes / sqrt_evals[np.newaxis, :] # sqrt_evals behaves like a row vector
    sqrt_evals = sqrt_evals[:, np.newaxis, np.newaxis] # turn it into 3D column vector for below broadcasting

    # Eigendecompose maps (coeffs is n_modes x n_maps)
    coeffs = decompose(data, emodes, method=decomp_method, mass=mass, check_ortho=check_ortho)
    # Turn coeffs into inv_coeffs, a 3D array of shape (n_modes, n_nulls, n_maps)
    # This is the precomputed inverse-transformed coefficients (spheroid -> ellipsoid for each eigengroup)
    if randomize:
        inv_coeffs = np.empty((n_modes, n_nulls, n_maps))
        for i, s in enumerate(null_seeds):
            rng = np.random.default_rng(s)
            for group in groups:
                inv_coeffs[group, i, :] = rng.permutation(coeffs[group, :], axis=0) 
        inv_coeffs *= sqrt_evals
    else:
        inv_coeffs = np.broadcast_to((coeffs[:, np.newaxis, :] * sqrt_evals), (n_modes, n_nulls, n_maps))

    # Generate nulls using tforms of shape (n_modes, n_nulls, n_maps)
    tforms = _rotate_coeffs(
        inv_coeffs=inv_coeffs,
        groups=groups,
        seeds=null_seeds
    )
    # tensordot appears faster than einsum
    nulls = np.tensordot(norm_emodes, tforms, axes=(1, 0)) # (n_verts, n_nulls, n_maps)

    # Optional post-processing steps
    # Optionally add residuals of reconstruction
    if residual is not None:
        residual_data = data - emodes @ coeffs # shape (n_verts, n_maps)
        if residual == 'add':
            nulls += residual_data[:, np.newaxis, :]
        elif residual == 'permute':
            for i, s in enumerate(null_seeds):
                nulls[:, i, :] += np.random.default_rng(s).permutation(residual_data, axis=0)

    # Optionally resample values to match stats of original data
    if resample == 'exact':
        sorted_data = np.sort(data, axis=0)[:, np.newaxis, :]
        ranks = np.argsort(np.argsort(nulls, axis=0), axis=0)
        nulls = np.take_along_axis(sorted_data, ranks, axis=0)
    elif resample == 'mean':
        nulls -= nulls.mean(axis=0, keepdims=True)
        nulls += data.mean(axis=0)
    elif resample == 'affine':
        nulls -= nulls.mean(axis=0, keepdims=True)
        nulls /= nulls.std(axis=0, keepdims=True)
        nulls *= data.std(axis=0)
        nulls += data.mean(axis=0)
    elif resample == 'range': # to match original
        nulls -= nulls.min(axis=0, keepdims=True)
        nulls /= nulls.max(axis=0, keepdims=True)
        nulls *= data.max(axis=0) - data.min(axis=0)
        nulls += data.min(axis=0)

    if is_vector_data:
        nulls = nulls.squeeze(axis=2)

    return nulls

def _rotate_coeffs_scipy(
    inv_coeffs: NDArray[floating],
    groups: list[NDArray[integer]],
    seeds: NDArray[integer]
) -> NDArray[floating]:
    """
    Rotate coefficients using `scipy.stats.special_ortho_group.rvs` to sample random orthogonal
    matrices from SO(N). This is largely a legacy option to match the original implementation of
    eigenstrapping, which uses this method. The QR method is generally faster, especially for larger
    numbers of modes and nulls, and is recommended for most users. However, the scipy method is
    recommended only for users who want to exactly match the original implementation of
    eigenstrapping.

    Parameters
    ----------
    inv_coeffs : array of shape (n_modes, n_nulls, n_maps)
        The inverse-transformed coefficients (spheroid -> ellipsoid) of shape (n_modes, n_nulls,
        n_maps) to rotate.
    groups : list of arrays
        A list of arrays, where each array contains the indices of modes belonging to the same
        eigengroup.
    seeds : array of shape (n_nulls,)
        An array of integer seeds of shape (n_nulls,) to use for reproducibility of the random
        rotations for each null map.
    """
    # Unlike qr method, have to pass None (to keep using global seed) to match original eigenstrapping
    rngs = [None if s is None else np.random.default_rng(s) for s in seeds]

    # Define helper
    def _get_so(k: int, rng: np.random.Generator | None) -> NDArray[floating]:
        return special_ortho_group.rvs(dim=k, random_state=rng) if k != 1 else np.array([[1.0]])

    tforms = np.empty_like(inv_coeffs)
    for i, rng in enumerate(rngs):  # Has to be in this order to match original functionality
        for group in groups:        # ie can't switch order of loops (even though it would be better)
            Q = _get_so(len(group), rng)
            # NumPy slicing trickery as second dims of tforms and inv_coeffs are collapsed:
            tforms[group, i, :] = Q @ inv_coeffs[group, i, :] 

    return tforms

def _rotate_coeffs_qr(
        inv_coeffs: NDArray[floating],
        groups: list[NDArray[integer]],
        seeds: NDArray[integer]
) -> NDArray[floating]:
    """
    Rotate coefficients using QR decomposition of random Gaussian matrices to generate random
    orthogonal matrices.
    
    Parameters
    ----------
    inv_coeffs : np.ndarray of shape (n_modes, n_nulls, n_maps)
        The inverse-transformed coefficients (spheroid -> ellipsoid) of shape (n_modes, n_nulls,
        n_maps) to rotate.
    groups : list of np.ndarrays
        A list of arrays, where each array contains the indices of modes belonging to the same
        eigengroup.
    seeds : np.ndarray (n_nulls,)
        An array of integer seeds of shape (n_nulls,) to use for reproducibility of the random
        rotations for each null map.

    References
    ----------
    ..  [1] Mezzadri. (2007). How to generate random matrices from the classical compact groups. 
        https://www.ams.org/notices/200705/fea-mezzadri-web.pdf
    """
    # Unlike scipy method, still have to return Generator even if seed is None (for generating random Gaussian matrices)
    rngs = [np.random.default_rng(s) for s in seeds]

    # Define helper
    def _generate_so(k: int, rngs: list[np.random.Generator]) -> NDArray[floating]:
        X = np.stack([rng.standard_normal((k, k)) for rng in rngs], axis=0) # rng progresses over each group
        Q, R = np.linalg.qr(X) # Q has shape (n_nulls, k, k)
        r = np.copysign(1.0, np.diagonal(R, axis1=1, axis2=2)) # copysign avoids chance of 0
        Q = Q * r[:, np.newaxis, :]
        dets = np.linalg.det(Q)
        Q[:, :, 0] *= dets[:, np.newaxis]
        return Q

    tforms = np.empty_like(inv_coeffs)
    for group in groups: # unlike scipy, can loop over groups only (don't also have to loop over nulls)
        Q = _generate_so(len(group), rngs) # as we can generate rotations for all nulls at once
        # Batched matmul (equiv of @) appears faster than einsum or tensordot:
        tforms[group, :, :] = np.matmul(Q, inv_coeffs[group, :, :], axes=[(1, 2), (0, 2), (0, 2)])

    return tforms
