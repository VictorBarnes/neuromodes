"""
Spatial null models via eigenmode rotation.

This module provides functions for generating null brain maps that preserve spatial
autocorrelation structure through random rotation of geometric eigenmodes.
"""

import numpy as np
from numpy.typing import NDArray, ArrayLike
from scipy.stats import special_ortho_group
from scipy import linalg, sparse
from joblib import Parallel, delayed
from typing import Union, Tuple

from neuromodes.basis import decompose
from neuromodes.eigen import get_eigengroup_inds, is_orthonormal_basis

match_eigenstrapping = False
# print(f"Match eigenstrapping mode: {match_eigenstrapping}")

def generate_nulls(
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
    check_ortho: bool = True
) -> NDArray:
    """
    Generate surface-based null maps via rotation of eigengroups.
    
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
        The eigenmodes array of shape (n_verts, n_modes), excluding the constant mode, 
        where n_verts is the number of vertices and n_modes is the number of eigenmodes. 
        If working on a masked surface, `emodes` must already be masked (see Notes).
    evals : array-like
        The eigenvalues array of shape (n_modes,) excluding the eval corresponding to the 
        constant mode.
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
        Notes for more details). Default is None.
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
        If residual is not None, 'add', or 'permute'.
    ValueError
        If number of modes is less than 3.
    ValueError
        If emodes columns are not orthonormal when method='project' and check_ortho=True.

    Notes
    -----
    This function does not apply any vertex masking. If working on a masked surface (e.g., 
    excluding the medial wall), the user must pass `data`, `emodes`, and `mass` (if used) 
    already restricted to the desired masked vertices.

    Seeding is handled in two stages to ensure reproducibility when parallelising. First 
    `seed` is used to initialize a master random number generator (RNG) that generates an 
    independent integer seed for each null map. Then each null uses that its allocated 
    integer to generate its own RNG to use for all rotations/permutations of that null. 
    This ensures that each null is independent of the number of parallel jobs used.
    
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
    if (np.std(emodes, axis=0) < 1e-6).any() or (evals < 1e-6).any():
        raise ValueError("Eigenmodes appear to include the constant mode; however, null "
                         "generation requires modes starting from the first non-constant "
                         "mode. Please exclude the constant mode and corresponding eval.")
    
    if residual is not None and residual not in ('add', 'permute'):
        raise ValueError(f"Invalid residual method '{residual}'; must be 'add', 'permute', or None.")
    
    # Compute decomposition coefficients and residuals
    if match_eigenstrapping:
        coeffs = np.linalg.solve(emodes.T @ emodes, emodes.T @ data)
    else:
        coeffs = decompose(data, emodes, method=method, mass=mass, check_ortho=check_ortho)
    coeffs = coeffs.squeeze()
    
    reconstructed = emodes @ coeffs
    residuals = data - reconstructed
    
    # TODO: remove first (constant) eigengroup from following calculation
    # Identify eigengroups
    groups = get_eigengroup_inds(emodes)
    
    if match_eigenstrapping:
        # Match eigenstrapping exactly: set global seed AND create RandomState
        if seed is not None:
            np.random.seed(seed)
        rng = _check_random_state(seed)
        null_seeds = rng.randint(np.iinfo(np.int32).max, size=n_nulls)
    else:
        # Simplified approach for non-eigenstrapping mode
        rng = np.random.RandomState(seed)
        null_seeds = rng.randint(np.iinfo(np.int32).max, size=n_nulls)
    
    # Generate nulls in parallel
    nulls = Parallel(n_jobs=n_jobs)(
        delayed(_generate_single_null)(
            data, emodes, evals, groups, coeffs, residuals, mass, resample, randomize, 
            residual, seed_i
        )
        for seed_i in null_seeds
    )
    
    if match_eigenstrapping:
        result = np.row_stack(nulls)
        result = np.asarray(result.squeeze()).T
    else:
        result = np.asarray(nulls).T

    return result

def _generate_single_null(
    data: NDArray,
    emodes: NDArray,
    evals: NDArray,
    groups: list[NDArray],
    coeffs: NDArray,
    residuals: NDArray,
    mass: Union[NDArray, None],
    resample: bool,
    randomize: bool,
    residual: Union[str, None],
    seed: Union[int, None]
) -> NDArray:
    """
    Generate a single null map.
    
    Parameters
    ----------
    data : array-like
        Empirical brain map of shape (n_verts,) to generate nulls from. 
    emodes : array-like
        The eigenmodes array of shape (n_verts, n_modes), excluding the constant mode, 
        where n_verts is the number of vertices and n_modes is the number of eigenmodes. 
    evals : array-like
        The eigenvalues array of shape (n_modes,) excluding the eval corresponding to the 
        constant mode.
    groups : list of array-like
        Eigengroup indices.
    coeffs : array-like
        Decomposition coefficients of shape (n_modes,)
    residuals : array-like
        Residuals from reconstruction of shape (n_verts,).
    mass: array-like
        The mass matrix of shape (n_verts, n_verts) used for the decomposition.
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
    # Create random state for seed generation
    rng = _check_random_state(seed) if match_eigenstrapping else np.random.RandomState(seed)
    
    # Initialize rotated modes
    rotated_emodes = np.zeros_like(emodes)
    
    # Rotate each eigengroup
    max_var = 0
    for group_idx in groups:
        group_emodes = emodes[:, group_idx]
        group_evals = evals[group_idx]

        eval_var = np.var(group_evals)
        if eval_var > max_var:
            max_var = eval_var
        
        # Transform to spheroid, rotate, transform back to ellipsoid
        # spheroid_modes = group_emodes / np.sqrt(group_evals)

        rotated_spheroid = _rotate_emodes(group_emodes, random_state=rng)
        
        # rotated_spheroid = rotated_spheroid * np.sqrt(group_evals)
        
        rotated_emodes[:, group_idx] = rotated_spheroid

    # print(f"Max eigenvalue variance across groups: {max_var}")

    # Check orthonormality of rotated modes
    if not is_orthonormal_basis(rotated_emodes, mass=mass):
        raise ValueError("Rotated eigenmodes are not orthonormal.")
    
    # Optionally shuffle coefficients within eigengroups
    null_coeffs = coeffs.copy()
    if randomize:
        for group_idx in groups:
            null_coeffs[group_idx] = rng.permutation(null_coeffs[group_idx])
    
    # TODO: figure out whether normalization is needed here
    # Reconstruct null
    if np.any(np.isnan(rotated_emodes)) or np.any(np.isinf(rotated_emodes)):
        raise ValueError("Rotated eigenmodes contain NaN or Inf values.")
    if np.any(np.isnan(null_coeffs)) or np.any(np.isinf(null_coeffs)):
        raise ValueError("Null coefficients contain NaN or Inf values.")
    # Check if null_coeffs contains any values close to 0
    if np.any(np.isclose(null_coeffs, 0)):
        raise ValueError("Null coefficients contain values close to zero, which may lead to invalid null maps.")
    null_map = rotated_emodes @ null_coeffs

    if np.any(np.isnan(null_map)) or np.any(np.isinf(null_map)):
        raise ValueError("Generated null map contains NaN or Inf values.")
    
    # Handle residuals
    if residual == 'add':
        null_map += residuals
    elif residual == 'permute':
        null_map += rng.permutation(residuals)
    
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

# TODO: decide whether we're going to use this function
def _compute_mass_sqrt(mass: Union[NDArray, sparse.spmatrix]) -> Tuple[NDArray, NDArray]:
    """
    Compute M^(1/2) and M^(-1/2) using eigendecomposition.
    
    For a symmetric positive definite mass matrix M, computes:
        M = Q Λ Q^T
        M^(1/2) = Q Λ^(1/2) Q^T
        M^(-1/2) = Q Λ^(-1/2) Q^T
    
    Parameters
    ----------
    mass : ndarray or sparse matrix
        The mass matrix of shape (n_verts, n_verts). Must be symmetric positive definite.
    
    Returns
    -------
    mass_sqrt : ndarray of shape (n_verts, n_verts)
        The matrix square root M^(1/2).
    mass_invsqrt : ndarray of shape (n_verts, n_verts)
        The inverse matrix square root M^(-1/2).
    """
    # Check if sparse
    is_sparse = sparse.issparse(mass)
    
    if is_sparse:
        # For sparse matrices, use eigsh (assumes symmetric)
        # Compute all eigenvalues for full transformation
        n = mass.shape[0]
        evals, evecs = sparse.linalg.eigsh(mass, k=n-1, which='LM')
    else:
        # For dense matrices, use eigh
        evals, evecs = linalg.eigh(mass)
    
    # Ensure all eigenvalues are positive (with numerical tolerance)
    if np.any(evals <= 0):
        min_eval = np.min(evals)
        if min_eval < -1e-10:  # Significant negative eigenvalue
            raise ValueError(f"Mass matrix has negative eigenvalue: {min_eval}")
        # Set small negative/zero eigenvalues to small positive value
        evals = np.maximum(evals, 1e-12)
    
    # Compute sqrt and inverse sqrt of eigenvalues
    sqrt_evals = np.sqrt(evals)
    invsqrt_evals = 1.0 / sqrt_evals
    
    # Reconstruct M^(1/2) and M^(-1/2)
    # M^(1/2) = Q * sqrt(Λ) * Q^T
    mass_sqrt = evecs @ np.diag(sqrt_evals) @ evecs.T
    mass_invsqrt = evecs @ np.diag(invsqrt_evals) @ evecs.T
    
    return mass_sqrt, mass_invsqrt

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
    random_state : int or numpy.random.RandomState, optional
        Random state passed through to `scipy.stats.special_ortho_group.rvs`.
        Prefer passing a RandomState instance so successive calls advance the RNG state.
        If an int is provided, SciPy will reseed a fresh RNG for each call.
    
    Returns
    -------
    ndarray of shape (n_vertices, n_modes)
        Rotated eigenmodes.
    """
    n_modes = emodes.shape[1]

    rotation_matrix = special_ortho_group.rvs(dim=n_modes, random_state=random_state)
    
    return emodes @ rotation_matrix

#TODO: remove this function
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
