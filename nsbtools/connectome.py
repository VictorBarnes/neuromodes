"""Module for generating structural connectomes using geometric eigenmodes of the cortex."""

import numpy as np
from warnings import warn
from typing import Union, List, Optional
from numpy.typing import NDArray

def generate_connectome(
    emodes: Union[NDArray, List[List[float]]],
    evals: Union[NDArray, List[float]],
    r: float = 9.53,
    k: int = 108,
    threshold: float = 0.0,
    resample_weights_gaussian: bool = False,
    seed: Optional[int] = None
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
    threshold : float, optional
        Percentile under which to zero out connection weights from the output matrix, between 0 and
        100. Default is 0.
    resample_weights_gaussian : bool, optional
        Whether to apply Gaussian resampling to the weights.
    seed : int, optional
        Random seed for weight resampling.

    Returns
    -------
    np.ndarray
        The generated vertex-wise structural connectivity matrix.
    """
    emodes = np.asarray(emodes)
    evals = np.asarray(evals)
    k = int(k)

    if r <= 0:
        raise ValueError("Parameter `r` must be positive.")
    if emodes.ndim != 2:
        raise ValueError("Parameter `emodes` must be a 2D array (vertices x modes).")
    if len(evals) != emodes.shape[1]:
        raise ValueError("Length of `evals` must match the number of modes (columns) in `emodes`.")
    if k <= 0 or k > len(evals):
        raise ValueError(f"Parameter `k` must be an integer in the range [1, {len(evals)}].")

    # Select the first k eigenmodes and eigenvalues
    k_evals = evals[:k]
    k_emodes = emodes[:, :k]

    # Compute structural connectivity
    denom = np.array([
        1/(1 + eval * r**2)
        for eval in k_evals])

    connectome = k_emodes @ np.diag(denom) @ np.linalg.pinv(k_emodes)

    # Replace diagonal and negative values with zero
    np.fill_diagonal(connectome, 0)
    connectome[connectome < 0] = 0
    
    # Symmetrize the matrix
    connectome = (connectome + connectome.T) / 2

    if threshold != 0:
        connectome = threshold_matrix(connectome, threshold)

    if resample_weights_gaussian:
        connectome = resample_matrix(connectome, seed=seed)
    else:
        connectome /= np.max(connectome)

    return connectome

def threshold_matrix(
    matrix: Union[NDArray, List[List[float]]],
    threshold: float
) -> NDArray:
    """
    Threshold a matrix by zeroing out values below a certain percentile.

    Parameters
    ----------
    matrix : array-like
        The input matrix to be thresholded.
    threshold : float
        The percentile threshold (between 0 and 100).

    Returns
    -------
    np.ndarray
        The thresholded matrix.
    """
    matrix = np.asarray(matrix)

    if threshold < 0 or threshold > 100:
        raise ValueError(f"`threshold` must be between 0 and 100.")
    
    if (matrix == matrix.T).all():
        # If matrix is symmetric, only consider upper triangle for thresholding
        triu_indices = np.triu_indices_from(matrix, k=1)
        triu_values = matrix[triu_indices]
        threshold_value = np.percentile(triu_values, threshold)

        thresholded_triu_values = np.where(triu_values >= threshold_value, triu_values, 0)

        thresholded_matrix = np.zeros_like(matrix)
        thresholded_matrix[triu_indices] = thresholded_triu_values

        thresholded_matrix += thresholded_matrix.T
    else:
        threshold_value = np.percentile(matrix, threshold)
        thresholded_matrix = np.where(matrix >= threshold_value, matrix, 0)
    
    return thresholded_matrix
    
def resample_matrix(
    template: Union[NDArray, List[List[float]]],
    noise: Union[str, NDArray, List[float]] = 'gaussian',
    rand_params: List[float] = [0.5, 0.1],
    ignore_repeats: bool = True,
    preserve_zeros: bool = True,
    resymmetrize: bool = True,
    seed: Optional[int] = None
) -> NDArray:
    """
    Generates a matrix of noise in the same pattern as a template.
    
    Parameters
    ----------
    template : np.ndarray
        Input template matrix.
    noise : str or np.ndarray, optional
        Type of noise ('gaussian', 'uniform', 'integers', or custom array).
    rand_params : list, optional
        Parameters for the random distribution. For 'gaussian', [mean, std]; 
        for 'uniform' and 'integers', [min, max].
    ignore_repeats : bool, optional
        Whether to ignore repeated values in the template.
    preserve_zeros : bool, optional
        Whether to preserve the template's zero entries in the output matrix.
    resymmetrize : bool, optional
        Whether to resymmetrize the noise matrix if the template is symmetric.
    seed : int, optional
        Random seed.

    Returns
    -------
    spatial_noise : np.ndarray
        The resampled noise matrix.
    """

    # Set random seed if specified
    if seed is not None:
        np.random.seed(seed)

    # Make noise
    if ignore_repeats:
        u, uc = np.unique(template, return_inverse=True)
        n_rand = len(u)
    else:
        n_rand = template.size

    if isinstance(noise, str):
        if noise == 'gaussian':
            mean, std = rand_params
            sorted_noise = np.sort(np.random.randn(n_rand) * std + mean)
        elif noise == 'uniform':
            min, max = rand_params
            sorted_noise = np.sort(np.random.rand(n_rand) * (max - min) + min)
        elif noise == 'integers':
            min, max = rand_params
            sorted_noise = np.sort(np.random.randint(min, max + 1, size=n_rand))
        else:
            raise ValueError("Invalid noise type")

    elif isinstance(noise, np.ndarray) or isinstance(noise, list):
        noise = np.asarray(noise)

        assert noise.size == n_rand, (
            f"Custom noise array must have {n_rand} elements, but has {noise.size}."
        )
        sorted_noise = np.sort(noise.flatten())
    else:
        raise ValueError("Noise input not valid")

    # Organise noise
    if ignore_repeats:
        spatial_noise = sorted_noise[uc].reshape(template.shape)
    else:
        idx = np.argsort(template.flatten())
        spatial_noise = np.zeros_like(template.flatten())
        spatial_noise[idx] = sorted_noise
        spatial_noise = spatial_noise.reshape(template.shape)

    if resymmetrize:
        if not np.allclose(template, template.T):
            warn("Template matrix is not approximately symmetric.")

        spatial_noise = (spatial_noise + spatial_noise.T) / 2

    if preserve_zeros:
        spatial_noise[template == 0] = 0

    return spatial_noise