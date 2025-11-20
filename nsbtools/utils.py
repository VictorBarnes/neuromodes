import numpy as np
from typing import Union, List, Optional
from warnings import warn
from numpy.typing import NDArray, ArrayLike

def unmask(
    data: ArrayLike,
    mask: ArrayLike,
    val: float = np.nan
) -> NDArray:
    """
    Unmasks data by inserting it into a full array with the same length as the medial wall mask.

    Parameters
    ----------
    data : numpy.ndarray
        The data to be unmasked. Can be 1D or 2D with shape (n_verts, n_maps).
    mask : numpy.ndarray
        A boolean array where True indicates the positions of the data in the full array.
    val : float, optional
        The value to fill in the positions outside the mask. Default is np.nan.

    Returns
    -------
    numpy.ndarray
        The unmasked data, with the same shape as the medial mask.
    """
    data = np.asarray(data)
    mask = mask.astype(bool)

    n_verts = len(mask)

    if data.ndim == 1:
        map_reshaped = np.full(n_verts, val)
        map_reshaped[mask] = data
    elif data.ndim == 2:
        nfeatures = np.shape(data)[1]
        map_reshaped = np.full((n_verts, nfeatures), val)
        map_reshaped[mask, :] = data

    return map_reshaped

def unparcellate(
    data: ArrayLike,
    parc: ArrayLike,
    val: float = np.nan
) -> NDArray:
    """
    Reconstructs a full array from parcellated data based on a parcellation map.

    Parameters
    ----------
    data : np.ndarray
        The parcellated data, where each element corresponds to a parcel. Can be 1D or 2D with shape
        (n_parcels, n_maps).
    parc : array-like
        An array indicating the parcellation ID for each vertex. A value of 0 indicates that the 
        vertex does not belong to any parcel.
    val : float, optional
        The value to assign to vertices that do not belong to any parcel, by default np.nan.

    Returns
    -------
    np.ndarray
        The reconstructed full array, where each vertex is assigned the value from the corresponding
        parcel in `data`, or `val` if it does not belong to any parcel.
    """
    data = np.asarray(data)
    parc = np.asarray(parc)

    unique_parcels = np.unique(parc)
    unique_parcels = unique_parcels[unique_parcels != 0]
    if data.shape[0] != len(unique_parcels):
        raise ValueError(f"Data length ({data.shape[0]}) does not match the number of non-zero "
                         f"parcels ({len(unique_parcels)}).")
    if data.ndim > 2:
        raise ValueError("Data must be 1D or 2D.")
    if parc.ndim != 1:
        raise ValueError("Parcellation map must be 1D.")

    data_2d = data[:, np.newaxis] if data.ndim == 1 else data

    out = np.full((len(parc), data_2d.shape[1]), val)
    for idx, parcel_id in enumerate(unique_parcels):
        out[parc == parcel_id, :] = data_2d[idx, :]

    return out.squeeze()

def threshold_matrix(
    matrix: ArrayLike,
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
    template: ArrayLike,
    noise: Union[str, ArrayLike] = 'gaussian',
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
    np.ndarray
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