import numpy as np
from typing import Union, List
from numpy.typing import NDArray

def unmask(
    data: Union[NDArray, List[List[float]]],
    mask: Union[NDArray, List[bool]],
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
    data: Union[NDArray, List[List[float]], List[float]],
    parc: Union[NDArray, List[int]],
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