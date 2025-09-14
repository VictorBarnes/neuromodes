import numpy as np

def unmask(data, medmask, val=np.nan):
    """
    Unmasks data by inserting it into a full array with the same length as the medial mask.

    Parameters
    ----------
    data : numpy.ndarray
        The data to be unmasked (n_verts, n_data). Can be 1D or 2D.
    medmask : numpy.ndarray
        A boolean array where True indicates the positions of the data in the full array.
    val : float, optional
        The value to fill in the positions outside the mask. Default is np.nan.

    Returns
    -------
    numpy.ndarray
        The unmasked data, with the same shape as the medial mask.
    """
    medmask = medmask.astype(bool)

    if data.ndim == 1:
        nverts = len(medmask)
        map_reshaped = np.full(nverts, val)
        map_reshaped[medmask] = data
    elif data.ndim == 2:
        nverts = len(medmask)
        nfeatures = np.shape(data)[1]
        map_reshaped = np.full((nverts, nfeatures), val)
        map_reshaped[medmask, :] = data

    return map_reshaped

def unparcellate(data, parc, val=np.nan):
    """
    Reconstructs a full array from parcellated data based on a parcellation map.

    Parameters
    ----------
    data : array-like
        The parcellated data, where each element corresponds to a parcel.
    parc : array-like
        An array indicating the parcellation ID for each vertex. A value of 0
        indicates that the vertex does not belong to any parcel.
    val : scalar, optional
        The value to assign to vertices that do not belong to any parcel, by default np.nan.

    Returns
    -------
    array-like
        The reconstructed full array, where each vertex is assigned the value from
        the corresponding parcel in `data`, or `val` if it does not belong to any parcel.
    """

    unique_parcels = np.unique(parc)
    unique_parcels = unique_parcels[unique_parcels != 0]
    if len(data) != len(unique_parcels):
        raise ValueError("Data length must match the number of non-zero parcels.")

    out = np.full(len(parc), val)
    for idx, parcel_id in enumerate(unique_parcels):
        out[parc == parcel_id] = data[idx]

    return out