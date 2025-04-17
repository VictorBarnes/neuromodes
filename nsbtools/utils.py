import numpy as np
from pathlib import Path
from dotenv import load_dotenv


def load_project_env(marker_file=".env", override=True):
    """
    Walk upward from the location of this script to find a .env file, and load its contents into 
    the environment.

    Parameters
    ----------
    marker_file : str, optional
        The name of the marker file to look for. Default is ".env".
    override : bool, optional
        If True, override existing environment variables with the same name. Default is True.
    """

    current = Path(Path.cwd()).resolve()
    for parent in [current] + list(current.parents):
        env_path = parent / marker_file
        if env_path.exists():
            load_dotenv(dotenv_path=env_path, override=override)
            return  # Exit after loading
    raise FileNotFoundError(f"Could not find {marker_file} in any parent directory.")

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
    out = np.full(len(parc), val)
    for i in range(len(parc)):
        if parc[i] > 0:
            out[i] = data[parc[i] - 1]
    
    return out

