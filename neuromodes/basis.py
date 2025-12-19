"""
Module for expressing brain maps as linear combinations of orthogonal basis vectors.
"""

from warnings import warn
from typing import Optional, Union, Tuple, TYPE_CHECKING
import numpy as np
from scipy.sparse import spmatrix
from scipy.spatial.distance import cdist, squareform
from neuromodes.eigen import is_mass_orthonormal_modes

if TYPE_CHECKING:
    from numpy.typing import NDArray, ArrayLike
    from scipy.spatial.distance import _MetricCallback, _MetricKind 

def decompose(
    data: ArrayLike,
    emodes: ArrayLike,
    method: str = 'project',
    mass: Optional[Union[ArrayLike, spmatrix]] = None
) -> NDArray:
    """
    Calculate the decomposition of the given data onto a basis set.

    Parameters
    ----------
    data : array-like
        The input data array of shape (n_verts, n_maps), where n_verts is the number of vertices and
        n_maps is the number of brain maps.
    emodes : array-like
        The vectors array of shape (n_verts, n_modes), where n_modes is the number of basis vectors.
    method : str, optional
        The method used for the decomposition, either 'project' to project data into a
        mass-orthonormal space or 'regress' for least-squares fitting. Note that the beta values
        from 'regress' tend towards those from 'project' when more basis vectors are provided. For a
        non-orthonormal basis set, 'regress' must be used. Default is 'project'.
    mass : array-like, optional
        The mass matrix of shape (n_verts, n_verts) used for the decomposition when method is
        'project'. If using EigenSolver, provide its self.mass. Default is None.

    Returns
    -------
    numpy.ndarray
        The beta coefficients array of shape (n_modes, n_maps), obtained from the decomposition.
    
    Raises
    ------
    ValueError
        If the number of vertices in `data` and `emodes` do not match, if `emodes` contain NaNs, or
        if an invalid method/mass matrix is specified.
    """
    # Format inputs
    data = np.asarray(data)
    data = np.expand_dims(data, axis=1) if data.ndim == 1 else data
    emodes = np.asarray(emodes)        
    if mass is not None and not isinstance(mass, spmatrix):
        mass = np.asarray(mass)

    # Check inputs (mass matrix shape checked in is_mass_orthonormal_modes later if needed)
    if data.shape[0] != emodes.shape[0]:
        raise ValueError(f"The number of elements in `data` ({data.shape[0]}) must match "
                            f"the number of vertices in `emodes` ({emodes.shape[0]}).")
    if np.isnan(emodes).any() or np.isinf(emodes).any():
        raise ValueError("`emodes` contains NaNs or Infs.")   
        
    # Decomposition
    if method == 'project':
        if is_mass_orthonormal_modes(emodes):
            if mass is not None and not is_mass_orthonormal_modes(emodes, mass):
                warn("Provided `emodes` are orthonormal in Euclidean space; ignoring mass matrix.")
            beta = emodes.T @ data
        elif mass is None:
            raise ValueError(f"Mass matrix must be provided when method is 'project' and "
                             "`emodes` is not an orthonormal basis set in Euclidean space.")
        elif not is_mass_orthonormal_modes(emodes, mass):
            raise ValueError(f"`emodes` are not mass-orthonormal (with or without provided mass " 
                                "matrix); cannot use 'project' method.")
        else:
            beta = emodes.T @ mass @ data
    elif method == 'regress':
        beta = np.linalg.lstsq(emodes, data)[0]
    else:
        raise ValueError(f"Invalid decomposition method '{method}'; must be 'project' "
                            "or 'regress'.")

    return beta

def reconstruct(
    data: ArrayLike,
    emodes: ArrayLike,
    method: str = 'project',
    mass: Optional[Union[ArrayLike, spmatrix]] = None,
    mode_counts: Optional[ArrayLike] = None,
    metric: Optional[Union['_MetricCallback', '_MetricKind']] = 'correlation'
) -> Tuple[NDArray, NDArray, list[NDArray]]:
    """
    Calculate and score the reconstruction of the given independent data using the provided
    orthogonal vectors.

    Parameters
    ----------
    data : array-like
        The input data array of shape (n_verts, n_maps), where n_verts is the number of vertices and
        n_maps is the number of maps.
    emodes : array-like
        The vectors array of shape (n_verts, n_modes), where n_modes is the number of orthogonal
        vectors.
    method : str, optional
        The method used for the decomposition, either 'project' to project data into a
        mass-orthonormal space or 'regress' for least-squares fitting. Note that the beta values
        from 'regress' tend towards those from 'project' when more basis vectors are provided. For a
        non-orthonormal basis set, 'regress' must be used. Default is 'project'.
    mass : array-like, optional
        The mass matrix of shape (n_verts, n_verts) used for the decomposition when method is
        'project'. If using EigenSolver, provide its self.mass. Default is None.
    mode_counts : array-like, optional
        The sequence of vectors to be used for reconstruction. For example,
        `mode_counts=np.array([10,20,30])` will run three analyses: with the first 10 vectors, with
        the first 20 vectors, and with the first 30 vectors. Default is None, which uses all vectors
        provided.
    metric : str, optional
        The metric used for calculating reconstruction error. Should be one of the options from
        scipy cdist, or None if no scoring is required. Default is 'correlation'.

    Returns
    -------
    recon : numpy.ndarray
        The reconstructed data array of shape (n_verts, nq, n_maps), where nq is the number of
        different reconstructions ordered in `mode_counts`. Each slice is the independent
        reconstruction of each map. Note that if `mode_counts` includes any constant vector (e.g.,
        the first geometric eigenmode), the reconstructions will be constant for that value of
        `mode_counts` (this may also result in warnings/nans for `recon_error`). 
    recon_error : numpy.ndarray
        The reconstruction error array of shape (nq, n_maps). Each value represents the
        reconstruction error of one map. If `metric` is None, this will be empty. 
    beta : list of numpy.ndarray
        A list of beta coefficients calculated for each vector.
    
    Raises
    ------
    ValueError
        If the number of vertices in `data` and `emodes` do not match, if `emodes` contain NaNs, or
        if an invalid method/mass matrix is specified.
    """

    # Format inputs
    data = np.asarray(data)
    data = np.expand_dims(data, axis=1) if data.ndim == 1 else data
    emodes = np.asarray(emodes)
    mode_counts = np.arange(emodes.shape[1])+1 if mode_counts is None else np.asarray(mode_counts)

    # Decompose the data to get beta coefficients
    if method == 'project':
        # only need to decompose once (with n=max modes) if using orthogonal method
        tmp = decompose(data, emodes[:, :np.max(mode_counts)], mass=mass, 
                          method=method)
        beta = [tmp[:mq,:] for mq in mode_counts]
    else:
        beta = [
            decompose(data, emodes[:, :mq], mass=mass, 
                        method=method)
            for mq in mode_counts
        ]

    # Reconstruct and calculate error
    recon = np.stack([emodes[:, :mode_counts[i]] @ beta[i] for i in range(len(beta))], axis=1)
    recon_error = np.stack([
        cdist(recon[:, :, i].T, data[:, [i]].T, metric=metric)
        for i in range(data.shape[1])
    ], axis=1) if metric is not None else np.empty(0)

    return recon, recon_error, beta

def reconstruct_timeseries(
    data: ArrayLike,
    emodes: ArrayLike,
    method: str = 'project',
    mass: Optional[Union[ArrayLike, spmatrix]] = None,
    mode_counts: Optional[ArrayLike] = None,
    metric: Optional[Union['_MetricCallback', '_MetricKind']] = 'correlation'
) -> Tuple[NDArray, NDArray, NDArray, NDArray, list[NDArray]]:
    """
    Calculate and score the reconstruction of the given time-series data using the provided
    orthogonal vectors.

    Parameters
    ----------
    data : array-like
        The input data array of shape (n_verts, n_timepoints), where n_verts is the number of
        vertices and n_timepoints is the number of timepoints.
    emodes : array-like
        The vectors array of shape (n_verts, n_modes), where n_modes is the number of orthogonal
        vectors.
    method : str, optional
        The method used for the decomposition, either 'project' to project data into a
        mass-orthonormal space or 'regress' for least-squares fitting. Note that the beta values
        from 'regress' tend towards those from 'project' when more basis vectors are provided. For a
        non-orthonormal basis set, 'regress' must be used. Default is 'project'.
    mass : array-like, optional
        The mass matrix of shape (n_verts, n_verts) used for the decomposition when method is
        'project'. If using EigenSolver, provide its self.mass. Default is None.
    mode_counts : array-like, optional
        The sequence of vectors to be used for reconstruction. For example, `mode_counts =
        np.array([10,20,30])` will run three analyses: with the first 10 vectors, with the first 20
        vectors, and with the first 30 vectors. Default is None, which uses all vectors provided.
    metric : str, optional
        The metric used for calculating reconstruction error. Should be one of the options from
        scipy cdist, or None if no scoring is required. Default is 'correlation'.

    Returns
    -------
    fc_recon : numpy.ndarray
        The functional connectivity reconstructed data array of shape (ne, nq). The FC matrix is
        r-to-z (arctanh) transformed and vectorized; ne is the number of edges
        (n_verts*(n_verts-1)/2) and nq is the number of different reconstructions ordered in
        `mode_counts`. Note that if `mode_counts` includes any constant vector (e.g., the first
        geometric eigenmode), the reconstructions will be constant for that value of `mode_counts`
        (this may also result in warnings/nans for `recon_error`). 
    fc_recon_error : numpy.ndarray
        The functional reconstruction accuracy of shape (nq,). If `metric` is None, this will be
        empty.
    recon : numpy.ndarray
        The reconstructed data array of shape (n_verts, nq, n_timepoints), where nq is the number of
        different reconstructions ordered in `mode_counts`. Each slice is the independent
        reconstruction of each timepoint. Note that if `mode_counts` includes any constant vector
        (e.g., the first geometric eigenmode), the reconstructions will be constant for that value
        of `mode_counts` (this may also result in warnings/nans for `recon_error`).
    recon_error : numpy.ndarray
        The reconstruction error array of shape (nq, n_timepoints). Each value represents the
        reconstruction error at one timepoint. If `metric` is None, this will be empty. 
    beta : list of numpy.ndarray
        A list of beta coefficients calculated for each vector.
    
    Raises
    ------
    ValueError
        If the number of vertices in `data` and `emodes` do not match, if `emodes` contain NaNs, or
        if an invalid method/mass matrix is specified.
    """

    # Format/check inputs    
    data = np.asarray(data)
    if data.ndim != 2: 
        raise ValueError("`data` must be a 2D array of shape (n_verts, n_timepoints).")
    
    # Use reconstruct to get independent reconstructions
    recon, recon_error, beta = reconstruct(
        data,
        emodes, 
        method=method,
        mass=mass,
        mode_counts=mode_counts,
        metric=metric
    )

    # Concatenate/get FC, reconstruct and calculate error
    calc_vec_FC = lambda x: np.arctanh(squareform(np.corrcoef(x), checks=False)) # returns 1D
    fc = calc_vec_FC(data)[np.newaxis,:] # original FC
    fc_recon = np.stack([calc_vec_FC(recon[:, i, :]) for i in range(recon.shape[1])], axis=1)
    fc_recon_error = cdist(fc_recon.T, fc, metric=metric).ravel() if metric is not None else np.empty(0)

    return fc_recon, fc_recon_error, recon, recon_error, beta


def calc_norm_power(
    beta: ArrayLike
) -> NDArray:
    """
    Transform beta coefficients from a decomposition into normalised power.

    Parameters
    ----------
    beta : array-like
        The beta coefficients array of shape (n_modes, n_maps), where n_modes is the number of 
        orthogonal vectors and n_maps is the number of brain maps.

    Returns
    -------
    numpy.ndarray
        The normalized power array of shape (n_modes, n_maps), where each element represents the 
        proportion of power contributed by the corresponding orthogonal vector to each brain map.
    """
    beta_sq = np.asarray(beta)**2
    total_power = np.sum(beta_sq, axis=0)

    return beta_sq / total_power

