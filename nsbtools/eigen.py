"""
Module for computing eigenmodes and performing eigen-decomposition/reconstruction on surface meshes.
"""

import warnings
import numpy as np
from pathlib import Path
from trimesh import Trimesh
from lapy import Solver, TriaMesh
from scipy.stats import zscore
from scipy import sparse
from scipy.sparse.linalg import LinearOperator, eigsh, splu
from numpy.typing import NDArray
from typing import Optional, Union, Any, List
from nsbtools.io import read_surf, mask_surf

class EigenSolver(Solver):
    """
    EigenSolver class for spectral analysis and simulation on surface meshes.

    This class computes the Laplace-Beltrami operator on a triangular mesh via the Finite Element
    Method, which discretizes the eigenvalue problem according to mass and stiffness matrices.
    Spatial heterogeneity and various normalization/scaling options are supported.
    """

    def __init__(
        self,
        mesh: Union[str, Path, Trimesh, TriaMesh],
        mask: Optional[Union[NDArray, List[bool]]] = None,
        hetero: Optional[Union[NDArray, List[float]]] = None,
        n_modes: int = 100,
        alpha: float = 1.0,
        beta: float = 1.0,
        r: float = 28.9,
        gamma: float = 0.116,
        scaling: str = "sigmoid",
        lump: bool = False,
        smoothit: int = 10,
        normalize: bool = False
    ):
        """
        Initialize the EigenSolver class.

        Parameters
        ----------
        mesh : str, pathlib.Path, trimesh.Trimesh, or lapy.TriaMesh
            The surface mesh to be used.
        mask : array-like, optional
            A boolean mask to exclude certain points (e.g., medial wall) from the surface mesh.
            Default is None.
        hetero : array-like, optional
            A heterogeneity map to scale the Laplace-Beltrami operator. Default is None.
        n_modes : int, optional
            Number of eigenmodes to compute. Default is 100.
        alpha : float, optional
            Scaling factor for the heterogeneity map. Default is 1.0.
        beta : float, optional
            Exponent for the sigmoid scaling of the heterogeneity map. Default is 1.0.
        r : float, optional
            Axonal length scale for wave propagation. Default is 28.9.
        gamma : float, optional
            Damping parameter for wave propagation. Default is 0.116.
        scaling : str, optional
            Scaling function to apply to the heterogeneity map. Must be "sigmoid" or "exponential". 
            Default is "sigmoid".
        lump : bool, optional
            Whether to use lumped mass matrix for the Laplace-Beltrami operator. Default is False.
        smoothit : int, optional
            Number of smoothing iterations for curvature calculation. Default is 10.
        normalize : bool, optional
            Whether to normalize the surface mesh. Default is False.

        Raises
        -------
        ValueError
            If the input mesh, mask, or parameters are not valid.
        """
        if r <= 0:
            raise ValueError("`r` must be positive.")
        if gamma <= 0:
            raise ValueError("`gamma` must be positive.")
        if beta < 0:
            raise ValueError("`beta` must be non-negative.")
        if smoothit < 0:
            raise ValueError("`smoothit` must be non-negative.")
        self.n_modes = int(n_modes)
        self._r = float(r)
        self._gamma = float(gamma)
        self.alpha = float(alpha) if hetero is not None else 0
        self.beta = float(beta) if hetero is not None else 0
        self.scaling = str(scaling)
        self.smoothit = int(smoothit)
        self.lump = bool(lump)

        # Initialize surface and convert to TriaMesh object
        mesh = read_surf(mesh)
        if mask is not None:
            self.mask = np.asarray(mask, dtype=bool)
            mesh = mask_surf(mesh, self.mask)
        else:
            self.mask = None
        self.geometry = TriaMesh(mesh.vertices, mesh.faces)
        if normalize:
            self.geometry.normalize_()
        self.n_verts = mesh.vertices.shape[0]
        self.hetero = hetero

        # Calculate the two matrices of the Laplace-Beltrami operator
        self.compute_lbo()

    @property
    def r(self) -> float:
        return self._r

    @r.setter
    def r(self, r: float) -> None:
        check_hetero(hetero=self.hetero, r=r, gamma=self.gamma)
        self._r = r

    @property
    def gamma(self) -> float:
        return self._gamma

    @gamma.setter
    def gamma(self, gamma: float) -> None:
        check_hetero(hetero=self.hetero, r=self.r, gamma=gamma)
        self._gamma = gamma

    @property
    def hetero(self) -> NDArray:
        return self._hetero

    @hetero.setter
    def hetero(self, hetero: Optional[Union[NDArray, List[float]]]) -> None:
        # Handle None case by setting to ones
        if hetero is None:
            if self.alpha != 0 or self.beta != 0:
                warnings.warn('Setting `alpha` and `beta` to 0 because `hetero` is None.')
                self.alpha = 0
                self.beta = 0
            self._hetero = np.ones(self.n_verts)
        else:
            hetero = np.asarray(hetero)

            # Ensure hetero has correct length
            n_expected = len(self.mask) if self.mask is not None else self.n_verts
            if len(hetero) != n_expected:
                raise ValueError(f"The number of elements in `hetero` ({len(hetero)}) must match "
                                 f"the number of vertices in the surface mesh ({n_expected}).")
                         
            if self.mask is not None:
                hetero = hetero[self.mask]
            # Check for NaN/Inf values
            if np.isnan(hetero).any() or np.isinf(hetero).any():
                raise ValueError("`hetero` must not contain NaNs or Infs.")
                
            # Scale the heterogeneity map
            hetero = scale_hetero(
                hetero=hetero, 
                alpha=self.alpha, 
                beta=self.beta,
                scaling=self.scaling
            )

            check_hetero(hetero=hetero, r=self.r, gamma=self.gamma)

            # Assign to private attribute
            self._hetero = hetero

    def compute_lbo(self) -> None:   
        """
        This method computes the Laplace-Beltrami operator using finite element methods on a
        triangular mesh, optionally incorporating spatial heterogeneity and smoothing of the
        curvature. The resulting stiffness and mass matrices are stored as attributes.

        Parameters
        ----------
        lump : bool, optional
            If True, use lumped mass matrix. Default is False.
        smoothit : int, optional
            Number of smoothing iterations to apply to the curvature computation. Default is 10.
        """
        hetero_tri = self.geometry.map_vfunc_to_tfunc(self.hetero)

        u1, u2, _, _ = self.geometry.curvature_tria(smoothit=self.smoothit)

        hetero_mat = np.tile(hetero_tri[:, np.newaxis], (1, 2))
        self.stiffness, self.mass = self._fem_tria_aniso(self.geometry, u1, u2,
                                                         hetero_mat, self.lump)

    def solve(
        self,
        standardize: bool = True,
        fix_mode1: bool = True,
        seed: Optional[int] = None
    ) -> None:
        """
        Solve the generalized eigenvalue problem for the Laplace-Beltrami operator and compute 
        eigenvalues and eigenmodes.

        Parameters
        ----------
        standardize : bool, optional
            If True, standardizes the sign of the eigenmodes so the first element is positive. 
            Default is False.
        fix_mode1 : bool, optional
            If True, sets the first eigenmode to a constant value and the first eigenvalue to zero. 
            Default is True. See the check_orthonorm_modes function for details.
        seed : int, optional
            Random seed for reproducibility. Default is None.

        Raises
        ------
        AssertionError
            If the computed eigenmodes or eigenvalues contain NaN values.
        """
        
        sigma = -0.01
        lu = splu(self.stiffness - sigma * self.mass)
        op_inv = LinearOperator(
            matvec=lu.solve,
            shape=self.stiffness.shape,
            dtype=self.stiffness.dtype,
        )

        # Set initial vector by sampling from uniform distribution over [0, 1)
        rng = np.random.default_rng(seed)
        v0 = rng.random(self.n_verts)

        # Solve the eigenvalue problem
        self.evals, self.emodes = eigsh(
            self.stiffness,
            k=self.n_modes,
            M=self.mass,
            sigma=sigma,
            OPinv=op_inv,
            v0=v0
        )

        assert not np.isnan(self.evals).any(), "Eigenvalues contain NaNs."
        if self.evals[0] / self.evals[1] >= 0.01:
            warnings.warn("Unfixed first eigenvalue (analytically expected to be 0) is not at least"
                          " 100 times smaller than the second.")

        check_orthonorm_modes(self.emodes, self.mass)

        if fix_mode1:
            self.emodes[:, 0] = np.full(self.n_verts, 1 / np.sqrt(self.mass.sum()))
            self.evals[0] = 0.0
        if standardize:
            self.emodes = standardize_modes(self.emodes)

def check_hetero(
    hetero: NDArray,
    r: float,
    gamma: float
) -> None:
    """
    Check if the heterogeneity map values result in physiologically plausible wave speeds.
    
    Parameters
    ----------
    hetero : array_like
        Heterogeneity map values.
    r : float
        Axonal length scale for wave propagation.
    gamma : float
        Damping parameter for wave propagation.
    
    Raises
    ------
    ValueError
        If the computed wave speed exceeds 150 m/s, indicating non-physiological values.
    """
    max_speed = np.max(r * gamma * np.sqrt(hetero))
    if max_speed > 150:
        raise ValueError(f"Alpha value results in non-physiological wave speeds of {max_speed:.2f} "
                         "m/s (> 150 m/s). Try using a smaller alpha value.")

def scale_hetero(
    hetero: NDArray,
    alpha: float = 1.0,
    beta: float = 1.0,
    scaling: str = "sigmoid"
) -> NDArray:
    """
    Scales a heterogeneity map using specified normalization and scaling functions.
    
    Parameters
    ----------
    hetero : array-like
        The heterogeneity map to be scaled.
    alpha : float, optional
        Scaling parameter controlling the strength of the transformation. Default is 1.0.
    scaling : str, optional
        The scaling function to apply to the heterogeneity map, either "sigmoid" or "exponential".
        Default is "sigmoid".
    
    Returns
    -------
    hetero : ndarray
        The scaled heterogeneity map.

    Raises
    ------
    ValueError
        If the scaling parameter is not a supported function.
    """
    # Z-score the heterogeneity map
    hetero = zscore(hetero)

    # Scale the heterogeneity map
    if scaling == "exponential":
        hetero = np.exp(alpha * hetero)
    elif scaling == "sigmoid":
        hetero = (2 / (1 + np.exp(-alpha * hetero)))**beta
    else:
        raise ValueError(f"Invalid scaling '{scaling}'. Must be 'exponential' or 'sigmoid'.")

    return hetero

def check_orthonorm_modes(
    emodes: Union[NDArray, List[List[float]]],
    mass: Union[NDArray, List[List[float]], sparse._csc.csc_matrix],
    atol: float = 1e-3
) -> None:
    """
    Check if eigenmodes are approximately mass-orthonormal. Raises a warning if not.

    Parameters
    ----------
    emodes : array-like
        The eigenmodes array of shape (n_verts, n_modes), where n_modes is the number of eigenmodes.
    mass : array-like
        The mass matrix of shape (n_verts, n_verts).

    Notes
    -----
    Under discretization, the set of solutions for the generalized eigenvalue problem is expected to
    be mass-orthogonal (mode_i^T * mass matrix * mode_j = 0 for i ≠ j), rather than orthogonal with
    respect to the standard inner (dot) product (mode_i^T * mode_j = 0 for i ≠ j). Eigenmodes are
    also expected to be mass-normal (mode_i^T * mass matrix * mode_i = 1). It follows that the first
    mode is expected to be a specific constant, but precision error during computation can introduce
    spurious spatial heterogeneity. Since many eigenmode analyses rely on mass-orthonormality (e.g.,
    decomposition, wave simulation), this function serves to ensure the validity of any calculated
    or provided eigenmodes.
    """
    emodes = np.asarray(emodes)
    mass = sparse.csc_matrix(mass)

    prod = emodes.T @ mass @ emodes
    if not np.allclose(prod, np.eye(prod.shape[0]), atol=atol):
        warnings.warn('Eigenmodes are not mass-orthonormal.')

def standardize_modes(
    emodes: Union[NDArray, List[List[float]]]
) -> NDArray:
    """
    Perform standardisation by flipping the modes such that the first element of each eigenmode is 
    positive. This is helpful when visualising eigenmodes.

    Parameters
    ----------
    emodes : array-like
        The eigenmodes array of shape (n_verts, n_modes), where n_modes is the number of eigenmodes.

    Returns
    -------
    numpy.ndarray
        The standardized eigenmodes array of shape (n_verts, n_modes), with the first element of
        each mode set to be positive.
    """
    emodes = np.asarray(emodes)

    # Find the sign of the first non-zero element in each column
    signs = np.sign(emodes[np.argmax(emodes != 0, axis=0), np.arange(emodes.shape[1])])
    
    # Apply the sign to the modes
    standardized_modes = emodes * signs
    
    return standardized_modes

def decompose(
    data: Union[NDArray, List[List[float]], List[float]],
    emodes: Union[NDArray, List[List[float]]],
    method: str = 'orthogonal',
    mass: Optional[Union[NDArray, List[List[float]], sparse._csc.csc_matrix]] = None,
    return_norm_power: bool = False,
    check_orthonorm: bool = True
) -> NDArray:
    """
    Calculate the eigen-decomposition of the given data using the specified method.

    Parameters
    ----------
    data : array-like
        The input data array of shape (n_verts, n_maps), where n_verts is the number of vertices 
        and n_maps is the number of brain maps.
    emodes : array-like
        The eigenmodes array of shape (n_verts, n_modes), where n_modes is the number of 
        eigenmodes.
    method : str, optional
        The method used for the eigen-decomposition, either 'orthogonal' or 'regress'. Default is
        'orthogonal'.
    mass : array-like, optional
        The mass matrix of shape (n_verts, n_verts) used for the eigen-decomposition when method
        is 'orthogonal'. If using EigenSolver, provide its self.mass. Default is None.
    return_norm_power : bool, optional
        If True, returns normalized power of each mode instead of beta coefficients. Default is
        False.
    check_orthonorm : bool, optional
        If True and mass is not None, checks that the eigenmodes are mass-orthonormal. Default 
        is True. See the check_orthonorm_modes function for details.

    Returns
    -------
    beta : numpy.ndarray
        The beta coefficients array of shape (n_modes, n_maps), obtained from the
        eigen-decomposition.
    norm_power : numpy.ndarray, optional
        The normalized power array of shape (n_modes, n_maps), obtained from the
        eigen-decomposition.

    Raises
    ------
    ValueError
        If the number of vertices in `data` and `emodes` do not match, if `emodes` contain NaNs,
        if an invalid method is specified, or if the `mass` matrix is not provided when
        required.
    """
    data = np.asarray(data)
    emodes = np.asarray(emodes)
    if mass is not None:
        mass = sparse.csc_matrix(mass)
    if method not in ['orthogonal', 'regress']:
        raise ValueError(f"Invalid eigen-decomposition method '{method}'; must be 'orthogonal' "
                            "or 'regress'.")
    
    n_verts = emodes.shape[0]

    if data.shape[0] != n_verts:
        raise ValueError(f"The number of elements in `data` ({data.shape[0]}) must match "
                            f"the number of vertices in `emodes` ({n_verts}).")
    if np.isnan(emodes).any() or np.isinf(emodes).any():
        raise ValueError("`emodes` contains NaNs or Infs.")
    if data.ndim == 1:
        data = np.expand_dims(data, axis=1)
    if check_orthonorm and mass is not None:
        check_orthonorm_modes(emodes, mass)

    if method == 'orthogonal':
        if mass is None or mass.get_shape() != (n_verts, emodes.shape[0]):
            raise ValueError(f"Mass matrix of shape ({emodes.shape[0]}, {emodes.shape[0]}) must "
                                "be provided when method is 'orthogonal'.")
        beta = emodes.T @ mass @ data
    elif method == 'regress':
        beta = np.linalg.solve(emodes.T @ emodes, emodes.T @ data)
    else:
        raise ValueError(f"Invalid eigen-decomposition method '{method}'; must be 'orthogonal' "
                            "or 'regress'.")

    if return_norm_power:
        total_power = np.sum(beta**2, axis=0)
        norm_power = beta**2 / total_power
        return norm_power
    else:
        return beta

def reconstruct(
    data: Union[NDArray, List[List[float]], List[float]],
    emodes: Union[NDArray, List[List[float]]],
    method: str = 'orthogonal',
    mass: Optional[Union[NDArray, List[List[float]], sparse._csc.csc_matrix]] = None,
    modesq: Optional[Union[NDArray, list]] = None,
    timeseries: bool = False,
    metric: str = "pearsonr",
    return_all: bool = False,
    check_orthonorm: bool = True
) -> Any:
    """
    Calculate the eigen-reconstruction of the given data using the provided eigenmodes.

    Parameters
    ----------
    data : array-like
        The input data array of shape (n_verts, n_maps), where n_verts is the number of vertices 
        and n_maps is the number of brain maps.
    emodes : array-like
        The eigenmodes array of shape (n_verts, n_modes), where n_modes is the number of 
        eigenmodes.
    method : str, optional
        The method used for the eigen-decomposition, either 'orthogonal' or 'regress'. Default is
        'orthogonal'.
    mass : array-like, optional
        The mass matrix used for the eigen-decomposition when method is 'orthogonal'. If using
        EigenSolver, provide its self.mass. Default is None.
    modesq : array-like, optional
        The sequence of modes to be used for reconstruction. Default is None, which uses all 
        modes.
    timeseries : bool, optional
        Whether to treat brain maps as a time series of activity and reconstruct the functional
        coupling matrix. Default is False.
    metric : str, optional
        The metric used for calculating reconstruction accuracy, either "pearsonr" or "mse".
        Default is "pearsonr".
    return_all : bool, optional
        Whether to return the reconstructed timepoints when timeseries is True. Default is
        False.
    check_orthonorm : bool, optional
        If True and mass is not None, checks that the eigenmodes are mass-orthonormal. Default 
        is True. See the check_orthonorm_modes function for details.

    Returns
    -------
    beta : list of numpy.ndarray
        A list of beta coefficients calculated for each mode.
    recon : numpy.ndarray
        The reconstructed data array of shape (n_verts, nq, n_maps).
    recon_score : numpy.ndarray
        The correlation coefficients array of shape (nq, n_maps).
    fc_recon : numpy.ndarray, optional
        The functional connectivity reconstructed data array of shape (n_verts, n_verts, nq). 
        Returned only if data_type is "timeseries".
    fc_recon_score : numpy.ndarray, optional
        The functional connectivity correlation coefficients array of shape (nq,). Returned only
        if data_type is "timeseries".
    
    Raises
    ------
    ValueError
        If the number of vertices in `data` and `emodes` do not match, if `emodes` contain NaNs,
        if an invalid method or metric is specified, or if the `mass` matrix is not provided
        when required.
    """
    data = np.asarray(data)
    emodes = np.asarray(emodes)
    if mass is not None:
        mass = sparse.csc_matrix(mass)

    if metric not in ["pearsonr", "mse"]:
        raise ValueError(f"Invalid metric '{metric}'; must be 'pearsonr' or 'mse'.")
    if data.ndim == 1:
        data = np.expand_dims(data, axis=1)
    if check_orthonorm and mass is not None:
        check_orthonorm_modes(emodes, mass)

    # Use all modes if not specified (except the first constant mode)
    modesq = np.arange(1, np.shape(emodes)[1] + 1) if modesq is None else np.asarray(modesq)
    nq = len(modesq)

    n_verts, n_maps = np.shape(data)

    # If data is timeseries, calculate the FC of the original data and initialize output arrays
    if timeseries:
        triu_inds = np.triu_indices(n_verts, k=1)
        fc_recon = np.empty((n_verts, n_verts, nq), dtype=data.dtype)
        fc_recon_score = np.empty((nq,), dtype=data.dtype)
        fc_orig = np.corrcoef(data)[triu_inds]
        if metric == "pearsonr":
            # Clip FC to exclude 1s and -1s, then Fisher r-to-z transform
            fc_orig_z = np.arctanh(np.clip(fc_orig, -1+1e-6, 1-1e-6))

    # Decompose the data to get beta coefficients
    if method == 'orthogonal':
        if mass is None or mass.get_shape() != (emodes.shape[0], emodes.shape[0]):
            raise ValueError(f"Mass matrix of shape ({emodes.shape[0]}, {emodes.shape[0]}) must "
                                "be provided when method is 'orthogonal'.")
        tmp = decompose(data, emodes[:, :np.max(modesq)], mass=mass,
                                    check_orthonorm=False)
        beta = [tmp[:mq] for mq in modesq]
    else:
        beta = [
            decompose(data, emodes[:, :modesq[i]], method=method)
            for i in range(nq)
        ]

    # Initialize the output arrays
    recon = np.empty((n_verts, nq, n_maps), dtype=data.dtype)
    recon_score = np.empty((nq, n_maps), dtype=data.dtype)
    for i in range(nq):
        # Reconstruct the data using the beta coefficients
        recon[:, i, :] = emodes[:, :modesq[i]] @ beta[i]

        # Score reconstruction
        if return_all or timeseries is False:
            if metric == "pearsonr":
                recon_score[i, :] = [
                    0 if modesq[i] == 1 else np.corrcoef(data[:, j],
                                                            np.squeeze(recon[:, i, j]))[0, 1]
                    for j in range(n_maps)
                ]
            elif metric == "mse":
                recon_score[i, :] = [
                    np.mean((data[:, j] - np.squeeze(recon[:, i, j]))**2)
                    for j in range(n_maps)
                ]

        if timeseries:
            # Calculate FC from the reconstruction
            fc_recon[:, :, i] = 0 if modesq[i] == 1 else np.corrcoef(recon[:, i, :])

            # Clip FC at (-1, 1), then Fisher r-to-z transform
            fc_recon_z = np.arctanh(np.clip(fc_recon[:, :, i][triu_inds], -1+1e-6, 1-1e-6))

            # Score reconstruction of FC
            if metric == "pearsonr":
                fc_recon_score[i] = 0 if modesq[i] == 1 else np.corrcoef(
                    fc_orig_z, fc_recon_z  
                )[0, 1]
            elif metric == "mse":
                fc_recon_score[i] = np.mean(
                    (fc_orig - np.squeeze(fc_recon[:, :, i][triu_inds]))**2
                    )
                
    beta = [beta[i].squeeze() for i in range(nq)]
    recon = recon.squeeze()
    recon_score = recon_score.squeeze()

    if timeseries:
        if return_all:
            return beta, recon, recon_score, fc_recon, fc_recon_score
        else:
            return beta, recon, fc_recon, fc_recon_score
    else:
        return beta, recon, recon_score