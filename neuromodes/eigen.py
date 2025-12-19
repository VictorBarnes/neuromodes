"""
Module for computing geometric eigenmodes on cortical surface meshes and decomposing/reconstructing 
cortical maps.
"""

from pathlib import Path
from warnings import warn
from typing import Optional, Union, Any, TYPE_CHECKING
import numpy as np
from numpy.typing import NDArray, ArrayLike
from scipy import sparse
from scipy.stats import zscore
from scipy.sparse.linalg import LinearOperator, eigsh, splu
from trimesh import Trimesh
from lapy import Solver, TriaMesh
from neuromodes.io import read_surf, mask_surf

if TYPE_CHECKING:
    from scipy.spatial.distance import _MetricCallback, _MetricKind 

class EigenSolver(Solver):
    """
    EigenSolver class for spectral analysis and simulation on surface meshes.

    This class computes the Laplace-Beltrami operator on a triangular mesh via the Finite Element
    Method, which discretizes the eigenvalue problem according to mass and stiffness matrices.
    Spatial heterogeneity and various normalization/scaling options are supported.
    """

    def __init__(
        self,
        surf: Union[str, Path, Trimesh, TriaMesh, dict],
        mask: Optional[ArrayLike] = None,
        normalize: bool = False,
        hetero: Optional[ArrayLike] = None,
        scaling: Optional[str] = None, # default to "sigmoid" if hetero given (and remains None)
        alpha: Optional[float] = None  # default to 1.0 if hetero given (and remains None)
    ):
        """
        Initialize the EigenSolver class with surface, and optionally with a heterogeneity map.

        Parameters
        ----------
        surf : str, pathlib.Path, trimesh.Trimesh, lapy.TriaMesh, or dict
            The surface mesh to be used. Can be a file path to a supported format (see
            io.read_surf), a supported mesh object, or a dictionary with 'vertices' and 'faces'
            keys.
        mask : array-like, optional
            A boolean mask to exclude certain points (e.g., medial wall) from the surface mesh.
            Default is None.
        normalize : bool, optional
            Whether to normalize the surface mesh to have unit surface area and centroid at the
            origin (modifies the vertices). Default is False.
        hetero : array-like, optional
            A heterogeneity map to scale the Laplace-Beltrami operator. Default is None.
        scaling : str, optional
            Scaling function to apply to the heterogeneity map. Must be "sigmoid" or "exponential".
            If a heterogenity map is specified, the default is "sigmoid". Otherwise, this value is
            ignored (and is set to None).
        alpha : float, optional
            Scaling parameter for the heterogeneity map. If a heterogenity map is specified, the
            default is 1.0. Otherwise, this value is ignored (and is set to None). 

        Raises
        -------
        ValueError
            Raised if any input parameter is invalid, such as negative or non-numeric values for `r`
            or `gamma`.
        """
        # Surface inputs and checks (check_surf called in read_surf and mask_surf)
        surf = read_surf(surf)
        if mask is not None:
            self.mask = np.asarray(mask, dtype=bool)
            surf = mask_surf(surf, self.mask)
        else:
            self.mask = None
        self.geometry = TriaMesh(surf.vertices, surf.faces)
        if normalize:
            self.geometry.normalize_()
        self.n_verts = surf.vertices.shape[0]

        # Hetero inputs
        self._raw_hetero = hetero
        if hetero is None: # Handle None case by setting to ones
            if scaling is not None: 
                warn("`scaling` is ignored (and set to None) as `hetero` is None.")
            if alpha is not None:
                warn("`alpha` is ignored (and set to None) as `hetero` is None.")
            self._scaling = None
            self._alpha = None
            self.hetero = np.ones(self.n_verts)
        else:
            hetero = np.asarray(hetero)
            alpha = 1.0 if alpha is None else float(alpha)
            scaling = "sigmoid" if scaling is None else scaling

            # Ensure hetero has correct length (masked or unmasked)
            if len(hetero) == self.n_verts:
                pass
            elif self.mask is None: 
                raise ValueError(f"The number of elements in `hetero` ({len(hetero)}) must match "
                                f"the number of vertices in the surface mesh ({self.n_verts}).")
            elif len(hetero) == len(self.mask):
                hetero = hetero[self.mask]
            else:
                raise ValueError(f"The number of elements in `hetero` ({len(hetero)}) must match "
                                f"the number of vertices in the surface mesh ({self.n_verts}) "
                                f"or the surface mask (of size {len(self.mask)}).") 

            # Scale and assign the heterogeneity map
            self._scaling = scaling    
            self._alpha = alpha
            self.hetero = scale_hetero(
                hetero=hetero, 
                alpha=self._alpha, 
                scaling=self._scaling
            )

    def compute_lbo(
        self, 
        lump: bool = False,
        smoothit: int = 10
    ) -> "EigenSolver":
        """
        This method computes the Laplace-Beltrami operator using finite element methods on a
        triangular mesh, optionally incorporating spatial heterogeneity and smoothing of the
        curvature. The resulting stiffness and mass matrices are stored as attributes.

        Parameters
        ----------
        lump : bool, optional
            Whether to use lumped mass matrix for the Laplace-Beltrami operator. Default is False.
        smoothit : int, optional
            Number of smoothing iterations for curvature calculation. Default is 10.

        Raises
        ------
        ValueError
            If `smoothit` is negative or not an integer.
        """
        if smoothit < 0 or not isinstance(smoothit, int):
            raise ValueError("`smoothit` must be a non-negative integer.")

        u1, u2, _, _ = self.geometry.curvature_tria(smoothit)

        hetero_tri = self.geometry.map_vfunc_to_tfunc(self.hetero)
        hetero_mat = np.tile(hetero_tri[:, np.newaxis], (1, 2))

        self.stiffness, self.mass = self._fem_tria_aniso(self.geometry, u1, u2,
                                                         hetero_mat, lump)
        return self

    def solve(
        self,
        n_modes: int, 
        standardize: bool = True,
        fix_mode1: bool = True,
        atol: float = 1e-3,
        sigma: float = -0.01,
        seed: Optional[Union[int, ArrayLike]] = None, 
        **kwargs
    ) -> "EigenSolver":
        """
        Solve the generalized eigenvalue problem for the Laplace-Beltrami operator and compute
        eigenvalues and eigenmodes.

        Parameters
        ----------
        standardize : bool, optional
            If True, standardizes the sign of the eigenmodes so the first element is positive.
            Default is False.
        fix_mode1 : bool, optional
            If True, sets the first eigenmode to a constant value and the first eigenvalue to zero,
            as is expected analytically. Default is True. See the is_mass_orthonormal_modes function
            for details.
        atol : float, optional
            Absolute tolerance for mass-orthonormality validation. Default is 1e-3.
        sigma : float, optional
            Shift-invert parameter to speed up the computation of eigenvalues close to this value.
            Default is -0.01.
        seed : int or array-like, optional
            Random seed for reproducibile generation of eigenvectors (which otherwise use an
            iterative algorithm that starts with a random vector, meaning that repeated generation
            of eigenmodes on the same surface can have different orientations). Specify as in int
            (to set the seed) or a vector with n_verts elements (to directly set the initialisation
            vector). Default is None (not reproducible).
        **kwargs
            Additional keyword arguments passed to compute_lbo (lump, smoothit).

        Returns
        -------
        numpy.ndarray
            The computed eigenmodes.
        numpy.ndarray
            The computed eigenvalues.

        Raises
        ------
        ValueError
            If `seed` is an array but does not have the correct shape of (n_verts,).
        AssertionError
            If any computed eigenvalues are NaN.
        """
        # Validate inputs
        if n_modes <= 0 or not isinstance(n_modes, int):
            raise ValueError("`n_modes` must be a positive integer.")
        
        for key in kwargs:
            if key not in {"lump", "smoothit"}:
                raise ValueError(f"Invalid keyword argument: {key}")

        # Compute the Laplace-Beltrami operator / set stiffness and mass matrices
        self.compute_lbo(**kwargs)
        
        # Set intitialization vector (if desired) for reproducibile eigenvectors 
        if seed is None or isinstance(seed, int):
            rng = np.random.default_rng(seed)
            v0 = rng.random(self.n_verts)
        else:
            v0 = np.asarray(seed)
            if v0.shape != (self.n_verts,):
                raise ValueError("`seed` must be either an integer or an array-like of shape "
                                f"({self.n_verts},).")

        # Solve the eigenvalue problem
        lu = splu(self.stiffness - sigma * self.mass)
        op_inv = LinearOperator( 
            matvec=lu.solve, # type: ignore
            shape=self.stiffness.shape,
            dtype=self.stiffness.dtype,
        )

        self.n_modes = n_modes
        self.evals, self.emodes = eigsh(
            self.stiffness,
            k=self.n_modes,
            M=self.mass,
            sigma=sigma,
            OPinv=op_inv,
            v0=v0
        )

        # Validate results
        assert not np.isnan(self.evals).any(), "Eigenvalues contain NaNs."
        if self.evals[0] / self.evals[1] >= 0.01:
            warn("Unfixed first eigenvalue (analytically expected to be 0) is not at least"
                          " 100 times smaller than the second.")

        if not is_mass_orthonormal_modes(self.emodes, self.mass, atol=atol):
            warn(f"Computed eigenmodes are not mass-orthonormal (atol={atol}).")

        # Post-process
        if fix_mode1:
            # Value given by mass-orthonormality condition
            self.emodes[:, 0] = np.full(self.n_verts, 1 / np.sqrt(self.mass.sum()))
            self.evals[0] = 0.0

        if standardize:
            self.emodes = standardize_modes(self.emodes)

        return self
    
    def decompose(
        self,
        data: ArrayLike,
        method: str = 'project'
    ) -> NDArray:
        """
        Calculate the decomposition of the given data onto a basis set.

        Parameters
        ----------
        data : array-like
            The input data array of shape (n_verts, n_maps), where n_verts is the number of vertices
            and n_maps is the number of brain maps.
        method : str, optional
            The method used for the decomposition, either 'project' to project data into a
            mass-orthonormal space or 'regress' for least-squares fitting. Note that the beta values
            from 'regress' tend towards those from 'project' when more basis vectors are provided.
            For a non-orthonormal basis set, 'regress' must be used. Default is 'project'.
        mass : array-like, optional

        Returns
        -------
        numpy.ndarray
            The beta coefficients array of shape (n_modes, n_maps), obtained from the decomposition.
        
        Raises
        ------
        ValueError
            If the number of vertices in `data` and `emodes` do not match, if `emodes` contain NaNs,
            or if an invalid method is specified.
        """
        from neuromodes.basis import decompose

        if not hasattr(self, 'emodes'):
            raise ValueError("Eigenmodes not found. Please run the solve() method first.")
    
        return decompose(
            data,
            self.emodes,
            method=method,
            mass=self.mass
        )
    
    def reconstruct(
        self,
        data: ArrayLike,
        method: str = 'project',
        mode_counts: Optional[ArrayLike] = None,
        metric: Optional[Union['_MetricCallback', '_MetricKind']] = 'correlation'
    ) -> Any:
        """
        Calculate and score the reconstruction of the given data using the provided orthogonal
        vectors.

        Parameters
        ----------
        data : array-like
            The input data array of shape (n_verts, n_maps), where n_verts is the number of vertices
            and n_maps is the number of maps.
        method : str, optional
            The method used for the decomposition, either 'project' to project data into a
            mass-orthonormal space or 'regress' for least-squares fitting. Note that the beta values
            from 'regress' tend towards those from 'project' when more basis vectors are provided.
            For a non-orthonormal basis set, 'regress' must be used. Default is 'project'.
        mode_counts : array-like, optional
            The sequence of vectors to be used for reconstruction. For example,
            `mode_counts=np.asarray([10,20,30])` will run three analyses: with the first 10 modes,
            with the first 20 modes, and with the first 30 modes. Default is None, which uses all
            vectors provided.
        metric : str, optional
            The metric used for calculating reconstruction error. Should be one of the options from
            scipy cdist, or None if no scoring is required. Default is 'correlation'.

        Returns
        -------
        recon : numpy.ndarray
            The reconstructed data array of shape (n_verts, nq, n_maps), where nq is the number of
            different reconstructions ordered in `mode_counts`. Each slice is the independent
            reconstruction of each map.
        recon_error : numpy.ndarray
            The reconstruction error array of shape (nq, n_maps). Each value represents the
            reconstruction error of one map. If `metric` is None, this will be empty. 
        beta : list of numpy.ndarray
            A list of beta coefficients calculated for each mode.
        
        Raises
        ------
        ValueError
            If the number of vertices in `data` and `emodes` do not match, if `emodes` contain NaNs,
            or if an invalid method/mass matrix is specified.
        """
        from neuromodes.basis import reconstruct
        
        if not hasattr(self, 'emodes'):
            raise ValueError("Eigenmodes not found. Please run the solve() method first.")
            
        return reconstruct(
            data,
            self.emodes,
            method=method,
            mass=self.mass,
            mode_counts=mode_counts,
            metric=metric
        )
    
    def reconstruct_timeseries(
        self,
        data: ArrayLike,
        method: str = 'project',
        mode_counts: Optional[ArrayLike] = None,
        metric: Optional[Union['_MetricCallback', '_MetricKind']] = 'correlation'
    ) -> Any:
        """
        Calculate and score the reconstruction of the given data using the provided orthogonal
        vectors.

        Parameters
        ----------
        data : array-like
            The input data array of shape (n_verts, n_timepoints), where n_verts is the number of
            vertices and n_timepoints is the number of timepoints.
        method : str, optional
            The method used for the decomposition, either 'project' to project data into a
            mass-orthonormal space or 'regress' for least-squares fitting. Note that the beta values
            from 'regress' tend towards those from 'project' when more basis vectors are provided.
            For a non-orthonormal basis set, 'regress' must be used. Default is 'project'.
        mode_counts : array-like, optional
            The sequence of vectors to be used for reconstruction. For example, `mode_counts =
            np.asarray([10,20,30])` will run three analyses: with the first 10 modes, with the first
            20 modes, and with the first 30 modes. Default is None, which uses all vectors provided.
        metric : str, optional
            The metric used for calculating reconstruction error. Should be one of the options from
            scipy cdist, or None if no scoring is required. Default is 'correlation'.

        Returns
        -------
        fc_recon : numpy.ndarray
            The functional connectivity reconstructed data array of shape (ne, nq). The FC matrix is
            r-to-z (arctanh) transformed and vectorized; ne is the number of edges
            (n_verts*(n_verts-1)/2) and nq is the number of different reconstructions ordered in
            `mode_counts`.
        fc_recon_error : numpy.ndarray
            The functional reconstruction accuracy of shape (nq,). If `metric` is None, this will be
            empty.
        recon : numpy.ndarray
            The reconstructed data array of shape (n_verts, nq, n_timepoints), where nq is the
            number of different reconstructions ordered in `mode_counts`. Each slice is the
            independent reconstruction of each timepoint.
        recon_error : numpy.ndarray
            The reconstruction error array of shape (nq, n_timepoints). Each value represents the
            reconstruction error at one timepoint. If `metric` is None, this will be empty. 
        beta : list of numpy.ndarray
            A list of beta coefficients calculated for each mode.
        
        Raises
        ------
        ValueError
            If the number of vertices in `data` and `emodes` do not match, if `emodes` contain NaNs,
            or if an invalid method is specified.
        """
        from neuromodes.basis import reconstruct_timeseries

        if not hasattr(self, 'emodes'):
            raise ValueError("Eigenmodes not found. Please run the solve() method first.")
            
        return reconstruct_timeseries(
            data,
            self.emodes,
            method=method,
            mass=self.mass,
            mode_counts=mode_counts,
            metric=metric
        )
    
    def model_connectome(
        self,
        r: float = 9.53,
        k: int = 108
    ) -> NDArray:
        """
        Generate a vertex-wise structural connectivity matrix using the Green's function approach
        described in Normand et al., 2025.

        Parameters
        ----------
        k : int, optional
            Number of eigenmodes to use. Default is 108.

        Returns
        -------
        numpy.ndarray
            The generated vertex-wise structural connectivity matrix.

        Raises
        ------
        ValueError
            If any input parameter is invalid, such as negative or non-numeric values for  
            `r`, or if `k` is not a positive integer within the valid range.

        Notes
        -----
        If comparing this model to empirical connectomes, consider thresholding the generated 
        connectome to match the density of the empirical data.
        """
        from neuromodes.connectome import model_connectome

        if not hasattr(self, 'emodes'):
            raise ValueError("Eigenmodes not found. Please run the solve() method first.")

        return model_connectome(
            emodes=self.emodes,
            evals=self.evals,
            r=r,
            k=k
        )
    
    def simulate_waves(
        self,
        ext_input: Optional[ArrayLike] = None,
        dt: float = 0.1,
        nt: int = 1000,
        r: float = 28.9,
        gamma: float = 0.116,
        bold_out: bool = False,
        decomp_method: str = "project",
        pde_method: str = "fourier",
        seed: Optional[int] = None
    ) -> NDArray:
        """
        Simulate neural activity or BOLD signals on the surface mesh using the eigenmode
        decomposition. The simulation uses a Neural Field Theory wave model and optionally the
        Balloon-Windkessel model for BOLD signal generation. 

        Parameters
        ----------
        ext_input : array-like, optional
            External input array of shape (n_verts, n_timepoints). If None, random input is
            generated.
        dt : float, optional
            Time step for simulation in milliseconds. Default is 0.1.
        nt : int, optional
            Number of time points to simulate Default is 1000.
        bold_out : bool, optional
            If True, simulate BOLD signal using the balloon model. If False, simulate neural
            activity. Default is False.
        decomp_method : str, optional
            The method used for the decomposition, either 'project' to project data into a
            mass-orthonormal space or 'regress' for least-squares fitting. Note that the beta
            coefficients from 'regress' tend towards those from 'project' when more basis vectors
            are provided. For a non-orthonormal basis set, 'regress' must be used. Default is
            'project'.
        pde_method : str, optional
            Method for solving the wave PDE. Either "fourier" or "ode". Default is "fourier".
        seed : int, optional
            Random seed for generating external input. Default is None.

        Returns
        -------
        numpy.ndarray
            Simulated neural or BOLD activity of shape (n_verts, n_timepoints).

        Raises
        ------
        ValueError
            If the shape of ext_input does not match (n_verts, n_timepoints), or if either the
            eigen-decomposition or PDE method is invalid.

        Notes
        -----
        Since the simulation begins at rest, consider discarding the first 50 timepoints to allow
        the system to reach a steady state.
        """
        from neuromodes.waves import simulate_waves

        if np.max(estimate_wave_speed(self.hetero, r, gamma)) > 150.0:
            warn("The combination of heterogeneity, r, and gamma may lead to "
                 "non-physiological wave speeds (>150 m/s). Consider adjusting "
                 "these parameters.")

        if not hasattr(self, 'emodes'):
            raise ValueError("Eigenmodes not found. Please run the solve() method first.")

        return simulate_waves(
            emodes=self.emodes,
            evals=self.evals,
            r=r,
            gamma=gamma,
            ext_input=ext_input,
            dt=dt,
            nt=nt,
            mass=self.mass,
            bold_out=bold_out,
            decomp_method=decomp_method,
            pde_method=pde_method,
            seed=seed
        )

def estimate_wave_speed(
    hetero: ArrayLike,
    r: float,
    gamma: float
) -> NDArray:
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
    
    Returns
    -------
    float
        Estimation of the speed at each vertex.
    """
    hetero = np.asarray(hetero)
    return r * gamma * np.sqrt(hetero)

def scale_hetero(
    hetero: ArrayLike,
    alpha: float = 1.0,
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
    ndarray
        The scaled heterogeneity map.

    Raises
    ------
    ValueError
        If the scaling parameter is not a supported function.
    """
    # z-score the heterogeneity map
    hetero = np.asarray(hetero)
    if np.any(np.isnan(hetero)) or np.any(np.isinf(hetero)):
        raise ValueError("`hetero` must not contain NaNs or Infs; check input values.")

    hetero_z = zscore(hetero)
    if np.any(np.isnan(hetero_z)):
        raise ValueError("z-scored `hetero` must not contain NaNs; check input values.")

    if alpha == 0:
        warn("`alpha` is set to 0, meaning heterogeneity map will have no effect.")

    # Scale the heterogeneity map
    if scaling == "exponential":
        hetero_scaled = np.exp(alpha * hetero_z)
    elif scaling == "sigmoid":
        hetero_scaled = (2 / (1 + np.exp(-alpha * hetero_z)))
    else:
        raise ValueError(f"Invalid scaling '{scaling}'. Must be 'exponential' or 'sigmoid'.")

    return hetero_scaled

def standardize_modes(
    emodes: ArrayLike
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

def is_mass_orthonormal_modes(
    emodes: ArrayLike,
    mass: Optional[Union[ArrayLike,sparse.spmatrix]] = None,
    rtol: float = 1e-05, atol: float = 1e-03
) -> bool:
    """
    Check if a set of eigenmodes is approximately mass-orthonormal (i.e., `emodes.T @ mass @ emodes
    == I`).

    Parameters
    ----------
    emodes : array-like
        The eigenmodes array of shape (n_verts, n_modes), where n_modes is the number of modes.
    mass : array-like, optional
        The mass matrix of shape (n_verts, n_verts). If using EigenSolver, provide its self.mass. If
        None, an identity matrix will be used, corresponding to Euclidean orthonormality. Default is
        None.
    atol : float, optional
        Absolute tolerance for the orthonormality check. Default is 1e-3.

    Notes
    -----
    Under discretization, the set of solutions for the generalized eigenvalue problem is expected to
    be mass-orthogonal (mode_i^T * mass matrix * mode_j = 0 for i ≠ j), rather than orthogonal with
    respect to the standard Euclidean inner (dot) product (mode_i^T * mode_j = 0 for i ≠ j).
    Eigenmodes are also expected to be mass-normal (mode_i^T * mass matrix * mode_i = 1). It follows
    that the first mode is expected to be a specific constant, but precision error during
    computation can introduce spurious spatial heterogeneity. Since many eigenmode analyses rely on
    mass-orthonormality (e.g., decomposition, wave simulation), this function serves to ensure the
    validity of any calculated or provided eigenmodes.
    """
    # Format inputs
    emodes = np.asarray(emodes)
    if mass is not None and not isinstance(mass,sparse.spmatrix):
        mass = np.asarray(mass)

    # Check inputs (ie mass matrix shape)
    n_verts = emodes.shape[0]
    if mass is not None and (mass.shape != (n_verts, n_verts)):
        raise ValueError(f"The mass matrix must have shape ({n_verts}, {n_verts}).")

    prod = emodes.T @ emodes if mass is None else emodes.T @ mass @ emodes
    return np.allclose(prod, np.eye(emodes.shape[1]), rtol=rtol, atol=atol, equal_nan=False)