"""
Module for computing geometric eigenmodes on cortical surface meshes and decomposing/reconstructing 
cortical maps.
"""

from pathlib import Path
from warnings import warn
from typing import Optional, Union, Any, Tuple, TYPE_CHECKING
import numpy as np
from numpy.typing import NDArray, ArrayLike
from scipy.stats import zscore
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import LinearOperator, eigsh, splu
from trimesh import Trimesh
from lapy import Solver, TriaMesh

from nsbtools.io import read_surf, mask_surf
from nsbtools.validation import is_mass_orthonormal_modes

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
        hetero: Optional[ArrayLike] = None,
        n_modes: int = 200,
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
        surf : str, pathlib.Path, trimesh.Trimesh, lapy.TriaMesh, or dict
            The surface mesh to be used. Can be a file path to a supported format 
            (see io.read_surf), a supported mesh object, or a dictionary with 'vertices' and 'faces'
            keys.
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
            Axonal length scale for wave propagation, used to ensure plausible wave speeds for 
            activity simulation when hetero is provided. Default is 28.9.
        gamma : float, optional
            Damping parameter for wave propagation, used to ensure plausible wave speeds for
            activity simulation when hetero is provided. Default is 0.116.
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
            Raised if any input parameter is invalid, such as negative or non-numeric values for  
            `r`, `gamma`, or `beta`, or if `n_modes` is not a positive integer.
        """
        r = float(r)
        gamma = float(gamma)
        beta = float(beta)

        if r <= 0:
            raise ValueError("`r` must be positive.")
        if gamma <= 0:
            raise ValueError("`gamma` must be positive.")
        if beta < 0:
            raise ValueError("`beta` must be non-negative.")
        if smoothit < 0 or not isinstance(smoothit, int):
            raise ValueError("`smoothit` must be a non-negative integer.")
        if n_modes <= 0 or not isinstance(n_modes, int):
            raise ValueError("`n_modes` must be a positive integer.")
        if not isinstance(lump, bool):
            raise ValueError("`lump` must be a boolean value.")
        
        self.n_modes = n_modes
        self._r = r
        self._gamma = gamma
        self.alpha = alpha if hetero is not None else 0
        self.beta = beta if hetero is not None else 0
        self.scaling = scaling
        self.smoothit = smoothit
        self.lump = lump

        # Initialize surface and convert to TriaMesh object
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
        self.hetero = hetero

        # Calculate the two matrices of the Laplace-Beltrami operator
        self.compute_lbo()

    @property
    def r(self) -> float:
        return self._r

    @r.setter
    def r(self, r: float) -> None:
        if not is_valid_hetero(hetero=self.hetero, r=r, gamma=self.gamma):
            raise ValueError(f"Alpha value results in non-physiological wave speeds above 150 m/s. "
                             "Try using a smaller alpha value.")
        self._r = r

    @property
    def gamma(self) -> float:
        return self._gamma

    @gamma.setter
    def gamma(self, gamma: float) -> None:
        if not is_valid_hetero(hetero=self.hetero, r=self.r, gamma=gamma):
            raise ValueError(f"Alpha value results in non-physiological wave speeds above 150 m/s. "
                             "Try using a smaller alpha value.")
        self._gamma = gamma

    @property
    def hetero(self) -> NDArray:
        return self._hetero

    @hetero.setter
    def hetero(self, hetero: Optional[ArrayLike]) -> None:
        # Handle None case by setting to ones
        if hetero is None:
            if self.alpha != 0 or self.beta != 0:
                warn('Setting `alpha` and `beta` to 0 because `hetero` is None.')
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

            if not is_valid_hetero(hetero=hetero, r=self.r, gamma=self.gamma):
                raise ValueError(f"Alpha value results in non-physiological wave speeds above "
                                 "150 m/s. Try using a smaller alpha value.")

            # Assign to private attribute
            self._hetero = hetero

    def compute_lbo(self) -> None:   
        """
        This method computes the Laplace-Beltrami operator using finite element methods on a
        triangular mesh, optionally incorporating spatial heterogeneity and smoothing of the
        curvature. The resulting stiffness and mass matrices are stored as attributes.
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
        atol: float = 1e-3,
        seed: Optional[Union[int, ArrayLike]] = None
    ) -> Tuple[NDArray, NDArray]:
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
        seed : int or array-like, optional
            Random seed for reproducibile generation of eigenvectors (which otherwise use an iterative 
            algorithm that starts with a random vector, meaning that repeated generation of eigenmodes 
            on the same surface can have different orientations). Specify as in int (to set the seed) 
            or a vector with n_verts elements (to directly set the initialisation vector). 
            Default is None (not reproducible).

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
        
        sigma = -0.01
        lu = splu(self.stiffness - sigma * self.mass)
        op_inv = LinearOperator( 
            matvec=lu.solve, # type: ignore
            shape=self.stiffness.shape,
            dtype=self.stiffness.dtype,
        )

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
            warn("Unfixed first eigenvalue (analytically expected to be 0) is not at least"
                          " 100 times smaller than the second.")

        if not is_mass_orthonormal_modes(self.emodes, self.mass, atol=atol):
            warn(f"Computed eigenmodes are not mass-orthonormal (atol={atol}).")

        if fix_mode1:
            self.emodes[:, 0] = np.full(self.n_verts, 1 / np.sqrt(self.mass.sum()))
            self.evals[0] = 0.0
        if standardize:
            self.emodes = standardize_modes(self.emodes)

        return self.emodes, self.evals
    
    def decompose(
        self,
        data: ArrayLike,
        method: str = 'project',
        **kwargs
    ) -> NDArray:
        """
        Calculate the decomposition of the given data onto a basis set.

        Parameters
        ----------
        data : array-like
            The input data array of shape (n_verts, n_maps), where n_verts is the number of vertices
            and n_maps is the number of brain maps.
        emodes : array-like
            The vectors array of shape (n_verts, n_vecs), where n_vecs is the number of basis vectors.
        method : str, optional
            The method used for the decomposition, either 'project' to project data into a 
            mass-orthonormal space or 'regress' for least-squares fitting. Note that the beta values
            from 'regress' tend towards those from 'project' when more basis vectors are provided. For a 
            non-orthonormal basis set, 'regress' must be used. Default is 'project'.
        mass : array-like, optional
            The mass matrix of shape (n_verts, n_verts) used for the decomposition when method
            is 'project'. If using EigenSolver, provide its self.mass. Default is None.

        Returns
        -------
        numpy.ndarray
            The beta coefficients array of shape (n_vecs, n_maps), obtained from the decomposition.
        
        Raises
        ------
        ValueError
            If the number of vertices in `data` and `emodes` do not match, if `emodes` contain NaNs,
            or if an invalid method is specified.
        """
        from nsbtools.basis import decompose

        if not hasattr(self, 'emodes'):
            _ = self.solve(**kwargs)
            print("Solved Laplace-Beltrami eigenvalue problem, "
                  "stored in self.emodes and self.evals.")
    
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
        mode_seq: Optional[ArrayLike] = None,
        metric: Optional[Union['_MetricCallback', '_MetricKind']] = 'euclidean', 
        **kwargs
    ) -> Any:
        """
        Calculate and score the reconstruction of the given data using the provided orthogonal vectors.

        Parameters
        ----------
        data : array-like
            The input data array of shape (n_verts, n_maps), where n_verts is the number of vertices 
            and n_maps is the number of maps.
        emodes : array-like
            The vectors array of shape (n_verts, n_vects), where n_vects is the number of orthogonal 
            vectors.
        method : str, optional
            The method used for the decomposition, either 'project' to project data into a 
            mass-orthonormal space or 'regress' for least-squares fitting. Note that the beta values
            from 'regress' tend towards those from 'project' when more basis vectors are provided. For a 
            non-orthonormal basis set, 'regress' must be used. Default is 'project'.
        mass : array-like, optional
            The mass matrix of shape (n_verts, n_verts) used for the decomposition when method is 
            'project'. If using EigenSolver, provide its self.mass. Default is None.
        mode_seq : array-like, optional
            The sequence of vectors to be used for reconstruction. Default is None, which uses all 
            vectors provided.
        metric : str, optional
            The metric used for calculating reconstruction error. Should be one of the options from 
            scipy cdist, or None if no scoring is required. Default is 'euclidean'.

        Returns
        -------
        recon : numpy.ndarray
            The reconstructed data array of shape (n_verts, nq, n_maps), where nq is the number of different 
            reconstructions ordered in `mode_seq`. Each slice is the independent reconstruction of each map.
        recon_error : numpy.ndarray
            The reconstruction error array of shape (nq, n_maps). Each value represents the reconstruction 
            error of one map. If `metric` is None, this will be empty. 
        beta : list of numpy.ndarray
            A list of beta coefficients calculated for each mode.
        
        Raises
        ------
        ValueError
            If the number of vertices in `data` and `emodes` do not match, if `emodes` contain NaNs,
            or if an invalid method is specified.
        """
        from nsbtools.basis import reconstruct
        
        if not hasattr(self, 'emodes'):
            _ = self.solve(**kwargs)
            print("Solved Laplace-Beltrami eigenvalue problem, "
                  "stored in self.emodes and self.evals.")
            
        return reconstruct(
            data,
            self.emodes,
            method=method,
            mass=self.mass,
            mode_seq=mode_seq,
            metric=metric
        )
    
    def reconstruct_timeseries(
        self,
        data: ArrayLike,
        method: str = 'project',
        mode_seq: Optional[ArrayLike] = None,
        metric: Optional[Union['_MetricCallback', '_MetricKind']] = 'euclidean', 
        **kwargs
    ) -> Any:
        """
        Calculate and score the reconstruction of the given data using the provided orthogonal vectors.

        Parameters
        ----------
        data : array-like
            The input data array of shape (n_verts, n_timepoints), where n_verts is the number of vertices 
            and n_timepoints is the number of timepoints.
        emodes : array-like
            The vectors array of shape (n_verts, n_vects), where n_vects is the number of orthogonal 
            vectors.
        method : str, optional
            The method used for the decomposition, either 'project' to project data into a 
            mass-orthonormal space or 'regress' for least-squares fitting. Note that the beta values
            from 'regress' tend towards those from 'project' when more basis vectors are provided. For a 
            non-orthonormal basis set, 'regress' must be used. Default is 'project'.
        mass : array-like, optional
            The mass matrix of shape (n_verts, n_verts) used for the decomposition when method is 
            'project'. If using EigenSolver, provide its self.mass. Default is None.
        mode_seq : array-like, optional
            The sequence of vectors to be used for reconstruction. Default is None, which uses all 
            vectors provided.
        metric : str, optional
            The metric used for calculating reconstruction error. Should be one of the options from 
            scipy cdist, or None if no scoring is required. Default is 'euclidean'.

        Returns
        -------
        fc_recon : numpy.ndarray
            The functional connectivity reconstructed data array of shape (ne, nq). The FC matrix is r-to-z 
            (arctanh) transformed and vectorized; ne is the number of edges (n_verts*(n_verts-1)/2) and nq is 
            the number of different reconstructions ordered in `mode_seq`.
        fc_recon_error : numpy.ndarray
            The functional reconstruction accuracy of shape (nq,). If `metric` is None, this will be empty.
        recon : numpy.ndarray
            The reconstructed data array of shape (n_verts, nq, n_timepoints), where nq is the number of different 
            reconstructions ordered in `mode_seq`. Each slice is the independent reconstruction of each timepoint.
        recon_error : numpy.ndarray
            The reconstruction error array of shape (nq, n_timepoints). Each value represents the reconstruction 
            error at one timepoint. If `metric` is None, this will be empty. 
        beta : list of numpy.ndarray
            A list of beta coefficients calculated for each mode.
        
        Raises
        ------
        ValueError
            If the number of vertices in `data` and `emodes` do not match, if `emodes` contain NaNs,
            or if an invalid method is specified.
        """
        from nsbtools.basis import reconstruct_timeseries

        if not hasattr(self, 'emodes'):
            _ = self.solve(**kwargs)
            print("Solved Laplace-Beltrami eigenvalue problem, "
                  "stored in self.emodes and self.evals.")
            
        return reconstruct_timeseries(
            data,
            self.emodes,
            method=method,
            mass=self.mass,
            mode_seq=mode_seq,
            metric=metric
        )
    
    def generate_connectome(
        self,
        r: float = 9.53,
        k: int = 108,
        **kwargs
    ) -> NDArray:
        """
        Generate a vertex-wise structural connectivity matrix using the Green's function approach
        described in Normand et al., 2025.

        Parameters
        ----------
        r : float, optional
            Spatial scale parameter for the Green's function. Default is 9.53.
        k : int, optional
            Number of eigenmodes to use. Default is 108.
        **kwargs
            Additional keyword arguments to be passed to the solve() method (standardize, fix_mode1,
            atol, seed).

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
        from nsbtools.connectome import generate_connectome

        if not hasattr(self, 'emodes'):
            _ = self.solve(**kwargs)
            print("Solved Laplace-Beltrami eigenvalue problem, "
                  "stored in self.emodes and self.evals.")

        return generate_connectome(
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
        bold_out: bool = False,
        decomp_method: str = "project",
        pde_method: str = "fourier",
        seed: Optional[int] = None,
        **kwargs
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
            The method used for the eigendecomposition, either 'project' to project data into a 
            mass-orthonormal space or 'regress' for least-squares fitting. Note that the beta values
            from 'regress' tend towards those from 'project' when EigenSolver is initialised with a 
            higher `n_modes`. For a non-orthonormal basis set, 'regress' must be used. Default is
            'project'.
        pde_method : str, optional
            Method for solving the wave PDE. Either "fourier" or "ode". Default is "fourier".
        seed : int, optional
            Random seed for generating external input. Default is None.
        **kwargs
            Additional keyword arguments to be passed to the solve() method (standardize, fix_mode1,
            atol, seed).

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
        from nsbtools.waves import simulate_waves

        if not hasattr(self, 'emodes'):
            _ = self.solve(**kwargs)
            print("Solved Laplace-Beltrami eigenvalue problem, "
                  "stored in self.emodes and self.evals.")

        return simulate_waves(
            emodes=self.emodes,
            evals=self.evals,
            r=self.r,
            gamma=self.gamma,
            ext_input=ext_input,
            dt=dt,
            nt=nt,
            mass=self.mass,
            bold_out=bold_out,
            decomp_method=decomp_method,
            pde_method=pde_method,
            seed=seed
        )

def is_valid_hetero(
    hetero: ArrayLike,
    r: float,
    gamma: float
) -> bool:
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
    hetero = np.asarray(hetero)
    max_speed = np.max(r * gamma * np.sqrt(hetero))

    return bool(max_speed <= 150)

def scale_hetero(
    hetero: ArrayLike,
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
    ndarray
        The scaled heterogeneity map.

    Raises
    ------
    ValueError
        If the scaling parameter is not a supported function.
    """
    hetero = np.asarray(hetero)

    # Z-score the heterogeneity map
    hetero_z = zscore(hetero)

    # Scale the heterogeneity map
    if scaling == "exponential":
        hetero_scaled = np.exp(alpha * hetero_z)
    elif scaling == "sigmoid":
        hetero_scaled = (2 / (1 + np.exp(-alpha * hetero_z)))**beta
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

def calc_norm_power(
    beta: ArrayLike
) -> NDArray:
    """
    Transform beta coefficients from a decomposition into normalised power.

    Parameters
    ----------
    beta : array-like
        The beta coefficients array of shape (n_vects, n_maps), where n_vects is the number of 
        orthogonal vectors and n_maps is the number of brain maps.

    Returns
    -------
    numpy.ndarray
        The normalized power array of shape (n_vects, n_maps), where each element represents the 
        proportion of power contributed by the corresponding orthogonal vector to each brain map.
    """
    beta_sq = np.asarray(beta)**2
    total_power = np.sum(beta_sq, axis=0)

    return beta_sq / total_power

