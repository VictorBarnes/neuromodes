import os
import warnings
import importlib
import numpy as np
from joblib import Memory
from pathlib import Path
import nibabel as nib
import trimesh
from lapy import Solver, TriaMesh
from scipy.stats import zscore
from scipy.integrate import solve_ivp
import scipy.sparse
from typing import Optional, Union, Any, List

# Set up joblib memory caching
CACHE_DIR = os.getenv("CACHE_DIR")
if CACHE_DIR is None or not os.path.exists(CACHE_DIR):
    nsbtools = importlib.resources.files('nsbtools')
    CACHE_DIR = Path(nsbtools) / '.eigencache'
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
memory = Memory(CACHE_DIR, verbose=0)

class EigenSolver(Solver):
    """
    EigenSolver class for spectral analysis and simulation on surface meshes.

    This class computes the Laplace-Beltrami operator on a triangular mesh via the Finite Element
    Method, which discretizes the eigenvalue problem according to mass and stiffness matrices.
    Spatial heterogeneity and various normalization/scaling options are supported. It provides
    methods for calculating eigen-decompositions and eigen-reconstructions of data, and for
    simulating neural or BOLD activity using the Neural Field Theory wave model and
    Balloon-Windkessel model.
    """

    def __init__(
        self,
        surf: Union[str, Path, trimesh.Trimesh, TriaMesh],
        medmask: Optional[Union[np.ndarray, list]] = None,
        hetero: Optional[np.ndarray] = None,
        n_modes: int = 100,
        alpha: float = 1.0,
        beta: float = 1.0,
        r: float = 28.9,
        gamma: float = 0.116,
        scaling: str = "sigmoid",
        lump: bool = False,
        smoothit: int = 10,
        normalize: bool = False,
        verbose: bool = False
    ):
        """
        Initialize the EigenSolver class.

        Parameters
        ----------
        surf : str, pathlib.Path, or BSPolyData
            The surface mesh to be used. Can be a file path to the surface mesh or a BSPolyData 
            object.
        medmask : numpy.ndarray, optional
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
        verbose : bool, optional
            Whether to print verbose output during initialization. Default is False.

        Raises
        -------
        ValueError
            If the input mesh, mask, or parameters are not valid.
        """
        self.nmodes = n_modes
        self._r = r
        self._gamma = gamma
        self.alpha = alpha if hetero is not None else 0
        self.beta = beta if hetero is not None else 0
        self.scaling = scaling
        self.verbose = verbose

        # Initialize surface and convert to TriaMesh object
        surf = read_surf(surf)
        if medmask is not None:
            self.medmask = np.asarray(medmask, dtype=bool)
            surf = mask_surf(surf, self.medmask)
        else:
            self.medmask = None
        self.geometry = TriaMesh(surf.vertices, surf.faces)
        if normalize:
            self.geometry.normalize_()
        self.n_verts = surf.vertices.shape[0]
        self.hetero = hetero

        # Calculate the two matrices of the Laplace-Beltrami operator
        self.laplace_beltrami(lump=lump, smoothit=smoothit)

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
    def hetero(self) -> np.ndarray:
        return self._hetero

    @hetero.setter
    def hetero(self, hetero: Optional[np.ndarray]) -> None:
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
            n_expected = len(self.medmask) if self.medmask is not None else self.n_verts
            if len(hetero) != n_expected:
                raise ValueError(f"The number of elements in `hetero` ({len(hetero)}) must match "
                                 f"the number of vertices in the surface mesh ({n_expected}).")
                         
            # If medmask is provided, check only the masked portion for NaN/Inf
            if self.medmask is not None:
                if np.isnan(hetero[self.medmask]).any() or np.isinf(hetero[self.medmask]).any():
                    raise ValueError("`hetero` must not contain NaNs or Infs.")
                hetero = hetero[self.medmask]  # Then apply the mask
            else:
                # If no medmask, check the entire array
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

    def laplace_beltrami(self, lump: bool = False, smoothit: int = 10) -> None:   
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

        if self.verbose:
            if self.alpha != 0:
                print("Solving eigenvalue problem with heterogeneous Laplace-Beltrami (⍺ = "
                      f"{self.alpha})")
            else:
                print("Solving eigenvalue problem with homogeneous Laplace-Beltrami")
        u1, u2, _, _ = self.geometry.curvature_tria(smoothit=smoothit)

        hetero_mat = np.tile(hetero_tri[:, np.newaxis], (1, 2))
        self.stiffness, self.mass = self._fem_tria_aniso(self.geometry, u1, u2, hetero_mat, lump)

    def solve(self, standardize: bool = True, fix_mode1: bool = True) -> None:
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

        Raises
        ------
        AssertionError
            If the computed eigenmodes or eigenvalues contain NaN values.
        """
        # Solve the eigenvalue problem
        self.use_cholmod = False
        self.evals, emodes = self.eigs(k=self.nmodes)

        assert not np.isnan(self.evals).any(), "Eigenvalues contain NaNs."
        if abs(self.evals[0]) > 1e-6:
            warnings.warn(f"First eigenvalue is {self.evals[0]}, expected to be 0 (< 1e-6 with "
                          "precision error).")

        check_orthonorm_modes(emodes, self.mass)

        if fix_mode1:
            emodes[:, 0] = np.full(self.n_verts, 1 / np.sqrt(self.mass.sum()))
            self.evals[0] = 0.0
        if standardize:
            emodes = standardize_modes(emodes)

        self.emodes = emodes
    
    @staticmethod
    def decompose(
        data: Union[np.ndarray, List[List[float]]],
        emodes: Union[np.ndarray, List[List[float]]],
        method: str = 'orthogonal',
        mass: Optional[Union[np.ndarray, List[List[float]], scipy.sparse._csc.csc_matrix]] = None,
        return_norm_power: bool = False,
        check_orthonorm: bool = True
    ) -> np.ndarray:
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
        if isinstance(mass, list):
            mass = np.array(mass)

        if np.shape(data)[0] != np.shape(emodes)[0]:
            raise ValueError(f"The number of elements in `data` ({np.shape(data)[0]}) must match "
                             f"the number of vertices in `emodes` ({np.shape(emodes)[0]}).")
        if np.isnan(emodes).any() or np.isinf(emodes).any():
            raise ValueError("`emodes` contains NaNs or Infs.")
        if data.ndim == 1:
            data = np.expand_dims(data, axis=1)
        if check_orthonorm and mass is not None:
            check_orthonorm_modes(emodes, mass)

        if method == 'orthogonal':
            if mass is None or mass.shape != (emodes.shape[0], emodes.shape[0]):
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

    @staticmethod
    def reconstruct(
        data: Union[np.ndarray, List[List[float]]],
        emodes: Union[np.ndarray, List[List[float]]],
        method: str = 'orthogonal',
        mass: Optional[Union[np.ndarray, List[List[float]], scipy.sparse._csc.csc_matrix]] = None,
        modesq: Optional[Union[np.ndarray, list]] = None,
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
            Whether the brain maps comprise a time series of activity. Default is False.
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
        if isinstance(mass, list):
            mass = np.array(mass)

        if metric not in ["pearsonr", "mse"]:
            raise ValueError(f"Invalid metric '{metric}'; must be 'pearsonr' or 'mse'.")
        if data.ndim == 1:
            data = np.expand_dims(data, axis=1)
        if check_orthonorm and mass is not None:
            check_orthonorm_modes(emodes, mass)

        if modesq is None:
            # Use all modes if not specified (except the first constant mode)
            modesq = np.arange(1, np.shape(emodes)[1] + 1)
        nq = len(modesq)

        n_verts, n_maps = np.shape(data)

        # If data is timeseries, calculate the FC of the original data and initialize output arrays
        if timeseries:
            triu_inds = np.triu_indices(n_verts, k=1)
            fc_recon = np.empty((n_verts, n_verts, nq))
            fc_recon_score = np.empty((nq,))
            fc_orig = np.corrcoef(data)[triu_inds]
            if metric == "pearsonr":
                # Clip FC at (-1, 1), then Fisher r-to-z transform
                fc_orig_z = np.arctanh(np.clip(fc_orig, -1+1e-6, 1-1e-6))

        # Decompose the data to get beta coefficients
        if method == 'orthogonal':
            if mass is None or mass.shape != (emodes.shape[0], emodes.shape[0]):
                raise ValueError(f"Mass matrix of shape ({emodes.shape[0]}, {emodes.shape[0]}) must "
                                 "be provided when method is 'orthogonal'.")
            tmp = EigenSolver.decompose(data, emodes[:, :np.max(modesq)], mass=mass,
                                        check_orthonorm=False)
            beta = [tmp[:mq] for mq in modesq]
        else:
            beta = [
                EigenSolver.decompose(data, emodes[:, :modesq[i]], method=method)
                for i in range(nq)
            ]

        # Initialize the output arrays
        recon = np.empty((n_verts, nq, n_maps))
        recon_score = np.empty((nq, n_maps))
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

    def simulate_waves(
        self,
        ext_input: Optional[Union[np.ndarray, List[List[float]]]] = None,
        dt: float = 0.1,
        nt: int = 1000,
        bold_out: bool = False,
        eig_method: str = "orthogonal",
        pde_method: str = "fourier",
        seed: Optional[int] = None
    ) -> np.ndarray:
        """
        Simulate neural activity or BOLD signals on the surface mesh using the eigenmode 
        decomposition. Consider discarding the first 50 time points to allow the system to reach a 
        steady state. The simulation uses a Neural Field Theory wave model and optionally the
        Balloon-Windkessel model for BOLD signal generation. 

        Parameters
        ----------
        ext_input : np.ndarray, optional
            External input array of shape (n_verts, n_timepoints). If None, random input is 
            generated.
        dt : float, optional
            Time step for simulation in milliseconds. Default is 0.1.
        nt : int, optional
            Number of time points to simulate (excluding steady-state period). Default is 1000.
        bold_out : bool, optional
            If True, simulate BOLD signal using the balloon model. If False, simulate neural 
            activity. Default is False.
        eig_method : str, optional
            Method for eigen-decomposition, either "orthogonal" or "regress". Default is
            "orthogonal".
        pde_method : str, optional
            Method for solving the wave PDE. Either "fourier" or "ode". Default is "fourier".
        seed : int, optional
            Random seed for generating external input. Default is None.

        Returns
        -------
        sim_activity : np.ndarray
            Simulated neural or BOLD activity of shape (n_verts, n_timepoints), starting after the 
            steady-state period.

        Raises
        ------
        ValueError
            If the shape of ext_input does not match (n_verts, n_timepoints), or if either the
            eigen-decomposition or PDE method is invalid.
        """

        # Ensure the eigenmodes are calculated
        if not hasattr(self, 'emodes'):
            self.solve()
        
        self.t = np.linspace(0, dt * (nt - 1), nt)

        # Check if external input is provided, otherwise generate random input
        if ext_input is None:
            ext_input = gen_random_input(self.n_verts, len(self.t), seed=seed)
        # Ensure the external input has the correct shape
        if ext_input.shape != (self.n_verts, len(self.t)):
            raise ValueError(f"External input shape is {ext_input.shape}, should be "
                             f"({self.n_verts}, {len(self.t)}).")

        # Mode decomposition of external input
        input_coeffs = self.decompose(ext_input, self.emodes, method=eig_method, mass=self.mass,
                                      check_orthonorm=False)

        # Initialize simulated activity vector
        mode_coeffs = np.zeros((self.nmodes, input_coeffs.shape[1]))
        for mode_ind in range(self.nmodes):
            input_coeffs_i = input_coeffs[mode_ind, :]
            eval = self.evals[mode_ind]

            # Calculate the neural activity for the mode
            if pde_method == "fourier":
                neural = model_wave_fourier(
                    mode_coeff=input_coeffs_i, 
                    dt=dt, 
                    r=self.r, 
                    gamma=self.gamma, 
                    eval=eval
                )
            elif pde_method == "ode":            
                neural = solve_wave_ode(
                    mode_coeff=input_coeffs_i, 
                    t=self.t,
                    gamma=self.gamma,
                    r=self.r,
                    eval=eval
                )
            else:
                raise ValueError(f"Invalid PDE method '{pde_method}'; must be 'fourier' or 'ode'.")

            # If bold_out is True, calculate the BOLD signal using the balloon model
            if bold_out:
                if pde_method == "fourier":
                    bold = model_balloon_fourier(mode_coeff=neural, dt=dt)
                elif pde_method == "ode":
                    bold = model_balloon_ode(mode_coeff=neural, t=self.t)
                mode_coeffs[mode_ind, :] = bold
            else:
                mode_coeffs[mode_ind, :] = neural

        # Combine the mode activities to get the total simulated activity
        sim_activity = self.emodes @ mode_coeffs

        return sim_activity

def read_surf(surf: Union[str, Path, trimesh.Trimesh, TriaMesh]) -> trimesh.Trimesh:
    """Validate surface type and load if a file name. Returns a validated trimesh.Trimesh object."""
    if isinstance(surf, trimesh.Trimesh):
        mesh = surf
    elif isinstance(surf, TriaMesh):
        mesh = trimesh.Trimesh(vertices=surf.v, faces=surf.t)
    else:
        try:
            surf_str = str(surf)
            if surf_str.endswith('.vtk'):
                mesh_data = TriaMesh.read_vtk(surf_str)
                mesh = trimesh.Trimesh(vertices=mesh_data.v, faces=mesh_data.t)
            else:
                mesh_data = nib.load(surf_str).darrays
                mesh = trimesh.Trimesh(vertices=mesh_data[0].data, faces=mesh_data[1].data)
        except Exception as e:
            raise ValueError('Surface must be a path-like string or an instance of either '
                             'trimesh.Trimesh or lapy.TriaMesh.') from e
    
    # Validate the mesh before returning
    check_surf(mesh)

    return mesh

def mask_surf(surf: trimesh.Trimesh, medmask: Union[np.ndarray, list]) -> trimesh.Trimesh:
    """Remove medial wall vertices from the surface mesh. Returns a validated trimesh.Trimesh object."""
    if len(medmask) != surf.vertices.shape[0]:
        raise ValueError(f"The number of elements in `medmask` ({len(medmask)}) must match "
                         f"the number of vertices in the surface mesh ({surf.vertices.shape[0]}).")
    
    # Mask vertices
    v_masked = surf.vertices[medmask]
    # Map old vertex indices to new
    idx_map = np.full(len(medmask), -1, dtype=int)
    idx_map[medmask] = np.arange(np.sum(medmask))
    # Keep only faces where all vertices are in mask
    f_masked = surf.faces[np.all(medmask[surf.faces], axis=1)]
    f_masked = idx_map[f_masked]
    mesh = trimesh.Trimesh(vertices=v_masked, faces=f_masked, process=False)
    
    # Validate the masked mesh before returning
    check_surf(mesh)

    return mesh

def check_surf(surf: trimesh.Trimesh) -> None:
    """Check if the surface mesh is valid."""

    # Check for unreferenced vertices
    referenced = np.zeros(len(surf.vertices), dtype=bool)
    referenced[surf.faces] = True
    if not np.all(referenced):
        raise ValueError(f'Surface mesh contains {np.sum(~referenced)} unreferenced '
                         'vertices (i.e., not part of any face).')

    # Check if the mesh is contiguous
    components = surf.split(only_watertight=False)
    if len(components) != 1:
        raise ValueError(f'Surface mesh is not contiguous: {len(components)} connected components '
                         'found.')

def check_hetero(hetero: np.ndarray, r: float, gamma: float) -> None:
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
    hetero: Optional[np.ndarray] = None,
    alpha: float = 1.0,
    beta: float = 1.0,
    scaling: str = "sigmoid"
) -> np.ndarray:
    """
    Scales a heterogeneity map using specified normalization and scaling functions.
    
    Parameters
    ----------
    hetero : array-like, optional
        The heterogeneity map to be scaled. If None, no operation is performed.
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
        emodes: Union[np.ndarray, List[List[float]]],
        mass: Union[np.ndarray, List[List[float]], scipy.sparse._csc.csc_matrix]
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
    respect to the standard inner (dot) product (mode_i^T * mode_j = 0 for i ≠ j). Eigenmodes are also 
    expected to be mass-normal (mode_i^T * mass matrix * mode_i = 1). It follows that the first mode
    is expected to be a specific constant, but precision error during computation can introduce
    spurious spatial heterogeneity. Since many eigenmode analyses rely on mass-orthonormality (e.g.,
    decomposition, wave simulation), this function serves to ensure the validity of any calculated
    or provided eigenmodes.
    """
    emodes = np.asarray(emodes)
    if isinstance(mass, list):
        mass = np.array(mass)

    prod = emodes.T @ mass @ emodes
    if not np.allclose(prod, np.eye(prod.shape[0]), atol=1e-3):
        warnings.warn('Eigenmodes are not mass-orthonormal.')

def standardize_modes(emodes: Union[np.ndarray, List[List[float]]]) -> np.ndarray:
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

@memory.cache
def gen_random_input(
    n_verts: int,
    n_timepoints: int,
    seed: Optional[int] = None
) -> np.ndarray:
    """Generates external input with caching to avoid redundant recomputation."""
    if seed is not None:
        np.random.seed(seed)
    return np.random.randn(n_verts, n_timepoints)

def model_wave_fourier(
    mode_coeff: np.ndarray,
    dt: float,
    r: float,
    gamma: float,
    eval: float
) -> np.ndarray:
    """
    Simulates the time evolution of a wave model based on one mode using a frequency-domain 
    approach. This method applies a Fourier transform to the input mode coefficients, computes the
    system's frequency response, and then applies an inverse Fourier transform to obtain the
    time-domain response of the mode.

    Parameters
    ----------
    mode_coeff : np.ndarray
        Array of mode coefficients at each time representing the input signal to the model.
    dt : float
        Time step for the simulation in milliseconds.
    r : float
        Spatial length scale of wave propagation.
    gamma : float
        Damping rate of wave propagation.
    eval : float or array_like
        The eigenvalue associated with the mode.

    Returns
    -------
    out : ndarray
        The real part of the time-domain response of the mode at the specified time points.
    
    Notes
    -----
    This function uses a frequency-domain method to simulate the damped wave response of a causal 
    input. To ensure causality (i.e., the input is zero for t < 0), the input is zero-padded on the 
    negative time axis and transformed using `ifft`, which mimics the forward Fourier transform of a 
    causal signal. The system's frequency response (transfer function) is then applied, and `fft` is 
    used to return to the time domain. This approach is standard for simulating linear 
    time-invariant causal systems and is equivalent to convolution with a Green's function.

    The sequence is:
      1. Zero-pad input for t < 0 (causality)
      2. Take ifft to get the frequency-domain representation for this causal signal
      3. Apply the frequency response (transfer function)
      4. Use fft to return to the time domain (with appropriate shifts)
    """
    nt = len(mode_coeff) - 1
    t_full = np.linspace(-nt * dt, nt * dt, 2 * nt + 1)  # Symmetric time vector
    nt_full = len(t_full)

    # Pad input with zeros on negative side to ensure causality (system is only driven for t >= 0)
    # This is required for the correct Green's function solution of the damped wave equation.
    mode_coeff_padded = np.concatenate([np.zeros(nt), mode_coeff])

    # Frequencies for full signal
    omega = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(nt_full, d=dt))

    # Apply inverse Fourier transform to get frequency-domain representation of the causal signal.
    mode_coeff_f = np.fft.fftshift(np.fft.ifft(mode_coeff_padded))

    # Compute transfer function
    denom = -omega**2 - 2j * omega * gamma + gamma**2 * (1 + r**2 * eval)
    H = gamma**2 / denom

    # Apply frequency response
    out_fft = H * mode_coeff_f

    # Inverse transform: use fft (not ifft) to return to the time domain, matching above convention
    out_full = np.real(np.fft.fft(np.fft.ifftshift(out_fft)))

    # Return only the non-negative time part (t >= 0)
    return out_full[nt:]

def solve_wave_ode(
    mode_coeff: np.ndarray,
    t: np.ndarray,
    gamma: float,
    r: float,
    eval: float
) -> np.ndarray:
    """
    Solves the damped wave ODE for one eigenmode j.

    Parameters
    ----------
    mode_coeff : array_like
        Input drive to the system with the same length as t (written as qj in equation below).
    t : array_like
        Time points (must be increasing).
    gamma : float
        Damping coefficient.
    r : float
        Spatial length scale.
    eval : float
        Eigenvalue for the j-th mode (written as lambdaj in equation below).

    Returns
    -------
    np.ndarray
        Time evolution of phi_j(t), solution to the wave equation.
    
    Notes
    -----
    The equation is derived from the damped wave equation:
    d^2 phi_j / dt^2 + 2 * gamma * d phi_j / dt + gamma^2 * (1 + r^2 * lambdaj) * phi_j = gamma^2 * qj
    
    Rearranging gives us the first-order system
        dx1/dt = x2
        dx2/dt = -2 * gamma * x2 - gamma^2 * (1 + r^2 * lambdaj) * x1 + gamma^2 * qval
    """
    eval = float(eval)  # Ensure eval is a float

    def q_interp_safe(t_):
        """Safely interpolate the driving term at time t_."""
        val = np.interp(t_, t, mode_coeff)
        return val.item() if isinstance(val, np.ndarray) else val

    def wave_rhs(t_, y):
        """Right-hand side of the wave equation in first-order form."""
        x1, x2 = y  # both should be scalars
        qval = q_interp_safe(t_)  # should be scalar

        dx1dt = x2
        dx2dt = -2 * gamma * x2 - gamma**2 * (1 + r**2 * eval) * x1 + gamma**2 * qval

        return [dx1dt, dx2dt]

    y0 = [0.0, 0.0]

    sol = solve_ivp(
        wave_rhs,
        t_span=(t[0], t[-1]),
        y0=y0,
        t_eval=t,
        method='RK45',
        rtol=1e-6,
        atol=1e-9
    )

    return sol.y[0]  # Return phi_j(t)

def model_balloon_fourier(
    mode_coeff: np.ndarray,
    dt: float
) -> np.ndarray:       
    """
    Simulates the hemodynamic response of one mode using the balloon model in the frequency domain. 
    This method applies a frequency-domain implementation of the balloon model to a given set of 
    mode coefficients, returning the modeled hemodynamic response over time.

    Parameters
    ----------
    mode_coeff : np.ndarray
        Array of mode coefficients representing the input signal to the model.
    dt : float
        Time step for the simulation in milliseconds.

    Returns
    -------
    np.ndarray
        The real part of the time-domain response of the mode at the specified time points.

    Notes
    -----
    This function uses a frequency-domain method to simulate the damped wave response of a causal 
    input. To ensure causality (i.e., the input is zero for t < 0), the input is zero-padded on the 
    negative time axis and transformed using `ifft`, which mimics the forward Fourier transform of a 
    causal signal. The system's frequency response (transfer function) is then applied, and `fft` is 
    used to return to the time domain. This approach is standard for simulating linear 
    time-invariant causal systems and is equivalent to convolution with a Green's function.

    The sequence is:
      1. Zero-pad input for t < 0 (causality)
      2. Take ifft to get the frequency-domain representation for this causal signal
      3. Apply the frequency response (transfer function)
      4. Use fft to return to the time domain (with appropriate shifts)
    """
    # Default independent model parameters
    kappa = 0.65   # signal decay rate [s^-1]
    gamma = 0.41   # rate of elimination [s^-1]
    tau = 0.98     # hemodynamic transit time [s]
    alpha = 0.32   # Grubb's exponent [unitless]
    rho = 0.34     # resting oxygen extraction fraction [unitless]
    V0 = 0.02      # resting blood volume fraction [unitless]

    # Other parameters
    w_f = 0.56
    Q0 = 1
    rho_f = 1000
    eta = 0.3
    Xi_0 = 1
    beta = 3
    V_0 = 0.02
    k1 = 3.72
    k2 = 0.527
    k3 = 0.48
    beta = (rho + (1 - rho) * np.log(1 - rho)) / rho

    # --- Use the same causal Fourier procedure as model_wave_fourier ---
    # Zero-pad input for t < 0 (causality)
    nt = len(mode_coeff) - 1
    t_full = np.arange(-nt * dt, nt * dt + dt, dt)  # Symmetric time vector
    nt_full = len(t_full)

    mode_coeff_padded = np.concatenate([np.zeros(nt), mode_coeff])

    # Frequencies for full signal
    omega = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(nt_full, d=dt))

    # Apply inverse Fourier transform to get frequency-domain representation of the causal signal.
    mode_coeff_f = np.fft.fftshift(np.fft.ifft(mode_coeff_padded))

    # Calculate the frequency response of the system
    phi_hat_Fz = 1 / (-(omega + 1j * 0.5 * kappa) ** 2 + w_f ** 2)
    phi_hat_yF = V_0 * (alpha * (k2 + k3) * (1 - 1j * tau * omega) 
                                - (k1 + k2) * (alpha + beta - 1 
                                - 1j * tau * alpha * beta * omega)) / ((1 - 1j * tau * omega)
                                *(1 - 1j * tau * alpha * omega))
    phi_hat = phi_hat_yF * phi_hat_Fz

    # Apply frequency response
    out_fft = phi_hat * mode_coeff_f

    # Inverse transform: use fft (not ifft) to return to the time domain, matching above convention
    out_full = np.real(np.fft.fft(np.fft.ifftshift(out_fft)))

    # Return only the non-negative time part (t >= 0)
    return out_full[nt:]

def model_balloon_ode(
    mode_coeff: np.ndarray,
    t: np.ndarray
) -> np.ndarray:
    """
    Simulates the hemodynamic response of one mode using the balloon model in the time domain (ODE 
    approach). This function numerically integrates the balloon model ODEs for a given input mode 
    time course.

    Parameters
    ----------
    mode_coeff : np.ndarray
        Array of mode coefficients representing the input signal to the model (neural activity, same 
        length as t).
    t : np.ndarray
        Array of time points (must be increasing, same length as mode_coeff).

    Returns
    -------
    np.ndarray
        The BOLD signal time course for the mode at the specified time points.
    """
    # Balloon model parameters (canonical values)
    kappa = 0.65   # signal decay rate [s^-1]
    gamma_h = 0.41 # rate of elimination [s^-1]
    tau = 0.98     # hemodynamic transit time [s]
    alpha = 0.32   # Grubb's exponent [unitless]
    rho = 0.34     # resting oxygen extraction fraction [unitless]
    V0 = 0.02      # resting blood volume fraction [unitless]
    E0 = rho       # resting oxygen extraction fraction
    TE = 0.04      # echo time [s]
    k1 = 7 * E0
    k2 = 2
    k3 = 2 * E0 - 0.2

    # ODE system: y = [s, f, v, q]
    # s: vasodilatory signal, f: blood inflow, v: blood volume, q: deoxyhemoglobin content
    def balloon_rhs(t_, y):
        s, f, v, q = y
        # Interpolate neural input at current time
        u = np.interp(t_, t, mode_coeff)
        dsdt = u - kappa * s - gamma_h * (f - 1)
        dfdt = s
        dvdt = (f - v ** (1 / alpha)) / tau
        dqdt = (f * (1 - (1 - E0) ** (1 / f)) / E0 - q * v ** (1 / alpha - 1) / v) / tau
        return [dsdt, dfdt, dvdt, dqdt]

    # Initial conditions: resting state
    y0 = [0.0, 1.0, 1.0, 1.0]

    sol = solve_ivp(
        balloon_rhs,
        t_span=(t[0], t[-1]),
        y0=y0,
        t_eval=t,
        method='RK45',
        rtol=1e-6,
        atol=1e-9
    )

    s, f, v, q = sol.y
    # BOLD signal (standard formula)
    bold = V0 * (k1 * (1 - q) + k2 * (1 - q / v) + k3 * (1 - v))
    
    return bold
