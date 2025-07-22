import os
import pathlib
import importlib
import numpy as np
import scipy as sp
from joblib import Memory
from pathlib import Path
from lapy import Solver, TriaMesh
from lapy.utils._imports import import_optional_dependency
from scipy.stats import zscore
from scipy.integrate import solve_ivp
from sklearn.preprocessing import QuantileTransformer
from brainspace.vtk_interface.wrappers import BSPolyData
from brainspace.mesh.mesh_operations import mask_points
from brainspace.mesh.mesh_io import read_surface
from brainspace.mesh.mesh_elements import get_cells, get_points
from nsbtools.utils import load_project_env

# Turn off VTK warning when using importing brainspace.mesh_operations:  
# "vtkThreshold.cxx:99 WARN| vtkThreshold::ThresholdBetween was deprecated for VTK 9.1 and will be 
# removed in a future version."
import vtk
vtk.vtkObject.GlobalWarningDisplayOff()

# Set up joblib memory caching
load_project_env()
CACHE_DIR = os.getenv("CACHE_DIR")
if CACHE_DIR is None or not os.path.exists(CACHE_DIR):
    CACHE_DIR = Path.cwd()
memory = Memory(Path(CACHE_DIR), verbose=0)


def _check_surf(surf):
    """Validate surface type and load if a file name. Adapted from `surfplot`."""
    if isinstance(surf, (str, pathlib.Path)):
        return read_surface(str(surf))
    elif isinstance(surf, BSPolyData) or (surf is None):
        return surf
    else:
        raise ValueError('Surface be a path-like string, an instance of '
                        'BSPolyData, or None')


class EigenSolver(Solver):
    """
    EigenSolver class for spectral analysis and simulation on surface meshes.

    This class computes the Laplace-Beltrami operator on a triangular mesh, and supports spatial 
    heterogeneity and various normalization/scaling options. It provides methods for calculating 
    eigen-decompositions and eigen-reconstructions of data, and for simulating neural or BOLD 
    activity using the Neural Field Theory wave model and Balloon-Windkessel model.
    """

    def __init__(self, surf, medmask=None, hetero=None, n_modes=100, alpha=1.0, beta=1.0, r=28.9, gamma=0.116, 
                 scaling="sigmoid", q_norm=None, lump=False, smoothit=10, normalize=False, 
                 verbose=False):
        """
        Initialize the EigenSolver class.

        Parameters
        ----------
        surf : str, pathlib.Path, or BSPolyData
            The surface mesh to be used. Can be a file path to the surface mesh or a BSPolyData 
            object.
        medmask : numpy.ndarray, optional
            A boolean mask to exclude certain points from the surface mesh. Default is None.
        hetero : numpy.ndarray, optional
            A heterogeneity map to scale the Laplace-Beltrami operator. Default is None.
        n_modes : int, optional
            Number of eigenmodes to compute. Default is 100.
        alpha : float, optional
            Scaling factor for the heterogeneity map. Only used if `hetero` is not None. Default is 
            1.0.
        beta : float, optional
            Exponent for the sigmoid scaling of the heterogeneity map. Only used if `hetero` is not 
            None. Default is 1.0.
        r : float, optional
            Axonal length scale for wave propagation. Default is 28.9.
        gamma : float, optional
            Damping parameter for wave propagation. Default is 0.116.
        scaling : str, optional
            Scaling function to apply to the heterogeneity map. Must be "sigmoid" or "exponential". 
            Default is "sigmoid".
        q_norm : str, optional
            Distribution type for quantile normalization of the heterogeneity map. Default is None.
        lump : bool, optional
            Whether to use lumped mass matrix for the Laplace-Beltrami operator. Default is False.
        smoothit : int, optional
            Number of smoothing iterations for curvature calculation. Default is 10.
        normalize : bool, optional
            Whether to normalize the surface mesh. Default is False.
        verbose : bool, optional
            Whether to print verbose output during initialization. Default is False.
        """
        self.nmodes = n_modes
        self._r = r
        self._gamma = gamma
        self.alpha = alpha if hetero is not None else 0
        self.beta = beta if hetero is not None else 0
        self.scaling = scaling
        self.q_norm = q_norm
        self.verbose = verbose

        # Initialise surface and convert to TriaMesh object
        surf = _check_surf(surf)
        if medmask is not None:
            surf = mask_points(surf, medmask)
            # TODO: check for floating points after mask and return modified mask and print warning
            if hetero is not None:
                hetero = hetero[medmask]
        self.geometry = TriaMesh(get_points(surf), get_cells(surf))
        if normalize:
            self.geometry.normalize_()
        self.surf = surf  
        self.hetero = hetero

        # Calculate the two matrices of the Laplace-Beltrami operator
        self.laplace_beltrami(lump=lump, smoothit=smoothit)

    @property
    def r(self):
        return self._r

    @r.setter
    def r(self, r):
        self.check_hetero(hetero=self.hetero, r=r, gamma=self.gamma)
        self._r = r

    @property
    def gamma(self):
        return self._gamma

    @gamma.setter
    def gamma(self, gamma):
        self.check_hetero(hetero=self.hetero, r=self.r, gamma=gamma)
        self._gamma = gamma

    @property
    def hetero(self):
        return self._hetero

    @hetero.setter
    def hetero(self, hetero):
        # Handle None case by setting to ones
        if hetero is None:
            if self.alpha != 0 or self.beta != 0:
                # TODO: raise warning instead of print
                print("Warning: Setting `alpha` and `beta` to 0 because `hetero` is None.")
                self.alpha = 0
                self.beta = 0
            self._hetero = np.ones(self.surf.n_points)
        else:
            # Ensure hetero is valid
            if not isinstance(hetero, np.ndarray):
                raise ValueError("Heterogeneity map must be a numpy array or None")
            if len(hetero) != self.surf.n_points:
                raise ValueError("Heterogeneity map must have the same number of elements as the "
                                 "number of vertices in the surface template.")
            if np.isnan(hetero).any() or np.isinf(hetero).any():
                raise ValueError("Heterogeneity map must not contain NaNs or Infs.")

            # Scale the heterogeneity map
            hetero = self.scale_hetero(
                hetero=hetero, 
                alpha=self.alpha, 
                beta=self.beta,
                scaling=self.scaling, 
                q_norm=self.q_norm
            )

            # Check the heterogeneity does not result in non-physiological wave speeds
            self.check_hetero(hetero=hetero, r=self.r, gamma=self.gamma)

            # Assign to private attribute
            self._hetero = hetero

    @staticmethod
    def check_hetero(hetero, r, gamma):
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
        
        # Check hmap values are physiologically plausible
        if np.max(r * gamma * np.sqrt(hetero)) > 150:
            raise ValueError("Alpha value results in non-physiological wave speeds (> 150 m/s). Try" 
                             " using a smaller alpha value.")

    @staticmethod
    def scale_hetero(hetero=None, alpha=1.0, beta=1.0, scaling="sigmoid", q_norm=None):
        """
        Scales a heterogeneity map using specified normalization and scaling functions.
        
        Parameters
        ----------
        hetero : array-like, optional
            The heterogeneity map to be scaled. If None, no operation is performed.
        alpha : float, default=1.0
            Scaling parameter controlling the strength of the transformation.
        scaling : {'sigmoid', 'exponential'}, default='sigmoid'
            The scaling function to apply to the heterogeneity map.
            - 'sigmoid': Applies a scaled sigmoid transformation.
            - 'exponential': Applies an exponential transformation.
        q_norm : {'uniform', 'normal'}, optional
            If specified, applies quantile normalization to the heterogeneity map.
            - 'uniform': Maps data to a uniform distribution.
            - 'normal': Maps data to a normal distribution.
        
        Returns
        -------
        hetero : ndarray
            The scaled heterogeneity map.
        """

        # Z-score the heterogeneity map
        hetero = zscore(hetero)

        # Apply quantile normalisation
        if q_norm is not None:
            scaler = QuantileTransformer(output_distribution=q_norm, random_state=0)
            hetero = scaler.fit_transform(hetero.reshape(-1, 1)).flatten()

        # Scale the heterogeneity map
        if scaling == "exponential":
            hetero = np.exp(alpha * hetero)
        elif scaling == "sigmoid":
            hetero = (2 / (1 + np.exp(-alpha * hetero)))**beta
        else:
            raise ValueError("Invalid scaling function. Must be 'exponential' or 'sigmoid'.")

        return hetero

    def laplace_beltrami(self, lump=False, smoothit=10):   
        """
        This method computes the Laplace-Beltrami operator using finite element methods
        on a triangular mesh, optionally incorporating spatial heterogeneity and smoothing
        of the curvature. The resulting stiffness and mass matrices are stored as attributes.

        Parameters
        ----------
        lump : bool, optional
            If True, use lumped mass matrix. Default is False.
        smoothit : int, optional
            Number of smoothing iterations to apply to the curvature computation. Default is 10.
        """

        hetero_tri = self.geometry.map_vfunc_to_tfunc(self.hetero)
        # Check that the length of the heterogeneity map matches the number of triangles
        if len(hetero_tri) != self.geometry.t.shape[0]:
            raise ValueError(f"Wrong hetero length: {len(hetero_tri)}. Should be: "
                                f"{self.geometry.t.shape[0]}")

        # heterogneous Laplace
        if self.verbose:
            print("TriaMesh with heterogeneous Laplace-Beltrami")
        u1, u2, _, _ = self.geometry.curvature_tria(smoothit=smoothit)

        hetero_mat = np.tile(hetero_tri[:, np.newaxis], (1, 2))
        self.stiffness, self.mass = self._fem_tria_aniso(self.geometry, u1, u2, hetero_mat, lump)

    def solve(self, fix_mode1=False, standardise=False, use_cholmod=False):
        """
        Solve the generalized eigenvalue problem for the Laplace-Beltrami operator and compute 
        eigenvalues and eigenmodes.

        Parameters
        ----------
        k : int, optional
            Number of eigenvalues and eigenmodes to compute. Default is 10.
        fix_mode1 : bool, optional
            If True, sets the first eigenmode to a constant value. Default is False.
        standardise : bool, optional
            If True, standardizes the sign of the eigenmodes so the first element is positive. 
            Default is False.
        use_cholmod : bool, optional
            If True, uses the CHOLMOD solver from sksparse for the eigenvalue problem. Default is 
            False.

        Raises
        ------
        AssertionError
            If the computed eigenmodes contain NaN values.
        """

        self.use_cholmod = use_cholmod
        if self.use_cholmod:
            self.sksparse = import_optional_dependency("sksparse", raise_error=True)
            importlib.import_module(".cholmod", self.sksparse.__name__)
        else:
            self.sksparse = None
        
        # Solve the eigenvalue problem
        self.evals, evecs = self.eigs(k=self.nmodes)
        
        # Set first mode to be constant (mean of first column)
        if fix_mode1:
            evecs[:, 0] = np.mean(evecs[:, 0])

        # Standardise sign of modes
        if standardise:
            evecs = standardise_modes(evecs)

        # Check for NaNs
        if np.isnan(evecs).any():
            raise AssertionError("`evecs` contain NaNs") 

        self.evecs = evecs
    
    @staticmethod
    def decompose(data, evecs, method='orthogonal', mass=None):
        """
        Calculate the eigen-decomposition of the given data using the specified method.

        Parameters
        ----------
        data : array-like
            The input data for the eigen-decomposition.
        evecs : array-like
            The eigenmodes used for the eigen-decomposition.
        method : str, optional
            The method used for the eigen-decomposition. Default is 'matrix'.
        mass : array-like, optional
            The mass matrix used for the eigen-decomposition when method is 'orthogonal'. Default is 
            None.

        Returns
        -------
        beta : numpy.ndarray of shape (n_modes, n_data)
            The beta coefficients obtained from the eigen-decomposition.
        """
        if not np.allclose(evecs[:, 0], np.full_like(evecs[:, 0], evecs[0, 0])):
            print("Warning: `evecs` should contain a constant eigenvector.")

        # Solve the linear system to get the beta coefficients
        if method == 'matrix':
            beta = np.linalg.solve((evecs.T @ evecs), (evecs.T @ data))
        elif method == 'orthogonal':
            if mass is None:
                raise ValueError("B must be specified when method is 'orthogonal'")

            beta = evecs.T @ mass @ data
        else:
            raise ValueError("Invalid method; must be 'matrix' or 'orthogonal'.")

        return beta

    @staticmethod
    def reconstruct(data, evecs, method='orthogonal', modesq=None, mass=None, data_type="maps", 
                    metric="pearsonr", return_all=False):
        """
        Calculate the eigen-reconstruction of the given data using the provided eigenmodes.

        Parameters
        ----------
        data : array-like
            The input data array of shape (n_verts, n_data), where n_verts is the number of vertices 
            and n_data is the number of data points.
        evecs : array-like
            The eigenmodes array of shape (n_verts, n_modes), where n_modes is the number of 
            eigenmodes.
        method : str, optional
            The method used for eigen-decomposition. Default is 'matrix'.
        modesq : array-like, optional
            The sequence of modes to be used for reconstruction. Default is None, which uses all 
            modes.
        mass : array-like, optional
            The mass matrix used for the eigen-decomposition when method is 'orthogonal'. Default is 
            None.
        data_type : str, optional
            The type of data, either "maps" or "timeseries". Default is "maps".
        metric : str, optional
            The metric used for calculating reconstruction accuracy. Default is "pearsonr".
        return_all : bool, optional
            Whether to return the reconstructed timepoints when data_type is "timeseries". Default 
            is False.

        Returns
        -------
        beta : list of numpy.ndarray
            A list of beta coefficients calculated for each mode.
        recon : numpy.ndarray
            The reconstructed data array of shape (n_verts, nq, n_data).
        recon_score : numpy.ndarray
            The correlation coefficients array of shape (nq, n_data).
        fc_recon : numpy.ndarray, optional
            The functional connectivity reconstructed data array of shape (n_verts, n_verts, nq). 
            Returned only if data_type is "timeseries".
        fc_recon_score : numpy.ndarray, optional
            The functional connectivity correlation coefficients array of shape (nq,). Returned only
            if data_type is "timeseries".
        """

        if np.shape(data)[0] != np.shape(evecs)[0]:
            raise ValueError("The number of vertices in `data` and `evecs` must be the same.")
        if method == "orthogonal" and mass is None:
            raise ValueError("B must be specified when method is 'orthogonal'")
        if metric not in ["pearsonr", "mse"]:
            raise ValueError("Invalid metric; must be 'pearsonr' or 'mse'.")
        if data_type not in ["maps", "timeseries"]:
            raise ValueError("Invalid data_type; must be 'maps' or 'timeseries'.")
        
        # Get the number of vertices and data points
        n_verts, n_data = np.shape(data)

        if modesq is None:
            # Use all modes if not specified (except the first constant mode)
            modesq = np.arange(1, np.shape(evecs)[1] + 1)
        nq = len(modesq)

        # If data is timeseries, calculate the FC of the original data and initialize output arrays
        if data_type == "timeseries":
            triu_inds = np.triu_indices(n_verts, k=1)
            fc_orig = np.corrcoef(data)[triu_inds]
            fc_recon = np.empty((n_verts, n_verts, nq))
            fc_recon_score = np.empty((nq,))

        # If method is 'orthogonal', then beta coefficients can be calucated at once
        if method == "orthogonal":
            tmp = EigenSolver.decompose(data, evecs[:, :np.max(modesq)], method=method, mass=mass)
            beta = [tmp[:mq] for mq in modesq]
        else:
            beta = [None] * nq

        # Initialize the output arrays
        recon = np.empty((n_verts, nq, n_data))
        recon_score = np.empty((nq, n_data))
        for i in range(nq):
            if method != "orthogonal":
                beta[i] = EigenSolver.decompose(data, evecs[:, :modesq[i]], method=method, mass=mass)

            # Reconstruct the data using the beta coefficients
            recon[:, i, :] = evecs[:, :modesq[i]] @ beta[i]
            if data_type == "maps":
                # Avoid division by zero
                if modesq[i] == 1:  
                    recon_score[i, :] = 0
                else:
                    if metric == "pearsonr":
                        recon_score[i, :] = [
                            np.corrcoef(data[:, j], np.squeeze(recon[:, i, j]))[0, 1] 
                            for j in range(n_data)
                        ]
                    elif metric == "mse":
                        recon_score[i, :] = np.mean((data - np.squeeze(recon[:, i, :]))**2, axis=0)
                    else:
                        raise ValueError("Invalid metric; must be 'pearsonr' or 'mse'")

            # Calculate FC of the reconstructed data
            elif data_type == "timeseries":
                # Calculate the functional connectivity of the reconstructed data
                fc_recon[:, :, i] = np.corrcoef(recon[:, i, :])
                
                # Avoid division by zero
                if modesq[i] == 1:
                    if return_all:
                        recon_score[i, :] = 0
                    fc_recon_score[i] = 0
                else:
                    if return_all:
                        recon_score[i, :] = [
                            np.corrcoef(data[:, j], np.squeeze(recon[:, i, j]))[0, 1] 
                            for j in range(n_data)
                        ]

                    if metric == "pearsonr":
                        fc_recon_score[i] = np.corrcoef(
                            np.arctanh(fc_orig), 
                            np.arctanh(np.squeeze(fc_recon[:, :, i][triu_inds]))
                        )[0, 1]
                    elif metric == "mse":
                        fc_recon_score[i] = np.mean(
                            (fc_orig - np.squeeze(fc_recon[:, :, i][triu_inds]))**2
                        )
                    else:
                        raise ValueError("Invalid metric; must be 'pearsonr' or 'mse'")
                    
        if data_type == "timeseries":
            if return_all:
                return beta, recon, recon_score, fc_recon, fc_recon_score
            else:
                return beta, recon, fc_recon, fc_recon_score
        else:
            return beta, recon, recon_score


    def simulate_waves(self, ext_input=None, dt=0.1, nt=1000, tsteady=0, eig_method="orthogonal", 
                       pde_method="fourier", seed=None, bold_out=False):
        """
        Simulate neural activity or BOLD signals on the surface mesh using the eigenmode 
        decomposition.

        Parameters
        ----------
        ext_input : np.ndarray, optional
            External input array of shape (n_points, n_timepoints). If None, random input is 
            generated.
        dt : float, optional
            Time step for simulation in milliseconds. Default is 0.1.
        nt : int, optional
            Number of time points to simulate (excluding steady-state period). Default is 1000.
        tsteady : float, optional
            Duration of steady-state period (in milliseconds) before simulation starts. Default is 
            0.
        eig_method : str, optional
            Method for eigen-decomposition. Either "orthogonal" or "matrix". Default is 
            "orthogonal".
        seed : int, optional
            Random seed for generating external input. Default is None.
        bold_out : bool, optional
            If True, simulate BOLD signal using the balloon model. If False, simulate neural 
            activity. Default is False.

        Returns
        -------
        sim_activity : np.ndarray
            Simulated neural or BOLD activity of shape (n_points, n_timepoints), starting after the 
            steady-state period.

        Raises
        ------
        ValueError
            If the shape of ext_input does not match (n_points, n_timepoints).
        """
        # Ensure the eigenmodes are calculated
        if not hasattr(self, 'evecs'):
            self.solve(fix_mode1=True)

        self.dt = dt
        self.tmax = self.dt * nt + tsteady
        self.t = np.arange(0, self.tmax + self.dt, self.dt)
        tsteady_ind = np.abs(self.t - tsteady).argmin() # index of the steady state time point

        # Check if external input is provided, otherwise generate random input
        if ext_input is None:
            ext_input = gen_random_input(self.surf.n_points, len(self.t), seed=seed)
        # Ensure the external input has the correct shape
        if ext_input.shape != (self.surf.n_points, len(self.t)):
            raise ValueError(f"External input shape {ext_input.shape} does not have the correct "
                              "shape ({self.surf.n_points}, {len(self.t)}).")

        # Mode decomposition of external input
        input_coeffs = self.decompose(ext_input, self.evecs, method=eig_method, mass=self.mass)

        # Initialize simulated activity vector
        mode_coeffs = np.zeros((self.nmodes, input_coeffs.shape[1]))
        for mode_ind in range(self.nmodes):
            input_coeffs_i = input_coeffs[mode_ind, :]
            eval = self.evals[mode_ind]

            # Calculate the neural activity for the mode
            if pde_method == "fourier":
                neural = model_wave_fourier(
                    mode_coeff=input_coeffs_i, 
                    dt=self.dt, 
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
                    lambdaj=eval
                )
            else:
                raise ValueError("Invalid PDE method; must be 'fourier' or 'ode'.")
            
            # If bold_out is True, calculate the BOLD signal using the balloon model
            if bold_out:
                if pde_method == "fourier":
                    bold = model_balloon_fourier(mode_coeff=neural, dt=self.dt)
                elif pde_method == "ode":
                    bold = model_balloon_ode(mode_coeff=neural, t=self.t)
                else:
                    raise ValueError("Invalid PDE method; must be 'fourier' or 'ode'.")
                mode_coeffs[mode_ind, :] = bold
            else:
                mode_coeffs[mode_ind, :] = neural

        # Combine the mode activities to get the total simulated activity
        sim_activity = self.evecs @ mode_coeffs

        return sim_activity[:, tsteady_ind:]


    # TODO:
    def pangcellation():
        pass

    # TODO:
    def francis_phd():
        pass


def standardise_modes(emodes):
    """
    Perform standardisation by flipping the modes such that the first element of each eigenmode is 
    positive. This is helpful when visualising eigenmodes.

    Parameters
    ----------
    emodes : numpy.ndarray
        The input array containing the modes.

    Returns
    -------
    numpy.ndarray
        The standardized eigenmodes with the first element of each mode set to be positive.
    """
    # Find the sign of the first non-zero element in each column
    signs = np.sign(emodes[np.argmax(emodes != 0, axis=0), np.arange(emodes.shape[1])])
    
    # Apply the sign to the modes
    standardized_modes = emodes * signs
    
    return standardized_modes


@memory.cache
def gen_random_input(n_points, n_timepoints, seed=None):
    """Generates external input with caching to avoid redundant recomputation."""
    if seed is not None:
        np.random.seed(seed)
    return np.random.randn(n_points, n_timepoints)


def model_wave_fourier(mode_coeff, dt, r, gamma, eval):
    """
    Simulates the time evolution of a wave model based on one mode using a frequency-domain 
    approach. This method applies a Fourier transform to the input mode coefficients, computes 
    the system's frequency response, and then applies an inverse Fourier transform to obtain the
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
    We use ifft (not fft) on the zero-padded input to obtain the correct frequency-domain 
    representation for a causal system (i.e., one where the input is zero for t < 0). This matches 
    the analytic solution for the time evolution of a damped wave equation with a causal input (see 
    e.g. convolution with a Green's function). Using fft here would not yield the correct result for
    this setup.
    
    The sequence is:
      1. Zero-pad input for t < 0 (causality)
      2. Take ifft to get the frequency-domain representation for this causal signal
      3. Apply the frequency response (transfer function)
      4. Use fft to return to the time domain (with appropriate shifts)
    
    This approach is standard in some signal processing and physics literature for causal systems.
    """

    nt = len(mode_coeff) - 1
    t_full = np.arange(-nt * dt, nt * dt + dt, dt)  # Symmetric time vector
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


def solve_wave_ode(mode_coeff, t, gamma, r, lambdaj):
    """
    Solves the damped wave ODE for one eigenmode j.

    Parameters
    ----------
    mode_coeff : array_like
        Input drive to the system with the same length as t (written as qj in equation below.)
    t : array_like
        Time points (must be increasing).
    gamma : float
        Damping coefficient.
    r : float
        Spatial length scale.
    lambdaj : float
        Eigenvalue for the j-th mode.

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
        dx2/dt = -2 * gamma * x2 - gamma^2 * (1 + r**2 * lambdaj) * x1 + gamma**2 * qval
    """

    lambdaj = float(lambdaj)  # Ensure lambdaj is a float

    def q_interp_safe(t_):
        val = np.interp(t_, t, mode_coeff)
        return val.item() if isinstance(val, np.ndarray) else val

    def wave_rhs(t_, y):
        x1, x2 = y  # both should be scalars
        qval = q_interp_safe(t_)  # should be scalar

        dx1dt = x2
        dx2dt = -2 * gamma * x2 - gamma**2 * (1 + r**2 * lambdaj) * x1 + gamma**2 * qval

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

def model_balloon_fourier(mode_coeff, dt):       
    """
    Simulates the hemodynamic response of one mode using the balloon model in the frequency 
    domain. This method applies a frequency-domain implementation of the balloon model to a 
    given set of mode coefficients, returning the modeled hemodynamic response over time.

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
    We use ifft (not fft) on the zero-padded input to obtain the correct frequency-domain 
    representation for a causal system (i.e., one where the input is zero for t < 0). This matches 
    the analytic solution for the time evolution of a damped wave equation with a causal input (see 
    e.g. convolution with a Green's function). Using fft here would not yield the correct result for
    this setup.
    
    The sequence is:
      1. Zero-pad input for t < 0 (causality)
      2. Take ifft to get the frequency-domain representation for this causal signal
      3. Apply the frequency response (transfer function)
      4. Use fft to return to the time domain (with appropriate shifts)
    
    This approach is standard in some signal processing and physics literature for causal systems.
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

def model_balloon_ode(mode_coeff, t):
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
    from scipy.integrate import solve_ivp
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
