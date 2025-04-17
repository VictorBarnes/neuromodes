import os
import pathlib
import importlib
import numpy as np
from joblib import Memory
from pathlib import Path
from lapy import Solver, TriaMesh
from lapy.utils._imports import import_optional_dependency
from scipy.stats import zscore
from sklearn.preprocessing import QuantileTransformer
from brainspace.vtk_interface.wrappers import BSPolyData
from brainspace.mesh.mesh_operations import mask_points
from brainspace.mesh.mesh_io import read_surface
from brainspace.mesh.mesh_elements import get_cells, get_points
from heteromodes.models import WaveModel, BalloonModel
from heteromodes.utils import load_project_env

# Turn off VTK warning when using importing brainspace.mesh_operations:  
# "vtkThreshold.cxx:99 WARN| vtkThreshold::ThresholdBetween was deprecated for VTK 9.1 and will be removed in a future version."
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

@memory.cache
def gen_random_input(n_points, n_timepoints, seed=None):
    """Generates external input with caching to avoid redundant recomputation."""
    if seed is not None:
        np.random.seed(seed)
    return np.random.randn(n_points, n_timepoints)

class EigenSolver(Solver):
    """
    Class to solve the eigenvalue problem for the Laplace-Beltrami operator on a surface mesh.
    The class allows for the calculation of eigenvalues and eigenmodes, as well as the simulation
    of BOLD signals using a balloon model.
    """
    def __init__(self, surf, medmask=None, hetero=None, alpha=0, r=28.9, gamma=0.116, scaling="sigmoid", 
                 q_norm=None, lump=False, smoothit=10, normalize=False, verbose=False):
        """
        Initialize the EigenSolver class.

        Parameters
        ----------
        surf : str, pathlib.Path, or BSPolyData
            The surface mesh to be used. Can be a file path to the surface mesh or a BSPolyData object.
        medmask : numpy.ndarray, optional
            A boolean mask to exclude certain points from the surface mesh. Default is None.
        hetero : numpy.ndarray, optional
            A heterogeneity map to scale the Laplace-Beltrami operator. Default is None.
        alpha : float, optional
            Scaling factor for the heterogeneity map. Default is 0.
        r : float, optional
            Wave propagation speed parameter. Default is 28.9.
        gamma : float, optional
            Damping parameter for wave propagation. Default is 0.116.
        scaling : str, optional
            Scaling function to apply to the heterogeneity map. Must be "sigmoid" or "exponential". Default is "sigmoid".
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
        self._r = r
        self._gamma = gamma
        self.alpha = alpha
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
            if self.alpha != 0:
                # TODO: raise warning instead of print
                print("Warning: Setting `alpha` to 0 because `hetero` is None.")
                self.alpha = 0
            self._hetero = np.ones(self.surf.n_points)
        else:
            # Ensure hetero is valid
            if not isinstance(hetero, np.ndarray):
                raise ValueError("Heterogeneity map must be a numpy array or None")
            if len(hetero) != self.surf.n_points:
                raise ValueError("Heterogeneity map must have the same number of elements as the number of vertices in the surface template.")
            if np.isnan(hetero).any() or np.isinf(hetero).any():
                raise ValueError("Heterogeneity map must not contain NaNs or Infs.")

            # Scale the heterogeneity map
            hetero = self.scale_hetero(hetero=hetero, alpha=self.alpha, scaling=self.scaling, q_norm=self.q_norm)

            # Check the heterogeneity does not result in non-physiological wave speeds
            self.check_hetero(hetero=hetero, r=self.r, gamma=self.gamma)

            # Assign to private attribute
            self._hetero = hetero

    @staticmethod
    def check_hetero(hetero, r, gamma):
        # Check hmap values are physiologically plausible
        if np.max(r * gamma * np.sqrt(hetero)) > 150:
            raise ValueError("Alpha value results in non-physiological wave speeds (> 150 m/s). Try" 
                             " using a smaller alpha value.")

    @staticmethod
    def scale_hetero(hetero=None, alpha=1.0, scaling="sigmoid", q_norm=None): 
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
            hetero = 2 / (1 + np.exp(-alpha * hetero))
        else:
            raise ValueError("Invalid scaling function. Must be 'exponential' or 'sigmoid'.")

        return hetero

    def laplace_beltrami(self, lump=False, smoothit=10):        
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

    def solve(self, k=10, fix_mode1=False, standardise=False, use_cholmod=False):
        """
        Solve for eigenvalues and eigenmodes

        Parameters
        ----------

        Returns
        -------

        """

        self.use_cholmod = use_cholmod
        if self.use_cholmod:
            self.sksparse = import_optional_dependency("sksparse", raise_error=True)
            importlib.import_module(".cholmod", self.sksparse.__name__)
        else:
            self.sksparse = None
        
        # Solve the eigenvalue problem
        self.evals, evecs = self.eigs(k=k)
        
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
        
    def simulate_bold(self, ext_input=None, dt=0.1, nt=1000, tsteady=0, solver_method="Fourier", eig_method="orthogonal", seed=None):
        tmax = dt * nt + tsteady
        wave = WaveModel(self.evecs, self.evals, r=self.r, gamma=self.gamma, tstep=dt, tmax=tmax)
        self.t = wave.t
        
        # Check if external input is provided, otherwise generate random input
        if ext_input is None:
            ext_input = gen_random_input(self.surf.n_points, len(self.t), seed=seed)
        # Ensure the external input has the correct shape
        if ext_input.shape != (self.surf.n_points, len(self.t)):
            raise ValueError(f"External input shape {ext_input.shape} does not have the correct shape ({self.surf.n_points}, {len(self.t)}).")
        
        # Simulate neural activity
        _, neural = wave.solve(ext_input, solver_method, eig_method, mass=self.mass)
        
        # Simulate BOLD activity
        balloon = BalloonModel(self.evecs, tstep=dt, tmax=tmax)
        _, bold = balloon.solve(neural, solver_method, eig_method, mass=self.mass)

        # Return only the steady state part
        tsteady_ind = np.abs(self.t - tsteady).argmin()

        return bold[:, tsteady_ind:]
    
    # TODO: check if this is the proper way to do this
    def decompose(self, data, method):
        beta = calc_eigendecomposition(data, self.evecs, method=method, mass=self.mass)

        return beta

    @staticmethod
    def reconstruct():
        pass

    # TODO:
    def pangcellation():
        pass

    # TODO:
    def francis_phd():
        pass


def standardise_modes(emodes):
    """
    Perform standardisation by flipping the modes such that the first element of each mode is 
    positive.

    Parameters
    ----------
    emodes : numpy.ndarray
        The input array containing the modes.

    Returns
    -------
    numpy.ndarray
        The standardized modes with the first element of each mode set to be positive.
    """
    # Find the sign of the first non-zero element in each column
    signs = np.sign(emodes[np.argmax(emodes != 0, axis=0), np.arange(emodes.shape[1])])
    
    # Apply the sign to the modes
    standardized_modes = emodes * signs
    
    return standardized_modes

def calc_eigendecomposition(data, evecs, method='orthogonal', mass=None):
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

def calc_eigenreconstruction(data, evecs, method='orthogonal', modesq=None, mass=None, data_type="maps", metric="pearsonr", return_all=False):
    """
    Calculate the eigen-reconstruction of the given data using the provided eigenmodes.

    Parameters
    ----------
    data : array-like
        The input data array of shape (n_verts, n_data), where n_verts is the number of vertices and 
        n_data is the number of data points.
    evecs : array-like
        The eigenmodes array of shape (n_verts, n_modes), where n_modes is the number of eigenmodes.
    method : str, optional
        The method used for eigen-decomposition. Default is 'matrix'.
    modesq : array-like, optional
        The sequence of modes to be used for reconstruction. Default is None, which uses all modes.
    mass : array-like, optional
        The mass matrix used for the eigen-decomposition when method is 'orthogonal'. Default is None.
    data_type : str, optional
        The type of data, either "maps" or "timeseries". Default is "maps".
    metric : str, optional
        The metric used for calculating reconstruction accuracy. Default is "pearsonr".
    return_all : bool, optional
        Whether to return the reconstructed timepoints when data_type is "timeseries". Default is False.

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
        The functional connectivity correlation coefficients array of shape (nq,). Returned only if 
        data_type is "timeseries".
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

    # If data is timeseries, calculate the FC of the original data and initialize the output arrays
    if data_type == "timeseries":
        triu_inds = np.triu_indices(n_verts, k=1)
        fc_orig = np.corrcoef(data)[triu_inds]
        fc_recon = np.empty((n_verts, n_verts, nq))
        fc_recon_score = np.empty((nq,))

    # If method is 'orthogonal', then beta coefficients can be calucated at once
    if method == "orthogonal":
        tmp = calc_eigendecomposition(data, evecs[:, :np.max(modesq)], method=method, mass=mass)
        beta = [tmp[:mq] for mq in modesq]
    else:
        beta = [None] * nq

    # Initialize the output arrays
    recon = np.empty((n_verts, nq, n_data))
    recon_score = np.empty((nq, n_data))
    for i in range(nq):
        if method != "orthogonal":
            beta[i] = calc_eigendecomposition(data, evecs[:, :modesq[i]], method=method, mass=mass)

        # Reconstruct the data using the beta coefficients
        recon[:, i, :] = evecs[:, :modesq[i]] @ beta[i]
        if data_type == "maps":
            # Avoid division by zero
            if modesq[i] == 1:  
                recon_score[i, :] = 0
            else:
                if metric == "pearsonr":
                    recon_score[i, :] = [np.corrcoef(data[:, j], np.squeeze(recon[:, i, j]))[0, 1] for j in range(n_data)]
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
                    recon_score[i, :] = [np.corrcoef(data[:, j], np.squeeze(recon[:, i, j]))[0, 1] for j in range(n_data)]

                if metric == "pearsonr":
                    fc_recon_score[i] = np.corrcoef(
                        np.arctanh(fc_orig), 
                        np.arctanh(np.squeeze(fc_recon[:, :, i][triu_inds]))
                    )[0, 1]
                elif metric == "mse":
                    fc_recon_score[i] = np.mean((fc_orig - np.squeeze(fc_recon[:, :, i][triu_inds]))**2)
                else:
                    raise ValueError("Invalid metric; must be 'pearsonr' or 'mse'")
                
    if data_type == "timeseries":
        if return_all:
            return beta, recon, recon_score, fc_recon, fc_recon_score
        else:
            return beta, recon, fc_recon, fc_recon_score
    else:
        return beta, recon, recon_score
    