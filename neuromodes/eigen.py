"""
Module for computing geometric eigenmodes on cortical surface meshes and decomposing/reconstructing 
cortical maps.
"""

from __future__ import annotations
from pathlib import Path
from typing import Union, Tuple, TYPE_CHECKING
from warnings import warn
from lapy import Solver, TriaMesh
from lapy.diffgeo import tria_compute_gradient
import numpy as np
from scipy.stats import zscore
from scipy.sparse import spmatrix
from scipy.sparse.linalg import LinearOperator, eigsh, splu
from trimesh import Trimesh
from neuromodes.io import read_surf, mask_surf, check_surf

if TYPE_CHECKING:
    from numpy.typing import NDArray, ArrayLike

class EigenSolver(Solver):
    """
    EigenSolver class for spectral analysis and simulation on surface meshes.

    This class computes the Laplace-Beltrami operator (LBO) [1] on a triangular mesh via 
    the Finite Element Method [2][3], which discretizes the eigenvalue problem according 
    to mass and stiffness matrices. It also provides an option to incorporate spatial 
    heterogeneity via the generalised LBO [4].

    Parameters
    ----------
    surf : str, pathlib.Path, trimesh.Trimesh, lapy.TriaMesh, or dict
        The surface mesh to be used. Can be a file path to a supported format (see
        `io.read_surf`), a supported mesh object, or a dictionary with `'vertices'` and
        `'faces'` keys.
    mask : array-like, optional
        A boolean mask to exclude certain points (e.g., medial wall) from the surface mesh.
        Vertices labelled as `False` in the mask will be excluded. Default is `None`.
    normalize : bool, optional
        Whether to normalize the surface mesh to have unit surface area and centroid at the
        origin (modifies the vertices). Default is `False`.
    hetero : array-like, optional
        Heterogeneity map to incorporate regional heterogeneity into the Laplace-Beltrami 
        operator. Default is `None`.
    alpha : float, optional
        Scaling parameter for the heterogeneity map. If a heterogenity map is specified, the
        default is `1.0`. Otherwise, this value is ignored (and is set to `None`).
    scaling : str, optional
        Scaling function to apply to the heterogeneity map. Must be `'sigmoid'` or
        `'exponential'`. If a heterogenity map is specified, the default is `'sigmoid'`.
        Otherwise, this value is ignored (and is set to `None`).
    aniso : array-like, optional
        Anisotropy map to incorporate regional anisotropy into the Laplace-Beltrami 
        operator. Default is `None`.
    beta: float, optional
        Anisotropy ratio parameter controlling directional bias of diffusion. When beta=0, 
        diffusion is isotropic. Positive beta enhances diffusion parallel to the gradient of 
        `aniso` while reducing it perpendicular to the gradient. Negative beta does the 
        opposite: enhances diffusion perpendicular to the gradient while reducing it parallel. 
        
    Raises
    ------
    ValueError
        If `hetero` length does not match the number of vertices (masked or unmasked).
    ValueError
        If `scaling` is not 'sigmoid' or 'exponential' (raised by `scale_hetero`).
    ValueError
        If `hetero` is constant (raised by `scale_hetero`).

    References
    ----------
    ..  [1] Pang, J. C., et al. (2023). Geometric constraints on human brain function. Nature.
        https://doi.org/10.1038/s41586-023-06098-1
    ..  [2] Reuter, M., et al. (2006). Laplace-Beltrami spectra as 'Shape-DNA' of surfaces and
        solids, Computer-Aided Design. https://doi.org/10.1016/j.cad.2005.10.011
    ..  [3] Wachinger, C., et al. (2015). BrainPrint: a discriminative characterization of brain
        morphology, Neuroimage. https://doi.org/10.1016/j.neuroimage.2015.01.032
    ..  [4] Barnes, V., et al. (2026). Regional heterogeneity shapes macroscopic wave dynamics of
        the human and non-human primate cortex. bioRxiv. https://doi.org/10.64898/2026.01.22.701178
    """
    def __init__(
        self,
        surf: Union[str, Path, Trimesh, TriaMesh, dict],
        mask: Union[ArrayLike, None] = None,
        normalize: bool = False,
        hetero: Union[ArrayLike, None] = None,
        alpha: Union[float, None] = None, # default to 1.0 if hetero given (and remains None)
        scaling: Union[str, None] = None,  # default to "sigmoid" if hetero given (and remains None)
        aniso: Union[ArrayLike, None] = None,
        beta: Union[float, None] = None
    ):
        # Surface inputs and checks
        surf = read_surf(surf)
        if mask is not None:
            self.mask = np.asarray(mask, dtype=bool)
            surf = mask_surf(surf, self.mask)
        else:
            self.mask = None
        check_surf(surf)
        
        # Convert to lapy TriaMesh and normalize if desired
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
            if hetero.shape == (self.n_verts,):
                pass
            elif self.mask is not None and hetero.shape == (len(self.mask),):
                hetero = hetero[self.mask]
            else:
                err_str = f"the number of vertices in the surface mesh ({self.n_verts})"
                if self.mask is not None:
                    err_str += f" or the masked surface mesh (of size {self.mask.sum()})"
                raise ValueError(
                    f"`hetero` must be a 1D array with length matching {err_str}."
                )

            # Scale and assign the heterogeneity map
            self._scaling = scaling    
            self._alpha = alpha
            self.hetero = scale_hetero(
                hetero=hetero, 
                alpha=self._alpha, 
                scaling=self._scaling
            )

        # Aniso inputs
        if aniso is None:
            if beta is not None:
                warn("`beta` is ignored as `aniso` is None.")
            self.aniso = None
            self._beta = None
        else:
            aniso = np.asarray(aniso)
            beta = 1.0 if beta is None else float(beta)

            # Ensure aniso has correct length (masked or unmasked)
            if mask is not None and aniso.shape == (len(mask),):
                aniso = aniso[mask]
            elif aniso.shape != (self.n_verts,):
                err_str = f"the number of vertices in the surface mesh ({self.n_verts})"
                if self.mask is not None:
                    err_str += f" or the masked surface mesh (of size {self.mask.sum()})"
                raise ValueError(
                    f"`aniso` must be a 1D array with length matching {err_str}."
                )
            
            aniso_check = np.asarray_chkfinite(aniso)   # check for NaNs and infs
            self.aniso = (aniso_check - np.mean(aniso_check)) / np.std(aniso_check)
            self._beta = beta


    def __str__(self) -> str:
        """String representation of the EigenSolver object."""
        str_out = f'EigenSolver\n-----------\nSurface mesh: {self.n_verts} vertices'
        if self.mask is not None:
            str_out += f' ({np.sum(self.mask == 0)} vertices masked out)'
        if self._raw_hetero is not None:
            str_out += f'\nHeterogeneity map scaling: {self._scaling} (alpha={self._alpha})'
        str_out += f'\n{self.n_modes if hasattr(self, "n_modes") else "No"} eigenmodes computed'

        return str_out

    def compute_lbo(
        self, 
        lump: bool = False,
        smoothit: int = 10,
    ) -> EigenSolver:
        """
        This method computes the Laplace-Beltrami operator using finite element methods on a
        triangular mesh, optionally incorporating spatial heterogeneity and smoothing of the
        curvature. The resulting `stiffness` and `mass` matrices are stored as attributes.

        Parameters
        ----------
        lump : bool, optional
            Whether to use lumped mass matrix for the Laplace-Beltrami operator. Default is `False`.
        smoothit : int, optional
            Number of smoothing iterations for curvature calculation. Default is 10.

        Returns
        -------
        EigenSolver
            The EigenSolver instance.

        Notes
        -----
        When anisotropy is applied, the gradient components are cross-coupled with the tangent
        basis directions due to the FEM formulation. Specifically, the gradient component in the 
        u1 direction scales diffusion in the u2 direction, and vice versa. This corresponds to 
        the tensor transformation R^T D R where R = [0,-1; 1,0], a standard rotation required 
        by the finite element discretization. Positive beta enhances diffusion parallel to the 
        gradient direction, while negative beta enhances it perpendicular to the gradient.
        
        The anisotropic diffusion tensor uses a symmetric exponential formulation which 
        preserves the determinant (geometric mean = 1). This decouples the effects of 
        hetero (overall diffusion magnitude) and beta (directional bias).
        """
        if self.hetero is None and self.aniso is None:
            stiffness, mass = self._fem_tria(self.geometry, lump)
        else:
            if self.hetero is not None:
                hetero_tri = self.geometry.map_vfunc_to_tfunc(self.hetero)

            u1, u2, _, _ = self.geometry.curvature_tria(smoothit=smoothit)
            
            if self.aniso is not None:
                # Compute 3D gradient at each triangle
                grad_3d = tria_compute_gradient(self.geometry, self.aniso)
                
                # Project gradient onto local (u1, u2) coordinates
                grad_u1 = np.sum(grad_3d * u1, axis=1)  # component in u1 direction
                grad_u2 = np.sum(grad_3d * u2, axis=1)  # component in u2 direction
                
                # Compute gradient magnitude and normalize to [0, 1] range
                grad_mag = np.sqrt(grad_u1**2 + grad_u2**2)
                grad_mag_norm = grad_mag / np.maximum(grad_mag.max(), 1e-10)
                
                # Compute unit gradient directions (avoid division by zero)
                grad_u1_unit = grad_u1 / np.maximum(grad_mag, 1e-10)
                grad_u2_unit = grad_u2 / np.maximum(grad_mag, 1e-10)
                
                # Symmetric exponential anisotropy: preserves determinant (aniso_u1 * aniso_u2 = 1)
                # FEM cross-coupling: grad_u2 component scales u1 diffusion (R^T D R, R=[0,-1;1,0])
                aniso_u1 = np.exp(self.beta * grad_mag_norm * (grad_u2_unit**2 - grad_u1_unit**2) / 2)
                aniso_u2 = np.exp(self.beta * grad_mag_norm * (grad_u1_unit**2 - grad_u2_unit**2) / 2)

                # Scale by heterogeneity to control overall diffusion magnitude
                aniso_mat = np.column_stack([
                    hetero_tri * aniso_u1,  # when beta=0, this becomes hetero_tri * 1
                    hetero_tri * aniso_u2   # when beta=0, this becomes hetero_tri * 1
                ])
            else:
                # Isotropic case
                aniso_mat = np.tile(hetero_tri[:, np.newaxis], (1, 2))

            stiffness, mass = self._fem_tria_aniso(self.geometry, u1, u2, aniso_mat, lump)
        
        self.stiffness = stiffness
        self.mass = mass

        return self

    def solve(
        self,
        n_modes: int, 
        standardize: bool = True,
        fix_mode1: bool = True,
        atol: float = 1e-3,
        rtol: float = 1e-5,
        sigma: Union[float, None] = -0.01,
        seed: Union[int, ArrayLike, None] = None, 
        lump: bool = False,
        smoothit: int = 10
    ) -> EigenSolver:
        """
        Solves the generalized eigenvalue problem for the Laplace-Beltrami operator and compute
        eigenvalues and eigenmodes, which are stored as attributes (`emodes` and `evals`).

        Parameters
        ----------
        n_modes : int
            Number of eigenmodes to compute.
        standardize : bool, optional
            If `True`, standardizes the sign of the eigenmodes so the first element is positive.
            Default is `False`.
        fix_mode1 : bool, optional
            If `True`, sets the first eigenmode to a constant value and the first eigenvalue to
            zero, as is expected analytically. Default is `True`. See the `is_orthonormal_basis`
            function for details.
        atol : float, optional
            Absolute tolerance for mass-orthonormality validation. Default is `1e-3`.
        rtol : float, optional
            Relative tolerance for mass-orthonormality validation. Default is `1e-5`.
        sigma : float, optional
            Shift-invert parameter to speed up the computation of eigenvalues close to this value.
            Default is `-0.01`.
        seed : int or array-like, optional
            Random seed for reproducibile generation of eigenvectors (which otherwise use an
            iterative algorithm that starts with a random vector, meaning that repeated generation
            of eigenmodes on the same surface can have different orientations). Specify as an `int`
            (to set the seed) or a vector with n_verts elements (to directly set the initialisation
            vector). Default is `None` (not reproducible).
        lump: bool = False
            Whether to use lumped mass matrix for the Laplace-Beltrami operator. Default is `False`.
        smoothit: int = 10
            Number of smoothing iterations for curvature calculation. Default is 10.

        Returns
        -------
        EigenSolver
            The EigenSolver instance.

        Raises
        ------
        ValueError
            If `n_modes` is not a positive integer.
        ValueError
            If `seed` is an array but does not have shape (n_verts,).
        AssertionError
            If computed eigenvalues or eigenmodes contain NaNs.
        """
        # Validate inputs
        if not isinstance(n_modes, int) or n_modes <= 0:
            raise ValueError("`n_modes` must be a positive integer.")

        # Compute the Laplace-Beltrami operator / set stiffness and mass matrices
        if not hasattr(self, 'stiffness') or not hasattr(self, 'mass'):
            self.compute_lbo(lump, smoothit)
        
        # Set intitialization vector (if desired) for reproducibile eigenvectors 
        if seed is None or isinstance(seed, int):
            rng = np.random.default_rng(seed)
            v0 = rng.random(self.n_verts)
        else:
            v0 = np.asarray_chkfinite(seed)
            if v0.shape != (self.n_verts,):
                raise ValueError("`seed` must be either an integer or an array of shape (n_verts,) "
                                 f"= {(self.n_verts,)}.")

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
        assert not (np.isnan(self.evals).any() or np.isnan(self.emodes).any()), (
            "Computed eigenvalues or eigenmodes contain NaNs. This may indicate numerical "
            "instability; consider adjusting `sigma` or checking mesh quality.")

        if not is_orthonormal_basis(self.emodes, self.mass, atol=atol, rtol=rtol):
            warn(f"Computed eigenmodes are not mass-orthonormal (atol={atol}, rtol={rtol}).")

        # Post-process
        if fix_mode1:
            # Value given by mass-orthonormality condition
            self.emodes[:, 0] = np.full(self.n_verts, 1 / np.sqrt(self.mass.sum()))
            self.evals[0] = 0.0

        if standardize:
            self.emodes = standardize_modes(self.emodes)

        return self
    
    def _check_for_emodes(self) -> None:
        if not hasattr(self, 'emodes'):
            raise ValueError("Eigenmodes not found. Please run the solve() method first.")
    
    def decompose(
        self,
        data: ArrayLike,
        **kwargs
    ) -> NDArray:
        """
        This is a wrapper for `neuromodes.basis.decompose`, see its documentation for details: 
        https://neuromodes.readthedocs.io/en/latest/generated/nsbtools.basis.decompose.html

        Note that `emodes`, `mass`, and `check_ortho` are passed automatically by the `EigenSolver`
        instance.
        """
        from neuromodes.basis import decompose

        self._check_for_emodes()
    
        return decompose(
            data,
            self.emodes,
            mass=self.mass,
            check_ortho=False,
            **kwargs
        )
    
    def reconstruct(
        self,
        data: ArrayLike,
        **kwargs
    ) -> Tuple[NDArray, NDArray, list[NDArray]]:
        """
        This is a wrapper for `neuromodes.basis.reconstruct`, see its documentation for details:
        https://neuromodes.readthedocs.io/en/latest/generated/nsbtools.basis.reconstruct.html

        Note that `emodes`, `mass`, and `check_ortho` are passed automatically by the `EigenSolver`
        instance.
        """
        from neuromodes.basis import reconstruct
        
        self._check_for_emodes()
            
        return reconstruct(
            data,
            self.emodes,
            mass=self.mass,
            check_ortho=False,
            **kwargs
        )
    
    def reconstruct_timeseries(
        self,
        data: ArrayLike,
        **kwargs
    ) -> Tuple[NDArray, NDArray, NDArray, NDArray, list[NDArray]]:
        """
        This is a wrapper for `neuromodes.basis.reconstruct_timeseries`, see its documentation for
        details:
        https://neuromodes.readthedocs.io/en/latest/generated/nsbtools.basis.reconstruct_timeseries.html

        Note that `emodes`, `mass`, and `check_ortho` are passed automatically by the `EigenSolver`
        instance.
        """
        from neuromodes.basis import reconstruct_timeseries

        self._check_for_emodes()
            
        return reconstruct_timeseries(
            data,
            self.emodes,
            mass=self.mass,
            check_ortho=False,
            **kwargs
        )
    
    def model_connectome(
        self,
        **kwargs
    ) -> NDArray:
        """
        This is a wrapper for `neuromodes.connectome.model_connectome`, see its documentation for
        details:
        https://neuromodes.readthedocs.io/en/latest/generated/nsbtools.connectome.model_connectome.html

        Note that `emodes` and `evals` are passed automatically by the `EigenSolver` instance.
        """
        from neuromodes.connectome import model_connectome

        self._check_for_emodes()

        return model_connectome(
            emodes=self.emodes,
            evals=self.evals,
            **kwargs
        )
    
    def simulate_waves(
        self,
        **kwargs
    ) -> NDArray:
        """
        This is a wrapper for `neuromodes.waves.simulate_waves`, see its documentation for details:
        https://neuromodes.readthedocs.io/en/latest/generated/nsbtools.waves.simulate_waves.html

        Note that `emodes`, `evals`, `mass`, `scaled_hetero`, and `check_ortho` are passed
        automatically by the `EigenSolver` instance.
        """
        from neuromodes.waves import simulate_waves

        self._check_for_emodes()

        return simulate_waves(
            emodes=self.emodes,
            evals=self.evals,
            mass=self.mass,
            scaled_hetero=(self.hetero if self._raw_hetero is not None else None),
            check_ortho=False,
            **kwargs
        )

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
        Scaling parameter controlling the strength of the transformation. Default is `1.0`.
    scaling : str, optional
        The scaling function to apply to the heterogeneity map, either `'sigmoid'` or
        `'exponential'`. Default is `'sigmoid'`.
    
    Returns
    -------
    ndarray
        The scaled heterogeneity map.

    Raises
    ------
    ValueError
        If `hetero` is not a 1D array.
    ValueError
        If `scaling` is not 'exponential' or 'sigmoid'.
    ValueError
        If `hetero` is constant.
    """
    # Format / validate arguments
    hetero = np.asarray_chkfinite(hetero)
    if hetero.ndim != 1:
        raise ValueError("`hetero` must be a 1D array.")
    if scaling not in ["exponential", "sigmoid"]:
        raise ValueError(f"Invalid scaling '{scaling}'. Must be 'exponential' or 'sigmoid'.")
    if alpha == 0:
        warn("`alpha` is set to 0, meaning heterogeneity map will have no effect.")
    std = np.std(hetero)
    if std == 0:
        warn("Provided `hetero` is constant; scaling `hetero` to a vector of ones.")
        hetero_scaled = np.ones_like(hetero)
    else:
        # Scale the heterogeneity map
        hetero_z = (hetero - np.mean(hetero)) / std
        hetero_scaled = (2 / (1 + np.exp(-alpha * hetero_z))
                         if scaling == 'sigmoid' else np.exp(alpha * hetero_z))
    
    return hetero_scaled

def standardize_modes(
    emodes: ArrayLike
) -> NDArray:
    """
    Flips the modes' signs such that the first element of each eigenmode has positive amplitude. 
    Note that the sign of each mode is arbitrary--standardisation is only helpful to compare sets of
    eigenmodes.

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
    emodes = np.asarray_chkfinite(emodes)

    # Find the sign of each mode's amplitude at the first vertex
    signs = np.sign(emodes[0, :])
    signs[signs == 0] = 1  # Treat zero as positive (unlikely case)
    
    # Flip modes where the first element is negative
    return emodes * signs

def is_orthonormal_basis(
    emodes: ArrayLike,
    mass: Union[spmatrix, ArrayLike, None] = None,
    atol: float = 1e-03,
    rtol: float = 1e-05
) -> bool:
    """
    Check if a set of vectors is orthonormal in Euclidean space (i.e., `emodes.T @ emodes == I`) or
    with respect to a mass matrix (i.e., `emodes.T @ mass @ emodes == I`), where `I` is the identity
    matrix. Mass-orthonormality is expected for the geometric eigenmodes (see notes).

    Parameters
    ----------
    emodes : array-like
        The vectors array of shape (n_verts, n_modes), where n_modes is the number of vectors.
    mass : array-like, optional
        The mass matrix of shape (n_verts, n_verts). If `None`, Euclidean orthonormality is checked.
        Default is `None`.
    atol : float, optional
        Absolute tolerance for the orthonormality check. Default is `1e-3`.
    rtol : float, optional
        Relative tolerance for the orthonormality check. Default is `1e-5`.

    Returns
    -------
    bool
        `True` if the set of vectors is orthonormal (Euclidean or mass-orthonormal), `False`
        otherwise.

    Raises
    ------
    ValueError
        If `emodes` does not have shape (n_verts, n_modes), where n_verts ≥ n_modes.
    ValueError
        If `mass` is provided but does not have shape (n_verts, n_verts).

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
    # Format / validate arguments
    emodes = np.asarray_chkfinite(emodes)
    if not isinstance(mass, (spmatrix, type(None))):
        mass = np.asarray_chkfinite(mass)

    if emodes.ndim != 2 or emodes.shape[0] < emodes.shape[1]:
        raise ValueError("`emodes` must have shape (n_verts, n_modes), where n_verts ≥ n_modes.")
    n_verts, n_modes = emodes.shape
    if mass is not None and (mass.shape != (n_verts, n_verts)):
        raise ValueError(f"`mass` must have shape (n_verts, n_verts) = {(n_verts, n_verts)}.")

    # Check Euclidean or mass-orthonormality
    prod = emodes.T @ emodes if mass is None else emodes.T @ mass @ emodes
    return np.allclose(prod, np.eye(n_modes), rtol=rtol, atol=atol, equal_nan=False)