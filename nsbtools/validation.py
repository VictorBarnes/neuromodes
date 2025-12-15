import numpy as np
from numpy.typing import ArrayLike
from scipy import sparse
from typing import Optional, Union

def check_orthonormal_matrix(
    matrix: ArrayLike,
    atol: float = 1e-6
) -> bool:
    """
    Check if a matrix is orthonormal, i.e., its rows and columns are both orthogonal and normalized.

    Parameters
    ----------
    matrix : array_like
        The matrix to be checked for orthonormality.
    tol : float, optional
        The tolerance value for checking orthogonality and normalization. Default is 1e-6.

    Returns
    -------
    bool
        True if the matrix is orthonormal, False otherwise.

    Raises
    ------
    TypeError
        If the input cannot be converted to a numpy array.
    AssertionError
        If the input does not meet the dimensionality or value requirements.
    """

    matrix = np.asarray(matrix)

    return (
        matrix.shape[0] == matrix.shape[1] and                      # short-circuit if not square
        check_orthonormal_vectors(matrix, axis=0, atol=atol) and
        check_orthonormal_vectors(matrix, axis=1, atol=atol)
        )

def check_orthonormal_vectors(
    matrix: ArrayLike,
    axis: int = 0,
    atol: float = 1e-6
) -> bool:
    """
    Check if a set of real-valued vectors (emodes) in a matrix (rows or columns) are orthonormal.
    
    Parameters
    ----------
    matrix : array_like
        The set of vectors (emodes) to be checked for orthonormality.
    axis : int, optional
        If 0, vectors (emodes) are the matrix's columns. If 1, they are the matrix's rows. Default
        is 0.
    atol : float, optional
        The tolerance value for checking orthonormality. Default is 1e-6.
    
    Returns
    -------
    bool
        True if the vectors are orthonormal, False otherwise.

    Raises
    ------
    ValueError
        If the matrix is not 2 dimensional, or if the axis parameter is not 0 or 1.
    """

    matrix = np.asarray(matrix)
    if matrix.ndim != 2: 
        raise ValueError("Input array must be 2-dimensional.")

    if axis==0: 
        gram = matrix.T @ matrix
    elif axis==1:
        gram = matrix @ matrix.T
    else:
        raise ValueError("Axis must be 0 (columns) or 1 (rows).")

    return np.allclose(gram, np.eye(gram.shape[0]), atol=atol)

def check_orthogonal_vectors(
    matrix: ArrayLike,
    axis: int = 0,
    atol: float = 1e-6
) -> bool:
    """
    Check if a set of real-valued vectors (emodes) in a matrix (rows or columns) are orthogonal.

    Parameters
    ----------
    matrix : array_like
        The set of vectors (emodes) to be checked for orthogonality.
    colvec : bool, optional
        If True, vectors (emodes) are the matrix's columns. If False, they are the matrix's rows.
        Default is True.
    tol : float, optional
        The tolerance value for checking orthogonality. Default is 1e-6.

    Returns
    -------
    bool
        True if the vectors are orthogonal, False otherwise.

    Raises
    ------
    TypeError
        If the input cannot be converted to a numpy array.
    AssertionError
        If the input does not meet the dimensionality or value requirements.
    """

    matrix = np.asarray(matrix)
    assert matrix.ndim == 2, "Input array must be 2-dimensional."

    if axis==0: 
        gram = matrix.T @ matrix
    elif axis==1:
        gram = matrix @ matrix.T
    else:
        raise ValueError("Axis must be 0 (columns) or 1 (rows).")

    np.fill_diagonal(gram, 0.0)
    return np.allclose(gram, 0.0, atol=atol)

def check_normalized_vectors(
    matrix: ArrayLike,
    axis: int = 0,
    atol: float = 1e-6
) -> bool:
    """
    Check if a set of real-valued vectors (emodes) in a matrix (rows or columns) have unit
    magnitude.

    Parameters
    ----------
    matrix : array_like
        The input matrix.
    colvec : bool, optional
        If True, vectors (emodes) are the matrix's columns. If False, they are the matrix's rows.
        Default is True.
    tol : float, optional
        The tolerance for comparing the magnitudes to 1. By default, tol=1e-6.

    Returns
    -------
    bool
        True if all vector magnitudes are close to 1 within the given tolerance, False otherwise.

    Raises
    ------
    TypeError
        If the input cannot be converted to a numpy array.
    AssertionError
        If the input does not meet the dimensionality or value requirements.
    """

    return np.allclose(np.linalg.norm(np.asarray(matrix), axis=axis), 1.0, atol=atol)

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
