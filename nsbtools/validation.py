import numpy as np
from numpy.typing import ArrayLike

def check_orthonormal_matrix(
    matrix: ArrayLike,
    tol: float = 1e-6
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
    return (check_orthogonal_vectors(matrix, tol=tol) and
           check_normalized_vectors(matrix, tol=tol) and
           check_orthogonal_vectors(matrix, colvec=False, tol=tol) and
           check_normalized_vectors(matrix, colvec=False, tol=tol))

def check_orthogonal_vectors(
    matrix: ArrayLike,
    colvec: bool = True,
    tol: float = 1e-6
) -> bool:
    """
    Check if a set of real-valued vectors in a matrix (rows or columns) are orthogonal.

    Parameters
    ----------
    matrix : array_like
        The set of vectors to be checked for orthogonality.
    colvec : bool, optional
        If True, vectors are the matrix's columns. If False, they are the matrix's rows. Default is True.
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

    try:
        matrix = np.array(matrix)
    except Exception:
        raise TypeError("Input must be convertible to a numpy array.")
    
    # Ensure that vectors are along columns
    if not colvec:
        matrix = matrix.T

    assert matrix.ndim == 2, "Input array must be 2-dimensional."
    assert matrix.shape[0] > 1 and matrix.shape[1] > 1, "Input array must contain at least two vectors."
    assert np.isrealobj(matrix), "Input array must contain only real values."

    # For an orthogonal set of vectors, the Gram matrix's off-diagonal elements should be zero
    gram = matrix.T @ matrix
    diag = np.diag(gram)
    off_diag = gram - np.diagflat(diag)

    return np.allclose(off_diag, 0, atol=tol)

def check_normalized_vectors(
    matrix: ArrayLike,
    colvec: bool = True,
    tol: float = 1e-6
) -> bool:
    """
    Check if a set of real-valued vectors in a matrix (rows or columns) have unit magnitude.

    Parameters
    ----------
    matrix : array_like
        The input matrix.
    colvec : bool, optional
        If True, vectors are the matrix's columns. If False, they are the matrix's rows. Default is True.
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
    
    try:
        matrix = np.array(matrix)
    except Exception:
        raise TypeError("Input must be convertible to a numpy array.")

    # Ensure that vectors are along columns
    if not colvec:
        matrix = matrix.T

    assert matrix.ndim < 3, "Input array must be 1- or 2-dimensional."
    assert matrix.shape[0] > 1, "Input array must contain vectors, not single values."
    assert np.isrealobj(matrix), "Input array must contain only real values."

    return np.allclose(np.linalg.norm(matrix, axis=0), 1.0, atol=tol)
