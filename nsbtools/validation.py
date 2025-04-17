import numpy as np

def check_orthogonal_vectors(matrix, colvar=True, tol=1e-6):
    """
    Check if a set of vectors in a matrix (rows or columns) are orthogonal.

    Parameters
    ----------
    matrix : array_like
        The matrix to be checked for orthogonality.
    colvar : bool, optional
        If True, check the columns of the matrix. If False, check the rows. Default is True.
    tol : float, optional
        The tolerance value for checking orthogonality. Default is 1e-6.

    Returns
    -------
    bool
        True if the matrix is orthogonal, False otherwise.
    """
    # Convert the matrix to a numpy array
    matrix = np.array(matrix)
    if colvar is False:
        # If colvar is False, transpose the matrix
        matrix = matrix.T
    # Calculate the dot product of the matrix with its transpose
    dot_product = matrix.T @ matrix
    
    # Check if the diagonal elements are close to 1
    return np.allclose(dot_product, np.eye(np.shape(dot_product)[0]), atol=tol)

def check_normal(matrix, axis=0, tol=0.01):
    """
    Check if the columns of a matrix are normalized.

    Parameters
    ----------
    matrix : array_like
        The input matrix.
    axis : int, optional
        The axis along which to calculate the norm. By default, axis=0.
    tol : float, optional
        The tolerance for comparing the column norms to 1. By default, tol=0.01.

    Returns
    -------
    bool
        True if all column norms are close to 1 within the given tolerance, False otherwise.
    """
    # Convert the matrix to a numpy array
    matrix = np.array(matrix)
    
    # Calculate the norm of each column
    column_norms = np.linalg.norm(matrix, axis=axis)
    
    # Check if all column norms are close to 1
    if not np.allclose(column_norms, 1.0, atol=tol):
        return False
    
    return True
