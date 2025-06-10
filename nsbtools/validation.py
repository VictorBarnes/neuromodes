import numpy as np


def check_orthogonal_matrix(matrix, tol=1e-6): 
    return (check_orthogonal_vectors(matrix, colvar=True,  tol=tol) and 
            check_orthogonal_vectors(matrix, colvar=False, tol=tol))
	
def check_orthogonal_vectors(matrix, colvar=True, tol=1e-6):
    """
    Check if a set of vectors in a matrix (rows or columns) are orthogonal.

    Parameters
    ----------
    matrix : array_like
        The set of vectors to be checked for orthogonality.
    colvar : bool, optional
        If True, check the columns of the matrix. If False, check the rows. Default is True.
    tol : float, optional
        The tolerance value for checking orthogonality. Default is 1e-6.

    Returns
    -------
    bool
        True if the matrix is orthogonal, False otherwise.
    """

    matrix = np.array(matrix)
    
    # If colvar is False, need to check rows instead of cols
    if colvar is False:
        matrix = matrix.T
    
    # Check if matrix product is close to identity
    return np.allclose(matrix.T @ matrix, np.eye(np.shape(matrix)[1]), atol=tol)

def check_normal(matrix, axis=0, tol=1e-6):
    """
    Check if the columns of a matrix are normalized.

    Parameters
    ----------
    matrix : array_like
        The input matrix.
    axis : int, optional
        The axis along which to calculate the norm. By default, axis=0.
    tol : float, optional
        The tolerance for comparing the column norms to 1. By default, tol=1e-6.

    Returns
    -------
    bool
        True if all column norms are close to 1 within the given tolerance, False otherwise.
    """
    
    matrix = np.array(matrix)

    return np.allclose(np.linalg.norm(matrix, axis=axis), 1.0, atol=tol)
