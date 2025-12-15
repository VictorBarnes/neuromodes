import numpy as np
import pytest
from nsbtools.validation import (check_normalized_vectors,
                                 check_orthogonal_vectors,
                                 check_orthonormal_vectors, 
                                 check_orthonormal_matrix)


def test_check_normalized_vectors_typical():
    """Tests check_normalized_vectors() with typical cases.
    """

    vectors = np.array([
        [1/np.sqrt(2), 1/np.sqrt(2)], 
        [np.sqrt(3)/2, -1/2],
        [0, 1], 
        [1, 0]
    ])

    assert not check_normalized_vectors(vectors)
    assert not check_normalized_vectors(vectors, axis=0)
    assert check_normalized_vectors(vectors, axis=1)

def test_check_normalized_vectors_zero():
    """Tests check_normalized_vectors() with a zero vector.
    """

    vector = [0, 0]

    assert not check_normalized_vectors(vector)

def test_check_orthogonal_vectors_typical():
    """Tests check_orthogonal_vectors() with typical cases.
    """

    vectors = np.array([
        [2*np.pi, np.pi, np.pi], 
        [-1, 1, 1], 
        [0, 2, -2]
    ])

    assert not check_orthogonal_vectors(vectors)
    assert check_orthogonal_vectors(vectors, axis=1)

def test_check_orthogonal_vectors_empty():
    """Tests check_orthogonal_vectors() with an empty set.
    """

    vectors = []

    with pytest.raises(AssertionError):
        check_orthogonal_vectors(vectors)

def test_check_orthogonal_vectors_1d():
    """Tests check_orthogonal_vectors() with a single vector.
    """

    vector = np.array([1,2,3,4,5])

    with pytest.raises(AssertionError):
        check_orthogonal_vectors(vector)

    with pytest.raises(AssertionError):
        check_orthogonal_vectors(vector, axis=1)

    check_orthogonal_vectors([vector])

def test_check_orthogonal_vectors_parallel():
    """Tests check_orthogonal_vectors() with parallel vectors.
    """

    vector_a = [1, 1]
    vector_b = [2, 2]
    vectors = [vector_a, vector_b]

    assert not check_orthogonal_vectors(vectors)
    assert not check_orthogonal_vectors(vectors, axis=1)

def test_check_orthogonal_vectors_zero():
    """Tests check_orthogonal_vectors() with a zero vector.
    """

    vector_a = [0, 0]
    vector_b = [1, 0]
    vectors = [vector_a, vector_b]

    assert check_orthogonal_vectors(vectors)

def test_check_orthonormal_vectors_typical():
    """Tests check_orthonormal_vectors() with typical cases.
    """

    vectors = np.array([
        [1/np.sqrt(2), 1/np.sqrt(2)], 
        [-1/np.sqrt(2), 1/np.sqrt(2)]
    ])

    assert check_orthonormal_vectors(vectors)
    assert check_orthonormal_vectors(vectors, axis=0)
    assert check_orthonormal_vectors(vectors, axis=1)

def test_check_orthonormal_vectors_non_orthogonal():
    """Tests check_orthonormal_vectors() with non-orthogonal vectors.
    """

    vectors = np.array([
        [1, 0],
        [1, 1]
    ])

    assert not check_orthonormal_vectors(vectors)
    assert not check_orthonormal_vectors(vectors, axis=0)
    assert not check_orthonormal_vectors(vectors, axis=1)

def test_check_orthonormal_vectors_non_normalized():
    """Tests check_orthonormal_vectors() with non-normalized vectors.
    """

    vectors = np.array([
        [2, 0],
        [0, 2]
    ])

    assert not check_orthonormal_vectors(vectors)
    assert not check_orthonormal_vectors(vectors, axis=0)
    assert not check_orthonormal_vectors(vectors, axis=1)

def test_check_orthonormal_vectors_1directional():
    """Tests check_orthonormal_vectors() with a matrix that is orthonormal in one direction only.
    """

    vectors = np.array([
        [1, 0], 
        [0, 1], 
        [0, 0]
    ])

    assert check_orthonormal_vectors(vectors, axis=0)
    assert not check_orthonormal_vectors(vectors, axis=1)

def test_check_orthonormal_matrix_typical():
    """Tests check_orthonormal_matrix() with a typical case.
    """

    # Rotate about the z-axis
    rot = np.random.uniform(0, 2*np.pi)

    vectors = np.array([
        [np.cos(rot), -np.sin(rot), 0],
        [np.sin(rot), np.cos(rot), 0],
        [0, 0, 1]
    ])

    assert check_orthonormal_matrix(vectors)

def test_check_orthonormal_matrix_non_orthogonal():
    """Tests check_orthonormal_matrix() with a non-orthogonal matrix.
    """

    matrix = np.array([
        [1, 1, 0],
        [0, 1, 1],
        [1, 0, 1]
    ]) / np.sqrt(2)

    assert not check_orthonormal_matrix(matrix)

def test_check_orthonormal_matrix_non_normalized():
    """Tests check_orthonormal_matrix() with a non-normalized matrix.
    """

    matrix = np.array([
        [2, 0, 0],
        [0, 2, 0],
        [0, 0, 2]
    ])

    assert not check_orthonormal_matrix(matrix)

def test_check_orthonormal_matrix_non_orthonormal():
    """Tests check_orthonormal_matrix() with a non-orthogonal matrix.
    """

    matrix = np.array([
        [1, 1, 0],
        [0, 1, 1],
        [1, 0, 1]
    ])

    assert not check_orthonormal_matrix(matrix)

