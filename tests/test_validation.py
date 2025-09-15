import numpy as np
import pytest
from nsbtools.validation import (check_normalized_vectors,
                                 check_orthogonal_vectors,
                                 check_orthonormal_matrix)

# TODO: maybe tests should be combined across functions for the same case?

def test_check_orthogonal_vectors_typical():
    """Tests check_orthogonal_vectors() with typical cases.
    """

    vector_a = [2*np.pi, np.pi, np.pi]
    vector_b = [-1, 1, 1]
    vector_c = [0, 2, -2]
    vectors = [vector_a, vector_b, vector_c]

    assert not check_orthogonal_vectors(vectors)
    assert check_orthogonal_vectors(vectors, colvec=False)

def test_check_orthogonal_vectors_empty():
    """Tests check_orthogonal_vectors() with an empty set.
    """

    vectors = []

    with pytest.raises(AssertionError):
        check_orthogonal_vectors(vectors)

def test_check_orthogonal_vectors_single_vector():
    """Tests check_orthogonal_vectors() with a single vector.
    """

    vector = [0, 1]
    vectors = [vector]

    with pytest.raises(AssertionError):
        check_orthogonal_vectors(vectors)

    with pytest.raises(AssertionError):
        check_orthogonal_vectors(vectors, colvec=False)

def test_check_orthogonal_vectors_complex():
    """Tests check_orthogonal_vectors() with a complex vector.
    """

    vector_a = [0, 1 * 1j]
    vector_b = [1, 0]
    vectors = [vector_a, vector_b]

    with pytest.raises(AssertionError):
        check_orthogonal_vectors(vectors)

def test_check_orthogonal_vectors_parallel():
    """Tests check_orthogonal_vectors() with parallel vectors.
    """

    vector_a = [1, 1]
    vector_b = [2, 2]
    vectors = [vector_a, vector_b]

    assert not check_orthogonal_vectors(vectors)
    assert not check_orthogonal_vectors(vectors, colvec=False)

def test_check_orthogonal_vectors_zero():
    """Tests check_orthogonal_vectors() with a zero vector.
    """

    vector_a = [0, 0]
    vector_b = [1, 0]
    vectors = [vector_a, vector_b]

    assert check_orthogonal_vectors(vectors)

def test_check_orthogonal_vectors_mismatch():
    """Tests check_orthogonal_vectors() with mismatched vector lengths.
    """

    vector_a = [1, 0, 0]
    vector_b = [0, 1]
    vectors = [vector_a, vector_b]

    with pytest.raises(TypeError):
        check_orthogonal_vectors(vectors, colvec=False)

def test_check_normalized_vectors_typical():
    """Tests check_normalized_vectors() with typical cases.
    """

    vector_a = [np.sqrt(3)/2, -1/2]
    vector_b = [1, 0]
    vectors = [vector_a, vector_b]

    assert not check_normalized_vectors(vectors)
    assert check_normalized_vectors(vectors, colvec=False)

def test_check_normalized_vectors_empty():
    """Tests check_normalized_vectors() with an empty set.
    """

    vectors = []

    with pytest.raises(AssertionError):
        check_normalized_vectors(vectors)

def test_check_normalized_vectors_single_vector():
    """Tests check_normalized_vectors() with a single vector.
    """

    vector = [0, 1]

    assert check_normalized_vectors(vector)

    with pytest.raises(AssertionError):
        check_normalized_vectors([vector])

def test_check_normalized_vectors_complex():
    """Tests check_normalized_vectors() with a complex vector.
    """

    vector = [0, 1j]

    with pytest.raises(AssertionError):
        check_normalized_vectors(vector)

def test_check_normalized_vectors_zero():
    """Tests check_normalized_vectors() with a zero vector.
    """

    vector = [0, 0]

    assert not check_normalized_vectors(vector)

def test_check_normalized_vectors_mismatch():
    """Tests check_normalized_vectors() with mismatched vector lengths.
    """

    vector_a = [1, 0, 0]
    vector_b = [0, 1]
    vectors = [vector_a, vector_b]

    with pytest.raises(TypeError):
        check_normalized_vectors(vectors, colvec=False)

def test_check_orthonormal_matrix():
    """Tests check_orthonormal_matrix() with a typical case.
    """

    # Rotate about the z-axis
    rot = np.pi/6

    vector_a = [np.cos(rot), -np.sin(rot), 0]
    vector_b = [np.sin(rot), np.cos(rot), 0]
    vector_c = [0, 0, 1]
    vectors = [vector_a, vector_b, vector_c]

    assert check_orthonormal_matrix(vectors)