import numpy as np
import pytest
from nsbtools.utils import unmask, unparcellate, resample_matrix

def test_unmask_1d():
    data = np.array([1, 2, 3])
    mask = np.array([False, True, False, True, True])
    expected = np.array([np.nan, 1, np.nan, 2, 3])
    result = unmask(data, mask)
    np.testing.assert_array_equal(np.isnan(result), np.isnan(expected))
    np.testing.assert_array_equal(result[~np.isnan(expected)], expected[~np.isnan(expected)])

def test_unmask_1d_fill():
    data = np.array([1, 2, 3])
    mask = np.array([False, True, False, True, True])
    expected = np.array([np.pi, 1, np.pi, 2, 3])
    result = unmask(data, mask, val=np.pi)
    np.testing.assert_array_equal(result, expected)

def test_unmask_2d():
    data = np.array([[1, 2], [3, 4]])
    mask = np.array([False, True, False, True])
    expected = np.array([[np.nan, np.nan], [1, 2], [np.nan, np.nan], [3, 4]])
    result = unmask(data, mask)
    np.testing.assert_array_equal(np.isnan(result), np.isnan(expected))
    np.testing.assert_array_equal(result[~np.isnan(expected)], expected[~np.isnan(expected)])

def test_unmask_2d_fill():
    data = np.array([[1, 2], [3, 4]])
    mask = np.array([False, True, False, True])
    expected = np.array([[np.pi, np.pi], [1, 2], [np.pi, np.pi], [3, 4]])
    result = unmask(data, mask, val=np.pi)
    np.testing.assert_array_equal(result, expected)

def test_unparcellate():
    data = np.array([5, 10, 15])
    parc = np.array([2, 0, 1, 1, 3])
    expected = np.array([10, np.nan, 5, 5, 15])
    result = unparcellate(data, parc)
    np.testing.assert_array_equal(result, expected)

def test_unparcellate_fill():
    data = np.array([5, 10, 15])
    parc = np.array([2, 0, 1, 1, 3])
    expected = np.array([10, np.pi, 5, 5, 15])
    result = unparcellate(data, parc, val=np.pi)
    np.testing.assert_array_equal(result, expected)

def test_unparcellate_overflow():
    data = np.array([5, 10, 15])
    parc = np.array([2, 0, 1, 1, 3, 4])
    with pytest.raises(ValueError, match=r"Data length \(3\) does not match the number of non-zero "
                                         r"parcels \(4\)."):
        unparcellate(data, parc)

def test_unparcellate_2d():
    data = np.array([[1, 2], [3, 4]])
    parc = np.array([2, 0, 1, 1])
    expected = np.array([[3, 4], [np.nan, np.nan], [1, 2], [1, 2]])
    result = unparcellate(data, parc)
    np.testing.assert_array_equal(np.isnan(result), np.isnan(expected))
    np.testing.assert_array_equal(result[~np.isnan(expected)], expected[~np.isnan(expected)])

def test_unparcellate_3d():
    data = np.array([[[1]], [[2]]])
    parc = np.array([1, 2, 0])
    with pytest.raises(ValueError, match="Data must be 1D or 2D."):
        unparcellate(data, parc)

def test_unparcellate_invalid_parc():
    data = np.array([1, 2])
    parc = np.array([[1, 2], [0, 1]])
    with pytest.raises(ValueError, match="Parcellation map must be 1D."):
        unparcellate(data, parc)

def test_resample_vector(): 
    a  = np.random.rand(50,)
    b = resample_matrix(a)
    ai = np.argsort(a)
    bi = np.argsort(b)
    assert all(ai == bi)

def test_resample_matrix(): 
    a  = np.random.rand(50,50)
    b = resample_matrix(a)
    ai = np.argsort(a,None)
    bi = np.argsort(b,None)
    assert all(ai == bi)

def test_resample_matrix_repeats(): 
    from scipy.stats import rankdata
    a  = np.random.rand(50,50)
    for i in range(a.shape[1]):
        a[:,i] = resample_matrix(a[:,i], noise=a[:,0])
    b = resample_matrix(a, preserve_repeats=True)
    ai = rankdata(a)
    bi = rankdata(b)
    assert all(ai == bi)

def test_resample_matrix_symmetric(): 
    a  = np.random.rand(50,50)
    a = a + a.T
    b = resample_matrix(a, preserve_symmetry=True)
    ai = np.argsort(a,None)
    bi = np.argsort(b,None)
    assert all(ai == bi)
    assert np.allclose(b, b.T)

def test_resample_matrix_zeros(): 
    a  = np.random.rand(50,50)
    a[a<0.25] = 0
    b = resample_matrix(a, preserve_zeros=True)
    ai = np.argsort(a,None)
    bi = np.argsort(b,None)
    assert all(ai == bi)
    assert all(b[a==0]==0)



