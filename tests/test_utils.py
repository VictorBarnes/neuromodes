import numpy as np
import pytest
from nsbtools.utils import unmask, unparcellate

# TODO: add more testing for bad inputs?

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
    with pytest.raises(ValueError):
        unparcellate(data, parc)
