import pytest
import numpy as np
from neuromodes.eigen import EigenSolver
from neuromodes.io import fetch_surf
from neuromodes.nulls import eigenstrap

# Params
n_nulls = 5
seed = 0

@pytest.fixture
def solver():
    mesh, medmask = fetch_surf(density='4k')
    return EigenSolver(mesh, mask=medmask).solve(n_modes=100, seed=seed)

@pytest.fixture
def test_data(solver):
    """Generate 1D test data"""
    rng = np.random.default_rng(seed)
    data = rng.standard_normal(size=solver.n_verts)  # random data
    return data

@pytest.fixture
def nulls(solver, test_data):
    """Generate nulls for 1D test data"""
    return solver.eigenstrap(test_data, n_nulls=n_nulls, seed=seed)

def test_output_shape(solver, nulls):
    """Output shape should be (n_verts, n_nulls)"""    
    assert nulls.shape == (solver.n_verts, n_nulls), \
        f"Expected shape {(solver.n_verts, n_nulls)}, got {nulls.shape}"

def test_reproducibility(solver, test_data, nulls):
    """Nulls with same seed should be identical"""
    nulls2 = solver.eigenstrap(test_data, n_nulls=n_nulls, seed=seed)
    
    assert np.allclose(nulls, nulls2, atol=1e-10), "Null spaces with the same seed should be identical"

def test_parallel_reproducibility(solver, test_data, nulls):
    """Serial and parallel execution should give same results with same seed"""    
    nulls_serial = solver.eigenstrap(test_data, n_nulls=n_nulls, n_jobs=1, seed=seed)
    
    assert np.allclose(nulls_serial, nulls, atol=1e-10), \
        "Parallel and serial execution should produce identical results with same seed"

def test_finite(nulls):
    """Output should not contain NaNs or Infs"""    
    assert np.isfinite(nulls).all(), "Nulls contain NaNs or Infs"

def test_invalid_residual_parameter(solver, test_data):
    """Should raise ValueError for invalid residual parameter"""
    with pytest.raises(ValueError, match="Invalid residual method"):
        solver.eigenstrap(test_data, residual='invalid')

def test_shape_mismatch_data_emodes(solver, test_data):
    """`decompose()` should raise error when data length doesn't match emodes rows"""
    wrong_data = test_data[:-100]  # Truncate data
    
    with pytest.raises((ValueError, IndexError)):
        solver.eigenstrap(wrong_data)

def test_shape_mismatch_emodes_evals(solver, test_data):
    """Should raise error when emodes columns doesn't match evals length"""
    wrong_evals = solver.evals[:-10]  # Truncate evals
    
    with pytest.raises(ValueError, match="must have shape"):
        eigenstrap(test_data, solver.emodes, wrong_evals, mass=solver.mass)

def test_resample_exact(solver, test_data):
    """With resample=True, nulls should have same values as original data"""
    nulls = solver.eigenstrap(test_data, n_nulls=n_nulls, resample="exact", seed=seed)
    
    # Check that each null has the exact same values as original data (just reordered)
    for i in range(nulls.shape[1]):
        null_sorted = np.sort(nulls[:, i])
        data_sorted = np.sort(test_data)
        assert np.allclose(null_sorted, data_sorted), \
            f"Null {i} doesn't preserve data distribution"

def test_resample_zscore(solver, test_data):
    """With resample='zscore', nulls should have mean and std that match the data"""
    nulls = solver.eigenstrap(test_data, n_nulls=n_nulls, resample="affine", seed=seed)
    
    for i in range(nulls.shape[1]):
        mean = np.mean(nulls[:, i])
        std = np.std(nulls[:, i])
        assert np.isclose(mean, test_data.mean()), f"Null {i} mean is not close to data mean"
        assert np.isclose(std, test_data.std()), f"Null {i} std is not close to data std"

def test_resample_mean(solver, test_data):
    """With resample='mean', nulls should have mean equal to original data mean"""
    nulls = solver.eigenstrap(test_data, n_nulls=n_nulls, resample="mean", seed=seed)
    
    data_mean = np.mean(test_data)
    
    for i in range(nulls.shape[1]):
        null_mean = np.mean(nulls[:, i])
        assert np.isclose(null_mean, data_mean), f"Null {i} mean is not close to data mean"

def test_resample_range(solver, test_data):
    """With resample='range', nulls should have same min and max as original data"""
    nulls = solver.eigenstrap(test_data, n_nulls=n_nulls, resample="range", seed=seed)
    
    data_min = np.min(test_data)
    data_max = np.max(test_data)
    
    for i in range(nulls.shape[1]):
        null_min = np.min(nulls[:, i])
        null_max = np.max(nulls[:, i])
        assert np.isclose(null_min, data_min), f"Null {i} min is not close to data min"
        assert np.isclose(null_max, data_max), f"Null {i} max is not close to data max"

def test_randomize_option(solver, test_data):
    """Test randomize parameter works without errors"""
    nulls = solver.eigenstrap(test_data, n_nulls=n_nulls, randomize=True, seed=seed)
    assert nulls.shape == (solver.n_verts, n_nulls)
    assert np.isfinite(nulls).all(), "Nulls contain non-finite values when randomize=True"

def test_residual_methods(solver, test_data):
    """Test different residual methods run without errors"""
    for method in ['add', 'permute']:
        nulls = solver.eigenstrap(test_data, n_nulls=n_nulls, residual=method, seed=seed)
        assert nulls.shape == (solver.n_verts, n_nulls)
        assert np.isfinite(nulls).all(), f"Nulls contain non-finite values for residual method '{method}'"

def test_non_square_modes(test_data):
    """Should handle non-square n_modes by truncating last eigengroup with warning"""
    # Use 8 modes (not a perfect square)
    mesh, medmask = fetch_surf(density='4k')
    non_square_solver = EigenSolver(mesh, mask=medmask).solve(n_modes=8, seed=seed)
    
    # Should complete with a warning about truncating last eigengroup
    with pytest.warns(UserWarning, match="These last 4 modes will be excluded."):
        nulls = non_square_solver.eigenstrap(test_data, n_nulls=n_nulls, seed=seed)
    
    assert nulls.shape == (non_square_solver.n_verts, n_nulls)

# 2D data tests

@pytest.fixture
def test_data_2d(solver):
    """Generate 2D test data with 3 maps"""
    rng = np.random.default_rng(seed)
    data_2d = rng.standard_normal(size=(solver.n_verts, 3))
    return data_2d

def test_output_shape_2d(solver, test_data_2d):
    """Output shape should be (n_verts, n_nulls, n_maps)"""
    n_maps = test_data_2d.shape[1]
    nulls = solver.eigenstrap(test_data_2d, n_nulls=n_nulls, seed=seed)
    
    assert nulls.shape == (solver.n_verts, n_nulls, n_maps), \
        f"Expected shape {(solver.n_verts, n_nulls, n_maps)}, got {nulls.shape}"

def test_resample_match_2d(solver, test_data_2d):
    """With resample=True, nulls should have same values as original data"""
    nulls = solver.eigenstrap(test_data_2d, n_nulls=n_nulls, resample="match", seed=seed)
    
    # Check that each null has the exact same values as original data (just reordered)
    # TODO: vectorise if slow?
    for j in range(test_data_2d.shape[1]):
        for i in range(nulls.shape[1]):
            null_sorted = np.sort(nulls[:, i, j])
            data_sorted = np.sort(test_data_2d[:, j])
            assert np.allclose(null_sorted, data_sorted), \
                f"Null {i} doesn't preserve data distribution for map {j}"

def test_resample_zscore_2d(solver, test_data_2d):
    """With resample='zscore', nulls should have mean and std that match the data"""
    nulls = solver.eigenstrap(test_data_2d, n_nulls=n_nulls, resample="zscore", seed=seed)
    
    for j in range(test_data_2d.shape[1]):
        for i in range(nulls.shape[1]):
            mean = np.mean(nulls[:, i, j])
            std = np.std(nulls[:, i, j])
            data_mean = np.mean(test_data_2d[:, j])
            data_std = np.std(test_data_2d[:, j])
            assert np.isclose(mean, data_mean), f"Null {i} map {j} mean is not close to data mean"
            assert np.isclose(std, data_std), f"Null {i} map {j} std is not close to data std"

def test_resample_mean_2d(solver, test_data_2d):
    """With resample='mean', nulls should have mean equal to original data mean"""
    nulls = solver.eigenstrap(test_data_2d, n_nulls=n_nulls, resample="mean", seed=seed)
    
    for j in range(test_data_2d.shape[1]):
        data_mean = np.mean(test_data_2d[:, j])
        for i in range(nulls.shape[1]):
            null_mean = np.mean(nulls[:, i, j])
            assert np.isclose(null_mean, data_mean), f"Null {i} map {j} mean is not close to data mean"

def test_resample_range_2d(solver, test_data_2d):
    """With resample='range', nulls should have same min and max as original data"""
    nulls = solver.eigenstrap(test_data_2d, n_nulls=n_nulls, resample="range", seed=seed)
    
    for j in range(test_data_2d.shape[1]):
        data_min = np.min(test_data_2d[:, j])
        data_max = np.max(test_data_2d[:, j])
        for i in range(nulls.shape[1]):
            null_min = np.min(nulls[:, i, j])
            null_max = np.max(nulls[:, i, j])
            assert np.isclose(null_min, data_min), f"Null {i} map {j} min is not close to data min"
            assert np.isclose(null_max, data_max), f"Null {i} map {j} max is not close to data max"

def test_randomize_option_2d(solver, test_data_2d):
    """Test randomize parameter works without errors"""
    nulls = solver.eigenstrap(test_data_2d, n_nulls=n_nulls, randomize=True, seed=seed)
    assert nulls.shape == (solver.n_verts, n_nulls, test_data_2d.shape[1])
    assert np.isfinite(nulls).all(), "Nulls contain non-finite values"

def test_residual_methods_2d(solver, test_data_2d):
    """Test different residual methods run without errors"""
    for method in ['add', 'permute']:
        nulls = solver.eigenstrap(test_data_2d, n_nulls=n_nulls, residual=method, seed=seed)
        assert nulls.shape == (solver.n_verts, n_nulls, test_data_2d.shape[1])
        assert np.isfinite(nulls).all(), f"Nulls contain non-finite values for residual method '{method}'"