import pytest
import numpy as np
from neuromodes.eigen import EigenSolver
from neuromodes.io import fetch_surf
from neuromodes.nulls import eigenstrap

@pytest.fixture
def solver():
    mesh, medmask = fetch_surf(density='4k')
    return EigenSolver(mesh, mask=medmask).solve(n_modes=100, seed=0)

@pytest.fixture
def test_data(solver):
    """Load test data"""
    medmask = solver.mask
    rng = np.random.default_rng(0)
    data = rng.standard_normal(size=medmask.sum())  # random data
    return data

def test_reproducibility(solver, test_data):
    """Nulls with same seed should be identical"""
    nulls1 = solver.eigenstrap(test_data, n_nulls=10, seed=0)
    nulls2 = solver.eigenstrap(test_data, n_nulls=10, seed=0)
    
    assert np.allclose(nulls1, nulls2), "Null spaces with the same seed should be identical"

def test_parallel_reproducibility(solver, test_data):
    """Serial and parallel execution should give same results with same seed"""
    seed = 0
    n_nulls = 10
    
    nulls_serial = solver.eigenstrap(test_data, n_nulls=n_nulls, n_jobs=1, seed=seed)
    
    nulls_parallel = solver.eigenstrap(test_data, n_nulls=n_nulls, n_jobs=2, seed=seed)
    
    assert np.allclose(nulls_serial, nulls_parallel), \
        "Parallel and serial execution should produce identical results with same seed"

def test_output_shape(solver, test_data):
    """Output shape should be (n_verts, n_nulls)"""
    n_nulls = 10
    nulls = solver.eigenstrap(test_data, n_nulls=n_nulls, seed=0)
    
    assert nulls.shape == (len(test_data), n_nulls), \
        f"Expected shape {(len(test_data), n_nulls)}, got {nulls.shape}"

def test_no_nans_or_infs(solver, test_data):
    """Output should not contain NaNs or Infs"""
    nulls = solver.eigenstrap(test_data, n_nulls=10, seed=0)
    
    assert not np.any(np.isnan(nulls)), "Nulls contain NaN values"
    assert not np.any(np.isinf(nulls)), "Nulls contain Inf values"

def test_invalid_residual_parameter(solver, test_data):
    """Should raise ValueError for invalid residual parameter"""
    with pytest.raises(ValueError, match="Invalid residual method"):
        solver.eigenstrap(test_data, residual='invalid')

def test_shape_mismatch_data_emodes(solver, test_data):
    """Should raise error when data length doesn't match emodes rows"""
    wrong_data = test_data[:-100]  # Truncate data
    
    with pytest.raises((ValueError, IndexError)):
        solver.eigenstrap(wrong_data)

def test_shape_mismatch_emodes_evals(solver, test_data):
    """Should raise error when emodes columns doesn't match evals length"""
    wrong_evals = solver.evals[:-10]  # Truncate evals
    
    with pytest.raises(ValueError, match="must have shape"):
        eigenstrap(test_data, solver.emodes, wrong_evals, mass=solver.mass)

def test_resample_match(solver, test_data):
    """With resample=True, nulls should have same values as original data"""
    nulls = solver.eigenstrap(test_data, n_nulls=10, resample="match", seed=0)
    
    # Check that each null has the exact same values as original data (just reordered)
    for i in range(nulls.shape[1]):
        null_sorted = np.sort(nulls[:, i])
        data_sorted = np.sort(test_data)
        assert np.allclose(null_sorted, data_sorted), \
            f"Null {i} doesn't preserve data distribution"

def test_resample_zscore(solver, test_data):
    """With resample='zscore', nulls should have mean and std that match the data"""
    nulls = solver.eigenstrap(test_data, n_nulls=10, resample="zscore", seed=0)
    
    for i in range(nulls.shape[1]):
        mean = np.mean(nulls[:, i])
        std = np.std(nulls[:, i])
        assert np.isclose(mean, test_data.mean()), f"Null {i} mean is not close to data mean"
        assert np.isclose(std, test_data.std()), f"Null {i} std is not close to data std"

def test_resample_mean(solver, test_data):
    """With resample='mean', nulls should have mean equal to original data mean"""
    nulls = solver.eigenstrap(test_data, n_nulls=10, resample="mean", seed=0)
    
    data_mean = np.mean(test_data)
    
    for i in range(nulls.shape[1]):
        null_mean = np.mean(nulls[:, i])
        assert np.isclose(null_mean, data_mean), f"Null {i} mean is not close to data mean"

def test_resample_range(solver, test_data):
    """With resample='range', nulls should have same min and max as original data"""
    nulls = solver.eigenstrap(test_data, n_nulls=10, resample="range", seed=0)
    
    data_min = np.min(test_data)
    data_max = np.max(test_data)
    
    for i in range(nulls.shape[1]):
        null_min = np.min(nulls[:, i])
        null_max = np.max(nulls[:, i])
        assert np.isclose(null_min, data_min), f"Null {i} min is not close to data min"
        assert np.isclose(null_max, data_max), f"Null {i} max is not close to data max"

def test_randomize_option(solver, test_data):
    """Test randomize parameter works without errors"""
    nulls = solver.eigenstrap(test_data, n_nulls=10, randomize=True, seed=0)
    assert nulls.shape == (len(test_data), 10)
    assert not np.any(np.isnan(nulls))

def test_residual_methods(solver, test_data):
    """Test different residual methods run without errors"""
    for method in ['add', 'permute']:
        nulls = solver.eigenstrap(test_data, n_nulls=10, residual=method, seed=0)
        assert nulls.shape == (len(test_data), 10)
        assert not np.any(np.isnan(nulls))

def test_non_square_modes(test_data):
    """Should handle non-square n_modes by truncating last eigengroup with warning"""
    # Use 95 modes (not a perfect square)
    mesh, medmask = fetch_surf(density='4k')
    non_square_solver = EigenSolver(mesh, mask=medmask).solve(n_modes=95, seed=0)
    
    # Should complete with a warning about truncating last eigengroup
    with pytest.warns(UserWarning, match="not a perfect square.*Last eigengroup.*will be excluded"):
        nulls = non_square_solver.eigenstrap(test_data, n_nulls=3, seed=0)
    
    assert nulls.shape == (len(test_data), 3)
        