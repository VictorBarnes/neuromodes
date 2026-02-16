import pytest
import numpy as np
from neuromodes.eigen import EigenSolver
from neuromodes.io import fetch_surf, fetch_map
from neuromodes.nulls import eigenstrap

@pytest.fixture
def solver():
    mesh, medmask = fetch_surf(density='32k')
    return EigenSolver(mesh, mask=medmask).solve(n_modes=100, seed=0)

@pytest.fixture
def test_data(solver):
    """Load test data"""
    medmask = solver.mask
    data = fetch_map('myelinmap', density='32k')[medmask]
    return data

def test_reproducibility(solver, test_data):
    """Nulls with same seed should be identical"""
    nulls1 = eigenstrap(test_data, solver.emodes, solver.evals, mass=solver.mass, seed=0)
    nulls2 = eigenstrap(test_data, solver.emodes, solver.evals, mass=solver.mass, seed=0)
    
    assert np.allclose(nulls1, nulls2), "Null spaces with the same seed should be identical"

def test_parallel_reproducibility(solver, test_data):
    """Serial and parallel execution should give same results with same seed"""
    seed = 0
    n_nulls = 10
    
    nulls_serial = eigenstrap(test_data, solver.emodes, solver.evals,
                             mass=solver.mass, n_nulls=n_nulls, 
                             n_jobs=1, seed=seed)
    
    nulls_parallel = eigenstrap(test_data, solver.emodes, solver.evals,
                               mass=solver.mass, n_nulls=n_nulls,
                               n_jobs=2, seed=seed)
    
    assert np.allclose(nulls_serial, nulls_parallel), \
        "Parallel and serial execution should produce identical results with same seed"

def test_output_shape(solver, test_data):
    """Output shape should be (n_verts, n_nulls)"""
    n_nulls = 10
    nulls = eigenstrap(test_data, solver.emodes, solver.evals, mass=solver.mass, 
                      n_nulls=n_nulls, seed=0)
    
    assert nulls.shape == (len(test_data), n_nulls), \
        f"Expected shape {(len(test_data), n_nulls)}, got {nulls.shape}"

def test_no_nans_or_infs(solver, test_data):
    """Output should not contain NaNs or Infs"""
    nulls = eigenstrap(test_data, solver.emodes, solver.evals, mass=solver.mass, 
                      n_nulls=5, seed=0)
    
    assert not np.any(np.isnan(nulls)), "Nulls contain NaN values"
    assert not np.any(np.isinf(nulls)), "Nulls contain Inf values"

def test_invalid_residual_parameter(solver, test_data):
    """Should raise ValueError for invalid residual parameter"""
    with pytest.raises(ValueError, match="Invalid residual method"):
        eigenstrap(test_data, solver.emodes, solver.evals, mass=solver.mass,
                  residual='invalid')

def test_shape_mismatch_data_emodes(solver, test_data):
    """Should raise error when data length doesn't match emodes rows"""
    wrong_data = test_data[:-100]  # Truncate data
    
    with pytest.raises((ValueError, IndexError)):
        eigenstrap(wrong_data, solver.emodes, solver.evals, mass=solver.mass)

def test_shape_mismatch_emodes_evals(solver, test_data):
    """Should raise error when emodes columns doesn't match evals length"""
    wrong_evals = solver.evals[:-10]  # Truncate evals
    
    with pytest.raises(ValueError, match="must have shape"):
        eigenstrap(test_data, solver.emodes, wrong_evals, mass=solver.mass)

def test_resample_preserves_distribution(solver, test_data):
    """With resample=True, nulls should have same values as original data"""
    nulls = eigenstrap(test_data, solver.emodes, solver.evals, mass=solver.mass,
                      n_nulls=5, resample=True, seed=0)
    
    # Check that each null has the exact same values as original data (just reordered)
    for i in range(nulls.shape[1]):
        null_sorted = np.sort(nulls[:, i])
        data_sorted = np.sort(test_data)
        assert np.allclose(null_sorted, data_sorted), \
            f"Null {i} doesn't preserve data distribution"

def test_non_square_modes(test_data):
    """Should handle non-square n_modes by truncating last eigengroup with warning"""
    # Use 95 modes (not a perfect square)
    mesh, medmask = fetch_surf(density='32k')
    non_square_solver = EigenSolver(mesh, mask=medmask).solve(n_modes=95, seed=0)
    
    # Should complete with a warning about truncating last eigengroup
    with pytest.warns(UserWarning, match="not a perfect square.*Last eigengroup will be excluded"):
        nulls = eigenstrap(test_data, non_square_solver.emodes, non_square_solver.evals,
                          mass=non_square_solver.mass, n_nulls=3, seed=0)
    
    assert nulls.shape == (len(test_data), 3)
    assert not np.any(np.isnan(nulls))

def test_randomize_option(solver, test_data):
    """Test randomize parameter works without errors"""
    nulls = eigenstrap(test_data, solver.emodes, solver.evals, 
                      mass=solver.mass, n_nulls=5, randomize=True, seed=0)
    assert nulls.shape == (len(test_data), 5)
    assert not np.any(np.isnan(nulls))

def test_residual_methods(solver, test_data):
    """Test different residual methods run without errors"""
    for method in ['add', 'permute']:
        nulls = eigenstrap(test_data, solver.emodes, solver.evals, 
                          mass=solver.mass, n_nulls=5, residual=method, seed=0)
        assert nulls.shape == (len(test_data), 5)
        assert not np.any(np.isnan(nulls))
        