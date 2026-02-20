import pytest
import numpy as np
from neuromodes.eigen import EigenSolver, get_eigengroup_inds
from neuromodes.io import fetch_surf, fetch_map
from neuromodes.nulls import eigenstrap

# Params
n_nulls = 1000
seed = 0

@pytest.fixture
def solver():
    mesh, medmask = fetch_surf(density='4k')
    return EigenSolver(mesh, mask=medmask).solve(n_modes=100, seed=seed)

@pytest.fixture
def test_data(solver):
    """Generate 1D test data"""
    rng = np.random.default_rng(seed)
    return rng.normal(loc=1, size=solver.n_verts)  # random normal data, non-zero mean

@pytest.fixture
def nulls(solver, test_data):
    """Generate nulls for 1D test data"""
    return solver.eigenstrap(test_data, n_nulls=n_nulls, seed=seed)

def test_output_shape(solver, nulls):
    """Output shape should be (n_verts, n_nulls)"""    
    assert nulls.shape == (solver.n_verts, n_nulls), \
        f"Expected shape {(solver.n_verts, n_nulls)}, got {nulls.shape}"

def test_internull_corrs(nulls):
    """Internull correlations should be centered around zero"""
    # Plot null correlation distribution
    inter_null_corrs = np.corrcoef(nulls.T)
    triu_inds = np.triu_indices_from(inter_null_corrs, k=1)
    mean_corr = inter_null_corrs[triu_inds].mean()
    assert np.abs(mean_corr) < 0.01, \
        f"Mean internull correlation should be close to zero, got {mean_corr:.3f}"
    
def test_mean_preservation(test_data, nulls):
    """Nulls should approximately preserve mean of original data"""
    data_mean = np.mean(test_data)
    null_means = np.mean(nulls, axis=0)
    
    for i, null_mean in enumerate(null_means):
        assert np.abs(null_mean - data_mean) < 0.02, \
            f"Null {i} mean {null_mean} is not close to data mean {data_mean}"
        
def test_psd_preservation():
    """Nulls should approximately preserve eigengroup power spectral density of original data"""
    # Need real data for this test since random data won't have meaningful PSD structure
    # TODO: add 4k myelinmap to avoid 32k solving, then use 4k myelinmap in all tests for realism
    surf, medmask = fetch_surf()
    myelinmap = fetch_map('myelinmap')[medmask]

    solver = EigenSolver(surf, mask=medmask).solve(n_modes=6**2, seed=seed)
    nulls = solver.eigenstrap(myelinmap, n_nulls=n_nulls, seed=seed)

    groups = get_eigengroup_inds(solver.n_modes)

    n_groups = len(groups)
    group_indices = np.concatenate([np.full(len(group), i) for i, group in enumerate(groups)])

    beta0 = solver.decompose(myelinmap)
    psd0 = np.bincount(group_indices, weights=beta0.ravel()**2)[1:]  # Exclude constant mode

    beta1 = solver.decompose(nulls)
    psd1 = np.array([np.bincount(group_indices, weights=beta1[:, i].ravel()**2)
                    for i in range(n_nulls)])[:, 1:]  # Exclude constant mode
    psd1_mean = psd1.mean(axis=0)

    for i in range(n_groups - 1):
        assert np.allclose(psd1_mean[i], psd0[i], rtol=0.2), \
                f"Nulls do not preserve PSD for eigengroup {i+2}: " \
                f"data PSD={psd0[i]:.3f}, nulls mean PSD={psd1_mean[i]:.3f}"
    
def test_reproducibility(solver, test_data, nulls):
    """Nulls with same seed should be identical"""
    nulls2 = solver.eigenstrap(test_data, n_nulls=n_nulls, seed=seed)
    
    assert np.allclose(nulls, nulls2, atol=1e-10), \
        "Null spaces with the same seed should be identical"

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

def test_resample_affine(solver, test_data):
    """With resample='affine', nulls should have mean and std that match the data"""
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
        assert np.isfinite(nulls).all(), \
            f"Nulls contain non-finite values for residual method '{method}'"

def test_non_square_modes(test_data):
    """Should handle non-square n_modes by truncating last eigengroup with warning"""
    # Use 8 modes (not a perfect square)
    mesh, medmask = fetch_surf(density='4k')
    non_square_solver = EigenSolver(mesh, mask=medmask).solve(n_modes=8, seed=seed)
    
    # Should complete with a warning about truncating last eigengroup
    with pytest.warns(UserWarning, match="These last 4 modes will be excluded."):
        nulls = non_square_solver.eigenstrap(test_data, n_nulls=n_nulls, seed=seed, residual='add')
    
    assert nulls.shape == (non_square_solver.n_verts, n_nulls)

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

def test_resample_exact_2d(solver, test_data_2d):
    """With resample=True, nulls should have same values as original data"""
    nulls = solver.eigenstrap(test_data_2d, n_nulls=n_nulls, resample="exact", seed=seed)
    
    # Check that each null has the exact same values as original data (just reordered)
    # TODO: vectorise if slow?
    for j in range(test_data_2d.shape[1]):
        for i in range(nulls.shape[1]):
            null_sorted = np.sort(nulls[:, i, j])
            data_sorted = np.sort(test_data_2d[:, j])
            assert np.allclose(null_sorted, data_sorted), \
                f"Null {i} doesn't preserve data distribution for map {j}"

def test_resample_affine_2d(solver, test_data_2d):
    """With resample='affine', nulls should have mean and std that match the data"""
    nulls = solver.eigenstrap(test_data_2d, n_nulls=n_nulls, resample="affine", seed=seed)
    
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
            assert np.isclose(null_mean, data_mean), \
                f"Null {i} map {j} mean is not close to data mean"

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
        assert np.isfinite(nulls).all(), \
            f"Nulls contain non-finite values for residual method '{method}'"