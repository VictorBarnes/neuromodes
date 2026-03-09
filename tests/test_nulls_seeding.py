from pathlib import Path
import pytest
import numpy as np
from neuromodes.eigen import EigenSolver
from neuromodes.io import fetch_surf, fetch_map

# Params
density = '4k'
n_modes = 100 # should be a square number
n_maps = 3
n_nulls = 20

@pytest.fixture(scope='module')
def solver(seed=None):
    mesh, medmask = fetch_surf(density=density)
    return EigenSolver(mesh, mask=medmask).solve(n_modes=n_modes, seed=seed)

@pytest.fixture(scope='module')
def test_data(solver):
    """Generate test data""" # random normal data, non-zero mean
    return np.random.default_rng(None).normal(loc=1, size=(solver.n_verts, n_maps))  

@pytest.mark.parametrize("seed", [None, 0, 42, np.random.randint(100,size=n_nulls)])
def test_seed_options(solver, test_data, seed):
    """Test different seed options run without errors (1D data)"""
    test_data_1d = test_data[:, 0]
    nulls = solver.eigenstrap(test_data_1d, n_nulls=n_nulls, seed=seed)
    assert nulls.shape == (solver.n_verts, n_nulls)
    assert np.isfinite(nulls).all(), \
        f"Nulls contain non-finite values for seed={seed}"

@pytest.mark.parametrize("seed", [None, 0, 42, np.random.randint(100,size=n_nulls)])
def test_seed_options_2d(solver, test_data, seed):
    """Test different seed options run without errors (2D data)"""
    nulls = solver.eigenstrap(test_data, n_nulls=n_nulls, seed=seed)
    assert nulls.shape == (solver.n_verts, n_nulls, test_data.shape[1])
    assert np.isfinite(nulls).all(), \
        f"Nulls contain non-finite values for seed={seed}"
    
@pytest.mark.parametrize("rotation_method", ['qr', 'scipy'])
@pytest.mark.parametrize("randomize", [True, False])
@pytest.mark.parametrize("residual", [None, 'add', 'permute']) # skip resample and decomp method as they are not related to seeding
@pytest.mark.parametrize("seed", [0, 42, np.arange(n_nulls)])
def test_same_and_different_seeds(solver, test_data, rotation_method, randomize, residual, seed): 
    """Test the results based on seeding"""
    a1 = solver.eigenstrap(test_data, n_nulls=n_nulls, seed=seed, rotation_method=rotation_method, randomize=randomize, residual=residual)
    a2 = solver.eigenstrap(test_data, n_nulls=n_nulls, seed=seed, rotation_method=rotation_method, randomize=randomize, residual=residual)
    b1 = solver.eigenstrap(test_data, n_nulls=n_nulls, seed=seed+1, rotation_method=rotation_method, randomize=randomize, residual=residual)
    
    assert np.allclose(a1, a2), \
        "Nulls generated with the same seed should be the same"
    assert not np.allclose(a1, b1), \
        "Nulls generated with different seeds should be different"
    
@pytest.mark.parametrize("rotation_method", ['qr', 'scipy'])
def test_seed_none(solver, test_data, rotation_method): 
    """Test the results based on seeding"""
    a1 = solver.eigenstrap(test_data, n_nulls=n_nulls, seed=None, rotation_method=rotation_method)
    b1 = solver.eigenstrap(test_data, n_nulls=n_nulls, seed=None, rotation_method=rotation_method)
    
    assert not np.allclose(a1, b1), \
        "Nulls generated with None seeds should be different by default"

def test_seed_global_scipy(solver, test_data): 
    """Setting the global seed should produce the same nulls (to maintain compatibility with original implementation)"""
    np.random.seed(1) # set global seed
    a1 = solver.eigenstrap(test_data, n_nulls=n_nulls, seed=None, rotation_method="scipy")
    np.random.seed(1) # reset global seed
    a2 = solver.eigenstrap(test_data, n_nulls=n_nulls, seed=None, rotation_method="scipy")
    np.random.seed(2) # change global seed
    b1 = solver.eigenstrap(test_data, n_nulls=n_nulls, seed=None, rotation_method="scipy")
    
    assert np.allclose(a1, a2), \
        f"Nulls generated with the same global seed should be identical for seed=None and rotation_method='scipy'"
    assert not np.allclose(a1, b1), \
        f"Nulls generated with different global seeds should be different for seed=None and rotation_method='scipy'"

def test_seed_global_qr(solver, test_data): 
    """Setting the global seed should not affect nulls generated with QR rotation"""
    np.random.seed(1) # set global seed
    np.random.default_rng(1) # neither of these change inner state for QR rotation
    a1 = solver.eigenstrap(test_data, n_nulls=n_nulls, seed=None, rotation_method="qr")
    np.random.seed(1)
    np.random.default_rng(1)
    b1 = solver.eigenstrap(test_data, n_nulls=n_nulls, seed=None, rotation_method="qr")
    
    assert not np.allclose(a1, b1), \
        f"Nulls generated with the same global seed should not be identical for seed=None and rotation_method='qr'"

@pytest.mark.parametrize("rotation_method", ['qr', 'scipy'])
@pytest.mark.parametrize("seed", [0, 42, np.arange(n_nulls)])
def test_specific_seed_not_affected_by_global_seed(solver, test_data, rotation_method, seed): 
    """Setting the global seed should not affect nulls generated with a specific seed (to maintain compatibility with original implementation)"""
    np.random.seed(1) # set global seed
    a1 = solver.eigenstrap(test_data, n_nulls=n_nulls, seed=seed, rotation_method=rotation_method)
    np.random.seed(2) # change global seed
    a2 = solver.eigenstrap(test_data, n_nulls=n_nulls, seed=seed, rotation_method=rotation_method)
    
    assert np.allclose(a1, a2), \
        f"Nulls generated with the same specific seed (seed={seed}) should be identical regardless of global seed for rotation_method={rotation_method}"

@pytest.mark.parametrize("rotation_method", ['scipy', 'qr'])
def test_reproducibility_number_nulls(solver, test_data, rotation_method):
    """Nulls with same seed should be identical, regardless of number of nulls requested"""
    a1 = solver.eigenstrap(test_data, n_nulls=n_nulls, seed=1, rotation_method=rotation_method)
    a2 = solver.eigenstrap(test_data, n_nulls=n_nulls-1, seed=1, rotation_method=rotation_method)
    
    assert np.allclose(a1[:,:-1], a2, atol=1e-10), \
        f"Nulls with the same seed should be identical"

@pytest.mark.parametrize("rotation_method", ['scipy', 'qr'])
def test_reproducibility_number_data(solver, test_data, rotation_method):
    """Nulls with same seed should be identical, regardless of number of input maps"""
    a1 = solver.eigenstrap(test_data, n_nulls=n_nulls, seed=1, rotation_method=rotation_method)
    a2 = solver.eigenstrap(test_data[:,:-1], n_nulls=n_nulls, seed=1, rotation_method=rotation_method)
    
    assert np.allclose(a1[:,:,:-1], a2, atol=1e-10), \
        f"Nulls with the same seed should be identical"

def test_compared_to_original(): 
    # These parameters are hard coded to match data saved in the repo and should not be changed
    density = '4k'
    hemi = 'L'
    surf_type = 'midthickness'
    n_modes = 10**2
    n_nulls = 100
    seed = 365
    data = 'myelinmap'

    # Load original nulls
    test_data = Path(__file__).parent / 'test_data'
    nulls_file = f"sp-human_tpl-fsLR_den-{density}_hemi-{hemi}_{surf_type}_eigenstrap-nulls-orig.npy"
    nulls_orig = np.load(test_data / nulls_file)

    # Load data
    mesh, medmask = fetch_surf(density=density, hemi=hemi, surf_type=surf_type)
    map = fetch_map(data, density=density)[medmask]
    map = (map - np.mean(map)) # to match original implementation which doesn't use the constant mode

    # Compute new nulls
    solver = EigenSolver(mesh, mask=medmask).solve(n_modes, fix_mode1=True)
    np.random.seed(seed)            # match eigenstrapping
    nulls_neuromodes = solver.eigenstrap(
        data=map,
        n_nulls=n_nulls,
        seed=None,                  # matches original seed=seed 
        residual=None,              # matches original add_res=False and permute=False
        resample="range",           # matches original resample=False
        decomp_method="regress",    # matches original decomp_method='matrix'
        rotation_method="scipy",    # matches original rotations.indirect_method (called by geometry.gen_eigensamples)
    )

    # Compare (on diagonal near 1, off diagonal near 0)
    null_corrs = np.corrcoef(nulls_neuromodes.T, nulls_orig.T)[:n_nulls, n_nulls:]

    diagonal_corrs = np.diagonal(null_corrs)
    assert np.allclose(diagonal_corrs, 1.0, atol=0.001), \
        'New nulls should be similar to corresponding old nulls'

    column_mean = np.mean(null_corrs - np.diag(diagonal_corrs), axis=0)
    assert np.allclose(column_mean, 0.0, atol=0.1), \
        'New nulls should not be similar to different old nulls'
    assert np.allclose(np.mean(column_mean), 0.0, atol=0.001), \
        'New nulls should not be similar to different old nulls'


