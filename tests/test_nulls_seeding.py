from pathlib import Path
import pytest
import numpy as np
from neuromodes.eigen import EigenSolver
from neuromodes.io import fetch_surf, fetch_map

# Params
density = '4k'
n_modes = 100
n_maps = 10
n_nulls = 20

@pytest.fixture(scope='module')
def solver(seed=None):
    mesh, medmask = fetch_surf(density=density)
    return EigenSolver(mesh, mask=medmask).solve(n_modes=n_modes, seed=seed)

@pytest.fixture(scope='module')
def test_data(solver):
    """Generate 1D test data"""
    rng = np.random.default_rng(1)
    return rng.normal(loc=1, size=(solver.n_verts, n_maps))  # random normal data, non-zero mean

def test_min(solver, test_data):
    """Brief test to ensure that everything still runs"""
    solver.eigenstrap(test_data, n_nulls=n_nulls)

@pytest.mark.parametrize("rotation_method", ['qr', 'scipy'])
def test_different_maps(solver, test_data, rotation_method): 
    """For different maps, each null should be different to each other null"""
    nulls = solver.eigenstrap(test_data, n_nulls=n_nulls, rotation_method=rotation_method)

    for i in range(n_maps): 
        for j in range(i): 
            assert not np.allclose(nulls[:,:,i], nulls[:,:,j]), \
                f"Nulls for map {i} should be different to nulls for map {j}"
    for i in range(n_nulls): 
        for j in range(i): 
            assert not np.allclose(nulls[:,i,:], nulls[:,j,:]), \
                f"Nulls {i} should be different to nulls {j}"

@pytest.mark.parametrize("rotation_method", ['qr', 'scipy'])
def test_same_map(solver, test_data, rotation_method): 
    """For the same map, nulls should be the same (and different for different maps)"""
    test_data = np.column_stack((test_data[:,0], test_data[:,0], np.random.standard_normal(size=solver.n_verts)))
    nulls = solver.eigenstrap(test_data, n_nulls=n_nulls, rotation_method=rotation_method)

    assert not np.allclose(nulls[:,:,1], nulls[:,:,2]), \
        f"Nulls for map 1 should be different to nulls for map 2"
    assert np.allclose(nulls[:,:,1], nulls[:,:,0]), \
        f"Nulls for map 1 should be the same as nulls for map 0"

@pytest.mark.parametrize("rotation_method", ['qr', 'scipy'])
def test_same_and_different_seeds(solver, test_data, rotation_method): 
    """Test the results based on seeding"""
    a1 = solver.eigenstrap(test_data, n_nulls=n_nulls, seed=1, rotation_method=rotation_method)
    a2 = solver.eigenstrap(test_data, n_nulls=n_nulls, seed=1, rotation_method=rotation_method)
    b1 = solver.eigenstrap(test_data, n_nulls=n_nulls, seed=2, rotation_method=rotation_method)
    assert np.allclose(a1, a2), \
        "Nulls generated with the same seed should be the same"
    assert not np.allclose(a1, b1), \
        "Nulls generated with different seeds should be different"

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

# TODO : more reproducibility tests
# test that resample and randomize functions are working correctly incl. with other parameters