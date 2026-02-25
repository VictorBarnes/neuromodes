import pytest
import numpy as np
from neuromodes.eigen import EigenSolver, get_eigengroup_inds
from neuromodes.io import fetch_surf, fetch_map
from neuromodes.nulls import eigenstrap

# Params
n_modes = 100
n_maps = 5
n_nulls = 1000
seed = 0

@pytest.fixture
def solver():
    mesh, medmask = fetch_surf(density='4k')
    return EigenSolver(mesh, mask=medmask).solve(n_modes=n_modes, seed=seed)

@pytest.fixture
def test_data(solver):
    """Generate 1D test data"""
    rng = np.random.default_rng(seed)
    return rng.normal(loc=1, size=(solver.n_verts, n_maps))  # random normal data, non-zero mean

def test_1_map(solver, test_data):
    """Generate nulls for 1D test data"""
    solver.eigenstrap(test_data[:,0], n_nulls=n_nulls, seed=seed)

def test_all_map(solver, test_data):
    """Generate nulls for 1D test data"""
    solver.eigenstrap(test_data, n_nulls=n_nulls, seed=seed)


