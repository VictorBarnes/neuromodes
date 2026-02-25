import pytest
import time
import numpy as np
from neuromodes.eigen import EigenSolver, get_eigengroup_inds
from neuromodes.io import fetch_surf, fetch_map
from neuromodes.nulls import eigenstrap

# This test should be run with the `-s` flag to print timing results, e.g. `pytest -s tests/test_nulls_speed.py`

# Params
# These will be used for the bulk of the profiling, in the `test_all` function. 
# A smaller set of tests on the 32k mesh is below. 
density = '4k'
n_groups = (10,15,20)
n_maps = (1,10,100)
n_nulls = (1,10,100)
seed = 0

@pytest.fixture(scope='module')
def solver():
    """Initialise solver and solve for eigenmodes, which will be used for all tests."""
    mesh, medmask = fetch_surf(density=density)
    print(f"\nInitilising mesh with {density} vertices and {max(n_groups)**2} modes.")
    tic = time.time()
    s = EigenSolver(mesh, mask=medmask).solve(n_modes=max(n_groups)**2, seed=seed)
    print(f"Time to solve eigenmodes: {time.time() - tic:.5f} seconds.\n")
    return s

@pytest.fixture(scope='module')
def test_data(solver):
    """Generate 2D test data"""
    rng = np.random.default_rng(seed)
    return rng.normal(loc=1, size=(solver.n_verts, max(n_maps)))  # random normal data, non-zero mean

def test_one(solver, test_data):
    """Run one test (profile with `pytest tests/test_nulls_speed.py::test_one --profile`)."""
    n_null = max(n_nulls)
    n_map = max(n_maps)
    n_group = max(n_groups)
    tic = time.time()
    solver.eigenstrap(test_data[:,:n_map], n_nulls=n_null, n_groups=n_group, seed=seed)
    toc = time.time()
    print(f"Time to generate {n_null} nulls for {n_map} maps with {n_group} groups: {toc - tic:.5f} seconds.")

def test_all(solver, test_data):
    """Generate nulls for test data"""
    s = 12 # for spacing only
    for n_group in n_groups:
        print(f"\n{n_group**2} modes:")
        print(f"{'Maps | Nulls'}", end="")
        for n_null in n_nulls:
            print(f"{n_null:>{s}}", end="")
        print()
        for n_map in n_maps:
            print(f"{n_map:<{s}}", end="")
            for n_null in n_nulls:
                tic = time.time()
                solver.eigenstrap(test_data[:,:n_map], n_nulls=n_null, n_groups=n_group, seed=seed)
                toc = time.time()
                print(f"{toc - tic:>{s}.6f}", end="")
            print()
              
def test_32k():
    """Test on 32k mesh with specific number of nulls/maps/modes."""
    density='32k'
    n_modes = 100 # should be square number
    n_maps = 100
    n_nulls = 100

    print(f"\nInitialising mesh with {density} vertices and {n_modes} modes.")
    tic = time.time()
    solver = EigenSolver(fetch_surf(density=density)[0]).solve(n_modes=n_modes, seed=seed)
    print(f"Time to solve eigenmodes: {time.time() - tic:.5f} seconds.")

    test_data = np.random.default_rng(seed).normal(size=(solver.n_verts, n_maps))

    tic = time.time()
    solver.eigenstrap(test_data, n_nulls=n_nulls, seed=seed)
    print(f"Nulls generated for {n_maps} maps with {n_nulls} nulls: {time.time() - tic:.5f} seconds.")

# TODO create a test that compares n_maps with n_nulls
# ie for a given density and n_modes, what is the difference between 1 map/100 nulls and 100 maps/1 null
