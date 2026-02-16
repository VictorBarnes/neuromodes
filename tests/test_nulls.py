import pytest
import numpy as np
from neuromodes.eigen import EigenSolver
from neuromodes.io import fetch_surf, fetch_map
from neuromodes.nulls import eigenstrap

@pytest.fixture
def solver():
    mesh, medmask = fetch_surf(density='32k')
    return EigenSolver(mesh, mask=medmask).solve(n_modes=100, seed=0)

def test_reproducibility(solver):
    emodes = solver.emodes
    evals = solver.evals
    mass = solver.mass
    medmask = solver.mask
    data = fetch_map('myelinmap', density='32k')[medmask]

    nulls1 = eigenstrap(data, emodes, evals, mass=mass, seed=0)
    nulls2 = eigenstrap(data, emodes, evals, mass=mass, seed=0)

    assert np.allclose(nulls1, nulls2), "Null spaces with the same seed should be identical"

# TODO: test when parallelization is implemented

# TODO: constant mode failure

# TODO: zero eval failure

# TODO: zero eval after removing constant mode failure

# TODO: NaN in data failure

# TODO: data shape mismatch failure

# TODO: n_modes is not square
