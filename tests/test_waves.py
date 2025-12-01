import pytest
import numpy as np
from nsbtools.io import fetch_surf
from nsbtools.eigen import EigenSolver
from nsbtools.waves import simulate_waves

@pytest.fixture
def solver():
    mesh, medmask = fetch_surf(density='4k')
    return EigenSolver(mesh, mask=medmask).solve(n_modes=100)

def test_simulate_waves_impulse(solver):

    # Simulate timeseries with a 10ms impulse of white noise to the cortex
    dt = 1 # ms
    nt = 200
    i_start = 10 # ms
    i_stop = 20 # ms
    np.random.seed(0)
    impulse = np.random.normal(loc=0.5, scale=0.5, size=solver.n_verts)
    ext_input = np.zeros((solver.n_verts, nt))
    ext_input[:, i_start:i_stop] = impulse[:, np.newaxis]

    fourier_ts = simulate_waves(
        solver.emodes,
        solver.evals,
        ext_input=ext_input,
        mass=solver.mass,
        nt=nt,
        dt=dt
    )
    ode_ts = simulate_waves(
        solver.emodes,
        solver.evals,
        ext_input=ext_input,
        mass=solver.mass,
        nt=nt,
        dt=dt,
        pde_method='ode'
    )

    # Check output shapes
    assert fourier_ts.shape == (solver.n_verts, nt), 'Fourier output shape is incorrect.'
    assert ode_ts.shape == (solver.n_verts, nt), 'ODE output shape is incorrect.'

    # Check that activity is negligible before impulse
    assert np.allclose(fourier_ts[:, :i_start], 0, atol=1e-4), \
        'Fourier activity is not negligible before impulse.'
    assert np.allclose(ode_ts[:, :i_start], 0, atol=1e-10), \
        'ODE activity is not negligible before impulse.'

    # Check that activity returns to negligible by 200ms
    assert np.allclose(fourier_ts[:, -1], 0, atol=1e-4), \
        'Fourier activity is not negligible after 200ms.'
    assert np.allclose(ode_ts[:, -1], 0, atol=1e-8), \
        'ODE activity is not negligible after 200ms.'

def test_simulate_waves_default_input(solver):

    nt = 200
    dt = 0.1

    fourier_ts = simulate_waves(
        solver.emodes,
        solver.evals,
        mass=solver.mass,
        nt=nt,
        dt=dt,
        seed=0
    )
    ode_ts = simulate_waves(
        solver.emodes,
        solver.evals,
        mass=solver.mass,
        nt=nt,
        dt=dt,
        seed=0,
        pde_method='ode'
    )

    # Check that Fourier and ODE methods produce similar activity at selected timepoints
    for t in range(60, nt):
        assert np.corrcoef(fourier_ts[:, t], ode_ts[:, t])[0, 1] > 0.99, \
            f'Fourier and ODE solutions are not correlated at r>.99 at t={t}.'

def test_simulate_waves_bold(solver):

    nt = 10
    dt = 0.1

    bold_fourier = simulate_waves(
        solver.emodes,
        solver.evals,
        mass=solver.mass,
        nt=nt,
        dt=dt,
        bold_out=True,
        seed=0
    )
    bold_ode = simulate_waves(
        solver.emodes,
        solver.evals,
        mass=solver.mass,
        nt=nt,
        dt=dt,
        bold_out=True,
        seed=0,
        pde_method='ode'
    )

    assert bold_fourier.shape == (solver.n_verts, nt), 'Fourier BOLD output shape is incorrect.'
    assert bold_ode.shape == (solver.n_verts, nt), 'ODE BOLD output shape is incorrect.'

def test_simulate_waves_invalid_input_shape(solver):

    with pytest.raises(ValueError, match=r".*shape is \(4002, 1000\), should be \(3636, 1000\)."):
        simulate_waves(
            solver.emodes,
            solver.evals,
            ext_input=np.ones((4002, 1000)),
            mass=solver.mass
        )

def test_simulate_waves_invalid_pde_method(solver):

    with pytest.raises(ValueError, match="Invalid PDE method 'zote'.*"):
        simulate_waves(
            solver.emodes,
            solver.evals,
            mass=solver.mass,
            pde_method='zote'
        )