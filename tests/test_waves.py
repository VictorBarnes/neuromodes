import pytest
import os
from tempfile import TemporaryDirectory
import numpy as np
from neuromodes.io import fetch_surf
from neuromodes.eigen import EigenSolver
from neuromodes.waves import simulate_waves, estimate_wave_speed, get_balloon_params

@pytest.fixture
def solver():
    rng = np.random.default_rng(0)
    mesh, medmask = fetch_surf(density='4k')
    hetero = rng.standard_normal(size=sum(medmask))
    return EigenSolver(mesh, mask=medmask, hetero=hetero).solve(n_modes=200, seed=0)

def test_unusual_wave_speed(solver):
    with pytest.warns(UserWarning, match='non-physiological wave speeds'):
        solver.simulate_waves(r=1000)

def test_unusual_wave_speed_no_hetero(solver):
    with pytest.warns(UserWarning, match='non-physiological wave speeds'):
        simulate_waves(
            solver.emodes,
            solver.evals,
            mass=solver.mass,
            r=1500
        )

def test_simulate_waves_impulse(solver):

    # Simulate timeseries with a 10ms impulse of white noise to the cortex
    dt = 1 # ms
    nt = 200
    i_start = 10 # ms
    i_stop = 20 # ms
    rng = np.random.default_rng(0)
    impulse = rng.standard_normal(size=solver.n_verts)
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
    assert np.allclose(fourier_ts[:, :i_start], 0, atol=1e-3), \
        'Fourier activity is not negligible before impulse.'
    assert np.allclose(ode_ts[:, :i_start], 0, atol=1e-10), \
        'ODE activity is not negligible before impulse.'

    # Check that activity returns to negligible by 200ms
    assert np.allclose(fourier_ts[:, -1], 0, atol=1e-4), \
        'Fourier activity is not negligible after 200ms.'
    assert np.allclose(ode_ts[:, -1], 0, atol=1e-8), \
        'ODE activity is not negligible after 200ms.'

def test_simulate_waves_default_input(solver):

    nt = 100
    dt = 0.1
    seed = 1

    # Check that Fourier and ODE methods produce similar neural activity at selected timepoints

    fourier_ts = simulate_waves(
        solver.emodes,
        solver.evals,
        mass=solver.mass,
        nt=nt,
        dt=dt,
        seed=seed
    )
    ode_ts = simulate_waves(
        solver.emodes,
        solver.evals,
        mass=solver.mass,
        nt=nt,
        dt=dt,
        seed=seed,
        pde_method='ode'
    )

    for t in range(50, nt):
        assert np.corrcoef(fourier_ts[:, t], ode_ts[:, t])[0, 1] > 0.9, \
            f'Fourier and ODE solutions are not correlated at r>.9 at t={t}.'

    # Check that Fourier and ODE methods produce similar BOLD signal at selected timepoints

    bold_fourier = simulate_waves(
        solver.emodes,
        solver.evals,
        mass=solver.mass,
        nt=nt,
        dt=dt,
        bold_out=True,
        seed=seed
    )
    bold_ode = simulate_waves(
        solver.emodes,
        solver.evals,
        mass=solver.mass,
        nt=nt,
        dt=dt,
        bold_out=True,
        seed=seed,
        pde_method='ode'
    )

    for t in range(75, nt):
        assert np.corrcoef(bold_fourier[:, t], bold_ode[:, t])[0, 1] > 0.85, \
            f'Fourier and ODE BOLD solutions are not correlated at r>.85 at t={t}.'
        
    # Check that BOLD and neural activity are correlated at selected timepoints

    for t in range(75, nt):
        assert np.corrcoef(fourier_ts[:, t], bold_fourier[:, t])[0, 1] > 0.6, \
            f'Fourier neural and BOLD solutions are not correlated at r>.6 at t={t}.'
        assert np.corrcoef(ode_ts[:, t], bold_ode[:, t])[0, 1] > 0.6, \
            f'ODE neural and BOLD solutions are not correlated at r>.6 at t={t}.'

def test_simulate_waves_seed_bold_reproducibility_fourier(solver):
    
    nt = 100
    dt = 100
    seed = 36

    ts1 = simulate_waves(
        solver.emodes,
        solver.evals,
        mass=solver.mass,
        nt=nt,
        dt=dt,
        seed=seed,
        bold_out=True
    )
    ts2 = simulate_waves(
        solver.emodes,
        solver.evals,
        mass=solver.mass,
        nt=nt,
        dt=dt,
        seed=seed,
        bold_out=True
    )
    ts3 = simulate_waves(
        solver.emodes,
        solver.evals,
        mass=solver.mass,
        nt=nt,
        dt=dt,
        seed=seed+1,
        bold_out=True
    )

    assert np.allclose(ts1, ts2), "Simulations with the same seed do not match."
    assert not np.allclose(ts1, ts3), "Simulations with different seeds match unexpectedly."

def test_simulate_waves_invalid_input_shape(solver):

    with pytest.raises(ValueError, match=r"shape is \(4002, 1000\), should be \(3636, 1000\)."):
        simulate_waves(
            solver.emodes,
            solver.evals,
            ext_input=np.ones((4002, 1000)),
            mass=solver.mass
        )

def test_simulate_waves_invalid_pde_method(solver):

    with pytest.raises(ValueError, match="Invalid PDE method 'zote'"):
        simulate_waves(
            solver.emodes,
            solver.evals,
            mass=solver.mass,
            pde_method='zote'
        )

def test_simulate_waves_cached(solver):
    # Get CACHE_DIR
    cache_dir = os.getenv("CACHE_DIR")

    # Test with temporary directory
    with TemporaryDirectory() as temp_cache_dir:
        os.environ["CACHE_DIR"] = temp_cache_dir
        _ = simulate_waves(
            solver.emodes,
            solver.evals,
            mass=solver.mass,
            nt=10,
            cache_input=True
        )

        # Check that the temp_cache_dir/neuromodes/waves subdirectory exists
        cache_dir_waves = os.path.join(
            temp_cache_dir,
            "neuromodes",
            "waves"
        )
        assert os.path.exists(cache_dir_waves), "Waves cache directory was not created."


    # Restore original CACHE_DIR
    if cache_dir is not None:
        os.environ["CACHE_DIR"] = cache_dir
    else:
        del os.environ["CACHE_DIR"]

def test_simulate_waves_balloon_param(solver):
    nt = 100
    dt = 10

    ts_default = simulate_waves(
        solver.emodes,
        solver.evals,
        mass=solver.mass,
        nt=nt,
        dt=dt,
        bold_out=True
    )

    ts_custom = simulate_waves(
        solver.emodes,
        solver.evals,
        mass=solver.mass,
        nt=nt,
        dt=dt,
        bold_out=True,
        rho = 0.5
    )

    assert not np.allclose(ts_default, ts_custom), \
        "BOLD signals with different balloon model parameters match unexpectedly."
    
def test_get_balloon_params():

    # Check a default
    params = get_balloon_params()
    assert params['rho'] == 0.34, "Default parameter 'rho' is incorrect."

    # Check an override
    params = get_balloon_params(rho=0.5)
    assert params['rho'] == 0.5, "Overridden parameter 'rho' is incorrect."

    # Check an invalid override
    with pytest.raises(ValueError, match=r"\(received rho=0\)."):
        _ = get_balloon_params(rho=0)

    # Check an invalid parameter name
    with pytest.raises(ValueError, match="Invalid Balloon model parameter 'yoyoyo'."):
        _ = get_balloon_params(yoyoyo=1.0)

def test_estimate_wave_speed(solver):

    # Homogeneous case
    speed = estimate_wave_speed(r=18.0, gamma=0.116)
    assert isinstance(speed, float), "Output type is not float for `hetero=None`."

    # Heterogeneous case
    speed = estimate_wave_speed(r=18.0, gamma=0.116, scaled_hetero=solver.hetero)
    assert np.all(speed > 0), "Output contains non-positive wave speeds when using `scaled_hetero`."
    assert speed.shape == (solver.n_verts,), "Output shape is incorrect when using `scaled_hetero`."
    