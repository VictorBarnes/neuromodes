import os
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import pytest
from scipy.sparse.linalg import splu

from neuromodes import EigenSolver
from neuromodes.io import fetch_example_map, fetch_example_surf
from neuromodes.stats import cdistw, gramw, sigmoid_rescale, zscorew
from neuromodes.waves import (
    _gen_noise,
    balloon_model,
    calc_nft_fc,
    calc_nft_mode_freqs,
    calc_nft_wave_speed,
    sim_nft_waves,
)


@pytest.fixture(scope="module")
def solver():
    mesh, medmask = fetch_example_surf(res='4k')
    myelinmap = fetch_example_map(data="myelinmap", res="4k")[medmask]
    solver = EigenSolver(mesh, mask=medmask)
    hetero = sigmoid_rescale(zscorew(myelinmap, solver.mass), steepness=1.0, upper=2.0)
    return solver.solve(n_modes=100, hetero=hetero)

@pytest.fixture(scope="module")
def solver_float32():
    mesh, medmask = fetch_example_surf(res='4k')
    myelinmap = fetch_example_map(data="myelinmap", res="4k")[medmask]
    solver = EigenSolver(mesh, mask=medmask, dtype=np.float32)
    hetero = sigmoid_rescale(zscorew(myelinmap, solver.mass), steepness=1.0, upper=2.0)
    return solver.solve(n_modes=100, hetero=hetero)

def test_float32_wave_dtype(solver_float32):
    assert solver_float32.emodes.dtype == np.float32
    assert solver_float32.evals.dtype == np.float32
    assert solver_float32.mass.dtype == np.float32
    assert solver_float32.stiffness.dtype == np.float32
    assert solver_float32.hetero.dtype == np.float32
    
    inputs = {
        'emodes': solver_float32.emodes,
        'evals': solver_float32.evals,
        'mass': solver_float32.mass,
        'stiffness': solver_float32.stiffness,
        'hetero': solver_float32.hetero,
        'checks': 'maps'
    }
    kwargs = {
        'ext_input': np.random.default_rng(0).standard_normal((solver_float32.n_verts, 100)),
        'dt': 0.01,
        'r': 0.1,
        'gamma': 1.0,
        'speed_limits': None,
        'fft_pad_tol': 'nt'
    }
    for method in ('fourier', 'ode', 'fem'):
        activity = solver_float32.sim_nft_waves(method=method, **kwargs)
        assert activity.dtype == np.float32

        standalone_activity = sim_nft_waves(
            **inputs,
            dtype=np.float32,
            method=method,
            **kwargs
        )
        np.testing.assert_array_equal(activity, standalone_activity)

        reference_activity = sim_nft_waves(
            **inputs,
            dtype=np.float64,
            method=method,
            **kwargs
        )
        # Check that float32 and float64 outputs are close to each other
        np.testing.assert_allclose(activity, reference_activity, rtol=1e-3, atol=1e-5)

        generated_activity = solver_float32.sim_nft_waves(
            nt=100,
            dt=kwargs['dt'],
            r=kwargs['r'],
            gamma=kwargs['gamma'],
            speed_limits=None,
            seed=0,
            fft_pad_tol=kwargs['fft_pad_tol'],
            method=method
        )
        assert generated_activity.dtype == np.float32
    activity = np.random.default_rng(0).standard_normal((solver_float32.n_verts, 100))
    for method in ('fourier', 'ode'):
        bold = solver_float32.balloon_model(
            activity, dt=0.01, method=method, fft_pad_tol='nt'
        )
        assert bold.dtype == np.float32

        standalone_bold = balloon_model(
            activity=activity,
            emodes=solver_float32.emodes,
            dt=0.01,
            method=method,
            fft_pad_tol='nt',
            mass=solver_float32.mass,
            checks='maps',
            dtype=np.float32
        )
        np.testing.assert_array_equal(bold, standalone_bold)
        reference_bold = balloon_model(
            activity=activity,
            emodes=solver_float32.emodes,
            dt=0.01,
            method=method,
            fft_pad_tol='nt',
            mass=solver_float32.mass,
            checks='maps',
            dtype=np.float64
        )
        # Check that float32 and float64 outputs are close to each other
        np.testing.assert_allclose(bold, reference_bold, rtol=1e-3, atol=1e-5)

def test_wave_modal_input_and_output_space(solver):
    vertex_input = np.random.default_rng(0).standard_normal((solver.n_verts, 100))
    modal_input = solver.decompose(vertex_input)
    kwargs = {
        'dt': 0.01,
        'r': 0.1,
        'gamma': 1.0,
        'fft_pad_tol': 'nt'
    }

    for method in ('fourier', 'ode'):
        modal_from_vertex = solver.sim_nft_waves(
            ext_input=vertex_input,
            method=method,
            output_space='modal',
            **kwargs
        )
        modal_from_modal = solver.sim_nft_waves(
            ext_input=modal_input,
            method=method,
            output_space='modal',
            **kwargs
        )
        vertex_output = solver.sim_nft_waves(
            ext_input=vertex_input,
            method=method,
            **kwargs
        )
        vertex_from_modal = solver.sim_nft_waves(
            ext_input=modal_input,
            method=method,
            **kwargs
        )
        generated_modal = solver.sim_nft_waves(
            nt=100, seed=0, method=method, output_space='modal', **kwargs
        )

        assert modal_from_modal.shape == (solver.n_modes, 100)
        assert generated_modal.shape == (solver.n_modes, 100)
        np.testing.assert_allclose(modal_from_vertex, modal_from_modal, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(vertex_output, vertex_from_modal, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(
            vertex_output, solver.emodes @ modal_from_modal, rtol=1e-5, atol=1e-8
        )

def test_balloon_modal_input_and_output_space(solver):
    vertex_activity = np.random.default_rng(0).standard_normal((solver.n_verts, 100))
    modal_activity = solver.decompose(vertex_activity)

    for method in ('fourier', 'ode'):
        modal_from_vertex = solver.balloon_model(
            vertex_activity, dt=0.01, method=method, fft_pad_tol='nt', output_space='modal'
        )
        modal_from_modal = solver.balloon_model(
            modal_activity, dt=0.01, method=method, fft_pad_tol='nt', output_space='modal'
        )
        vertex_output = solver.balloon_model(
            modal_activity, dt=0.01, method=method, fft_pad_tol='nt'
        )

        assert modal_from_modal.shape == (solver.n_modes, 100)
        np.testing.assert_allclose(modal_from_vertex, modal_from_modal, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(
            vertex_output, solver.emodes @ modal_from_modal, rtol=1e-5, atol=1e-8
        )

def test_modal_wave_to_bold_pipeline(solver):
    vertex_input = np.random.default_rng(1).standard_normal((solver.n_verts, 100))
    kwargs = {
        'dt': 0.01,
        'r': 0.1,
        'gamma': 1.0,
        'fft_pad_tol': 'nt'
    }

    activity = solver.sim_nft_waves(ext_input=vertex_input, **kwargs)
    bold_vertex = solver.balloon_model(activity, dt=kwargs['dt'], fft_pad_tol='nt')

    activity_modal = solver.sim_nft_waves(
        ext_input=vertex_input, output_space='modal', **kwargs
    )
    bold_modal = solver.balloon_model(
        activity_modal, dt=kwargs['dt'], fft_pad_tol='nt', output_space='modal'
    )
    sampled_bold = solver.emodes @ bold_modal[:, 50::2]

    np.testing.assert_allclose(sampled_bold, bold_vertex[:, 50::2], rtol=1e-5, atol=1e-8)

def test_modal_input_shape_validation():
    # Define dummy inputs with ambiguous first dimension (n_verts vs n_modes)
    emodes = np.ones((10, 10))  # 10 vertices, 10 modes
    evals = np.ones((10,)) # 10 modes
    mass = np.ones((10, 10))  # 10 vertices
    ext_input = np.ones((10, 10))  # Ambiguous: could be 10 vertices or 10 modes
    with pytest.warns(UserWarning, match='ambiguous first dimension'):
        wave_output = sim_nft_waves(
            emodes=emodes,
            evals=evals,
            mass=mass,
            ext_input=ext_input,
            checks=False
        )

    activity = np.ones((10, 10))  # Ambiguous: could be 10 vertices or 10 modes
    with pytest.warns(UserWarning, match='ambiguous first dimension'):
        bold_output = balloon_model(
            activity=activity,
            emodes=emodes,
            dt=0.01,
            mass=mass,
            checks=False
        )
    assert wave_output.shape == (10, 10)
    assert bold_output.shape == (10, 10)

def test_modal_space_is_not_supported_by_fem(solver):
    with pytest.raises(ValueError, match="ext_input must be in vertex space"):
        solver.sim_nft_waves(
            ext_input=np.ones((solver.n_modes, 8)),
            method='fem',
            speed_limits=None
        )
    with pytest.raises(ValueError, match="output_space='modal'"):
        solver.sim_nft_waves(
            ext_input=np.ones((4, 8)),
            method='fem',
            output_space='modal',
            speed_limits=None
        )

def test_invalid_output_space(solver):
    with pytest.raises(ValueError, match="output_space must be either"):
        solver.sim_nft_waves(nt=100, output_space='invalid')
    with pytest.raises(ValueError, match="output_space must be either"):
        solver.balloon_model(
            np.ones((solver.n_verts, 100)),
            dt=0.01, 
            output_space='invalid'
        )

def test_unusual_wave_speed(solver):
    with pytest.warns(UserWarning, match=r'range of 0-150 m/s \(calculated 47.1-162.6 m/s\).'):
        solver.sim_nft_waves(r=1000, nt=10)

def test_unusual_wave_speed_no_hetero(solver):
    with pytest.warns(UserWarning, match=r'range of 0-115 m/s \(calculated 116.0 m/s\).'):
        sim_nft_waves(
            solver.emodes,
            solver.evals,
            mass=solver.mass,
            r=1000,
            speed_limits=(0, 115),
            nt=10
            )

def test_single_speed_limit(solver):
    with pytest.raises(ValueError, match="speed_limits must be a tuple"):
        solver.sim_nft_waves(nt=10, r=18.0, speed_limits=150)

def test_reversed_speed_limits(solver):
    with pytest.raises(ValueError, match="speed_limits must be a tuple"):
        solver.sim_nft_waves(nt=10, r=18.0, speed_limits=(150, 0))

def test_sim_nft_waves_impulse(solver):

    # Simulate timeseries with a 10ms impulse of white noise to the cortex
    dt = 1e-3
    nt = 200
    i_start = 10 # ms
    i_stop = 20 # ms
    rng = np.random.default_rng(0)
    impulse = rng.standard_normal(size=solver.n_verts)
    ext_input = np.zeros((solver.n_verts, nt))
    ext_input[:, i_start:i_stop] = impulse[:, np.newaxis]

    fourier_ts = solver.sim_nft_waves(ext_input=ext_input, dt=dt)
    ode_ts = solver.sim_nft_waves(ext_input=ext_input, dt=dt, method='ode')

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

def test_sim_nft_waves_methods(solver):

    nt = 100
    dt = 1e-4
    seed = 0

    # Check that Fourier and ODE methods produce similar neural activity at selected timepoints
    fourier_ts = solver.sim_nft_waves(nt=nt, dt=dt, seed=seed)
    ode_ts = solver.sim_nft_waves(nt=nt, dt=dt, seed=seed, method='ode')

    for t in range(50, nt):
        assert np.corrcoef(fourier_ts[:, t], ode_ts[:, t])[0, 1] > 0.85, \
            f'Fourier and ODE solutions are not correlated at r>.85 at t={t}.'

def test_sim_nft_waves_methods_bold(solver):

    nt = 100
    dt = 1e-2
    seed = 0

    # Check that Fourier and ODE methods produce similar BOLD signal at selected timepoints
    activity_fourier = solver.sim_nft_waves(nt=nt, dt=dt, seed=seed)
    activity_ode = solver.sim_nft_waves(nt=nt, dt=dt, seed=seed, method='ode')
    bold_fourier = solver.balloon_model(activity_fourier, dt=dt)
    bold_ode = solver.balloon_model(activity_ode, dt=dt, method='ode')

    # Methods converge to r=.98 by t=500, but this takes too long to run, so just anchor the test
    # to a lower value to catch if the alignment ever drops (TODO: add to validation?)
    for t in range(76, nt):
        cos = 1-cdistw(bold_fourier[:, t], bold_ode[:, t], solver.mass, metric='cosine')[0][0]
        assert cos > 0.6, \
            f'Fourier and ODE BOLD solutions are not correlated at r>.6 at t={t}.'
        
def test_gen_noise_reproducibility():
    seed = 0
    noise1 = _gen_noise(5, 10, seed=seed)
    noise2 = _gen_noise(5, 20, seed=seed)
    assert (noise1 == noise2[:, :10]).all(), \
        "Noise generated with the same seed does not match across different nt."
    noise32 = _gen_noise(5, 10, seed=seed, dtype=np.float32)
    noise32_long = _gen_noise(5, 20, seed=seed, dtype=np.float32)
    assert noise32.dtype == np.float32
    assert (noise32 == noise32_long[:, :10]).all(), \
        "Float32 noise does not match across different nt for the same seed."

def test_gen_noise_gram(solver):
    seed = 0
    nt = 500
    triu = np.triu_indices(nt, k=1)

    # Modal white noise
    noise = solver.emodes @ _gen_noise(solver.n_modes, nt, seed=seed)
    gram = gramw(noise, mass=solver.mass)  # shape (nt, nt)

    offdiag_mean = gram[triu].mean()  # expected to be 0
    diag_mean = np.diag(gram).mean()  # expected to be n_samples = n_modes
    assert np.abs(offdiag_mean) < 0.05     
    assert np.abs(diag_mean - solver.n_modes) < 1

    # Lumped mass-normalised white noise
    mass_lumped = EigenSolver(solver.geometry).compute_lbo(lump=True).mass
    # remove mass-weighting by taking inverse mass as reciprocal of diagonal
    mass_inv = 1/mass_lumped.diagonal()[:, None]
    noise = mass_inv * _gen_noise(solver.n_verts, nt, mass=mass_lumped, seed=seed)  
    gram = gramw(noise, mass=mass_lumped)  # shape (nt, nt)

    offdiag_mean = gram[triu].mean()  # expected to be 0
    diag_mean = np.diag(gram).mean()  # expected to be n_samples = n_verts
    assert np.abs(offdiag_mean) < 0.35
    assert np.abs(diag_mean - solver.n_verts) < 2

    noise = _gen_noise(solver.n_verts, nt, mass=solver.mass, seed=seed)
    # remove mass-weighting by solving mass * x = noise
    noise = splu(solver.mass).solve(noise)
    cov_mat = gramw(noise, mass=solver.mass)  # shape (nt, nt)
    cov_mean = cov_mat[triu].mean()  # expected to be 0  
    var_mean = np.diag(cov_mat).mean()  # expected to be n_samples = n_verts
    assert np.abs(cov_mean) < 0.35
    assert np.abs(var_mean - solver.n_verts) < 2

def test_sim_nft_waves_reproducibility_fourier(solver):
    
    nt = 100
    dt = 1e-2
    seed = 0
    
    # Same seed should be reproducible across different nt
    ts0 = solver.sim_nft_waves(nt=nt, dt=dt, seed=seed)
    ts1 = solver.sim_nft_waves(nt=2*nt, dt=dt, seed=seed)

    # Different seed should produce different results
    ts2 = solver.sim_nft_waves(nt=nt, dt=dt, seed=seed+1)

    mse01 = np.mean((ts0 - ts1[:, :nt])**2)
    mse02 = np.mean((ts0 - ts2)**2)
    assert mse01 < 2e-7, \
        f"Simulated timeseries with the same seed do not match (MSE={mse01:.4e})."
    assert mse02 > 3e-4, \
        f"Simulated timeseries with different seeds match unexpectedly (MSE={mse02:.4f})."

def test_invalid_input_shape(solver):

    with pytest.raises(ValueError, match=r"ext_input must have shape \(3619,"):
        solver.sim_nft_waves(ext_input=np.ones((4002, 1000)))
    with pytest.raises(ValueError, match=r"activity must have shape \(3619,"):
        solver.balloon_model(np.ones((4002, 1000)), dt=0.01)    

def test_sim_nft_waves_invalid_method(solver):

    with pytest.raises(ValueError, match="Invalid PDE method 'zote'"):
        solver.sim_nft_waves(nt=10, method='zote')

@pytest.mark.filterwarnings("ignore:invalid value encountered in dot:RuntimeWarning")
@pytest.mark.filterwarnings("ignore:overflow encountered in power:RuntimeWarning")
@pytest.mark.filterwarnings("ignore:invalid value encountered in subtract:RuntimeWarning")
def test_sim_nft_waves_ode_balloon_overflow(solver):

    # Large dt can cause overflow errors in the dqdt expression for the ODE balloon model, so
    # test that our error is raised
    dt = 1

    with pytest.raises(RuntimeError, match="message: Required step size is less than spacing"):
        with pytest.warns(UserWarning, match="dt=1 is too large"):
            activity = solver.sim_nft_waves(dt=dt, nt=10, method='ode')
        solver.balloon_model(activity, method='ode', dt=dt)

def test_sim_nft_waves_cached(solver):
    # Get CACHE_DIR
    cache_dir = os.getenv("CACHE_DIR")

    # Test with temporary directory
    try:
        with TemporaryDirectory() as temp_cache_dir:
            os.environ["CACHE_DIR"] = temp_cache_dir
            _ = solver.sim_nft_waves(nt=10, cache_input=True, seed=0)

            # Check that the temp_cache_dir/neuromodes/waves subdirectory exists
            cache_dir_waves = os.path.join(
                temp_cache_dir,
                "neuromodes",
                "waves"
            )
            assert os.path.exists(cache_dir_waves), "Waves cache directory was not created."
    finally:
        # Restore original CACHE_DIR
        if cache_dir is not None:
            os.environ["CACHE_DIR"] = cache_dir
        else:
            del os.environ["CACHE_DIR"]

def test_sim_nft_waves_balloon_param(solver):
    nt = 100
    dt = 1e-2

    activity = solver.sim_nft_waves(nt=nt, dt=dt)
    bold_default = solver.balloon_model(activity, dt=dt)
    bold_custom = solver.balloon_model(activity, dt=dt, rho=0.5)

    assert not np.allclose(bold_default, bold_custom), \
        "BOLD signals with different balloon model parameters match unexpectedly."

def test_calc_balloon_nyquist(solver):
    dt = 0.01
    activity = solver.sim_nft_waves(nt=10, dt=dt)
    with pytest.warns(UserWarning, match='dt=0.01 is too large'):
        _ = solver.balloon_model(activity, dt, gamma_h=1e5)

def test_calc_nft_wave_speed(solver):

    # Homogeneous case
    speed = calc_nft_wave_speed(r=18.0, gamma=116)
    assert isinstance(speed, float), "Output type is not float for hetero=None."

    # Heterogeneous case
    speed = calc_nft_wave_speed(r=18.0, gamma=116, hetero=solver.hetero)
    assert np.all(speed > 0), "Output contains non-positive wave speeds when using hetero."
    assert speed.shape == (solver.n_verts,), "Output shape is incorrect when using hetero." # type: ignore

def test_calc_nft_mode_freqs(solver):
    # This test is similar to what's in docs/validation/waves_mode_freqs.ipynb

    # Set parameters (these are a bit p-hacked to ensure that there's enough oscillatory activity
    # for FFT to detect the correct frequency)
    r = 100
    gamma = 10
    dt = 1e-3  # s
    t = 2.0  # s
    nt = round(t / dt)
    t0 = nt // 2

    # Set each mode's input to 1 from 0 to t0
    ext_input = np.zeros((solver.n_verts, nt))
    ext_input[:, :t0] = solver.emodes.sum(axis=1)[:, None]

    # Simulate waves using ODE method to avoid potential circularity
    sim_ts = solver.sim_nft_waves(dt=dt, r=r, gamma=gamma, ext_input=ext_input, method='ode')

    # Decompose timeseries into modal basis
    modes_ts = solver.decompose(sim_ts)

    # Check that system reaches theoretical steady state by t0
    steady_state = 1 / (1 + r**2 * solver.evals)
    np.testing.assert_allclose(
        modes_ts[:, t0-1],
        steady_state,
        rtol=0.05,
        err_msg="System does not reach theoretical steady state by t0"
        )

    # Get power spectrum of latter half of modes' timeseries via FFT
    modes_fft = np.abs(np.fft.rfft(modes_ts[:, t0:], axis=1))**2
    freqs_fft = np.fft.rfftfreq(t0, dt)

    # Get each mode's highest-power frequency from FFT of simulated activity
    freqs_detected = freqs_fft[np.argmax(modes_fft, axis=1)]

    # Get theoretical frequencies
    freqs_theo = calc_nft_mode_freqs(solver.evals, r=r, gamma=gamma)

    # Check that each mode's detected frequency is within 1 FFT bin of its theoretical frequency
    bin_width = freqs_fft[1] - freqs_fft[0]
    np.testing.assert_allclose(
        freqs_detected,
        freqs_theo,
        atol=bin_width,
        err_msg="Detected frequencies do not match theoretical frequencies within 1 FFT bin"
        )

def test_calc_nft_fc(solver):  
    sim_ts = solver.sim_nft_waves(nt=5000, dt=0.01, seed=0)
    # Check that simulated FC from waves aligns with the analytical FC
    ana_fc = calc_nft_fc(solver.emodes, solver.evals, r=17.4)
    sim_fc = np.corrcoef(sim_ts)
    mse = np.mean((ana_fc - sim_fc)**2)
    assert mse < 0.01, f"Analytical FC does not align with simulated FC (MSE={mse:.4f})."

def test_fem_default(solver):
    # Check that FEM simulation runs with default parameters   
    nt = 50
    ts = solver.sim_nft_waves(nt=nt, method='fem', n_jobs=-1)
    assert ts.shape == (solver.n_verts, nt), "FEM output shape is incorrect with default parameters."

def test_fem_alignment(solver):
    # Check that modal approximation aligns with FEM solution
    nt=50
    
    # construct input using first modes only to remove truncation error
    ext_input = solver.emodes @ _gen_noise(solver.n_modes, nt, seed=0)

    dt=0.01  # usually insufficient for FEM, but fine since ext_input is truncated

    fourier_ts = solver.sim_nft_waves(dt=dt, ext_input=ext_input)

    # Run FEM simulation
    with pytest.warns(UserWarning, match="dt=0.01 is too large"):
        fem_ts = solver.sim_nft_waves(dt=dt, ext_input=ext_input, method='fem')

    # Assess
    for t in range(10, nt):
        cos = 1-cdistw(fourier_ts[:, t], fem_ts[:, t], solver.mass, metric='cosine')[0][0]
        assert cos > 1-1e-9, \
            f'Modal and FEM solutions are not identical (cos > 1-1e-9) at t={t}.'

def test_fem_no_joblib(solver):
    # Check that FEM simulation runs without joblib installed
    nt=50
    dt=0.1
    seed=0

    with (patch.dict('sys.modules', {'joblib': None}),
          pytest.raises(ImportError, match='joblib must be installed')):
        fem_ts = solver.sim_nft_waves(nt=nt, dt=dt, seed=seed, n_jobs=-1, method='fem')

        assert fem_ts.shape == (solver.n_verts, nt), \
            "FEM output shape is incorrect when joblib is not installed."

# TODO: add test that BOLD FC is very similar to analytical FC
# TODO: check n_jobs=-1 gives same result as n_jobs=1