import trimesh
import pytest
import numpy as np
import nibabel as nib
import importlib.resources
from nsbtools.eigen import EigenSolver, check_surf

data_dir = importlib.resources.files('nsbtools.data')

@pytest.fixture
def surf_medmask_hetero():
    surfpath = data_dir / 'sp-human_tpl-fsLR_den-4k_hemi-L_midthickness.surf.gii'
    medmask = nib.load(
        data_dir / 'sp-human_tpl-fsLR_den-4k_hemi-L_medmask.label.gii').darrays[0].data.astype(bool)
    np.random.seed(0)
    hetero = np.random.normal(loc=0, scale=0.5, size=len(medmask))
    return surfpath, medmask, hetero

def test_init_params(surf_medmask_hetero):
    surfpath, medmask, hetero = surf_medmask_hetero
    solver = EigenSolver(surfpath, medmask=medmask, hetero=hetero, n_modes=2, alpha=0.5, beta=3,
                         scaling='exponential')
    assert hasattr(solver, 'stiffness') and hasattr(solver, 'mass'), \
        'Solver did not initialize correctly.'

def test_invalid_surf():
    with pytest.raises(ValueError, match="Surface must be a .*"):
        EigenSolver([1,2,3])

def test_surf_unreferenced_verts():
    # Create an invalid mesh with unreferenced vertices
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [2, 2, 2]])  # Last vertex unreferenced
    faces = np.array([[0, 1, 2], [0, 2, 3]])  # Only uses first 4 vertices, vertex 4 is unreferenced
    
    # IMPORTANT: Use process=False to prevent trimesh from automatically cleaning up unreferenced vertices    
    invalid_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
   
    # check_surf should raise ValueError due to unreferenced vertex
    with pytest.raises(ValueError, match="Surface mesh contains .* unreferenced vertices"):
        check_surf(invalid_mesh)

def test_surf_not_contiguous():
    # Create two separate triangles (disconnected components)
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [2, 0, 0], [3, 0, 0], [2, 1, 0]])
    faces = np.array([[0, 1, 2], [3, 4, 5]])  # Two separate triangles
    disconnected_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    
    # check_surf should raise ValueError due to multiple components
    with pytest.raises(ValueError, match="Surface mesh is not contiguous.*connected components"):
        check_surf(disconnected_mesh)

def test_no_medmask(surf_medmask_hetero):
    surfpath, _, hetero = surf_medmask_hetero
    EigenSolver(surfpath, hetero=hetero)

def test_invalid_mask_shape(surf_medmask_hetero):
    surfpath, _, _ = surf_medmask_hetero
    bad_mask = np.ones(10)
    with pytest.raises(ValueError, match=r"`medmask` \(10\) must match .* mesh \(4002\)."):
        EigenSolver(surfpath, medmask=bad_mask)

def test_invalid_hetero_shape(surf_medmask_hetero):
    surfpath, _, _ = surf_medmask_hetero
    bad_hetero = np.ones(10)
    with pytest.raises(ValueError, match=r"`hetero` \(10\) must match .* mesh \(4002\)."):
        EigenSolver(surfpath, hetero=bad_hetero)

def test_nan_inf_hetero(surf_medmask_hetero):
    surfpath, _, hetero = surf_medmask_hetero

    hetero[0] = np.nan
    with pytest.raises(ValueError, match="`hetero` must not contain NaNs or Infs."):
        EigenSolver(surfpath, hetero=hetero)

    hetero[0] = np.inf
    with pytest.raises(ValueError, match="`hetero` must not contain NaNs or Infs."):
        EigenSolver(surfpath, hetero=hetero)

def test_nan_inf_hetero_medmask(surf_medmask_hetero):
    # Inject NaN/Inf at a cortical vertex (should raise error)
    surfpath, medmask, hetero = surf_medmask_hetero
    cortical_vertex = np.where(medmask)[0][0]
    print(cortical_vertex)
    hetero[cortical_vertex] = np.nan
    with pytest.raises(ValueError, match="`hetero` must not contain NaNs or Infs."):
        EigenSolver(surfpath, hetero=hetero)
    hetero[cortical_vertex] = np.inf
    with pytest.raises(ValueError, match="`hetero` must not contain NaNs or Infs."):
        EigenSolver(surfpath, hetero=hetero)

def test_nan_inf_hetero_medmask_ignored(surf_medmask_hetero):
    # Inject NaN/Inf at a medial vertex (should be ignored)
    surfpath, medmask, hetero = surf_medmask_hetero
    medial_vertex = np.where(~medmask)[0][0]
    print(medial_vertex)
    hetero[medial_vertex] = np.nan
    EigenSolver(surfpath, medmask=medmask, hetero=hetero)
    hetero[medial_vertex] = np.inf  
    EigenSolver(surfpath, medmask=medmask, hetero=hetero)

def test_init_invalid_wave_speed(surf_medmask_hetero):
    surfpath, medmask, hetero = surf_medmask_hetero
    solver = EigenSolver(surfpath, medmask=medmask, hetero=hetero, beta=11)

    # This alpha scaling should result in implausibly fast waves at some vertices
    solver.alpha = 2
    with pytest.raises(ValueError, match='.*non-physiological wave speeds.*'):
        solver.hetero = hetero

def test_init_invalid_scaling(surf_medmask_hetero):
    surfpath, medmask, hetero = surf_medmask_hetero
    with pytest.raises(ValueError, match="Invalid scaling 'plantasia'.*"):
        EigenSolver(surfpath, medmask=medmask, hetero=hetero, scaling='plantasia')

@pytest.fixture
def init_solver(surf_medmask_hetero):
    surfpath, medmask, hetero = surf_medmask_hetero
    solver = EigenSolver(surfpath, medmask=medmask, hetero=hetero, n_modes=10)
    return solver

def test_symmetric_mass_matrix(init_solver):
    solver = init_solver
    mass = solver.mass
    diff = mass - mass.T
    assert diff.nnz == 0, 'Mass matrix is not symmetric.'

def test_no_hetero(init_solver):
    solver = init_solver

    with pytest.warns(UserWarning, match="Setting `alpha` and `beta` to 0.*"):
        solver.hetero = None
    solver.solve()
    emodes = solver.emodes
    evals = solver.evals    

    # Check that eigenmodes and eigenvalues 2-10 highly correlate with previously calculated set
    prior_modes = np.load(data_dir / 'sp-human_tpl-fsLR_den-4k_hemi-L_midthickness-emodes.npy')
    prior_evals = np.load(data_dir / 'sp-human_tpl-fsLR_den-4k_hemi-L_midthickness-evals.npy')

    for i in range(1, 10):
        assert np.abs(np.corrcoef(emodes[:, i], prior_modes[:, i])[0, 1]) > 0.99, \
            f'Eigenmode {i} does not match the prior set.'
        assert np.allclose(evals[i], prior_evals[i], rtol=0.1), \
            f'Eigenvalue {i} does not match the prior set.'

@pytest.fixture
def solve_modes(init_solver):
    solver = init_solver
    solver.solve()
    return solver, solver.emodes

def test_nonstandard_modes(solve_modes):
    solver, emodes = solve_modes

    solver.solve(standardize=False)
    emodes_nonstd = solver.emodes
    assert np.all(emodes[0, :] >= 0), 'Standardized first element has negative values.'
    assert not np.all(emodes_nonstd[0, :] >= 0), \
        'Non-standardized first element has no negative first elements.'
    assert np.allclose(abs(emodes), abs(emodes_nonstd),
                       atol=1e-6), 'Non-standardized modes do not match standardized modes.'

def test_solve_lumped_mass(solve_modes, surf_medmask_hetero):
    _, emodes = solve_modes
    surfpath, medmask, hetero = surf_medmask_hetero

    # Get first 10 modes after solving with lumped mass matrix
    solver = EigenSolver(surfpath, medmask=medmask, hetero=hetero, lump=True, n_modes=10)
    solver.solve()
    emodes_lumped = solver.emodes

    assert np.allclose(abs(emodes), abs(emodes_lumped), atol=1e-3), \
        'Lumped mass modes do not approximately match original modes.'
    for i in range(1, solver.nmodes):
        assert np.corrcoef(emodes[:, i], emodes_lumped[:, i])[0, 1] > 0.99, \
            'Lumped mass modes do not match original modes.'

def test_solutions(solve_modes):
    solver, emodes = solve_modes
    evals = solver.evals

    assert emodes.shape == (solver.n_verts,
                           solver.nmodes), (f'Eigenmodes have shape {emodes.shape}, should be '
                                            f'{(solver.n_verts, solver.nmodes)}.')
    assert len(evals) == solver.nmodes, (f'Eigenvalues has length {len(evals)}, should be '
                                         f'{solver.nmodes}.')
    assert np.all(np.diff(evals) > 0), 'Eigenvalues are not sorted in descending order.'

def test_constant_mode1(solve_modes):
    solver, emodes = solve_modes
    emode1 = emodes[:, 0]
    solver.solve(fix_mode1=False)
    emode1_unfixed = solver.emodes[:, 0]
    eval1_unfixed = solver.evals[0]

    assert (emode1 == emode1[0]).all(), 'Fixed first mode is not exactly constant.'
    assert np.allclose(emode1_unfixed, emode1[0],
                       atol=1e-4), 'Unfixed first mode is not approximately constant.'
    assert np.isclose(np.mean(emode1_unfixed), emode1[0],
                      atol=1e-6), 'Mean of unfixed first mode is not close to fixed value.'
    assert eval1_unfixed < 1e-6, 'First eigenvalue of unfixed first mode is not close to 0.'

def test_warn_orthonorm(solve_modes):
    solver, emodes = solve_modes
    emodes[:, 0] += 0.1 # Destroy mass-orthonormality by changing first mode's value

    # Solver and static methods should warn by default
    with pytest.warns(Warning, match='Eigenmodes are not mass-orthonormal.*'):
        _ = solver.decompose(emodes[:, 1], emodes, mass=solver.mass)
    with pytest.warns(Warning, match='Eigenmodes are not mass-orthonormal.*'):
        _ = solver.reconstruct(emodes[:, 1], emodes, mass=solver.mass)

def test_decompose_eigenmodes(solve_modes):
    solver, emodes = solve_modes

    for i in range(solver.nmodes):
        data = emodes[:, i]  # Use an eigenmode as data
        beta = solver.decompose(data, emodes, mass=solver.mass)

        # The mode should load onto only itself due to orthogonality
        beta_expected = np.zeros((solver.nmodes, 1))
        beta_expected[i, 0] = 1
        assert np.allclose(beta, beta_expected, atol=1e-4), f'Decomposition of mode {i+1} failed.'

def test_decompose_return_norm_power(solve_modes):
    solver, emodes = solve_modes

    # Decompose a random map
    map = np.random.normal(size=solver.n_verts)
    norm_power = solver.decompose(map, emodes, mass=solver.mass, return_norm_power=True)

    # Powers should be positive and add to 1
    assert np.all(norm_power > 0), 'Normalized powers contain negative values.'
    assert np.isclose(np.sum(norm_power), 1, atol=1e-8), 'Normalized powers do not sum to 1.'

def test_decompose_invalid_data_shape(solve_modes):
    solver, emodes = solve_modes

    with pytest.raises(ValueError, match=r".*`data` \(4002\) must match .* `emodes` \(3636\)."):
        solver.decompose(np.ones(4002), emodes, mass=solver.mass)

def test_decompose_nan_inf_mode(solve_modes):
    solver, emodes = solve_modes
    data = np.ones(solver.n_verts)

    emodes[0,0] = np.nan
    with pytest.raises(ValueError, match="`emodes` contains NaNs or Infs."):
        solver.decompose(data, emodes, mass=solver.mass)

    emodes[0,0] = np.inf
    with pytest.raises(ValueError, match="`emodes` contains NaNs or Infs."):
        solver.decompose(data, emodes, mass=solver.mass)

def test_decompose_massless(solve_modes):
    solver, emodes = solve_modes

    with pytest.raises(ValueError, match=r"Mass matrix of shape \(3636, 3636\) must be provided .*"):
        solver.decompose(emodes, emodes)

def test_decompose_invalid_method(solve_modes):
    solver, emodes = solve_modes

    with pytest.raises(ValueError, match="Invalid eigen-decomposition method 'fornitonian'.*"):
        solver.decompose(emodes, emodes, method='fornitonian')

@pytest.fixture
def gen_eigenmap(solve_modes):
    solver, emodes = solve_modes

    # Use randomly weighted sums of modes to generate maps
    n_maps = 3
    np.random.seed(0)
    weights = np.random.normal(loc=0, scale=0.5, size=(solver.nmodes, n_maps))
    eigenmaps = emodes @ weights

    return solver, emodes, weights, eigenmaps

def test_reconstruct_mode_superposition(gen_eigenmap):
    solver, emodes, weights, eigenmaps = gen_eigenmap

    beta, recon, pearsonr = solver.reconstruct(eigenmaps, emodes, mass=solver.mass)

    # Pearson's r should increase from 0 to 1 when using mode 1 only versus all relevant modes
    assert np.allclose(recon[:,-1,:], eigenmaps,
                       atol=1e-6), 'Final reconstructions do not match input maps.'
    assert (pearsonr[0,:] == 0).all(), 'Pearson r is not 0 when using only mode 1.'
    assert np.allclose(pearsonr[-1,:], 1,
                       atol=1e-10), 'Pearson r is not close to 1 when using all modes.'

    assert np.allclose(beta[-1], weights, atol=1e-4), \
        'Beta values do not match input mode weights when using all modes.'

    # MSE should be 0 when using all modes
    _, _, mse = solver.reconstruct(eigenmaps, emodes, mass=solver.mass, metric='mse')
    assert np.allclose(mse[-1,:], 0,
                       atol=1e-10), 'MSE is not close to 0 when using all modes.'

    # Reconstruct using the first 5 modes, then the first 2 modes
    _, _, pearsonr_modesq = solver.reconstruct(eigenmaps, emodes, mass=solver.mass, modesq=[5,2])
    assert (pearsonr_modesq[0,:] == pearsonr[4,:]).all(), \
        'Reconstruction scores do not match for 5 modes.'
    assert (pearsonr_modesq[1,:] == pearsonr[1,:]).all(), \
        'Reconstruction scores do not match for 2 modes.'

def test_reconstruct_regress_method(gen_eigenmap):
    solver, emodes, _, eigenmaps = gen_eigenmap

    _, _, pearsonr = solver.reconstruct(eigenmaps, emodes, method='regress')
    _, _, mse = solver.reconstruct(eigenmaps, emodes, method='regress', metric='mse')

    # Pearson's r should strictly increase and MSE should strictly decrease when adding modes
    assert np.all(np.diff(pearsonr, axis=0) > 0), \
        'Pearson r does not strictly increase when adding modes.'
    assert np.all(np.diff(mse, axis=0) < 0), \
        'MSE does not strictly decrease when adding modes.'

def test_reconstruct_mode_superposition_timeseries(gen_eigenmap):
    solver, emodes, _, eigen_ts = gen_eigenmap
    fc = np.corrcoef(eigen_ts)

    # Treat eigenmaps as timepoints of activity
    _, _, fc_recon, pearsonr = solver.reconstruct(eigen_ts, emodes, mass=solver.mass,
                                                        timeseries=True)

    assert np.allclose(fc_recon[:,:,-1], fc, atol=1e-3), 'Reconstructed FC does not match original.'
    assert pearsonr[0] == 0, \
        'FC reconstruction score is not close to 0 when using only the constant mode.'
    assert pearsonr[-1] > 0.9999, 'FC reconstruction score is not close to 1 when using all modes.'
    
    _, _, _, mse = solver.reconstruct(eigen_ts, emodes, mass=solver.mass, timeseries=True,
                                      metric='mse')
    assert mse[-1] < 1e-9, 'MSE is not close to 0 when using all modes.'

def test_reconstruct_real_map_32k():
    # Get modes of fsLR 32k midthickness (data is in 32k)
    surfpath = data_dir / 'sp-human_tpl-fsLR_den-32k_hemi-L_midthickness.surf.gii'
    medmask = nib.load(data_dir / 'sp-human_tpl-fsLR_den-32k_hemi-L_medmask.label.gii').darrays[0].data.astype(bool)
    np.random.seed(0)
    hetero = np.random.normal(loc=0, scale=0.5, size=len(medmask))
    solver = EigenSolver(surfpath, medmask=medmask, hetero=hetero, n_modes=10)
    solver.solve()
    emodes = solver.emodes

    # Load FC gradient from Margulies 2016 PNAS
    map = nib.load(data_dir / 'sp-human_tpl-fsLR_den-32k_hemi-L_fcgradient1.gii').darrays[0].data
    map = map[medmask]
    _, _, recon_score = solver.reconstruct(map, emodes, mass=solver.mass)

    # Pearson's r should strictly increase from 0, but not reach 1
    assert np.all(np.diff(recon_score) > 1e-6), \
        r'Reconstruction score \(r\) does not strictly increase.'
    assert np.isclose(recon_score[0], 0, atol=1e-6), \
        r'Reconstruction score \(r\) is non-zero when using only the constant mode.'
    assert not np.isclose(recon_score[-1], 1, atol=1e-6), \
        r'Reconstruction score \(r\) is unexpectedly close to 1 for only 10 modes.'

def test_reconstruct_invalid_map_shape(solve_modes):
    solver, emodes = solve_modes

    with pytest.raises(ValueError, match=r".*`data` \(4002\) must match .* `emodes` \(3636\)."):
        solver.reconstruct(np.ones(4002), emodes, mass=solver.mass)

def test_reconstruct_invalid_metric(solve_modes):
    solver, emodes = solve_modes

    with pytest.raises(ValueError, match="Invalid metric 'barnesv'.*"):
        solver.reconstruct(emodes, emodes, mass=solver.mass, metric='barnesv')

def test_reconstruct_massless(solve_modes):
    solver, emodes = solve_modes

    with pytest.raises(ValueError, match=r"Mass matrix of shape \(3636, 3636\) must be provided .*"):
        solver.reconstruct(emodes, emodes)

def test_simulate_waves_impulse(solve_modes):
    solver, _ = solve_modes

    # Simulate timeseries with a 10ms impulse of white noise to the cortex
    dt = 1 # ms
    nt = 200
    i_start = 10 # ms
    i_stop = 20 # ms
    np.random.seed(0)
    impulse = np.random.normal(loc=0.5, scale=0.5, size=solver.n_verts)
    ext_input = np.zeros((solver.n_verts, nt))
    ext_input[:, i_start:i_stop] = impulse[:, np.newaxis]
    fourier_ts = solver.simulate_waves(ext_input=ext_input, nt=nt, dt=dt)
    ode_ts = solver.simulate_waves(ext_input=ext_input, nt=nt, dt=dt, pde_method='ode')

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

def test_simulate_waves_default_input(solve_modes):
    solver, _ = solve_modes

    nt = 200
    dt = 0.1
    fourier_ts = solver.simulate_waves(nt=nt, dt=dt, seed=0)
    ode_ts = solver.simulate_waves(nt=nt, dt=dt, seed=0, pde_method='ode')
    
    # Check that Fourier and ODE methods produce similar activity at selected timepoints
    for t in range(60, nt):
        assert np.corrcoef(fourier_ts[:, t], ode_ts[:, t])[0, 1] > 0.99, \
            f'Fourier and ODE solutions are not correlated at r>.99 at t={t}.'
        
def test_simulate_waves_bold(solve_modes):
    solver, _ = solve_modes

    nt = 10
    dt = 0.1
    bold_fourier = solver.simulate_waves(nt=nt, dt=dt, bold_out=True, seed=0)
    bold_ode = solver.simulate_waves(nt=nt, dt=dt, bold_out=True, seed=0, pde_method='ode')

    assert bold_fourier.shape == (solver.n_verts, nt), 'Fourier BOLD output shape is incorrect.'
    assert bold_ode.shape == (solver.n_verts, nt), 'ODE BOLD output shape is incorrect.'

def test_simulate_waves_solveless(init_solver):
    solver = init_solver

    solver.simulate_waves()

def test_simulate_waves_invalid_input_shape(solve_modes):
    solver, _ = solve_modes

    with pytest.raises(ValueError, match=r".*shape is \(4002, 1000\), should be \(3636, 1000\)."):
        solver.simulate_waves(ext_input=np.ones((4002, 1000)))

def test_simulate_waves_invalid_pde_method(solve_modes):
    solver, _ = solve_modes

    with pytest.raises(ValueError, match="Invalid PDE method 'zote'.*"):
        solver.simulate_waves(pde_method='zote')