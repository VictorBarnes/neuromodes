"""Module for simulating neural activity and BOLD signals on cortical surfaces using geometric eigenmodes."""

import os
import numpy as np
from scipy import sparse
from pathlib import Path
from joblib import Memory
from numpy.typing import NDArray, ArrayLike
from scipy.integrate import solve_ivp
from typing import Optional
from nsbtools.eigen import decompose

# Set up joblib memory caching
CACHE_DIR = os.getenv("CACHE_DIR")
if CACHE_DIR is None or not os.path.exists(CACHE_DIR):
    CACHE_DIR = Path.home() / ".nsbtools_cache"
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Using default cache directory at {CACHE_DIR}")
memory = Memory(CACHE_DIR, verbose=0)

def simulate_waves(
    emodes: ArrayLike,
    evals: ArrayLike,
    ext_input: Optional[ArrayLike] = None,
    dt: float = 0.1,
    nt: int = 1000,
    r: float = 28.9,
    gamma: float = 0.116,
    mass: Optional[ArrayLike] = None,
    bold_out: bool = False,
    decomp_method: str = "project",
    pde_method: str = "fourier",
    seed: Optional[int] = None
) -> NDArray:
    """
    Simulate neural activity or BOLD signals on the surface mesh using the eigenmode 
    decomposition. The simulation uses a Neural Field Theory wave model and optionally the
    Balloon-Windkessel model for BOLD signal generation. 

    Parameters
    ----------
    emodes : array-like
        The eigenmodes array of shape (n_verts, n_modes), where n_verts is the number of vertices
        and n_modes is the number of eigenmodes.
    evals : array-like
        The eigenvalues array of shape (n_modes,).
    ext_input : array-like, optional
        External input array of shape (n_verts, n_timepoints). If None, random input is generated.
    dt : float, optional
        Time step for simulation in milliseconds. Default is 0.1.
    nt : int, optional
        Number of time points to simulate Default is 1000.
    r : float, optional
        Spatial length scale of wave propagation. Default is 28.9.
    gamma : float, optional
        Damping rate of wave propagation. Default is 0.116.
    mass : array-like, optional
        The mass matrix of shape (n_verts, n_verts) used for the decomposition when method is 
        'project'. If using EigenSolver, provide its self.mass. Default is None.
    bold_out : bool, optional
        If True, simulate BOLD signal using the balloon model. If False, simulate neural activity.
        Default is False.
    decomp_method : str, optional
        The method used for the eigendecomposition, either 'project' to project data into a 
        mass-orthonormal space or 'regress' for least-squares fitting. Note that the beta values
        from 'regress' tend towards those from 'project' when more modes are provided. Default is 
        'project'.
    mass : array-like, optional
        The mass matrix of shape (n_verts, n_verts) used for the decomposition when method is 
        'project'. If using EigenSolver, provide its self.mass. Default is None.
    pde_method : str, optional
        Method for solving the wave PDE. Either "fourier" or "ode". Default is "fourier".
    seed : int, optional
        Random seed for generating external input. Default is None.

    Returns
    -------
    np.ndarray
        Simulated neural or BOLD activity of shape (n_verts, n_timepoints).

    Raises
    ------
    ValueError
        If the shape of ext_input does not match (n_verts, n_timepoints), or if either the
        eigen-decomposition or PDE method is invalid.

    Notes
    -----
    Since the simulation begins at rest, consider discarding the first 50 timepoints to allow the
    system to reach a steady state.
    """
    emodes = np.asarray(emodes)
    evals = np.asarray(evals)
    r = float(r)
    gamma = float(gamma)
    if mass is not None:
        mass = sparse.csc_matrix(mass)
    
    n_verts, n_modes = emodes.shape
    if r <= 0:
        raise ValueError("Parameter `r` must be positive.")
    if gamma <= 0:
        raise ValueError("Parameter `gamma` must be positive.")
    if len(evals) != n_modes:
        raise ValueError(f"The number of eigenvalues ({len(evals)}) must match the number of "
                            f"eigenmodes ({n_modes}).")
    if dt <= 0:
        raise ValueError("`dt` must be positive.")
    if nt <= 0 or not isinstance(nt, int):
        raise ValueError("`nt` must be a positive integer.")

    if ext_input is None:
        ext_input = _gen_random_input(n_verts, nt, seed=seed)
    else:
        ext_input = np.asarray(ext_input)
        if ext_input.shape != (n_verts, nt):
            raise ValueError(f"External input shape is {ext_input.shape}, should be ({n_verts}, "
                             f"{nt}).")

    # Mode decomposition of external input
    input_coeffs = decompose(ext_input, emodes, method=decomp_method, mass=mass)
    
    t = np.linspace(0, dt * (nt - 1), nt)

    # Initialize simulated activity vector
    mode_coeffs = np.zeros((n_modes, nt))
    for mode_ind in range(n_modes):
        input_coeffs_i = input_coeffs[mode_ind, :]
        eval = evals[mode_ind]

        # Calculate the neural activity for the mode
        if pde_method == "fourier":
            neural = _model_wave_fourier(
                mode_coeff=input_coeffs_i, 
                dt=dt, 
                r=r, 
                gamma=gamma, 
                eval=eval
            )
        elif pde_method == "ode":            
            neural = _solve_wave_ode(
                mode_coeff=input_coeffs_i, 
                t=t,
                gamma=gamma,
                r=r,
                eval=eval
            )
        else:
            raise ValueError(f"Invalid PDE method '{pde_method}'; must be 'fourier' or 'ode'.")

        # If bold_out is True, calculate the BOLD signal using the balloon model
        if bold_out:
            if pde_method == "fourier":
                bold = _model_balloon_fourier(mode_coeff=neural, dt=dt)
            elif pde_method == "ode":
                bold = _model_balloon_ode(mode_coeff=neural, t=t)
            mode_coeffs[mode_ind, :] = bold
        else:
            mode_coeffs[mode_ind, :] = neural

    # Combine the mode activities to get the total simulated activity
    sim_activity = emodes @ mode_coeffs

    return sim_activity

@memory.cache
def _gen_random_input(
    n_verts: int,
    n_timepoints: int,
    seed: Optional[int] = None
) -> NDArray:
    """Generates external input with caching to avoid redundant recomputation."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal(size=(n_verts, n_timepoints))

def _model_wave_fourier(
    mode_coeff: NDArray,
    dt: float,
    r: float,
    gamma: float,
    eval: float
) -> NDArray:
    """
    Simulates the time evolution of a wave model based on one mode using a frequency-domain 
    approach. This method applies a Fourier transform to the input mode coefficients, computes the
    system's frequency response, and then applies an inverse Fourier transform to obtain the
    time-domain response of the mode.

    Parameters
    ----------
    mode_coeff : np.ndarray
        Array of mode coefficients at each time representing the input signal to the model.
    dt : float
        Time step for the simulation in milliseconds.
    r : float
        Spatial length scale of wave propagation.
    gamma : float
        Damping rate of wave propagation.
    eval : float or array_like
        The eigenvalue associated with the mode.

    Returns
    -------
    out : ndarray
        The real part of the time-domain response of the mode at the specified time points.
    
    Notes
    -----
    This function uses a frequency-domain method to simulate the damped wave response of a causal 
    input. To ensure causality (i.e., the input is zero for t < 0), the input is zero-padded on the 
    negative time axis and transformed using `ifft`, which mimics the forward Fourier transform of a 
    causal signal. The system's frequency response (transfer function) is then applied, and `fft` is 
    used to return to the time domain. This approach is standard for simulating linear 
    time-invariant causal systems and is equivalent to convolution with a Green's function.

    The sequence is:
      1. Zero-pad input for t < 0 (causality)
      2. Take ifft to get the frequency-domain representation for this causal signal
      3. Apply the frequency response (transfer function)
      4. Use fft to return to the time domain (with appropriate shifts)
    """
    nt = len(mode_coeff)

    # Pad input with zeros on negative side to ensure causality (system is only driven for t >= 0)
    # This is required for the correct Green's function solution of the damped wave equation.
    mode_coeff_padded = np.concatenate([np.zeros(nt-1), mode_coeff])

    # Apply inverse Fourier transform to get frequency-domain representation of the causal signal.
    mode_coeff_f = np.fft.fftshift(np.fft.ifft(mode_coeff_padded))

    # Frequencies for full signal
    omega = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(2*nt-1, d=dt))

    # Compute transfer function
    denom = -omega**2 - 2j * omega * gamma + gamma**2 * (1 + r**2 * eval)
    H = gamma**2 / denom

    # Apply frequency response
    out_fft = H * mode_coeff_f

    # Inverse transform: use fft (not ifft) to return to the time domain, matching above convention
    out_full = np.real(np.fft.fft(np.fft.ifftshift(out_fft)))

    # Return only the non-negative time part (t >= 0)
    return out_full[nt-1:]

def _solve_wave_ode(
    mode_coeff: NDArray,
    t: NDArray,
    gamma: float,
    r: float,
    eval: float
) -> NDArray:
    """
    Solves the damped wave ODE for one eigenmode j.

    Parameters
    ----------
    mode_coeff : np.ndarray
        Input drive to the system with the same length as t (written as qj in equation below).
    t : np.ndarray
        Time points (must be increasing).
    gamma : float
        Damping coefficient.
    r : float
        Spatial length scale.
    eval : float
        Eigenvalue for the j-th mode (written as lambdaj in equation below).

    Returns
    -------
    np.ndarray
        Time evolution of phi_j(t), solution to the wave equation.
    
    Notes
    -----
    The equation is derived from the damped wave equation:
    d^2 phi_j / dt^2 + 2 * gamma * d phi_j / dt + gamma^2 * (1 + r^2 * lambdaj) * phi_j = gamma^2 * qj
    
    Rearranging gives us the first-order system
        dx1/dt = x2
        dx2/dt = -2 * gamma * x2 - gamma^2 * (1 + r^2 * lambdaj) * x1 + gamma^2 * qval
    """
    eval = float(eval)  # Ensure eval is a float

    def q_interp_safe(t_):
        """Safely interpolate the driving term at time t_."""
        val = np.interp(t_, t, mode_coeff)
        return val.item() if isinstance(val, np.ndarray) else val

    def wave_rhs(t_, y):
        """Right-hand side of the wave equation in first-order form."""
        x1, x2 = y  # both should be scalars
        qval = q_interp_safe(t_)  # should be scalar

        dx1dt = x2
        dx2dt = -2 * gamma * x2 - gamma**2 * (1 + r**2 * eval) * x1 + gamma**2 * qval

        return [dx1dt, dx2dt]

    y0 = [0.0, 0.0]

    sol = solve_ivp(
        wave_rhs,
        t_span=(t[0], t[-1]),
        y0=y0,
        t_eval=t,
        method='RK45',
        rtol=1e-6,
        atol=1e-9
    )

    return sol.y[0]  # Return phi_j(t)

def _model_balloon_fourier(
    mode_coeff: NDArray,
    dt: float
) -> NDArray:       
    """
    Simulates the hemodynamic response of one mode using the balloon model in the frequency domain. 
    This method applies a frequency-domain implementation of the balloon model to a given set of 
    mode coefficients, returning the modeled hemodynamic response over time.

    Parameters
    ----------
    mode_coeff : np.ndarray
        Array of mode coefficients representing the input signal to the model.
    dt : float
        Time step for the simulation in milliseconds.

    Returns
    -------
    np.ndarray
        The real part of the time-domain response of the mode at the specified time points.

    Notes
    -----
    This function uses a frequency-domain method to simulate the damped wave response of a causal 
    input. To ensure causality (i.e., the input is zero for t < 0), the input is zero-padded on the 
    negative time axis and transformed using `ifft`, which mimics the forward Fourier transform of a 
    causal signal. The system's frequency response (transfer function) is then applied, and `fft` is 
    used to return to the time domain. This approach is standard for simulating linear 
    time-invariant causal systems and is equivalent to convolution with a Green's function.

    The sequence is:
      1. Zero-pad input for t < 0 (causality)
      2. Take ifft to get the frequency-domain representation for this causal signal
      3. Apply the frequency response (transfer function)
      4. Use fft to return to the time domain (with appropriate shifts)
    """
    # Default independent model parameters
    kappa = 0.65   # signal decay rate [s^-1]
    gamma = 0.41   # rate of elimination [s^-1]
    tau = 0.98     # hemodynamic transit time [s]
    alpha = 0.32   # Grubb's exponent [unitless]
    rho = 0.34     # resting oxygen extraction fraction [unitless]
    V0 = 0.02      # resting blood volume fraction [unitless]

    # Other parameters
    w_f = 0.56
    Q0 = 1
    rho_f = 1000
    eta = 0.3
    Xi_0 = 1
    beta = 3
    V_0 = 0.02
    k1 = 3.72
    k2 = 0.527
    k3 = 0.48
    beta = (rho + (1 - rho) * np.log(1 - rho)) / rho

    # --- Use the same causal Fourier procedure as model_wave_fourier ---
    # Zero-pad input for t < 0 (causality)
    nt = len(mode_coeff) - 1
    t_full = np.arange(-nt * dt, nt * dt + dt, dt)  # Symmetric time vector
    nt_full = len(t_full)

    mode_coeff_padded = np.concatenate([np.zeros(nt), mode_coeff])

    # Frequencies for full signal
    omega = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(nt_full, d=dt))

    # Apply inverse Fourier transform to get frequency-domain representation of the causal signal.
    mode_coeff_f = np.fft.fftshift(np.fft.ifft(mode_coeff_padded))

    # Calculate the frequency response of the system
    phi_hat_Fz = 1 / (-(omega + 1j * 0.5 * kappa) ** 2 + w_f ** 2)
    phi_hat_yF = V_0 * (alpha * (k2 + k3) * (1 - 1j * tau * omega) 
                                - (k1 + k2) * (alpha + beta - 1 
                                - 1j * tau * alpha * beta * omega)) / ((1 - 1j * tau * omega)
                                *(1 - 1j * tau * alpha * omega))
    phi_hat = phi_hat_yF * phi_hat_Fz

    # Apply frequency response
    out_fft = phi_hat * mode_coeff_f

    # Inverse transform: use fft (not ifft) to return to the time domain, matching above convention
    out_full = np.real(np.fft.fft(np.fft.ifftshift(out_fft)))

    # Return only the non-negative time part (t >= 0)
    return out_full[nt:]

def _model_balloon_ode(
    mode_coeff: NDArray,
    t: NDArray
) -> NDArray:
    """
    Simulates the hemodynamic response of one mode using the balloon model in the time domain (ODE 
    approach). This function numerically integrates the balloon model ODEs for a given input mode 
    time course.

    Parameters
    ----------
    mode_coeff : np.ndarray
        Array of mode coefficients representing the input signal to the model (neural activity, same 
        length as t).
    t : np.ndarray
        Array of time points (must be increasing, same length as mode_coeff).

    Returns
    -------
    np.ndarray
        The BOLD signal time course for the mode at the specified time points.
    """
    # Balloon model parameters (canonical values)
    kappa = 0.65   # signal decay rate [s^-1]
    gamma_h = 0.41 # rate of elimination [s^-1]
    tau = 0.98     # hemodynamic transit time [s]
    alpha = 0.32   # Grubb's exponent [unitless]
    rho = 0.34     # resting oxygen extraction fraction [unitless]
    V0 = 0.02      # resting blood volume fraction [unitless]
    E0 = rho       # resting oxygen extraction fraction
    TE = 0.04      # echo time [s]
    k1 = 7 * E0
    k2 = 2
    k3 = 2 * E0 - 0.2

    # ODE system: y = [s, f, v, q]
    # s: vasodilatory signal, f: blood inflow, v: blood volume, q: deoxyhemoglobin content
    def balloon_rhs(t_, y):
        s, f, v, q = y
        # Interpolate neural input at current time
        u = np.interp(t_, t, mode_coeff)
        dsdt = u - kappa * s - gamma_h * (f - 1)
        dfdt = s
        dvdt = (f - v ** (1 / alpha)) / tau
        dqdt = (f * (1 - (1 - E0) ** (1 / f)) / E0 - q * v ** (1 / alpha - 1) / v) / tau
        return [dsdt, dfdt, dvdt, dqdt]

    # Initial conditions: resting state
    y0 = [0.0, 1.0, 1.0, 1.0]

    sol = solve_ivp(
        balloon_rhs,
        t_span=(t[0], t[-1]),
        y0=y0,
        t_eval=t,
        method='RK45',
        rtol=1e-6,
        atol=1e-9
    )

    s, f, v, q = sol.y
    # BOLD signal (standard formula)
    bold = V0 * (k1 * (1 - q) + k2 * (1 - q / v) + k3 * (1 - v))
    
    return bold
