"""
Module for using neural field theory to simulate neural activity and BOLD signals on cortical 
surfaces.
"""

from __future__ import annotations
from typing import Union, TYPE_CHECKING
from warnings import warn
import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse import spmatrix
from neuromodes.basis import decompose

if TYPE_CHECKING:
    from numpy.typing import NDArray, ArrayLike

def simulate_waves(
    emodes: ArrayLike,
    evals: ArrayLike,
    nt: int = 1000,
    bold_out: bool = False,
    ext_input: Union[ArrayLike, None] = None,
    dt: float = 1e-4,
    r: float = 18.0,
    gamma: float = 116.0,
    pde_method: str = "fourier",
    decomp_method: str = "project",
    mass: Union[spmatrix, ArrayLike, None] = None,
    speed_limits: Union[tuple[float, float], None] = (0, 150),
    scaled_hetero: Union[ArrayLike, None] = None,
    check_ortho: bool = True,
    seed: Union[int, None] = None,
    cache_input: bool = False,
    **balloon_params
) -> NDArray:
    """
    Simulate neural activity or BOLD signals on the surface mesh using the eigenmode decomposition.
    The simulation uses a Neural Field Theory wave model and optionally the Balloon-Windkessel model
    for BOLD signal generation.

    Parameters
    ----------
    emodes : array-like
        The eigenmodes array of shape (n_verts, n_modes), where n_verts is the number of vertices
        and n_modes is the number of eigenmodes.
    evals : array-like
        The eigenvalues array of shape (n_modes,).
    nt : int, optional
        Number of time points to simulate Default is `1000`.
    bold_out : bool, optional
        If `True`, simulate BOLD signal using the balloon model. If `False`, simulate neural
        activity. Default is `False`.
    ext_input : array-like, optional
        External input array of shape (n_verts, n_timepoints). If `None`, random input is generated.
        Default is `None`.
    dt : float, optional
        Time step for simulation in seconds. Default is `1e-4`.
    r : float, optional
        Spatial length scale of wave propagation in millimeters. Default is `18.0`.
    gamma : float, optional
        Damping rate of wave propagation in seconds^-1. Default is `116.0`.
    pde_method : str, optional
        Method for solving the wave PDE. Either `'fourier'` or `'ode'`. Default is `'fourier'`.
    decomp_method : str, optional
        The method used for the eigendecomposition, either `'project'` to project data into a
        mass-orthonormal space or `'regress'` for least-squares fitting. Note that the beta values
        from `'regress'` tend towards those from `'project'` when more modes are provided. Default
        is `'project'`.
    mass : array-like, optional
        The mass matrix of shape (n_verts, n_verts) used for the decomposition when method is
        `'project'`. If using `EigenSolver`, provide its `self.mass`. Default is `None`.
    speed_limits : tuple, optional
        If any wave speeds are outside this range (in m/s), a warning is raised. If `None`, no
        warning is raised. Default is `(0, 150)`.
    scaled_hetero : array-like, optional
        Scaled heterogeneity map of shape (n_verts,), used only to check wave speeds (see
        `speed_limits` above). If not provided, wave speed is assumed to be spatially uniform. To
        scale a heterogeneity map, use the `eigen.scale_hetero` function.
        Default is `None`.
    check_ortho : bool, optional
        Whether to check if `emodes` are mass-orthonormal before using the `'project'` method for
        decomposition. Default is `True`.
    seed : int, optional
        Random seed for generating external input. Default is `None`.
    cache_input : bool, optional
        If `True` and `ext_input` is `None`, cache the generated random input to avoid
        recomputation for the same values of `nt`, `seed`, and number of rows (vertices) in
        `emodes`. Inputs are cached in the directory specified by the `CACHE_DIR` environment
        variable. If not set, the user's home directory is chosen. Default is `False`.
    **balloon_params
        Optional balloon model parameters to override defaults (e.g., `rho`, `k1`). See
        `get_balloon_params()` for available parameters. Only used when `bold_out=True`.

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
    Since the simulation begins at rest, consider discarding the first ~50 timepoints to allow the
    system to reach a steady state.
    """
    # Format / validate arguments
    emodes = np.asarray_chkfinite(emodes)
    evals = np.asarray_chkfinite(evals)
    r = float(r)
    gamma = float(gamma)
    if mass is not None and not isinstance(mass, spmatrix):
        mass = np.asarray_chkfinite(mass)
    
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
    if speed_limits is not None:
        if not isinstance(speed_limits, tuple) or not len(speed_limits) == 2 \
            or speed_limits[0] < 0 or speed_limits[0] >= speed_limits[1]:
            raise ValueError("`speed_limits` must be a tuple of (min_speed, max_speed), where "
                             "min_speed is non-negative and less than max_speed.")
        speed = calc_wave_speed(r, gamma, scaled_hetero=scaled_hetero)
        if np.min(speed) < speed_limits[0] or np.max(speed) > speed_limits[1]:
            warn("The combination of `r`, `gamma`, and `scaled_hetero` leads to wave speeds " \
                 f"outside the specified `speed_limits` range ({speed_limits[0]}-{speed_limits[1]}"\
                 " m/s). Consider changing these parameters to ensure physiologically plausible " \
                 "wave speeds.")
    if len(balloon_params) > 0 and not bold_out:
        warn("Balloon model parameters will be ignored as `bold_out` is False.")
    if pde_method not in ['fourier', 'ode']:
        raise ValueError(f"Invalid PDE method '{pde_method}'; must be 'fourier' or 'ode'.")

    if ext_input is not None:
        ext_input = np.asarray_chkfinite(ext_input)
        if ext_input.shape != (n_verts, nt):
            raise ValueError(f"`ext_input` has shape {ext_input.shape}, should be ({n_verts}, "
                             f"{nt}).")

        if seed is not None:
            warn("`seed` is ignored when `ext_input` is provided.")
        if cache_input:
            warn("`cache_input` is ignored when `ext_input` is provided.")
    else:
        # Use Gaussian white noise input if none provided
        if cache_input:
            if seed is None:
                warn("`cache_input` is ignored when `seed` is None.")
            else:
                from neuromodes.io import _set_cache

                memory = _set_cache()
                gen_input = memory.cache(_gen_noise)
        else:
            gen_input = _gen_noise
        
        ext_input = gen_input((n_verts, nt), seed)

    # ==========================================================================
    # Load extra parameters and precompute arrays if needed
    if bold_out:
        all_balloon_params = get_balloon_params(**balloon_params)
        if pde_method == 'fourier':
            balloon_freq_response = _calc_balloon_freq_response(nt, dt, all_balloon_params)
    if pde_method == 'ode':
        t = np.linspace(0, dt * (nt - 1), nt)

    # Eigendecomposition of external input
    input_coeffs = decompose(ext_input, emodes, method=decomp_method,
                             mass=mass, check_ortho=check_ortho)

    # Compute activity timeseries for each mode
    mode_coeffs = np.zeros((n_modes, nt))
    for mode_idx in range(n_modes):
        mode_coeff = _model_wave_fourier(input_coeffs[mode_idx, :], dt, r, gamma, evals[mode_idx]) \
                      if pde_method == "fourier" else \
                     _model_wave_ode(input_coeffs[mode_idx, :], t, r, gamma, evals[mode_idx])

        if bold_out:
            # Apply Balloon-Windkessel model
            mode_coeff = _model_balloon_fourier(mode_coeff, balloon_freq_response) \
                if pde_method == "fourier" else \
                    _model_balloon_ode(mode_coeff, t, all_balloon_params)

        mode_coeffs[mode_idx, :] = mode_coeff

    # Transform timeseries from modal coefficients back to vertex space
    return emodes @ mode_coeffs

def calc_wave_speed(
    r: float,
    gamma: float,
    scaled_hetero: Union[ArrayLike, None] = None
) -> Union[float, NDArray]:
    """
    Calculate wave speed based on the two parameters of the wave model. If a scaled
    heterogeneity map is provided, wave speeds are calculated for each cortical vertex.
    
    Parameters
    ----------
    r : float
        Axonal length scale for wave propagation in millimeters.
    gamma : float
        Damping parameter for wave propagation in seconds^-1.
    scaled_hetero : array-like, optional
        Scaled heterogeneity map of shape (n_verts,). If `None`, wave speed is assumed to be
        spatially uniform. To scale a heterogeneity map, use the `eigen.scale_hetero` function.
        Default is `None`.
    
    Returns
    -------
    float or np.ndarray
        Wave speed across the whole surface, or at each cortical vertex if `scaled_hetero` is
        provided.
    """
    speed = (r / 1000) * gamma # Convert r to meters
    if scaled_hetero is not None:
        speed *= np.sqrt(scaled_hetero)

    return speed

def _gen_noise(size, seed):
    return np.random.default_rng(seed).standard_normal(size=size)

def _model_wave_fourier(
    input_coeff: NDArray,
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
    input_coeff : np.ndarray
        Array of mode coefficients at each time representing the input signal to the model, with
        shape (nt,).
    dt : float
        Time step for the simulation in seconds.
    r : float
        Spatial length scale of wave propagation in millimeters.
    gamma : float
        Damping rate of wave propagation in seconds^-1.
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
    nt = len(input_coeff)

    # Pad input with zeros on negative side to ensure causality (system is only driven for t >= 0)
    # This is required for the correct Green's function solution of the damped wave equation.
    input_coeff_padded = np.concatenate([np.zeros(nt), input_coeff])

    # Apply inverse Fourier transform to get frequency-domain representation of the causal signal.
    input_coeff_f = np.fft.fftshift(np.fft.ifft(input_coeff_padded))

    # Frequencies for full signal
    omega = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(2*nt, d=dt))

    # Compute transfer function and apply it to frequency-domain input
    H = gamma**2 / (-omega**2 - 2j * omega * gamma + gamma**2 * (1 + r**2 * eval))
    out_fft = H * input_coeff_f

    # Inverse transform to time domain, implemented as forward FFT for causality
    out_full = np.real(np.fft.fft(np.fft.ifftshift(out_fft)))

    # Return only the non-negative time part (t >= 0)
    return out_full[nt:]

def _model_wave_ode(
    input_coeff: NDArray,
    t: NDArray,
    r: float,
    gamma: float,
    eval: float
) -> NDArray:
    """
    Solves the damped wave ODE for one eigenmode j.

    Parameters
    ----------
    input_coeff : np.ndarray
        Input drive to the system with shape (nt,) (written as `qj` in equation below).
    t : np.ndarray
        Vector of timepoints in seconds.
    gamma : float
        Damping coefficient seconds^-1.
    r : float
        Spatial length scale in millimeters.
    eval : float
        Eigenvalue for the j-th mode (written as `lambdaj` in equation below).

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
    def q_interp_safe(t_):
        """Safely interpolate the driving term at time t_."""
        val = np.interp(t_, t, input_coeff)
        return val.item() if isinstance(val, np.ndarray) else val

    def wave_rhs(t_, y):
        """Right-hand side of the wave equation in first-order form."""
        x1, x2 = y  # both should be scalars
        qval = q_interp_safe(t_)  # should be scalar

        dx1dt = x2
        dx2dt = -2 * gamma * x2 - gamma**2 * (1 + r**2 * eval) * x1 + gamma**2 * qval

        return [dx1dt, dx2dt]

    y0 = [0.0, 0.0] # Initial conditions: phi_j(0) = 0, dphi_j/dt(0) = 0

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

def get_balloon_params(**overrides) -> dict:
    """
    Return balloon model parameters with optional overrides.
    
    Parameters
    ----------
    **overrides
        Balloon model parameters to override default values. Must be positive.
        
    Returns
    -------
    dict
        Balloon model parameters.
        - `kappa`: Signal decay rate [s^-1]. Default is `0.65`.
        - `gamma_h`: Rate of elimination [s^-1]. Default is `0.41`.
        - `tau`: Hemodynamic transit time [s]. Default is `0.98`.
        - `alpha`: Grubb's exponent [unitless]. Default is `0.32`.
        - `rho`: Resting oxygen extraction fraction [unitless]. Default is `0.34`.
        - `V_0`: Resting blood volume fraction [unitless]. Default is `0.02`.
        - `w_f`: Frequency of blood flow response [rad/s]. Default is `0.56`.
        - `k1`, `k2`, `k3`: Coefficients for BOLD signal equation [unitless]. Defaults are `3.72`,
        `0.527`, and `0.48`, respectively.
    
    Raises
    ------
    ValueError
        If any provided balloon model parameter is non-positive.
    """
    
    # Get default values
    params = {
        'kappa': 0.65,
        'gamma_h': 0.41,
        'tau': 0.98,
        'alpha': 0.32,
        'rho': 0.34,
        'V_0': 0.02,
        'w_f': 0.56,
        'k1': 3.72,
        'k2': 0.527,
        'k3': 0.48
    }

    # Validate and apply overrides
    for param, value in overrides.items():
        if param not in params:
            raise ValueError(f"Invalid Balloon model parameter '{param}'.")
        if value <= 0 or np.isnan(value) or np.isinf(value):
            raise ValueError("All Balloon model parameters must be positive and finite (received "
                             f"{param}={value}).")

    params.update(overrides)

    return params

def _model_balloon_fourier(
    activity_coeff: NDArray,
    balloon_freq_response: NDArray,
) -> NDArray:
    """
    Simulates the hemodynamic response of one mode using the balloon model in the frequency domain. 
    This method applies a frequency-domain implementation of the balloon model to a given set of 
    mode coefficients, returning the modeled hemodynamic response over time.

    Parameters
    ----------
    activity_coeff : np.ndarray
        Array of mode coefficients representing the input signal to the model, with shape (nt,).
    balloon_freq_response : np.ndarray
        Frequency response of the balloon model at each frequency.

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
    # Zero-pad input at t < 0 for causality)
    nt = len(activity_coeff)
    activity_coeff_padded = np.concatenate([np.zeros(nt), activity_coeff])

    # Apply Fourier transform (implemented as inverse FFT for causality)
    activity_coeff_f = np.fft.fftshift(np.fft.ifft(activity_coeff_padded))

    # Apply frequency response
    out_fft = balloon_freq_response * activity_coeff_f

    # Inverse transform back to timeseries (implemented as forward FFT for causality)
    out_full = np.real(np.fft.fft(np.fft.ifftshift(out_fft)))

    # Remove zero padding
    return out_full[nt:]

def _model_balloon_ode(
    activity_coeff: NDArray,
    t: NDArray,
    params: dict
) -> NDArray:
    """
    Simulates the hemodynamic response of one mode using the balloon model in the time domain (ODE 
    approach). This function numerically integrates the balloon model ODEs for a given input mode 
    time course.

    Parameters
    ----------
    activity_coeff : np.ndarray
        Array of mode coefficients representing the input signal to the model, with shape (nt,).
    t : np.ndarray
        Vector of timepoints in seconds with shape (nt,).
    params: dict
        Balloon model parameters. See the `get_balloon_params` function for default parameters.

    Returns
    -------
    np.ndarray
        The BOLD signal time course for the mode at the specified time points.
    """    
    # Extract base parameters
    kappa = params['kappa']
    gamma_h = params['gamma_h']
    tau = params['tau']
    alpha = params['alpha']
    V_0 = params['V_0']
    rho = params['rho']
    k1 = params['k1']
    k2 = params['k2']
    k3 = params['k3']

    # ODE system: y = [s, f, v, q]
    # s: vasodilatory signal, f: blood inflow, v: blood volume, q: deoxyhemoglobin content
    def balloon_rhs(t_, y):
        s, f, v, q = y
        # Interpolate neural input at current time
        u = np.interp(t_, t, activity_coeff)
        dsdt = u - kappa * s - gamma_h * (f - 1)
        dfdt = s
        dvdt = (f - v ** (1 / alpha)) / tau
        dqdt = (f * (1 - (1 - rho) ** (1 / f)) / rho - q * v ** (1 / alpha - 1)) / tau
        return [dsdt, dfdt, dvdt, dqdt]

    # Initial conditions for [s, f, v, q]
    y0 = [0.0, 1.0, 1.0, 1.0]

    # Solve ODEs, raise error if overflows occur
    sol = solve_ivp(
        balloon_rhs,
        t_span=(t[0], t[-1]),
        y0=y0,
        t_eval=t,
        method='RK45',
        rtol=1e-6,
        atol=1e-9
    )

    if not sol.success:
        raise RuntimeError("Balloon model ODE solver failed. Try using `pde_method='fourier'` or " \
                           "a smaller `dt` timestep without altering balloon model parameters. " \
                           f"`scipy.integrate.solve_ivp` message: {sol.message}")

    # Apply standard BOLD signal equation
    _, _, v, q = sol.y
    return V_0 * (k1 * (1 - q) + k2 * (1 - q / v) + k3 * (1 - v))

def _calc_balloon_freq_response(
    nt: int,
    dt: float,
    params: dict
) -> NDArray:
    """
    Calculate the frequency response of the balloon model for use in the Fourier implementation.
    
    Parameters
    ----------
    nt : int
        Number of time points.
    dt : float
        Time step in seconds.
    params : dict
        Balloon model parameters. See the `get_balloon_params` function for default parameters.
    
    Returns
    -------
    np.ndarray
        Frequency response of the balloon model at each frequency, of shape (2*nt,).
    """
    # Extract parameters
    kappa = params['kappa']
    tau = params['tau']
    alpha = params['alpha']
    w_f = params['w_f']
    V_0 = params['V_0']
    k1 = params['k1']
    k2 = params['k2']
    k3 = params['k3']
    rho = params['rho']

    # Frequencies for full signal
    omega = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(2*nt, d=dt))

    # Calculate the frequency response of the system
    beta = (rho + (1 - rho) * np.log(1 - rho)) / rho
    phi_hat_Fz = 1 / (-(omega + 1j * 0.5 * kappa) ** 2 + w_f ** 2)
    phi_hat_yF = V_0 * (alpha * (k2 + k3) * (1 - 1j * tau * omega) 
                                - (k1 + k2) * (alpha + beta - 1 - 1j * tau * alpha * beta * omega)
                                ) / ((1 - 1j * tau * omega) * (1 - 1j * tau * alpha * omega))
    
    return phi_hat_yF * phi_hat_Fz