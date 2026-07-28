import numpy as np
import warnings
from scipy.optimize import minimize
from scipy.optimize import minimize_scalar
from scipy.linalg import solve_banded
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline
from typing import Literal, Tuple


def add_regularizing_noise(
    data,
    R_add,
    rng,
    *,
    copy=True,
    key="yk_reg",
    validate=False,
):
    """
    Add Gaussian regularizing noise to measured observations.

    This is intended as a preprocessing step before phase/parameter
    inference. The inferred observation variance R_eff then absorbs:

        R_eff ≈ R_meas + R_add + amplitude-fluctuation contributions

    Parameters
    ----------
    data : dict or tuple
        Either:
            {"tk": ..., "yk": ...}
        or
            (tk, yk)

    R_add : float
        Variance of the added Gaussian regularizing noise.

    rng : np.random.Generator
        Random number generator.

    copy : bool, default True
        If data is a dict:
            True  -> return a copied dict
            False -> modify in place

    key : str, default "yk_reg"
        Key used to store the regularized observations when
        data is a dict.

    validate : bool, default False
        Run basic validation checks.

    Returns
    -------
    out : dict or tuple
        If input is a dict:
            returns a dict containing

                out[key]          regularized observations
                out["R_add"]      added noise variance

        If input is a tuple:
            returns

                (tk, yk_reg)

    Notes
    -----
    The added noise is

        epsilon_k ~ N(0, R_add)

    and the returned observations are

        yk_reg = yk + epsilon_k

    The added noise realization is fixed once generated and should
    then be reused throughout inference.
    """
    if R_add < 0:
        raise ValueError("R_add must be nonnegative.")

    if isinstance(data, dict):
        tk = np.asarray(data["tk"], dtype=float)
        yk = np.asarray(data["yk"], dtype=float)

        out = data.copy() if copy else data

    else:
        tk, yk = data
        tk = np.asarray(tk, dtype=float)
        yk = np.asarray(yk, dtype=float)

    if validate:
        if tk.ndim != 1:
            raise ValueError("tk must be one-dimensional.")
        if yk.shape[0] != tk.shape[0]:
            raise ValueError("tk and yk must have matching leading dimension.")

    if R_add == 0:
        yk_reg = yk.copy()
    else:
        yk_reg = yk + np.sqrt(R_add) * rng.standard_normal(size=yk.shape)

    if isinstance(data, dict):
        out[key] = yk_reg
        out["R_add"] = float(R_add)
        return out

    return tk, yk_reg
    
def transition_matrix_from_kernel(kernel):
    """
    Construct the circulant transition matrix corresponding to a
    wrapped transition kernel.

    Parameters
    ----------
    kernel : ndarray, shape (L,)
        Circular transition kernel such that

            T[i,j] = kernel[(j-i) % L]

        represents

            p(theta_j | theta_i).

    Returns
    -------
    T : ndarray, shape (L, L)
        Row-stochastic transition matrix.
    """
    kernel = np.asarray(kernel, dtype=float)

    if kernel.ndim != 1:
        raise ValueError("kernel must be one-dimensional.")

    L = len(kernel)

    T = np.empty((L, L), dtype=float)

    idx = np.arange(L)

    for i in range(L):
        T[i] = kernel[(idx - i) % L]

    # numerical guard
    row_sums = T.sum(axis=1, keepdims=True)
    T /= row_sums

    return T
    

def q_fourier(theta, A, B):
    """
    Evaluate the scalar Fourier waveform

        q(theta)
        = sum_m A[m] cos(m theta)
        + sum_m B[m] sin(m theta)

    with gauge convention
        B[0] = 0
        B[1] = 0

    Parameters
    ----------
    theta : array_like
        Phase values.

    A, B : ndarray, shape (M+1,)
        Fourier coefficients.

    Returns
    -------
    q : ndarray
        Waveform evaluated at theta.
    """
    theta = np.asarray(theta, dtype=float)

    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)

    if A.shape != B.shape:
        raise ValueError("A and B must have same shape.")

    M = len(A) - 1

    q = np.zeros_like(theta, dtype=float)

    for m in range(M + 1):
        q += A[m] * np.cos(m * theta)

    for m in range(1, M + 1):
        q += B[m] * np.sin(m * theta)

    return q


def q_fourier_prime(theta, A, B):
    """
    First derivative dq/dtheta of the Fourier waveform.
    """
    theta = np.asarray(theta, dtype=float)

    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)

    if A.shape != B.shape:
        raise ValueError("A and B must have same shape.")

    M = len(A) - 1

    qp = np.zeros_like(theta, dtype=float)

    for m in range(1, M + 1):
        qp += -m * A[m] * np.sin(m * theta)

    for m in range(1, M + 1):
        qp += m * B[m] * np.cos(m * theta)

    return qp


def pack_param(omega, lnDtheta, lnR, A, B):
    """
    Pack structured parameters into a flat optimization vector.

    Gauge convention
    ----------------
    Fixed coefficients:
        B[0] = 0
        B[1] = 0

    Free coefficients:
        A[0], A[1], ..., A[M]
        B[2], ..., B[M]
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)

    if A.ndim != 1 or B.ndim != 1:
        raise ValueError("A and B must be one-dimensional arrays.")
    if A.shape != B.shape:
        raise ValueError(f"A and B must have same shape; got {A.shape} and {B.shape}")
    if len(A) < 2:
        raise ValueError("A and B must have length at least 2.")

    Bvec = B[2:]  # skip fixed B[0], B[1]

    return np.concatenate([
        [omega, lnDtheta, lnR],
        A,
        Bvec,
    ])


def unpack_param(x, M):
    """
    Unpack a flat optimization vector into structured parameters.

    The flat vector is interpreted as
        [omega, lnDtheta, lnR, A[0], ..., A[M], B[2], ..., B[M]].
    """
    x = np.asarray(x, dtype=float)

    omega = x[0]
    lnDtheta = x[1]
    lnR = x[2]

    nA = M + 1
    nB = max(M - 1, 0)

    expected = 3 + nA + nB
    if x.shape[0] != expected:
        raise ValueError(f"Expected flat parameter vector of length {expected}; got {x.shape[0]}")

    Avec = x[3:3 + nA]
    Bvec = x[3 + nA:3 + nA + nB]

    A = np.zeros(M + 1, dtype=float)
    B = np.zeros(M + 1, dtype=float)

    A[:] = Avec
    if M >= 2:
        B[2:] = Bvec

    # Fixed gauge coefficients.
    B[0] = 0.0
    B[1] = 0.0

    return omega, lnDtheta, lnR, A, B


def glycolytic_rhs(t, x, a, b):
    x1, x2 = x
    return np.array([
        -x1 + a * x2 + x1**2 * x2,
        b - a * x2 - x1**2 * x2,
    ], dtype=float)


def find_glycolytic_limit_cycle(
    a,
    b,
    *,
    x0=None,
    t_transient=500.0,
    t_cycle=200.0,
    n_cycle=2000,
    rtol=1e-9,
    atol=1e-11,
):
    """
    Integrate the noise-free glycolytic oscillator, identify one limit-cycle
    period, and set phase zero at the point where x1 is minimal.
    """
    if x0 is None:
        # Perturb away from the fixed point.
        x0 = np.array([b, b / (a + b**2)]) + np.array([0.1, 0.1])
    else:
        x0 = np.asarray(x0, dtype=float)

    # First converge to the attracting limit cycle.
    sol_tr = solve_ivp(
        lambda t, x: glycolytic_rhs(t, x, a, b),
        (0.0, t_transient),
        x0,
        rtol=rtol,
        atol=atol,
    )
    x_start = sol_tr.y[:, -1]

    # Then integrate long enough to identify a clean cycle.
    t_eval = np.linspace(0.0, t_cycle, max(5000, 5 * n_cycle))
    sol = solve_ivp(
        lambda t, x: glycolytic_rhs(t, x, a, b),
        (0.0, t_cycle),
        x_start,
        t_eval=t_eval,
        rtol=rtol,
        atol=atol,
    )

    t = sol.t
    x = sol.y.T
    x1 = x[:, 0]

    # Local minima of x1 define phase-zero crossings.
    minima = np.where((x1[1:-1] < x1[:-2]) & (x1[1:-1] < x1[2:]))[0] + 1

    if len(minima) < 3:
        raise RuntimeError(
            "Could not identify enough x1 minima to determine the limit cycle. "
            "Try increasing t_cycle or t_transient."
        )

    # Use the last two minima to define one period.
    i0, i1 = minima[-2], minima[-1]
    T0 = t[i1] - t[i0]
    omega0 = 2.0 * np.pi / T0

    # Extract and resample one cycle, with phase zero at x1 minimum.
    t_seg = t[i0:i1 + 1] - t[i0]
    x_seg = x[i0:i1 + 1].copy()
    
    # Force exact periodic closure for CubicSpline.
    # The endpoint represents the same phase as the initial point.
    x_seg[-1] = x_seg[0]
    
    phase_grid = np.linspace(0.0, 2.0 * np.pi, n_cycle, endpoint=False)
    t_grid = phase_grid / omega0
    
    cs1 = CubicSpline(t_seg, x_seg[:, 0], bc_type="periodic")
    cs2 = CubicSpline(t_seg, x_seg[:, 1], bc_type="periodic")

    x_cycle = np.column_stack([cs1(t_grid), cs2(t_grid)])

    return {
        "T0": T0,
        "omega0": omega0,
        "phase_grid": phase_grid,
        "t_grid": t_grid,
        "x_cycle": x_cycle,
        "q_cycle": x_cycle[:, 0],
        "interp_x1": cs1,
        "interp_x2": cs2,
    }


def simulate_glycolytic_data(
    N,
    dt,
    R,
    a,
    b,
    Dx,
    dt_integrate,
    rng,
    *,
    initial_phase=0.0,
    x0=None,
    t_transient=500.0,
    t_cycle=200.0,
    n_cycle=2000,
    validate=False,
):
    """
    Simulate partial observations of a noisy glycolytic oscillator.

    Stochastic dynamics
    -------------------
        dx = f(x) dt + sqrt(2 Dx) dW

    Measurement model
    -----------------
        y_k = x_1(t_k) + eta_k,
        eta_k ~ N(0, R)

    Phase convention
    ----------------
    Phase zero is the point on the deterministic limit cycle where x1 is minimal.

    Returns
    -------
    out : dict
        tk : shape (N,)
            Sampling times.

        xk : shape (N, 2)
            Noisy latent state sampled at tk.

        yk_clean : shape (N,)
            Noise-free partial observation x1(tk) from the noisy state.

        yk : shape (N,)
            Partial observation with measurement noise.

        omega0 : float
            Deterministic limit-cycle angular frequency.

        T0 : float
            Deterministic limit-cycle period.

        phase_grid : shape (n_cycle,)
            Phase grid for the deterministic waveform.

        x_cycle : shape (n_cycle, 2)
            Deterministic limit cycle with phase zero at min x1.

        q_cycle : shape (n_cycle,)
            Noise-free waveform q(theta) = x_{0,1}(theta).
    """
    if validate:
        if not (isinstance(N, (int, np.integer)) and N > 0):
            raise ValueError("N must be a positive integer.")
        if dt <= 0 or dt_integrate <= 0:
            raise ValueError("dt and dt_integrate must be positive.")
        if R < 0 or Dx < 0:
            raise ValueError("R and Dx must be nonnegative.")
        if a <= 0 or b <= 0:
            raise ValueError("a and b must be positive.")

    lc = find_glycolytic_limit_cycle(
        a,
        b,
        x0=x0,
        t_transient=t_transient,
        t_cycle=t_cycle,
        n_cycle=n_cycle,
    )

    T0 = lc["T0"]
    omega0 = lc["omega0"]

    # Initialize on the deterministic limit cycle at requested phase.
    phi0 = np.mod(initial_phase, 2.0 * np.pi)
    t0_cycle = phi0 / omega0
    x = np.array([
        lc["interp_x1"](t0_cycle),
        lc["interp_x2"](t0_cycle),
    ], dtype=float)

    tk = dt * np.arange(1, N + 1, dtype=float)
    xk = np.empty((N, 2), dtype=float)

    t = 0.0
    next_sample = dt
    sample_index = 0

    noise_prefactor = np.sqrt(2.0 * Dx)

    while sample_index < N:
        h = min(dt_integrate, next_sample - t)

        # Euler--Maruyama step.
        drift = glycolytic_rhs(t, x, a, b)
        if Dx > 0:
            x = x + drift * h + noise_prefactor * np.sqrt(h) * rng.standard_normal(2)
        else:
            x = x + drift * h

        t += h

        # Store sample. The tolerance avoids floating-point misses.
        if abs(t - next_sample) <= 1e-12 * max(1.0, next_sample):
            xk[sample_index] = x
            sample_index += 1
            next_sample += dt

    yk_clean = xk[:, 0].copy()

    if R > 0:
        yk = yk_clean + np.sqrt(R) * rng.standard_normal(N)
    else:
        yk = yk_clean.copy()

    return {
        "tk": tk,
        "xk": xk,
        "yk_clean": yk_clean,
        "yk": yk,
        "omega0": omega0,
        "T0": T0,
        "phase_grid": lc["phase_grid"],
        "x_cycle": lc["x_cycle"],
        "q_cycle": lc["q_cycle"],
        "limit_cycle": lc,
    }


def estimate_frequency_fft(data, pad_factor=4, detrend=True, window=True):
    """
    Estimate dominant angular frequency ω using FFT with zero padding.

    Parameters
    ----------
    data : (tk, yk)
        tk : array, shape (N,)
        yk : array, shape (N,) or (N, d)
             If multi-dimensional, will be reduced to 1D signal.

    pad_factor : int (default 4)
        Zero-padding factor (FFT length = pad_factor * N)

    detrend : bool (default True)
        Remove mean before FFT

    window : bool (default True)
        Apply Hann window to reduce spectral leakage

    Returns
    -------
    omega_hat : float
        Estimated angular frequency (rad / unit time)

    f_hat : float
        Estimated frequency (cycles / unit time)

    spectrum : (freqs, power)
        For debugging/plotting
    """

    tk, yk = data
    tk = np.asarray(tk)
    yk = np.asarray(yk)

    N = len(tk)
    dt = tk[1] - tk[0]

    # --- Convert to scalar signal ---
    if yk.ndim == 2:
        # Robust choice: use principal component (generalizable)
        y_mean = yk.mean(axis=0)
        U, S, Vt = np.linalg.svd(yk - y_mean, full_matrices=False)
        y = U[:, 0] * S[0]
    else:
        y = yk.copy()

    # --- Detrend ---
    if detrend:
        y = y - np.mean(y)

    # --- Windowing ---
    if window:
        w = np.hanning(N)
        y = y * w

    # --- Zero padding ---
    Nfft = int(pad_factor * N)

    Y = np.fft.rfft(y, n=Nfft)
    freqs = np.fft.rfftfreq(Nfft, d=dt)

    power = np.abs(Y)**2

    # Ignore DC component
    idx = np.argmax(power[1:]) + 1

    f_hat = freqs[idx]
    omega_hat = 2 * np.pi * f_hat

    return omega_hat, f_hat, (freqs, power)



def estimate_initial_phase_regression(
    data,
    omega_hat,
    window_size,
    R,
    *,
    M=3,
    start_index=0,
    validate=False,
):
    """
    Estimate initial phase and a scalar noise-free waveform by finite-window
    Fourier regression.

    Regression model
    ----------------
    Over a short fitting window, assume phase diffusion is negligible and

        theta_raw_k = omega_hat * t_k

        y_k = sum_{m=0}^M A_raw[m] cos(m theta_raw_k)
            + sum_{m=1}^M B_raw[m] sin(m theta_raw_k)
            + eta_k.

    The coefficients are then rotated to define the anchored phase

        theta_k = theta_raw_k + phi,

    with gauge convention

        B[0] = 0,
        B[1] = 0.

    This makes the first harmonic purely cosine.  No sign convention on A[1]
    is imposed here; in particular, A[1] is allowed to be negative.
    """
    if isinstance(data, dict):
        tk = np.asarray(data["tk"], dtype=float)
        yk = np.asarray(data["yk"], dtype=float)
    else:
        tk, yk = data
        tk = np.asarray(tk, dtype=float)
        yk = np.asarray(yk, dtype=float)

    yk = np.squeeze(yk)

    if validate:
        if tk.ndim != 1:
            raise ValueError(f"tk must have shape (N,); got {tk.shape}")
        if yk.ndim != 1:
            raise ValueError(f"yk must be scalar observations with shape (N,); got {yk.shape}")
        if tk.shape[0] != yk.shape[0]:
            raise ValueError(f"tk and yk must have same length; got {tk.shape[0]} and {yk.shape[0]}")
        if M < 1:
            raise ValueError("M must be >= 1")
        if not np.isscalar(window_size) or int(window_size) != window_size:
            raise ValueError("window_size must be an integer")
        if int(window_size) < 2 * M + 1:
            raise ValueError("window_size must be >= 2*M + 1")
        if not np.isscalar(start_index) or int(start_index) != start_index:
            raise ValueError("start_index must be an integer")
        if int(start_index) < 0:
            raise ValueError("start_index must be nonnegative")
        if int(start_index) + int(window_size) > len(tk):
            raise ValueError("start_index + window_size exceeds number of samples")
        if R <= 0:
            raise ValueError(f"R must be > 0; got {R!r}")

    m_window = int(window_size)
    i0 = int(start_index)
    idx = np.arange(i0, i0 + m_window)

    t = tk[idx]
    y = yk[idx]

    theta_raw = omega_hat * t

    # Design matrix: coeff_raw = [A0, A1, B1, A2, B2, ..., AM, BM]
    Z = np.ones((m_window, 1 + 2 * M), dtype=float)
    for m in range(1, M + 1):
        Z[:, 2 * m - 1] = np.cos(m * theta_raw)
        Z[:, 2 * m] = np.sin(m * theta_raw)

    JTJ = Z.T @ Z
    JTy = Z.T @ y

    coeff_raw = np.linalg.solve(JTJ, JTy)
    cov_raw = R * np.linalg.inv(JTJ)

    A_raw = np.zeros(M + 1, dtype=float)
    B_raw = np.zeros(M + 1, dtype=float)

    A_raw[0] = coeff_raw[0]
    for m in range(1, M + 1):
        A_raw[m] = coeff_raw[2 * m - 1]
        B_raw[m] = coeff_raw[2 * m]

    # Gauge fixing: choose phi so that B[1] = 0 in the rotated frame.
    # A1_raw cos(u) + B1_raw sin(u) = A1 cos(u + phi).
    # With theta = u + phi, the rotated first-harmonic sine coefficient is
    # B1 = A1_raw sin(phi) + B1_raw cos(phi), so B1=0 for
    # phi = atan2(-B1_raw, A1_raw).  A1 may be positive or negative.
    r1_sq = A_raw[1] ** 2 + B_raw[1] ** 2
    if r1_sq <= 0:
        raise RuntimeError(
            "Degenerate first harmonic: cannot anchor phase because A_raw[1] = B_raw[1] = 0."
        )

    phi_mean = np.arctan2(-B_raw[1], A_raw[1])

    # Delta-method uncertainty for phi = atan2(-B_raw[1], A_raw[1]).
    grad_phi_raw = np.zeros_like(coeff_raw)
    grad_phi_raw[1] = B_raw[1] / r1_sq       # d phi / d A_raw[1]
    grad_phi_raw[2] = -A_raw[1] / r1_sq      # d phi / d B_raw[1]

    phi_var = float(grad_phi_raw @ cov_raw @ grad_phi_raw)
    phi_std = np.sqrt(max(phi_var, 0.0))

    A = np.zeros(M + 1, dtype=float)
    B = np.zeros(M + 1, dtype=float)

    A[0] = A_raw[0]
    for m in range(1, M + 1):
        cm = np.cos(m * phi_mean)
        sm = np.sin(m * phi_mean)

        A[m] = A_raw[m] * cm - B_raw[m] * sm
        B[m] = A_raw[m] * sm + B_raw[m] * cm

    # Fixed gauge coefficients.
    B[0] = 0.0
    B[1] = 0.0

    theta_fit = theta_raw + phi_mean

    y_fit = np.zeros_like(theta_fit, dtype=float)
    for m in range(0, M + 1):
        y_fit += A[m] * np.cos(m * theta_fit)
    for m in range(2, M + 1):
        y_fit += B[m] * np.sin(m * theta_fit)

    return {
        "phi_mean": phi_mean,
        "phi_std": phi_std,
        "A": A,
        "B": B,
        "A_raw": A_raw,
        "B_raw": B_raw,
        "coeff_raw": coeff_raw,
        "cov_raw": cov_raw,
        "fit_indices": idx,
        "theta_fit": theta_fit,
        "y_fit": y_fit,
    }


def wrap_to_pi(x):
    return (x + np.pi) % (2.0 * np.pi) - np.pi


def wrapped_gaussian_kernel(theta_grid, mean_step, Q):
    """
    Circular transition kernel for

        theta_k = theta_{k-1} + mean_step + noise,
        noise ~ N(0, Q)

    Returns kernel[d] proportional to p(theta_j | theta_i)
    where d corresponds to theta_j - theta_i.
    """
    dtheta = theta_grid[1] - theta_grid[0]
    delta = wrap_to_pi(theta_grid - mean_step)

    if Q <= 0:
        kernel = np.zeros_like(theta_grid)
        idx = np.argmin(np.abs(delta))
        kernel[idx] = 1.0
    else:
        kernel = np.exp(-0.5 * delta**2 / Q)
        kernel /= np.sum(kernel)

    return kernel


def circular_convolve(kernel, p):
    """
    Circular convolution:

        out[j] = sum_i kernel[j-i] p[i]

    Used for the forward prediction step.
    """
    return np.fft.ifft(np.fft.fft(kernel) * np.fft.fft(p)).real


def circular_backward_apply(kernel, g):
    """
    Backward transition application:

        out[i] = sum_j T[i,j] g[j]
               = sum_d kernel[d] g[i+d]

    where T[i,j] = kernel[j-i].
    """
    L = len(kernel)
    kernel_rev = kernel[(-np.arange(L)) % L]
    return circular_convolve(kernel_rev, g)


def global_phase_filter(
    param,
    data,
    state_priors,
    *,
    n_theta=256,
    validate=False,
):
    """
    Grid-based global phase filter for scalar periodic observations.

    param = (omega, lnDtheta, lnR, A, B)
    """
    omega, lnDtheta, lnR, A, B = param
    Dtheta = np.exp(lnDtheta)
    R = np.exp(lnR)

    if isinstance(data, dict):
        tk = np.asarray(data["tk"], dtype=float)
        yk = np.asarray(data["yk"], dtype=float)
    else:
        tk, yk = data
        tk = np.asarray(tk, dtype=float)
        yk = np.asarray(yk, dtype=float)

    yk = np.squeeze(yk)
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    param = (omega, lnDtheta, lnR, A, B)

    mu_theta, sigma_theta = state_priors[:2]

    N = len(yk)
    if N == 0:
        return {
            "theta_grid": np.empty(0),
            "alpha_pred": np.empty((0, 0)),
            "alpha_filt": np.empty((0, 0)),
            "likelihood": np.empty((0, 0)),
            "loglik_offsets": np.empty(0),
            "kernel": np.empty(0),
            "q_grid": np.empty(0),
            "loglik": 0.0,
            "scales": np.empty(0),
            "param": param,
        }

    dt = tk[1] - tk[0] if N > 1 else 0.0

    if validate:
        if n_theta < 16:
            raise ValueError("n_theta should be at least 16.")
        if yk.ndim != 1:
            raise ValueError(f"yk must be scalar with shape (N,); got {yk.shape}")
        if tk.shape != (N,):
            raise ValueError(f"tk must have shape ({N},); got {tk.shape}")
        if N > 2 and not np.allclose(np.diff(tk), dt):
            raise ValueError("tk must be uniformly spaced.")
        if N > 1 and dt <= 0:
            raise ValueError("dt must be > 0.")
        if not np.isfinite(lnDtheta):
            raise ValueError("lnDtheta must be finite.")
        if not np.isfinite(lnR):
            raise ValueError("lnR must be finite.")
        if Dtheta <= 0:
            raise ValueError("Dtheta must be > 0.")
        if R <= 0:
            raise ValueError("R must be > 0.")
        if sigma_theta <= 0:
            raise ValueError("sigma_theta must be > 0.")
        if A.shape != B.shape:
            raise ValueError("A and B must have same shape.")
        if A.ndim != 1:
            raise ValueError("A and B must be one-dimensional arrays.")
        if len(A) < 2:
            raise ValueError("A and B must have length at least 2.")
        if not np.isclose(B[0], 0.0):
            raise ValueError("B[0] must be fixed to 0.")
        if len(B) > 1 and not np.isclose(B[1], 0.0):
            raise ValueError("B[1] must be fixed to 0.")

    theta_grid = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)

    Q = 2.0 * Dtheta * dt
    mean_step = omega * dt

    kernel = wrapped_gaussian_kernel(theta_grid, mean_step, Q)
    q_grid = q_fourier(theta_grid, A, B)

    # Measurement likelihoods, stabilized by subtracting max log likelihood.
    likelihood = np.empty((N, n_theta), dtype=float)
    loglik_offsets = np.empty(N, dtype=float)

    for k in range(N):
        resid = yk[k] - q_grid
        logL = -0.5 * np.log(2.0 * np.pi * R) - 0.5 * resid**2 / R

        logL_max = np.max(logL)
        if not np.isfinite(logL_max):
            raise RuntimeError(f"Invalid measurement likelihood at k={k}.")

        likelihood[k] = np.exp(logL - logL_max)
        loglik_offsets[k] = logL_max

    prior_delta = wrap_to_pi(theta_grid - mu_theta)
    prior = np.exp(-0.5 * prior_delta**2 / sigma_theta**2)
    prior_sum = np.sum(prior)

    if prior_sum <= 0 or not np.isfinite(prior_sum):
        raise RuntimeError("Invalid wrapped-Gaussian prior normalization.")

    prior /= prior_sum

    alpha_pred = np.empty((N, n_theta), dtype=float)
    alpha_filt = np.empty((N, n_theta), dtype=float)
    scales = np.empty(N, dtype=float)

    alpha_pred[0] = prior
    unnorm = likelihood[0] * alpha_pred[0]
    scales[0] = np.sum(unnorm)

    if scales[0] <= 0 or not np.isfinite(scales[0]):
        raise RuntimeError("Filtering failed at k=0: nonpositive normalization.")

    alpha_filt[0] = unnorm / scales[0]

    for k in range(1, N):
        alpha_pred[k] = circular_convolve(kernel, alpha_filt[k - 1])
        alpha_pred[k] = np.maximum(alpha_pred[k], 0.0)

        pred_sum = np.sum(alpha_pred[k])
        if pred_sum <= 0 or not np.isfinite(pred_sum):
            raise RuntimeError(f"Prediction failed at k={k}: nonpositive normalization.")

        alpha_pred[k] /= pred_sum

        unnorm = likelihood[k] * alpha_pred[k]
        scales[k] = np.sum(unnorm)

        if scales[k] <= 0 or not np.isfinite(scales[k]):
            raise RuntimeError(f"Filtering failed at k={k}: nonpositive normalization.")

        alpha_filt[k] = unnorm / scales[k]

    loglik = float(np.sum(np.log(scales) + loglik_offsets))

    return {
        "theta_grid": theta_grid,
        "alpha_pred": alpha_pred,
        "alpha_filt": alpha_filt,
        "likelihood": likelihood,
        "loglik_offsets": loglik_offsets,
        "kernel": kernel,
        "q_grid": q_grid,
        "loglik": loglik,
        "scales": scales,
        "param": param,
    }
    

def global_phase_smoother(filter_out, *, validate=False):
    """
    Backward smoother for the global phase filter.

    Parameters
    ----------
    filter_out : dict
        Output of global_phase_filter(...).

    Returns
    -------
    out : dict
        gamma : ndarray, shape (N, n_theta)
            Smoothed marginal distributions
            p(theta_k | y_1:N).

        beta : ndarray, shape (N, n_theta)
            Backward messages, normalized.

        xi_pair : optional not returned by default
            Pairwise distributions can be computed later as needed for EM.
    """
    alpha_filt = np.asarray(filter_out["alpha_filt"], dtype=float)
    likelihood = np.asarray(filter_out["likelihood"], dtype=float)
    kernel = np.asarray(filter_out["kernel"], dtype=float)

    N, n_theta = alpha_filt.shape

    if validate:
        if likelihood.shape != (N, n_theta):
            raise ValueError("likelihood must have same shape as alpha_filt.")
        if kernel.shape != (n_theta,):
            raise ValueError("kernel must have shape (n_theta,).")
        if not np.allclose(np.sum(alpha_filt, axis=1), 1.0):
            raise ValueError("Filtered distributions must be normalized.")

    beta = np.empty_like(alpha_filt)
    gamma = np.empty_like(alpha_filt)

    beta[-1] = np.ones(n_theta, dtype=float)
    beta[-1] /= np.sum(beta[-1])

    gamma[-1] = alpha_filt[-1] * beta[-1]
    gamma[-1] /= np.sum(gamma[-1])

    for k in range(N - 2, -1, -1):
        g_next = likelihood[k + 1] * beta[k + 1]

        beta[k] = circular_backward_apply(kernel, g_next)
        beta[k] = np.maximum(beta[k], 0.0)

        s = np.sum(beta[k])
        if s <= 0 or not np.isfinite(s):
            raise RuntimeError(f"Smoothing failed at k={k}: nonpositive normalization.")

        beta[k] /= s

        gamma[k] = alpha_filt[k] * beta[k]
        gamma[k] = np.maximum(gamma[k], 0.0)
        gamma[k] /= np.sum(gamma[k])

    return {
        "theta_grid": filter_out["theta_grid"],
        "gamma": gamma,
        "beta": beta,
        "alpha_filt": alpha_filt,
        "alpha_pred": filter_out["alpha_pred"],
        "likelihood": likelihood,
        "kernel": kernel,
    }


def sample_phase_trajectories_global(filter_out, n_samples, rng=None):
    """
    Sample phase-index trajectories from the global phase smoothing distribution
    using forward-filter backward-sampling.

    Returns
    -------
    theta_samples : ndarray, shape (n_samples, N)
    index_samples : ndarray, shape (n_samples, N)
    """
    if rng is None:
        rng = np.random.default_rng()

    theta = filter_out["theta_grid"]
    alpha = filter_out["alpha_filt"]
    kernel = filter_out["kernel"]

    N, L = alpha.shape
    T = transition_matrix_from_kernel(kernel)

    index_samples = np.empty((n_samples, N), dtype=int)

    for s in range(n_samples):
        # sample final state
        index_samples[s, -1] = rng.choice(L, p=alpha[-1])

        # sample backward
        for k in range(N - 2, -1, -1):
            j = index_samples[s, k + 1]

            # p(theta_k=i | theta_{k+1}=j, y_1:k)
            p = alpha[k] * T[:, j]
            p_sum = np.sum(p)

            if p_sum <= 0 or not np.isfinite(p_sum):
                raise RuntimeError(f"Invalid backward sampling distribution at k={k}")

            p /= p_sum
            index_samples[s, k] = rng.choice(L, p=p)

    theta_samples = theta[index_samples]
    return theta_samples, index_samples


def phase_msd_from_samples(
    theta_samples,
    lags,
    *,
    omega=None,
    dt=None,
    subtract_drift=True,
):
    lags = np.asarray(lags, dtype=int)
    S, N = theta_samples.shape

    msd = np.empty(len(lags), dtype=float)

    for a, lag in enumerate(lags):
        if lag < 1 or lag >= N:
            raise ValueError(f"Invalid lag {lag}; require 1 <= lag < N.")

        if subtract_drift:
            if omega is None or dt is None:
                raise ValueError("omega and dt are required when subtract_drift=True.")
            drift = omega * lag * dt
        else:
            drift = 0.0

        dtheta = wrap_to_pi(theta_samples[:, lag:] - theta_samples[:, :-lag] - drift)
        msd[a] = np.mean(dtheta**2)

    return msd
    
def neg_log_marginal_likelihood(
    param,
    data,
    state_priors,
    *,
    n_theta=256,
    validate=False,
):
    """
    Negative log marginal likelihood for the global phase filter.

    Parameters
    ----------
    param : tuple
        (omega, lnDtheta, lnR, A, B), where
            Dtheta = exp(lnDtheta)
            R = exp(lnR)

    data : tuple or dict
        Either (tk, yk) or dict with keys "tk", "yk".

    state_priors : tuple
        (mu_theta, sigma_theta)

    n_theta : int
        Number of phase grid points.

    Returns
    -------
    nll : float
        Negative log marginal likelihood,

            -log p(y_{1:N} | lambda),

        approximated by the global phase filter.
    """
    filt = global_phase_filter(
        param,
        data,
        state_priors,
        n_theta=n_theta,
        validate=validate,
    )

    return float(-filt["loglik"])

    
def neg_log_prior(param, param_priors, *, validate=False):
    """
    Negative log prior for (omega, lnDtheta, lnR, A, B).

    Fixed Fourier coefficients:
        B[0] = 0
        B[1] = 0

    Priors
    ------
    omega ~ Normal(mu_omega, sigma_omega^2)

    Dtheta ~ Inv-Gamma(alpha_D, beta_D)
    R      ~ Inv-Gamma(alpha_R, beta_R)

    A[0] has its own prior:
        A[0] ~ Normal(mu_A, sigma_A^2)

    All nonconstant free Fourier coefficients share one zero-mean prior:
        A[m] ~ Normal(0, sigma_q^2), m = 1,...,M
        B[m] ~ Normal(0, sigma_q^2), m = 2,...,M

    Dtheta and R have inverse-gamma priors in natural coordinates, with
    Jacobian corrections included for the log-coordinate posterior.
    """
    (
        mu_omega,
        sigma_omega,
        alpha_D,
        beta_D,
        alpha_R,
        beta_R,
        mu_A,
        sigma_A,
        sigma_q,
    ) = param_priors

    omega, lnDtheta, lnR, A, B = param

    Dtheta = np.exp(lnDtheta)
    R = np.exp(lnR)

    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)

    if validate:
        if sigma_omega <= 0:
            raise ValueError("sigma_omega must be > 0")
        if alpha_D <= 0 or beta_D <= 0:
            raise ValueError("alpha_D and beta_D must be > 0")
        if alpha_R <= 0 or beta_R <= 0:
            raise ValueError("alpha_R and beta_R must be > 0")
        if sigma_A <= 0:
            raise ValueError("sigma_A must be > 0")
        if sigma_q <= 0:
            raise ValueError("sigma_q must be > 0")
        if not np.isfinite(lnDtheta):
            raise ValueError("lnDtheta must be finite")
        if not np.isfinite(lnR):
            raise ValueError("lnR must be finite")
        if Dtheta <= 0 or R <= 0:
            raise ValueError("Dtheta and R must be > 0")
        if A.shape != B.shape:
            raise ValueError(f"A and B must have same shape; got {A.shape} and {B.shape}")
        if A.ndim != 1:
            raise ValueError("A and B must be one-dimensional arrays")
        if len(A) < 2:
            raise ValueError("A and B must have length at least 2")
        if not np.isclose(B[0], 0.0):
            raise ValueError("B[0] must be fixed to 0")
        if not np.isclose(B[1], 0.0):
            raise ValueError("B[1] must be fixed to 0")

    nlp = 0.0

    # Normal prior on omega.
    nlp += 0.5 * ((omega - mu_omega) / sigma_omega) ** 2

    # Inverse-gamma priors in natural coordinates:
    # IG(alpha, beta): -log p(x) = (alpha + 1) log x + beta/x + const.
    nlp += (alpha_D + 1.0) * lnDtheta + beta_D / Dtheta
    nlp += (alpha_R + 1.0) * lnR + beta_R / R

    # Jacobian correction for log-coordinate posterior.
    nlp -= lnDtheta
    nlp -= lnR

    # Prior on the waveform mean A[0].
    nlp += 0.5 * ((A[0] - mu_A) / sigma_A) ** 2

    # Shared zero-mean Gaussian prior on all nonconstant free Fourier modes.
    nlp += 0.5 * np.sum((A[1:] / sigma_q) ** 2)

    if len(B) > 2:
        nlp += 0.5 * np.sum((B[2:] / sigma_q) ** 2)

    return float(nlp)


def neg_log_posterior(param, data, priors, *, n_theta=256, validate=False):
    """
    Negative log posterior for (omega, lnDtheta, lnR, A, B), up to an additive
    constant.

    This combines:
        1. neg_log_marginal_likelihood(...)
        2. neg_log_prior(...)

    Parameters
    ----------
    param : tuple
        (omega, lnDtheta, lnR, A, B)

    data : tuple or dict
        Either (tk, yk) or dict with keys "tk", "yk".

    priors : tuple
        (
            mu_theta, sigma_theta,
            mu_omega, sigma_omega,
            alpha_D, beta_D,
            alpha_R, beta_R,
            mu_A, sigma_A,
            sigma_q,
        )

    Notes
    -----
    The phase prior is on theta_1 = theta(t_1), not on the phase intercept.

    Fixed Fourier coefficients:
        B[0] = 0
        B[1] = 0
    """
    (
        mu_theta,
        sigma_theta,
        mu_omega,
        sigma_omega,
        alpha_D,
        beta_D,
        alpha_R,
        beta_R,
        mu_A,
        sigma_A,
        sigma_q,
    ) = priors

    state_priors = (mu_theta, sigma_theta)

    param_priors = (
        mu_omega,
        sigma_omega,
        alpha_D,
        beta_D,
        alpha_R,
        beta_R,
        mu_A,
        sigma_A,
        sigma_q,
    )

    nll = neg_log_marginal_likelihood(
        param,
        data,
        state_priors,
        n_theta=n_theta,
        validate=validate,
    )

    nlp = neg_log_prior(
        param,
        param_priors,
        validate=validate,
    )

    return float(nll + nlp)


def _build_global_waveform_mstep_matrices(y, smoother_out, M):
    theta = smoother_out["theta_grid"]
    gamma = smoother_out["gamma"]

    y = np.asarray(y, dtype=float).ravel()
    N, L = gamma.shape

    labels = [("A", m) for m in range(M + 1)]
    labels += [("B", m) for m in range(2, M + 1)]

    F = np.empty((L, len(labels)), dtype=float)
    for j, (kind, m) in enumerate(labels):
        F[:, j] = np.cos(m * theta) if kind == "A" else np.sin(m * theta)

    Ephi = gamma @ F
    rhs = Ephi.T @ y

    weights = np.sum(gamma, axis=0)
    G = F.T @ (weights[:, None] * F)

    y2 = float(np.sum(y**2))
    return labels, G, rhs, y2


def _coeff_arrays_from_vector(coeff, labels, M):
    A = np.zeros(M + 1, dtype=float)
    B = np.zeros(M + 1, dtype=float)

    for c, (kind, m) in zip(coeff, labels):
        if kind == "A":
            A[m] = c
        else:
            B[m] = c

    B[0] = 0.0
    B[1] = 0.0
    return A, B


def _waveform_map_given_R_global(y, smoother_out, M, R, param_priors):
    (
        mu_omega, sigma_omega,
        alpha_D, beta_D,
        alpha_R, beta_R,
        mu_A, sigma_A,
        sigma_q,
    ) = param_priors

    labels, G, rhs, y2 = _build_global_waveform_mstep_matrices(
        y, smoother_out, M
    )

    P = len(labels)
    prior_mean = np.zeros(P, dtype=float)
    prior_prec = np.zeros(P, dtype=float)

    for j, (kind, m) in enumerate(labels):
        if kind == "A" and m == 0:
            prior_mean[j] = mu_A
            prior_prec[j] = 1.0 / sigma_A**2
        else:
            prior_mean[j] = 0.0
            prior_prec[j] = 1.0 / sigma_q**2

    lhs = G / R + np.diag(prior_prec)
    rhs_map = rhs / R + prior_prec * prior_mean

    coeff = np.linalg.solve(lhs, rhs_map)
    A_new, B_new = _coeff_arrays_from_vector(coeff, labels, M)

    Sy = float(y2 - 2.0 * coeff @ rhs + coeff @ G @ coeff)
    coeff_prior = 0.5 * np.sum(prior_prec * (coeff - prior_mean) ** 2)

    return A_new, B_new, Sy, coeff_prior


def _waveform_R_map_global(y, smoother_out, M, R0, param_priors):
    (
        mu_omega, sigma_omega,
        alpha_D, beta_D,
        alpha_R, beta_R,
        mu_A, sigma_A,
        sigma_q,
    ) = param_priors

    y = np.asarray(y, dtype=float).ravel()
    N = len(y)

    def objective(lnR):
        R = float(np.exp(lnR))
        _, _, Sy, coeff_prior = _waveform_map_given_R_global(
            y, smoother_out, M, R, param_priors
        )

        # Negative expected log posterior in log-R coordinates.
        return (
            0.5 * N * lnR
            + 0.5 * Sy / R
            + alpha_R * lnR
            + beta_R / R
            + coeff_prior
        )

    res = minimize_scalar(
        objective,
        bracket=(np.log(R0) - 2.0, np.log(R0), np.log(R0) + 2.0),
        method="brent",
    )

    if not res.success:
        raise RuntimeError(f"R/profile waveform M-step failed: {res.message}")

    R_new = float(np.exp(res.x))
    A_new, B_new, Sy, coeff_prior = _waveform_map_given_R_global(
        y, smoother_out, M, R_new, param_priors
    )

    return A_new, B_new, R_new, Sy, coeff_prior


def _phase_dynamics_map_log(ES0, ES1, n_trans, dt, param_priors):
    (
        mu_omega, sigma_omega,
        alpha_D, beta_D,
        alpha_R, beta_R,
        mu_A, sigma_A,
        sigma_q,
    ) = param_priors

    M = n_trans

    if np.isinf(sigma_omega):
        omega_new = ES1 / (M * dt)
    else:
        a2 = M * dt**2
        a1 = -2.0 * ES1 * dt
        a0 = ES0 + 4.0 * beta_D * dt

        Aprof = 0.5 * M + alpha_D

        # Stationary equation:
        # (omega - mu_omega) C(omega) + Aprof sigma^2 C'(omega) = 0
        # where C = a2 omega^2 + a1 omega + a0.
        coeffs = np.array([
            a2,
            a1 - mu_omega * a2,
            a0 - mu_omega * a1 + Aprof * sigma_omega**2 * 2.0 * a2,
            -mu_omega * a0 + Aprof * sigma_omega**2 * a1,
        ])

        roots = np.roots(coeffs)
        candidates = [mu_omega, ES1 / (M * dt)]

        for r in roots:
            if abs(r.imag) < 1e-10:
                candidates.append(float(r.real))

        def profile_neg_logpost(omega):
            C = a2 * omega**2 + a1 * omega + a0
            if C <= 0 or not np.isfinite(C):
                return np.inf
            return Aprof * np.log(C) + 0.5 * ((omega - mu_omega) / sigma_omega) ** 2

        omega_new = min(candidates, key=profile_neg_logpost)

    B_process = ES0 - 2.0 * ES1 * omega_new * dt + M * (omega_new * dt) ** 2

    Dtheta_new = (
        B_process + 4.0 * beta_D * dt
    ) / (
        2.0 * (M + 2.0 * alpha_D) * dt
    )

    return float(omega_new), float(Dtheta_new), float(B_process)


def global_phase_em_update(
    old_param,
    data,
    priors,
    *,
    n_theta=256,
    update_waveform=True,
    validate=False,
    return_aux=False,
):
    """
    One EM update using the global phase filter/smoother.

    Always uses log parameterization:
        old_param = (omega, lnDtheta, lnR, A, B)

    Here R is always inferred/updated. If regularizing noise is desired,
    add it to the data once beforehand, e.g. with add_regularizing_noise(...).
    """
    if isinstance(data, dict):
        tk = np.asarray(data["tk"], dtype=float)
        yk = np.asarray(data["yk"], dtype=float)
    else:
        tk, yk = data
        tk = np.asarray(tk, dtype=float)
        yk = np.asarray(yk, dtype=float)

    yk = np.squeeze(yk)

    (
        mu_theta, sigma_theta,
        mu_omega, sigma_omega,
        alpha_D, beta_D,
        alpha_R, beta_R,
        mu_A,
        sigma_A,
        sigma_q,
    ) = priors

    state_priors = (mu_theta, sigma_theta)
    param_priors = (
        mu_omega,
        sigma_omega,
        alpha_D,
        beta_D,
        alpha_R,
        beta_R,
        mu_A,
        sigma_A,
        sigma_q,
    )

    omega, lnDtheta, lnR, A, B = old_param
    Dtheta = float(np.exp(lnDtheta))
    R = float(np.exp(lnR))

    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)

    N = len(yk)
    dt = float(tk[1] - tk[0])
    n_trans = N - 1
    M = len(A) - 1

    if validate:
        if N < 2:
            raise ValueError("Need N >= 2.")
        if Dtheta <= 0:
            raise ValueError("Dtheta must be > 0.")
        if R <= 0:
            raise ValueError("R must be > 0.")
        if sigma_theta <= 0:
            raise ValueError("sigma_theta must be > 0.")
        if sigma_omega <= 0:
            raise ValueError("sigma_omega must be > 0.")
        if alpha_D <= 0 or beta_D <= 0:
            raise ValueError("alpha_D and beta_D must be > 0.")
        if alpha_R <= 0 or beta_R <= 0:
            raise ValueError("alpha_R and beta_R must be > 0.")
        if sigma_A <= 0:
            raise ValueError("sigma_A must be > 0.")
        if sigma_q <= 0:
            raise ValueError("sigma_q must be > 0.")
        if A.shape != B.shape:
            raise ValueError("A and B must have same shape.")
        if not np.isclose(B[0], 0.0):
            raise ValueError("B[0] must be fixed to 0.")
        if not np.isclose(B[1], 0.0):
            raise ValueError("B[1] must be fixed to 0.")

    old_param_for_filter = (omega, lnDtheta, lnR, A, B)

    filt = global_phase_filter(
        old_param_for_filter,
        data,
        state_priors,
        n_theta=n_theta,
        validate=validate,
    )

    smooth = global_phase_smoother(
        filt,
        validate=validate,
    )

    inc_stats = global_phase_increment_statistics(filt, smooth)
    ES0 = float(inc_stats["ES0"])
    ES1 = float(inc_stats["ES1"])

    omega_new, Dtheta_new, B_process = _phase_dynamics_map_log(
        ES0, ES1, n_trans, dt, param_priors
    )

    if update_waveform:
        A_new, B_new, R_new, ESy, coeff_prior = _waveform_R_map_global(
            yk, smooth, M, R, param_priors
        )
    else:
        A_new = A.copy()
        B_new = B.copy()
        ESy = expected_observation_error_global(yk, smooth, A_new, B_new)
        coeff_prior = np.nan

        # MAP update for R in log-R coordinates.
        R_new = (ESy + 2.0 * beta_R) / (N + 2.0 * alpha_R)

    B_new[0] = 0.0
    B_new[1] = 0.0

    new_param = (
        omega_new,
        np.log(Dtheta_new),
        np.log(R_new),
        A_new,
        B_new,
    )

    if not return_aux:
        return new_param

    aux = {
        "filter": filt,
        "smoother": smooth,
        "increment_stats": inc_stats,
        "expected_stats": {
            "S0": ES0,
            "S1": ES1,
            "Sy": ESy,
            "B_process": B_process,
            "coeff_prior": coeff_prior,
        },
        "old_param_log": old_param_for_filter,
        "new_param": new_param,
    }

    return new_param, aux


def global_phase_em_map_parameters(
    initial_params,
    data,
    priors,
    *,
    max_iter=500,
    tol=1e-6,
    n_theta=256,
    update_waveform=True,
    verbose=True,
    print_every=10,
    validate=False,
):
    """
    MAP estimation using exact EM with a global phase filter/smoother.

    Always uses log parameterization:
        params = (omega, lnDtheta, lnR, A, B)

    Here R is always inferred/updated. If regularizing noise is desired,
    add it to the data once beforehand, e.g. with add_regularizing_noise(...).
    """
    lambda_iter = [initial_params]

    if verbose:
        print("iter\tomega\t\tlnDtheta\tlnR\t\t|dparam|")

    converged = False
    rel_change = np.nan

    for i in range(1, max_iter + 1):
        new_param = global_phase_em_update(
            lambda_iter[-1],
            data,
            priors,
            n_theta=n_theta,
            update_waveform=update_waveform,
            validate=validate,
            return_aux=False,
        )

        old_vec = pack_param(*lambda_iter[-1])
        new_vec = pack_param(*new_param)

        denom = np.maximum(np.abs(old_vec), 1e-12)
        rel_change = np.max(np.abs(new_vec - old_vec) / denom)

        lambda_iter.append(new_param)

        if verbose and (i == 1 or i % print_every == 0):
            print(
                "%d\t%.5e\t%.5e\t%.5e\t%.3e"
                % (
                    i,
                    new_param[0],
                    new_param[1],
                    new_param[2],
                    rel_change,
                )
            )

        if rel_change < tol:
            converged = True
            if verbose:
                print(
                    f"\nConverged at iteration {i} "
                    f"(max relative change = {rel_change:.2e})"
                )
            break

    if not converged:
        warnings.warn(
            f"Global phase EM did not converge within {max_iter} iterations "
            f"(last relative change = {rel_change:.2e}).",
            RuntimeWarning,
        )

    return lambda_iter
    
    
def global_phase_increment_statistics(filter_out, smoother_out):
    """
    Compute wrapped phase-increment statistics from smoothed pairwise distributions.

    Returns
    -------
    stats : dict
        E_delta : shape (N,)
            E[wrap(theta_k - theta_{k-1}) | y], with E_delta[0] unused.

        E_delta2 : shape (N,)
            E[wrap(theta_k - theta_{k-1})^2 | y], with E_delta2[0] unused.

        ES1 : float
            Sum of expected wrapped increments.

        ES0 : float
            Sum of expected squared wrapped increments.
    """
    theta = filter_out["theta_grid"]
    alpha = filter_out["alpha_filt"]
    likelihood = filter_out["likelihood"]
    beta = smoother_out["beta"]
    kernel = filter_out["kernel"]

    N, L = alpha.shape

    idx = np.arange(L)
    T = kernel[(idx[None, :] - idx[:, None]) % L]

    delta = wrap_to_pi(theta[None, :] - theta[:, None])

    E_delta = np.zeros(N, dtype=float)
    E_delta2 = np.zeros(N, dtype=float)

    for k in range(N - 1):
        xi = (
            alpha[k, :, None]
            * T
            * likelihood[k + 1, None, :]
            * beta[k + 1, None, :]
        )

        s = np.sum(xi)
        if s <= 0 or not np.isfinite(s):
            raise RuntimeError(f"Invalid pairwise normalization at k={k}")

        xi /= s

        E_delta[k + 1] = np.sum(xi * delta)
        E_delta2[k + 1] = np.sum(xi * delta**2)

    return {
        "E_delta": E_delta,
        "E_delta2": E_delta2,
        "ES1": float(np.sum(E_delta[1:])),
        "ES0": float(np.sum(E_delta2[1:])),
    }

def global_phase_fourier_statistics(smoother_out, M):
    """
    Compute Fourier moments under smoothed marginal phase distributions.

    Returns
    -------
    stats : dict
        Ecos[n] : E[cos(n theta_k)] over time, shape (N,)
        Esin[n] : E[sin(n theta_k)] over time, shape (N,)
    """
    theta = smoother_out["theta_grid"]
    gamma = smoother_out["gamma"]

    Ecos = {}
    Esin = {}

    for n in range(0, 2 * M + 1):
        Ecos[n] = gamma @ np.cos(n * theta)
        Esin[n] = gamma @ np.sin(n * theta)

    return {
        "Ecos": Ecos,
        "Esin": Esin,
    }


def expected_observation_error_global(y, smoother_out, A, B):
    """
    Compute E[sum_k (y_k - q(theta_k))^2 | y] using smoothed marginals.
    """
    y = np.asarray(y, dtype=float).ravel()
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)

    M = len(A) - 1

    labels = [("A", m) for m in range(0, M + 1)]
    labels += [("B", m) for m in range(2, M + 1)]

    coeff = np.array(
        [A[m] if kind == "A" else B[m] for kind, m in labels],
        dtype=float,
    )

    fourier_stats = global_phase_fourier_statistics(smoother_out, M)
    Ecos = fourier_stats["Ecos"]
    Esin = fourier_stats["Esin"]

    def E_feature(kind, m):
        return Ecos[m] if kind == "A" else Esin[m]

    def E_product(label1, label2):
        kind1, m = label1
        kind2, n = label2

        if kind1 == "A" and kind2 == "A":
            return 0.5 * (Ecos[abs(m - n)] + Ecos[m + n])

        if kind1 == "B" and kind2 == "B":
            return 0.5 * (Ecos[abs(m - n)] - Ecos[m + n])

        if kind1 == "A" and kind2 == "B":
            if n >= m:
                diff = Esin[n - m]
            else:
                diff = -Esin[m - n]
            return 0.5 * (Esin[n + m] + diff)

        if kind1 == "B" and kind2 == "A":
            return E_product(label2, label1)

        raise RuntimeError("Unrecognized label.")

    N = len(y)

    q_mean = np.zeros(N, dtype=float)
    q2_mean = np.zeros(N, dtype=float)

    for c, lab in zip(coeff, labels):
        q_mean += c * E_feature(*lab)

    for i, lab_i in enumerate(labels):
        for j, lab_j in enumerate(labels):
            q2_mean += coeff[i] * coeff[j] * E_product(lab_i, lab_j)

    ESy = float(np.sum(y**2 - 2.0 * y * q_mean + q2_mean))

    return ESy
