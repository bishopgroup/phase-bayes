import numpy as np
import warnings
from scipy.optimize import minimize
from scipy.optimize import minimize_scalar
from scipy.linalg import solve_banded
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline
from typing import Literal, Tuple


def sample_laplace_parameters(x_map, cov, n_samples, rng, jitter=1e-10):
    """Sample parameter vectors from the Laplace approximation p(lambda | y)."""
    x_map = np.asarray(x_map, float)
    cov = np.asarray(cov, float)
    cov = 0.5 * (cov + cov.T)

    # small numerical guard
    cov = cov + jitter * np.eye(cov.shape[0])

    return rng.multivariate_normal(x_map, cov, size=n_samples)


def simulate_relative_phase(Dtheta, N, dt, rng):
    """Simulate drift-subtracted phase phi(t)=theta(t)-omega t."""
    dphi = np.sqrt(2.0 * Dtheta * dt) * rng.standard_normal(N - 1)
    phi = np.empty(N)
    phi[0] = 0.0
    phi[1:] = np.cumsum(dphi)
    return phi
    

def sample_phase_populations(
    x_samples,
    fit_data,
    priors,
    M,
    rng,
    *,
    n_phase_per_param=2,
    n_theta=128,
):
    """
    For each parameter sample, generate:
      inferred phase from p(theta | y, lambda_s)
      simulated phase from p(theta | lambda_s)
    Returns relative phases phi = theta - omega t.
    """
    tk = np.asarray(fit_data["tk"], float)
    dt = float(tk[1] - tk[0])
    N = len(tk)

    n_param = x_samples.shape[0]

    mu_theta, sigma_theta = priors[:2]
    
    state_priors = (
        mu_theta,
        sigma_theta,
    )

    theta_inf = np.empty((N, n_param, n_phase_per_param))
    phi_inf   = np.empty((N, n_param, n_phase_per_param))
    phi_sim   = np.empty((N, n_param, n_phase_per_param))
    D_samples = np.empty(n_param)
    
    for s in range(n_param):
        x_s = x_samples[s]
        omega_s = x_s[0]
        Dtheta_s = np.exp(x_s[1])
        D_samples[s] = Dtheta_s
    
        filt = global_phase_filter(
            unpack_param(x_s, M),
            fit_data,
            state_priors,
            n_theta=n_theta,
            transition="spectral",
            validate=False,
        )
    
        theta_s, _ = sample_phase_trajectories_global(
            filt,
            n_samples=n_phase_per_param,
            rng=rng,
        )
        # theta_s shape: (n_phase_per_param, N)
    
        theta_s = theta_s.T  # now (N, n_phase_per_param)
    
        theta_inf[:, s, :] = theta_s
    
        theta_unwrapped = np.unwrap(theta_s, axis=0)
        phi_inf[:, s, :] = theta_unwrapped - omega_s * tk[:, None]
    
        for p in range(n_phase_per_param):
            phi_sim[:, s, p] = simulate_relative_phase(
                Dtheta_s,
                N,
                dt,
                rng,
            )

    return {
        "theta_inf": theta_inf,
        "phi_inf": phi_inf,
        "phi_sim": phi_sim,
        "D_samples": D_samples,
        "tk": tk,
        "dt": dt,
    }


def nongaussian_interval_ppc(intervals, *, alpha=0.05):
    """
    PPC based on the non-Gaussian parameter

        alpha2 = <r^4> / (3 <r^2>^2) - 1
    """

    r_inf = intervals["r_inf"]
    r_sim = intervals["r_sim"]

    # shape (n_param, n_phase)
    T_inf = (
        np.mean(r_inf**4, axis=0)
        / (3 * np.mean(r_inf**2, axis=0)**2)
        - 1
    ).ravel()

    T_sim = (
        np.mean(r_sim**4, axis=0)
        / (3 * np.mean(r_sim**2, axis=0)**2)
        - 1
    ).ravel()

    delta = T_inf - T_sim

    ppc = interval_ppc_pvalue(
        delta,
        alpha=alpha,
        alternative="two-sided",
    )

    return {
        **{f"ng_{k}": v for k, v in ppc.items()},
        "ng_T_inf": T_inf,
        "ng_T_sim": T_sim,
        "ng_delta": delta,
    }

    
def lower_tail_ppc_pvalue(delta, *, alpha=0.05, finite_sample_correction=True):
    """
    One-sided PPC p-value for excess negative tail.

    delta = T^-_inf - T^-_sim.
    Small p means inferred trajectories have systematically larger lower tails.
    """
    delta = np.asarray(delta, float)
    B = delta.size

    if finite_sample_correction:
        p = (1.0 + np.sum(delta <= 0.0)) / (B + 1.0)
    else:
        p = np.mean(delta <= 0.0)

    return {
        "p_lower_tail": float(p),
        "passes_lower_tail": bool(p >= alpha),
        "alpha": float(alpha),
        "delta_mean": float(np.mean(delta)),
        "delta_median": float(np.median(delta)),
        "delta_q25": float(np.percentile(delta, 25)),
        "delta_q75": float(np.percentile(delta, 75)),
    }


def lower_tail_interval_ppc(intervals, *, z_tail=1.5, alpha=0.05):
    T_inf = np.mean(intervals["r_inf"] < -z_tail, axis=0).ravel()
    T_sim = np.mean(intervals["r_sim"] < -z_tail, axis=0).ravel()

    delta = T_inf - T_sim

    ppc = interval_ppc_pvalue(
        delta,
        alpha=alpha,
        alternative="less",
    )

    return {
        **{f"tail_{k}": v for k, v in ppc.items()},
        "tail_z": float(z_tail),
        "tail_T_inf": T_inf,
        "tail_T_sim": T_sim,
        "tail_delta": delta,
    }


def interval_ppc_pvalue(delta, *, alpha=0.05, alternative="less"):
    """
    Monte Carlo PPC p-value with Beta(1,1) finite-sample correction.

    alternative:
        "less"      p = P(delta <= 0)
        "greater"   p = P(delta >= 0)
        "two-sided" p = 2 min[P(delta <= 0), P(delta >= 0)]
    """
    delta = np.asarray(delta, float)
    B = delta.size

    if alternative == "less":
        k = np.sum(delta <= 0.0)
        p = (1.0 + k) / (B + 2.0)

    elif alternative == "greater":
        k = np.sum(delta >= 0.0)
        p = (1.0 + k) / (B + 2.0)

    elif alternative == "two-sided":
        k_less = np.sum(delta <= 0.0)
        k_greater = np.sum(delta >= 0.0)

        p_less = (1.0 + k_less) / (B + 2.0)
        p_greater = (1.0 + k_greater) / (B + 2.0)

        p = min(1.0, 2.0 * min(p_less, p_greater))

    else:
        raise ValueError("alternative must be 'less', 'greater', or 'two-sided'.")

    return {
        "p_value": float(p),
        "passes": bool(p >= alpha),
        "alpha": float(alpha),
        "alternative": alternative,
        "delta_mean": float(np.mean(delta)),
        "delta_median": float(np.median(delta)),
        "delta_q25": float(np.percentile(delta, 25)),
        "delta_q75": float(np.percentile(delta, 75)),
    }


def phase_interval_statistics(phases, *, D_ref):
    phi_inf = phases["phi_inf"]  # shape (N, n_param, n_phase)
    phi_sim = phases["phi_sim"]
    D_samples = phases["D_samples"]
    dt = phases["dt"]

    N, n_param, n_phase = phi_inf.shape

    lag = int(np.round((1.0 / D_ref) / dt))
    lag = int(np.clip(lag, 1, N - 1))
    tau = lag * dt

    dphi_inf = phi_inf[lag:, :, :] - phi_inf[:-lag, :, :]
    dphi_sim = phi_sim[lag:, :, :] - phi_sim[:-lag, :, :]

    scale = np.sqrt(2.0 * D_samples[None, :, None] * tau)

    r_inf = dphi_inf / scale
    r_sim = dphi_sim / scale

    return {
        "lag": lag,
        "tau": tau,
        "dphi_inf": dphi_inf,
        "dphi_sim": dphi_sim,
        "r_inf": r_inf,
        "r_sim": r_sim,
    }


def msd_interval_ppc(intervals, *, alpha=0.05):
    T_inf = np.mean(intervals["dphi_inf"]**2, axis=0).ravel()
    T_sim = np.mean(intervals["dphi_sim"]**2, axis=0).ravel()

    delta = T_inf - T_sim

    ppc = interval_ppc_pvalue(
        delta,
        alpha=alpha,
        alternative="two-sided",
    )

    return {
        **{f"msd_{k}": v for k, v in ppc.items()},
        "msd_T_inf": T_inf,
        "msd_T_sim": T_sim,
        "msd_delta": delta,
    }

    
def run_interval_ppcs(
    fit_out,
    M=None,
    rng=None,
    *,
    n_param_samples=50,
    n_phase_per_param=2,
    n_theta=128,
    z_tail=1.5,
    alpha=0.05,
):
    if rng is None:
        rng = np.random.default_rng()

    if fit_out["cov"] is None:
        raise ValueError(
            "PPC requires full Laplace covariance; set compute_uncertainty=True."
        )

    if M is None:
        M = int(fit_out["row"]["M"])

    x_map = fit_out["fit"]["x"]
    cov = fit_out["cov"]["cov"]

    x_samples = sample_laplace_parameters(
        x_map,
        cov,
        n_samples=n_param_samples,
        rng=rng,
    )

    phases = sample_phase_populations(
        x_samples,
        fit_out["fit_data"],
        fit_out["priors"],
        M=M,
        rng=rng,
        n_phase_per_param=n_phase_per_param,
        n_theta=n_theta,
    )

    intervals = phase_interval_statistics(
        phases,
        D_ref=np.exp(x_map[1]),
    )

    msd_ppc = msd_interval_ppc(
        intervals,
        alpha=alpha,
    )

    tail_ppc = lower_tail_interval_ppc(
        intervals,
        z_tail=z_tail,
        alpha=alpha,
    )

    ng_ppc = nongaussian_interval_ppc(
        intervals,
        alpha=alpha,
    )

    return {
        "lag": intervals["lag"],
        "tau": intervals["tau"],
        "x_samples": x_samples,
        "phases": phases,
        "intervals": intervals,
        **msd_ppc,
        **tail_ppc,
        **ng_ppc,
    }
    
def lower_tail_statistics(
    phi_inf,
    phi_sim,
    D_samples,
    dt,
    *,
    D_ref,
    z_tail=1.5,
):
    N, n_param, n_phase = phi_inf.shape

    lag = int(np.round((1.0 / D_ref) / dt))
    lag = int(np.clip(lag, 1, N - 1))
    tau = lag * dt

    T_inf = np.empty((n_param, n_phase))
    T_sim = np.empty((n_param, n_phase))

    for s in range(n_param):
        scale = np.sqrt(2.0 * D_samples[s] * tau)

        for p in range(n_phase):
            r_inf = (phi_inf[lag:, s, p] - phi_inf[:-lag, s, p]) / scale
            r_sim = (phi_sim[lag:, s, p] - phi_sim[:-lag, s, p]) / scale

            T_inf[s, p] = np.mean(r_inf < -z_tail)
            T_sim[s, p] = np.mean(r_sim < -z_tail)

    delta = (T_inf - T_sim).ravel()

    return {
        "T_inf": T_inf.ravel(),
        "T_sim": T_sim.ravel(),
        "delta": delta,
        "lag": lag,
        "tau": tau,
        "z_tail": z_tail,
    }

    
def spectral_backward_apply(K_hat, g, *, real_output=True):
    """
    Backward transition application for the spectral heat kernel.

    Forward:
        p_next = ifft(K_hat * fft(p))

    Backward/adjoint:
        h_prev = ifft(conj(K_hat) * fft(h_next))

    This corresponds to:
        h_k(theta_k) = E[h_{k+1}(theta_{k+1}) | theta_k]
    """
    g = np.asarray(g, dtype=float)

    out = np.fft.ifft(np.conj(K_hat) * np.fft.fft(g))

    if real_output:
        out = out.real

    return out
    

def fft_integer_modes(n_theta):
    """
    Integer Fourier modes compatible with np.fft.fft on a 2π-periodic grid.

    For theta_j = 2π j / n_theta, this returns
        m = 0, 1, ..., n/2, ..., -2, -1
    in FFT ordering.
    """
    return np.fft.fftfreq(n_theta, d=1.0 / n_theta)


def spectral_heat_kernel_hat(n_theta, mean_step, Q):
    """
    Fourier multiplier for wrapped drift-diffusion on the circle.

    Dynamics:
        theta_k = theta_{k-1} + mean_step + noise
        noise variance = Q = 2 Dtheta dt

    With the np.fft convention and the existing circular_convolve convention,

        K_hat[m] = exp(-i m mean_step) exp(-0.5 m^2 Q)

    equivalently,

        K_hat[m] = exp(-i m omega dt) exp(-m^2 Dtheta dt)
    """
    m = fft_integer_modes(n_theta)

    return np.exp(
        -1j * m * mean_step
        -0.5 * (m ** 2) * Q
    )


def spectral_predict(p, K_hat, *, real_output=True):
    """
    Predict a circular density using the spectral heat kernel.

        p_pred = ifft(K_hat * fft(p))

    This replaces circular_convolve(kernel, p) for the transition step.
    """
    p = np.asarray(p, dtype=float)

    p_pred = np.fft.ifft(K_hat * np.fft.fft(p))

    if real_output:
        p_pred = p_pred.real

    return p_pred


def normalize_probability(p, *, floor=0.0):
    """
    Normalize a nonnegative probability vector.

    floor=0 preserves current behavior.
    A tiny positive floor can be used for diagnostics or robustness.
    """
    p = np.asarray(p, dtype=float)

    if floor > 0:
        p = p + floor

    s = np.sum(p)

    if s <= 0 or not np.isfinite(s):
        raise RuntimeError("Probability normalization failed.")

    return p / s


def spectral_prediction_diagnostics(p_pred):
    """
    Diagnostics for spectral grid prediction.
    """
    p_pred = np.asarray(p_pred)

    return {
        "min_alpha_pred": float(np.min(p_pred.real)),
        "mass_pred": float(np.sum(p_pred.real)),
        "imag_norm": float(np.linalg.norm(np.imag(p_pred))),
        "real_norm": float(np.linalg.norm(np.real(p_pred))),
    }

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


def transition_matrix_from_spectral_kernel(K_hat):
    """
    Construct transition matrix T[i,j] = p(theta_j | theta_i)
    from spectral multiplier K_hat.
    """
    K_hat = np.asarray(K_hat)

    if K_hat.ndim != 1:
        raise ValueError("K_hat must be one-dimensional.")

    L = len(K_hat)
    T = np.empty((L, L), dtype=float)

    for i in range(L):
        e_i = np.zeros(L)
        e_i[i] = 1.0

        row = spectral_predict(e_i, K_hat)
        row = np.maximum(row, 0.0)
        row = row / np.sum(row)

        T[i] = row

    return T

    
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
    transition="spectral",
    probability_floor=0.0,
    return_transition_diagnostics=False,
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

    if transition == "sampled":
        kernel = wrapped_gaussian_kernel(theta_grid, mean_step, Q)
        K_hat = None
    elif transition == "spectral":
        kernel = None
        K_hat = spectral_heat_kernel_hat(n_theta, mean_step, Q)
    else:
        raise ValueError("transition must be 'sampled' or 'spectral'.")
    
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
        if transition == "sampled":
            alpha_pred[k] = circular_convolve(kernel, alpha_filt[k - 1])
            alpha_pred[k] = np.maximum(alpha_pred[k], 0.0)
        
        elif transition == "spectral":
            alpha_pred[k] = spectral_predict(alpha_filt[k - 1], K_hat)
        
            # For NumPy testing, keep this conservative.
            # Later in JAX, we can revisit whether this guard is needed.
            alpha_pred[k] = np.maximum(alpha_pred[k], 0.0)

        alpha_pred[k] = normalize_probability(
            alpha_pred[k],
            floor=probability_floor,
        )

        unnorm = likelihood[k] * alpha_pred[k]
        scales[k] = np.sum(unnorm)

        if scales[k] <= 0 or not np.isfinite(scales[k]):
            raise RuntimeError(f"Filtering failed at k={k}: nonpositive normalization.")

        alpha_filt[k] = unnorm / scales[k]

    loglik = float(np.sum(np.log(scales) + loglik_offsets))

    out = {
        "theta_grid": theta_grid,
        "alpha_pred": alpha_pred,
        "alpha_filt": alpha_filt,
        "likelihood": likelihood,
        "loglik_offsets": loglik_offsets,
        "kernel": kernel,
        "K_hat": K_hat,
        "transition": transition,
        "q_grid": q_grid,
        "loglik": loglik,
        "scales": scales,
        "param": param,
    }

    return out

    
def circular_mean_std_from_grid(gamma, theta_grid):
    """
    Compute circular mean and circular standard deviation from grid marginals.
    """
    z = gamma @ np.exp(1j * theta_grid)

    theta_mean_wrapped = np.angle(z)
    theta_mean = np.unwrap(theta_mean_wrapped)

    Rbar = np.abs(z)
    theta_std = np.sqrt(-2.0 * np.log(np.clip(Rbar, 1e-15, 1.0)))

    return {
        "z": z,
        "theta_mean_wrapped": theta_mean_wrapped,
        "theta_mean": theta_mean,
        "Rbar": Rbar,
        "theta_std": theta_std,
    }


def compare_sampled_vs_spectral_filter(
    param,
    data,
    state_priors,
    *,
    n_theta=256,
    validate=False,
):
    """
    Run sampled-kernel and spectral-transition filters side by side.
    """
    filt_sampled = global_phase_filter(
        param,
        data,
        state_priors,
        n_theta=n_theta,
        transition="sampled",
        validate=validate,
    )

    filt_spectral = global_phase_filter(
        param,
        data,
        state_priors,
        n_theta=n_theta,
        transition="spectral",
        validate=validate,
    )

    dloglik = filt_spectral["loglik"] - filt_sampled["loglik"]

    alpha_diff = filt_spectral["alpha_filt"] - filt_sampled["alpha_filt"]

    return {
        "sampled": filt_sampled,
        "spectral": filt_spectral,
        "dloglik": float(dloglik),
        "max_abs_alpha_diff": float(np.max(np.abs(alpha_diff))),
        "rms_alpha_diff": float(np.sqrt(np.mean(alpha_diff ** 2))),
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
    transition = filter_out.get("transition", "sampled")
    kernel = filter_out.get("kernel", None)
    K_hat = filter_out.get("K_hat", None)
    
    if kernel is not None:
        kernel = np.asarray(kernel, dtype=float)
    
    if K_hat is not None:
        K_hat = np.asarray(K_hat)

    N, n_theta = alpha_filt.shape

    if validate:
        if likelihood.shape != (N, n_theta):
            raise ValueError("likelihood must have same shape as alpha_filt.")
        if transition == "sampled":
            if kernel is None or kernel.shape != (n_theta,):
                raise ValueError("kernel must have shape (n_theta,) for sampled transition.")
        
        elif transition == "spectral":
            if K_hat is None or K_hat.shape != (n_theta,):
                raise ValueError("K_hat must have shape (n_theta,) for spectral transition.")
        
        else:
            raise ValueError("transition must be 'sampled' or 'spectral'.")
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

        if transition == "sampled":
            beta[k] = circular_backward_apply(kernel, g_next)
        
        elif transition == "spectral":
            beta[k] = spectral_backward_apply(K_hat, g_next)
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
        "K_hat": K_hat,
        "transition": transition,
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

    transition = filter_out.get("transition", "sampled")
    kernel = filter_out.get("kernel", None)
    K_hat = filter_out.get("K_hat", None)

    N, L = alpha.shape

    if transition == "sampled":
        T = transition_matrix_from_kernel(kernel)

    elif transition == "spectral":
        T = transition_matrix_from_spectral_kernel(K_hat)

    else:
        raise ValueError("transition must be 'sampled' or 'spectral'.")

    index_samples = np.empty((n_samples, N), dtype=int)

    for s in range(n_samples):
        index_samples[s, -1] = rng.choice(L, p=alpha[-1])

        for k in range(N - 2, -1, -1):
            j = index_samples[s, k + 1]

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
    # nlp += 0.5 * np.sum((A[1:] / sigma_q) ** 2)

    # if len(B) > 2:
    #     nlp += 0.5 * np.sum((B[2:] / sigma_q) ** 2)
    p_wave = 1.0
    
    for m in range(1, len(A)):
        sigma_m = sigma_q / (m ** p_wave)
        nlp += 0.5 * (A[m] / sigma_m) ** 2
    
    for m in range(2, len(B)):
        sigma_m = sigma_q / (m ** p_wave)
        nlp += 0.5 * (B[m] / sigma_m) ** 2
        
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

def get_observation_key(data, preferred=("yk_reg", "yk")):
    for key in preferred:
        if isinstance(data, dict) and key in data:
            return key
    raise KeyError(f"None of {preferred} found in data.")


def as_inference_data(data, *, y_key=None):
    if isinstance(data, dict):
        if y_key is None:
            y_key = get_observation_key(data)
        return np.asarray(data["tk"], float), np.asarray(data[y_key], float)
    tk, yk = data
    return np.asarray(tk, float), np.asarray(yk, float)


def initialize_map_parameters(
    data,
    M,
    *,
    R0=None,
    Dtheta0=None,
    omega0=None,
    window_size=None,
    window_cycles=3.0,
    y_key=None,
    validate=False,
):
    """
    Build an initial MAP parameter tuple

        (omega, lnDtheta, lnR, A, B)

    from FFT frequency estimation and short-window Fourier regression.
    """
    tk, yk = as_inference_data(data, y_key=y_key)
    dt = float(tk[1] - tk[0])
    N = len(tk)

    if omega0 is None:
        omega0, _, _ = estimate_frequency_fft((tk, yk))

    if window_size is None:
        period_samples = int(np.round((2.0 * np.pi / omega0) / dt))
        window_size = int(max(2 * M + 3, window_cycles * period_samples))
        window_size = min(window_size, N)

    if R0 is None:
        R0 = 0.1 * np.var(yk)

    if Dtheta0 is None:
        Dtheta0 = 1e-3 * omega0

    init = estimate_initial_phase_regression(
        (tk, yk),
        omega0,
        window_size,
        R0,
        M=M,
        validate=validate,
    )

    A0 = init["A"]
    B0 = init["B"]

    param0 = (
        float(omega0),
        float(np.log(Dtheta0)),
        float(np.log(R0)),
        A0,
        B0,
    )

    return {
        "param0": param0,
        "omega0": float(omega0),
        "Dtheta0": float(Dtheta0),
        "R0": float(R0),
        "A0": A0,
        "B0": B0,
        "window_size": int(window_size),
        "regression": init,
    }


def make_default_priors(
    data,
    param0,
    *,
    y_key=None,
    mu_theta=None,
    sigma_theta=np.pi,
    sigma_omega=None,
    alpha_D=1.0,
    Dtheta_mode=None,
    alpha_R=1.0,
    R_mode=None,
    sigma_A_factor=10.0,
    sigma_q_factor=10.0,
):
    """
    Construct priors for the MAP posterior.

    Important convention
    --------------------
    The inverse-gamma priors are used in log-parameter coordinates.

    In neg_log_prior, the log-coordinate posterior includes the Jacobian
    correction

        - lnDtheta - lnR.

    Therefore, if we want the prior mode in log coordinates to occur at

        Dtheta = Dtheta_mode
        R      = R_mode,

    then we set

        beta_D = alpha_D * Dtheta_mode
        beta_R = alpha_R * R_mode

    not beta = (alpha + 1) * mode.

    Parameters
    ----------
    data : dict or tuple
        Used only to estimate waveform scale for weak Gaussian priors.

    param0 : tuple
        Initial parameter tuple (omega, lnDtheta, lnR, A, B).

    mu_theta, sigma_theta : float
        Prior mean and width for the initial phase theta_1.
        In the hippocampal workflow these should usually come from the
        raw-data phase regression:
            mu_theta = phi_mean + omega_FFT * tk[0]
            sigma_theta = 2 * phi_std

    sigma_omega : float or None
        Absolute standard deviation of the frequency prior. If None, use
        np.inf, i.e. effectively no frequency prior.

    Dtheta_mode : float
        Desired prior mode for Dtheta in log coordinates.

    R_mode : float
        Desired prior mode for R_eff in log coordinates. In the added-noise
        sweep, this should usually be the current R_add.
    """
    tk, yk = as_inference_data(data, y_key=y_key)

    omega0, lnD0, lnR0, A0, B0 = param0
    D0 = float(np.exp(lnD0))
    R0 = float(np.exp(lnR0))

    y_scale = float(np.std(yk))
    if not np.isfinite(y_scale) or y_scale <= 0:
        y_scale = 1.0

    if mu_theta is None:
        mu_theta = 0.0

    if sigma_omega is None:
        sigma_omega = np.inf

    if Dtheta_mode is None:
        Dtheta_mode = D0

    if R_mode is None:
        R_mode = R0

    beta_D = float(alpha_D * Dtheta_mode)
    beta_R = float(alpha_R * R_mode)

    mu_A = float(A0[0])
    sigma_A = float(sigma_A_factor * y_scale)
    sigma_q = float(sigma_q_factor * y_scale)

    return (
        float(mu_theta),
        float(sigma_theta),
        float(omega0),
        float(sigma_omega),
        float(alpha_D),
        beta_D,
        float(alpha_R),
        beta_R,
        mu_A,
        sigma_A,
        sigma_q,
    )


def safe_neg_log_posterior_flat(
    x,
    M,
    data,
    priors,
    *,
    n_theta=256,
    bad_value=1e300,
    validate=False,
):
    """
    Flat-vector objective for scipy.optimize.minimize.
    """
    try:
        param = unpack_param(x, M)

        val = neg_log_posterior(
            param,
            data,
            priors,
            n_theta=n_theta,
            validate=validate,
        )

        if not np.isfinite(val):
            return bad_value

        return float(val)

    except Exception:
        return bad_value


def fit_map_parameters(
    data,
    priors,
    param0,
    M,
    *,
    n_theta=256,
    method="Powell",
    options=None,
    validate=False,
):
    """
    Direct gradient-free MAP optimization.
    """
    if options is None:
        options = {
            "maxiter": 1000,
            "xtol": 1e-4,
            "ftol": 1e-4,
            "disp": False,
        }

    x0 = pack_param(*param0)

    def objective(x):
        return safe_neg_log_posterior_flat(
            x,
            M,
            data,
            priors,
            n_theta=n_theta,
            validate=validate,
        )

    res = minimize(
        objective,
        x0,
        method=method,
        options=options,
    )

    x_map = np.asarray(res.x, dtype=float)
    param_map = unpack_param(x_map, M)

    return {
        "result": res,
        "param": param_map,
        "x": x_map,
        "fun": float(res.fun),
        "success": bool(res.success),
        "message": res.message,
        "nfev": getattr(res, "nfev", np.nan),
        "nit": getattr(res, "nit", np.nan),
    }


def summarize_map(param, *, prefix=""):
    """
    Convert structured parameters to a flat summary dictionary.
    """
    omega, lnDtheta, lnR, A, B = param

    out = {
        prefix + "omega": float(omega),
        prefix + "Dtheta": float(np.exp(lnDtheta)),
        prefix + "R_eff": float(np.exp(lnR)),
        prefix + "lnDtheta": float(lnDtheta),
        prefix + "lnR": float(lnR),
        prefix + "M": int(len(A) - 1),
    }

    for m, val in enumerate(A):
        out[prefix + f"A{m}"] = float(val)

    for m, val in enumerate(B):
        out[prefix + f"B{m}"] = float(val)

    return out


def run_added_noise_fit(
    data,
    R_add,
    M,
    rng,
    *,
    n_theta=256,
    param0=None,
    priors=None,
    initialize_kwargs=None,
    prior_kwargs=None,
    optimizer_kwargs=None,
    y_key="yk",
    regularized_key="yk_reg",
    validate=False,
):
    """
    One complete MAP fit at fixed added-noise variance.

    Steps:
        1. Add one fixed realization of regularizing noise.
        2. Initialize parameters if needed.
        3. Build priors if needed.
        4. Optimize the marginal posterior.
        5. Return summary and full objects.
    """
    if initialize_kwargs is None:
        initialize_kwargs = {}

    if prior_kwargs is None:
        prior_kwargs = {}

    if optimizer_kwargs is None:
        optimizer_kwargs = {}

    data_reg = add_regularizing_noise(
        data,
        R_add,
        rng,
        copy=True,
        key=regularized_key,
        validate=validate,
    )

    fit_data = {
        "tk": np.asarray(data_reg["tk"], float),
        "yk": np.asarray(data_reg[regularized_key], float),
    }

    if param0 is None:
        init = initialize_map_parameters(
            fit_data,
            M,
            validate=validate,
            **initialize_kwargs,
        )
        param0 = init["param0"]
    else:
        init = None

    if priors is None:
        priors = make_default_priors(
            fit_data,
            param0,
            **prior_kwargs,
        )

    fit = fit_map_parameters(
        fit_data,
        priors,
        param0,
        M,
        n_theta=n_theta,
        validate=validate,
        **optimizer_kwargs,
    )

    cov_dyn = estimate_dynamic_parameter_covariance(
        fit["param"],
        fit_data,
        priors,
        n_theta=n_theta,
        validate=validate,
    )
    
    row = summarize_map(fit["param"])
    row.update({
        "R_add": float(R_add),
        "success": fit["success"],
        "fun": fit["fun"],
        "nfev": fit["nfev"],
        "nit": fit["nit"],
    })
    row.update({
        "omega_se": cov_dyn["omega_se"],
        "lnDtheta_se": cov_dyn["lnDtheta_se"],
        "lnR_se": cov_dyn["lnR_se"],
        "Dtheta_mult_se": cov_dyn["Dtheta_mult_se"],
        "R_eff_mult_se": cov_dyn["R_eff_mult_se"],
    })

    return {
        "row": row,
        "fit": fit,
        "param": fit["param"],
        "param0": param0,
        "priors": priors,
        "init": init,
        "data_reg": data_reg,
        "fit_data": fit_data,
        "cov_dyn": cov_dyn,
    }


def sweep_added_noise(
    data,
    R_add_values,
    M,
    rng,
    *,
    n_reps=1,
    n_theta=256,
    warm_start=True,
    initialize_kwargs=None,
    prior_kwargs=None,
    optimizer_kwargs=None,
    validate=False,
):
    """
    Sweep over added-noise variances.

    Continuation is used only for the MAP initial guess.
    Priors:
        omega prior fixed across sweep
        Dtheta prior fixed across sweep
        R_eff prior mode set to current R_add
    """
    import pandas as pd

    if initialize_kwargs is None:
        initialize_kwargs = {}

    if prior_kwargs is None:
        prior_kwargs = {}

    if optimizer_kwargs is None:
        optimizer_kwargs = {}

    rows = []
    results = []

    R_add_values = list(R_add_values)

    # Estimate omega prior center once from the original data.
    tk0, yk0 = as_inference_data(data)
    omega_fft, _, _ = estimate_frequency_fft((tk0, yk0))

    for rep in range(n_reps):
        param0 = None

        for i, R_add in enumerate(R_add_values):

            init_kwargs_i = dict(initialize_kwargs)
            init_kwargs_i["omega0"] = omega_fft

            prior_kwargs_i = dict(prior_kwargs)
            prior_kwargs_i["R_mode"] = R_add

            out = run_added_noise_fit(
                data,
                R_add,
                M,
                rng,
                n_theta=n_theta,
                param0=param0 if warm_start else None,
                priors=None,
                initialize_kwargs=init_kwargs_i,
                prior_kwargs=prior_kwargs_i,
                optimizer_kwargs=optimizer_kwargs,
                validate=validate,
            )

            row = dict(out["row"])
            row["rep"] = rep
            row["sweep_index"] = i
            row["omega_prior_mode"] = omega_fft
            row["R_mode"] = R_add
            rows.append(row)
            results.append(out)

            if warm_start and out["fit"]["success"]:
                param0 = out["param"]

    return {
        "summary": pd.DataFrame(rows),
        "results": results,
    }


def finite_difference_hessian(f, x0, *, step=None):
    """
    Central finite-difference Hessian for a scalar function f(x).

    Parameters
    ----------
    f : callable
        Scalar objective.

    x0 : ndarray
        Point at which to evaluate Hessian.

    step : float or ndarray, optional
        Finite-difference step size.

    Returns
    -------
    H : ndarray
        Hessian matrix.
    """
    x0 = np.asarray(x0, dtype=float)
    n = len(x0)

    if step is None:
        step = 1e-4 * np.maximum(1.0, np.abs(x0))
    else:
        step = np.asarray(step, dtype=float)
        if step.ndim == 0:
            step = np.full(n, float(step))

    H = np.empty((n, n), dtype=float)
    f0 = float(f(x0))

    for i in range(n):
        ei = np.zeros(n)
        ei[i] = step[i]

        fp = float(f(x0 + ei))
        fm = float(f(x0 - ei))

        H[i, i] = (fp - 2.0 * f0 + fm) / step[i]**2

        for j in range(i + 1, n):
            ej = np.zeros(n)
            ej[j] = step[j]

            fpp = float(f(x0 + ei + ej))
            fpm = float(f(x0 + ei - ej))
            fmp = float(f(x0 - ei + ej))
            fmm = float(f(x0 - ei - ej))

            Hij = (fpp - fpm - fmp + fmm) / (4.0 * step[i] * step[j])

            H[i, j] = Hij
            H[j, i] = Hij

    return 0.5 * (H + H.T)


def estimate_dynamic_parameter_covariance(
    param_map,
    data,
    priors,
    *,
    n_theta=256,
    step=None,
    validate=False,
):
    """
    Estimate local covariance for [omega, lnDtheta, lnR] by finite-difference
    Hessian, holding waveform coefficients A, B fixed.

    Returns standard errors for omega, lnDtheta, lnR.
    """
    omega_map, lnDtheta_map, lnR_map, A_map, B_map = param_map

    x0 = np.array([omega_map, lnDtheta_map, lnR_map], dtype=float)

    def objective(x):
        omega, lnDtheta, lnR = x
        param = (omega, lnDtheta, lnR, A_map, B_map)

        return neg_log_posterior(
            param,
            data,
            priors,
            n_theta=n_theta,
            validate=validate,
        )

    H = finite_difference_hessian(objective, x0, step=step)

    cov = np.linalg.inv(H)
    se = np.sqrt(np.diag(cov))

    return {
        "x": x0,
        "H": H,
        "cov": cov,
        "omega_se": float(se[0]),
        "lnDtheta_se": float(se[1]),
        "lnR_se": float(se[2]),
        "Dtheta": float(np.exp(lnDtheta_map)),
        "Dtheta_mult_se": float(np.exp(se[1])),
        "R_eff": float(np.exp(lnR_map)),
        "R_eff_mult_se": float(np.exp(se[2])),
    }

def phase_msd_from_samples_diffusive(
    theta_samples,
    lags,
    *,
    omega,
    dt,
):
    scalar_input = np.isscalar(lags)
    lags = np.atleast_1d(lags).astype(int)

    theta_unwrapped = np.unwrap(theta_samples, axis=1)

    S, N = theta_unwrapped.shape
    msd = np.empty(len(lags))

    for a, lag in enumerate(lags):
        if lag < 1 or lag >= N:
            raise ValueError(f"Invalid lag {lag}; require 1 <= lag < N.")

        tau = lag * dt
        dtheta = theta_unwrapped[:, lag:] - theta_unwrapped[:, :-lag] - omega * tau
        msd[a] = np.mean(dtheta**2)

    return msd[0] if scalar_input else msd