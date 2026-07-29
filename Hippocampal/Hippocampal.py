import numpy as np
from scipy.optimize import minimize


def sample_laplace_parameters(x_map, cov, n_samples, rng, jitter=1e-10):
    """
    Sample parameter vectors from a Gaussian Laplace approximation.

    Parameters
    ----------
    x_map : array_like, shape (P,)
        MAP parameter vector.
    cov : array_like, shape (P, P)
        Laplace covariance matrix.
    n_samples : int
        Number of draws.
    rng : numpy.random.Generator
        Random number generator.
    jitter : float, optional
        Diagonal regularizer added before sampling.

    Returns
    -------
    samples : ndarray, shape (n_samples, P)
        Parameter draws from the regularized approximation.
    """
    x_map = np.asarray(x_map, float)
    cov = np.asarray(cov, float)
    cov = 0.5 * (cov + cov.T)

    # small numerical guard
    cov = cov + jitter * np.eye(cov.shape[0])

    return rng.multivariate_normal(x_map, cov, size=n_samples)


def simulate_relative_phase(Dtheta, N, dt, rng):
    """
    Simulate a drift-subtracted diffusive phase trajectory.

    Parameters
    ----------
    Dtheta : float
        Phase diffusivity.
    N : int
        Number of samples.
    dt : float
        Sampling interval.
    rng : numpy.random.Generator
        Random number generator.

    Returns
    -------
    phi : ndarray, shape (N,)
        Relative phase with ``phi[0] = 0``.
    """
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
    Sample inferred and prior-predictive relative phase populations.

    Parameters
    ----------
    x_samples : ndarray, shape (n_param, P)
        Flat parameter samples.
    fit_data : dict
        Fitting data containing ``"tk"`` and ``"yk"``.
    priors : tuple
        State and parameter prior hyperparameters.
    M : int
        Maximum Fourier mode.
    rng : numpy.random.Generator
        Random number generator.
    n_phase_per_param : int, optional
        Phase trajectories drawn per parameter sample.
    n_theta : int, optional
        Number of phase-grid points.

    Returns
    -------
    out : dict
        Inferred phases, inferred and simulated relative phases, sampled
        diffusivities, times, and sampling interval.
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

    Parameters
    ----------
    intervals : dict
        Normalized inferred and simulated phase increments.
    alpha : float, optional
        Posterior-predictive test threshold.

    Returns
    -------
    out : dict
        Non-Gaussian statistics, differences, p-value, and pass indicator.
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

    
def lower_tail_interval_ppc(intervals, *, z_tail=1.5, alpha=0.05):
    """
    Test excess negative tails in inferred phase increments.

    Parameters
    ----------
    intervals : dict
        Normalized inferred and simulated phase increments.
    z_tail : float, optional
        Negative-tail threshold in diffusion standard deviations.
    alpha : float, optional
        Posterior-predictive test threshold.

    Returns
    -------
    out : dict
        Tail frequencies, differences, p-value, and pass indicator.
    """
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

    Parameters
    ----------
    delta : array_like
        Replicate-level differences between inferred and simulated statistics.
    alpha : float, optional
        Test threshold.
    alternative : {"less", "greater", "two-sided"}, optional
        Tail convention for the Monte Carlo p-value.

    Returns
    -------
    out : dict
        P-value, pass indicator, test settings, and difference summaries.
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
    """
    Construct normalized phase increments for interval PPCs.

    Parameters
    ----------
    phases : dict
        Inferred and simulated relative phases, diffusivity samples, and
        sampling interval.
    D_ref : float
        Reference diffusivity used to select the comparison lag.

    Returns
    -------
    out : dict
        Lag, elapsed time, raw increments, and diffusion-normalized increments.
    """
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
    """
    Compare inferred and simulated interval mean-squared displacements.

    Parameters
    ----------
    intervals : dict
        Raw inferred and simulated phase increments.
    alpha : float, optional
        Posterior-predictive test threshold.

    Returns
    -------
    out : dict
        MSD statistics, differences, p-value, and pass indicator.
    """
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
    """
    Run MSD, lower-tail, and non-Gaussian interval PPCs.

    Parameters
    ----------
    fit_out : dict
        Successful JAX fit output containing full Laplace covariance.
    M : int, optional
        Maximum Fourier mode; inferred from ``fit_out`` when omitted.
    rng : numpy.random.Generator, optional
        Random number generator.
    n_param_samples : int, optional
        Laplace parameter draws.
    n_phase_per_param : int, optional
        Phase trajectories per parameter draw.
    n_theta : int, optional
        Number of phase-grid points.
    z_tail : float, optional
        Negative-tail threshold.
    alpha : float, optional
        Posterior-predictive test threshold.

    Returns
    -------
    out : dict
        Sampled populations, interval statistics, and all PPC summaries.
    """
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
    
def spectral_backward_apply(K_hat, g, *, real_output=True):
    """
    Backward transition application for the spectral heat kernel.

    Forward:
        p_next = ifft(K_hat * fft(p))

    Backward/adjoint:
        h_prev = ifft(conj(K_hat) * fft(h_next))

    This corresponds to:
        h_k(theta_k) = E[h_{k+1}(theta_{k+1}) | theta_k]

    Parameters
    ----------
    K_hat : ndarray, shape (L,)
        Spectral transition multiplier.
    g : array_like, shape (L,)
        Backward message.
    real_output : bool, optional
        Return only the real component.

    Returns
    -------
    out : ndarray, shape (L,)
        Backward transition applied to ``g``.
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

    Parameters
    ----------
    n_theta : int
        Number of points in the periodic grid.

    Returns
    -------
    modes : ndarray, shape (n_theta,)
        Integer Fourier modes in FFT ordering.
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

    Parameters
    ----------
    n_theta : int
        Number of phase-grid points.
    mean_step : float
        Mean phase increment.
    Q : float
        Phase-increment variance.

    Returns
    -------
    K_hat : ndarray, shape (n_theta,)
        Fourier multiplier for wrapped drift-diffusion.
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

    Parameters
    ----------
    p : array_like, shape (L,)
        Circular probability distribution.
    K_hat : ndarray, shape (L,)
        Spectral transition multiplier.
    real_output : bool, optional
        Return only the real component.

    Returns
    -------
    p_pred : ndarray, shape (L,)
        Predicted circular distribution.
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

    Parameters
    ----------
    p : array_like
        Nonnegative probability weights.
    floor : float, optional
        Positive value added to every weight.

    Returns
    -------
    normalized : ndarray
        Probability vector with unit sum.
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
    Summarize numerical artifacts in a spectral prediction.

    Parameters
    ----------
    p_pred : array_like
        Possibly complex spectral prediction.

    Returns
    -------
    out : dict
        Minimum real value, total real mass, and real/imaginary norms.
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

    Parameters
    ----------
    K_hat : ndarray, shape (L,)
        Spectral transition multiplier.

    Returns
    -------
    T : ndarray, shape (L, L)
        Dense row-stochastic transition matrix.
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

    Parameters
    ----------
    theta : array_like
        Phase values.
    A, B : ndarray, shape (M+1,)
        Fourier coefficients.

    Returns
    -------
    qp : ndarray
        Waveform derivative evaluated at ``theta``.
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

    Parameters
    ----------
    omega, lnDtheta, lnR : float
        Phase frequency, log diffusivity, and log observation variance.
    A, B : array_like, shape (M+1,)
        Fourier coefficients satisfying the gauge convention.

    Returns
    -------
    x : ndarray
        Flat optimization vector.
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

    Parameters
    ----------
    x : array_like
        Flat optimization vector.
    M : int
        Maximum Fourier mode.

    Returns
    -------
    omega, lnDtheta, lnR : float
        Phase frequency, log diffusivity, and log observation variance.
    A, B : ndarray, shape (M+1,)
        Fourier coefficients with ``B[0] = B[1] = 0``.
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

    # Reduce multidimensional measurements to a scalar signal.
    if yk.ndim == 2:
        # Robust choice: use principal component (generalizable)
        y_mean = yk.mean(axis=0)
        U, S, Vt = np.linalg.svd(yk - y_mean, full_matrices=False)
        y = U[:, 0] * S[0]
    else:
        y = yk.copy()

    # Remove the mean.
    if detrend:
        y = y - np.mean(y)

    # Apply a Hann window.
    if window:
        w = np.hanning(N)
        y = y * w

    # Zero-pad before the FFT.
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

    Parameters
    ----------
    data : tuple or dict
        ``(tk, yk)`` or a dictionary containing observation arrays.
    omega_hat : float
        Estimated angular frequency.
    window_size : int
        Number of observations used in the regression.
    R : float
        Observation-noise variance used for coefficient uncertainty.
    M : int, optional
        Maximum Fourier mode.
    start_index : int, optional
        First observation included in the fitting window.
    validate : bool, optional
        Perform basic input and parameter checks.

    Returns
    -------
    result : dict
        Anchored phase estimate and uncertainty, rotated and raw Fourier
        coefficients, coefficient covariance, fit indices, and fitted values.
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
    """
    Wrap angles to the interval ``[-pi, pi)``.

    Parameters
    ----------
    x : array_like
        Angles in radians.

    Returns
    -------
    wrapped : ndarray
        Wrapped angles.
    """
    return (x + np.pi) % (2.0 * np.pi) - np.pi


def wrapped_gaussian_kernel(theta_grid, mean_step, Q):
    """
    Circular transition kernel for

        theta_k = theta_{k-1} + mean_step + noise,
        noise ~ N(0, Q)

    Parameters
    ----------
    theta_grid : ndarray, shape (L,)
        Uniform circular phase grid.
    mean_step : float
        Mean phase increment.
    Q : float
        Phase-increment variance.

    Returns
    -------
    kernel : ndarray, shape (L,)
        Normalized sampled transition kernel.
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

    Parameters
    ----------
    kernel, p : ndarray, shape (L,)
        Circular kernel and input distribution.

    Returns
    -------
    out : ndarray, shape (L,)
        Circular convolution of ``kernel`` and ``p``.
    """
    return np.fft.ifft(np.fft.fft(kernel) * np.fft.fft(p)).real


def circular_backward_apply(kernel, g):
    """
    Backward transition application:

        out[i] = sum_j T[i,j] g[j]
               = sum_d kernel[d] g[i+d]

    where T[i,j] = kernel[j-i].

    Parameters
    ----------
    kernel, g : ndarray, shape (L,)
        Sampled transition kernel and backward message.

    Returns
    -------
    out : ndarray, shape (L,)
        Backward transition applied to ``g``.
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

    Parameters
    ----------
    param : tuple
        ``(omega, lnDtheta, lnR, A, B)``.
    data : tuple or dict
        Observation times and scalar measurements.
    state_priors : tuple
        ``(mu_theta, sigma_theta)`` defining the initial phase prior.
    n_theta : int, optional
        Number of phase-grid points.
    transition : {"spectral", "sampled"}, optional
        Spectral heat-kernel propagation or sampled wrapped-Gaussian
        convolution.
    probability_floor : float, optional
        Floor added during predicted-probability normalization.
    return_transition_diagnostics : bool, optional
        Reserved compatibility option; retained without changing behavior.
    validate : bool, optional
        Perform basic input and parameter checks.

    Returns
    -------
    out : dict
        Phase grid, filter distributions, likelihoods, transition
        representation, waveform, normalizers, and marginal log likelihood.
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
    Compute circular summaries from grid-based phase marginals.

    Parameters
    ----------
    gamma : ndarray, shape (N, L)
        Phase marginal distributions.
    theta_grid : ndarray, shape (L,)
        Circular phase grid.

    Returns
    -------
    out : dict
        Complex resultants, wrapped and unwrapped means, resultant lengths,
        and circular standard deviations.
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

    Parameters
    ----------
    param : tuple
        ``(omega, lnDtheta, lnR, A, B)``.
    data : tuple or dict
        Observation times and scalar measurements.
    state_priors : tuple
        Initial phase prior.
    n_theta : int, optional
        Number of phase-grid points.
    validate : bool, optional
        Perform filter input checks.

    Returns
    -------
    out : dict
        Both filter outputs and likelihood/distribution differences.
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
        Output from :func:`global_phase_filter`.
    validate : bool, optional
        Check filter-array shapes and normalization.

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

    Parameters
    ----------
    filter_out : dict
        Output from :func:`global_phase_filter`.
    n_samples : int
        Number of phase trajectories to draw.
    rng : numpy.random.Generator, optional
        Random number generator.

    Returns
    -------
    theta_samples : ndarray, shape (n_samples, N)
        Sampled phase trajectories.
    index_samples : ndarray, shape (n_samples, N)
        Corresponding phase-grid indices.
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
    """
    Compute MSD from principal-interval wrapped phase differences.

    Parameters
    ----------
    theta_samples : ndarray, shape (S, N)
        Sampled wrapped phase trajectories.
    lags : array_like
        Positive integer lags smaller than ``N``.
    omega : float, optional
        Angular frequency used to subtract deterministic drift.
    dt : float, optional
        Sampling interval.
    subtract_drift : bool, optional
        Subtract deterministic drift before wrapping increments.

    Returns
    -------
    msd : ndarray
        Mean-squared wrapped increments for each lag.

    Notes
    -----
    This circular statistic suppresses displacements beyond ``[-pi, pi)``.
    Use :func:`phase_msd_from_samples_diffusive` for long-lag diffusion.
    """
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

    Parameters
    ----------
    param : tuple
        ``(omega, lnDtheta, lnR, A, B)``.
    param_priors : tuple
        Frequency, diffusivity, observation-noise, waveform-mean, and
        mode-dependent waveform prior hyperparameters.
    validate : bool, optional
        Perform basic parameter and prior checks.

    Returns
    -------
    nlp : float
        Unnormalized negative log prior in log coordinates.
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
    n_theta : int, optional
        Number of phase-grid points.
    validate : bool, optional
        Pass validation through to the likelihood and prior.

    Returns
    -------
    nlp : float
        Unnormalized negative log posterior.

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


def get_observation_key(data, preferred=("yk_reg", "yk")):
    """
    Select the first available observation key.

    Parameters
    ----------
    data : dict
        Observation-data mapping.
    preferred : tuple[str, ...], optional
        Keys in descending priority.

    Returns
    -------
    key : str
        First preferred key present in ``data``.
    """
    for key in preferred:
        if isinstance(data, dict) and key in data:
            return key
    raise KeyError(f"None of {preferred} found in data.")


def as_inference_data(data, *, y_key=None):
    """
    Convert tuple or dictionary observations to NumPy arrays.

    Parameters
    ----------
    data : tuple or dict
        Observation times and values.
    y_key : str, optional
        Dictionary key for observations; selected automatically when omitted.

    Returns
    -------
    tk, yk : ndarray
        Observation times and values.
    """
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

    Parameters
    ----------
    data : tuple or dict
        Observation times and values.
    M : int
        Maximum Fourier mode.
    R0, Dtheta0, omega0 : float, optional
        Initial observation variance, diffusivity, and frequency.
    window_size : int, optional
        Regression-window length.
    window_cycles : float, optional
        Default regression-window duration in estimated cycles.
    y_key : str, optional
        Dictionary observation key.
    validate : bool, optional
        Perform initialization checks.

    Returns
    -------
    out : dict
        Structured initial parameters and initialization diagnostics.
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

    y_key : str, optional
        Dictionary observation key.
    alpha_D, alpha_R : float, optional
        Inverse-gamma shape parameters.
    sigma_A_factor, sigma_q_factor : float, optional
        Waveform-prior scale factors relative to the data scale.

    Returns
    -------
    priors : tuple
        Initial-phase, dynamic-parameter, and waveform prior hyperparameters.
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
    Evaluate the CPU posterior safely for SciPy optimization.

    Parameters
    ----------
    x : array_like
        Flat parameter vector.
    M : int
        Maximum Fourier mode.
    data : tuple or dict
        Observation times and values.
    priors : tuple
        Model prior hyperparameters.
    n_theta : int, optional
        Number of phase-grid points.
    bad_value : float, optional
        Finite penalty returned for invalid evaluations.
    validate : bool, optional
        Perform posterior input checks.

    Returns
    -------
    value : float
        Negative log posterior or ``bad_value``.
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
    Fit MAP parameters with a JAX-free SciPy optimizer.

    Parameters
    ----------
    data : tuple or dict
        Observation times and values.
    priors : tuple
        Model prior hyperparameters.
    param0 : tuple
        Initial ``(omega, lnDtheta, lnR, A, B)`` estimate.
    M : int
        Maximum Fourier mode.
    n_theta : int, optional
        Number of phase-grid points.
    method : str, optional
        SciPy optimization method; defaults to gradient-free Powell.
    options : dict, optional
        Optimizer options.
    validate : bool, optional
        Perform posterior input checks.

    Returns
    -------
    out : dict
        SciPy result, structured and flat MAP estimates, objective, and
        convergence diagnostics.

    Notes
    -----
    This is the CPU fallback for :func:`HippocampalJAX.fit_map_parameters_jax`;
    it does not use autodifferentiated gradients.
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

    Parameters
    ----------
    param : tuple
        ``(omega, lnDtheta, lnR, A, B)``.
    prefix : str, optional
        Prefix added to every output key.

    Returns
    -------
    out : dict
        Natural/log dynamic parameters and Fourier coefficients.
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

    Parameters
    ----------
    data : dict
        Original observation data.
    R_add : float
        Added regularizing-noise variance.
    M : int
        Maximum Fourier mode.
    rng : numpy.random.Generator
        Random number generator.
    n_theta : int, optional
        Number of phase-grid points.
    param0, priors : tuple, optional
        Explicit initialization and priors.
    initialize_kwargs, prior_kwargs, optimizer_kwargs : dict, optional
        Options forwarded to the CPU pipeline stages.
    y_key, regularized_key : str, optional
        Input and regularized observation keys.
    validate : bool, optional
        Perform input and posterior checks.

    Returns
    -------
    out : dict
        Summary, CPU fit, parameters, priors, regularized data, and
        conditional dynamic-parameter covariance.

    Notes
    -----
    Unlike the JAX workflow, uncertainty is estimated only for
    ``(omega, lnDtheta, lnR)`` while holding waveform coefficients fixed.
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

    Parameters
    ----------
    data : dict
        Original observation data.
    R_add_values : iterable[float]
        Added-noise variances.
    M : int
        Maximum Fourier mode.
    rng : numpy.random.Generator
        Random number generator.
    n_reps : int, optional
        Replicate sweeps.
    n_theta : int, optional
        Number of phase-grid points.
    warm_start : bool, optional
        Initialize each fit from the preceding successful MAP.
    initialize_kwargs, prior_kwargs, optimizer_kwargs : dict, optional
        Options forwarded to each CPU fit.
    validate : bool, optional
        Perform input and posterior checks.

    Returns
    -------
    out : dict
        Summary data frame and full fit results.
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

    Parameters
    ----------
    param_map : tuple
        Structured MAP parameters.
    data : tuple or dict
        Observation times and values.
    priors : tuple
        Model prior hyperparameters.
    n_theta : int, optional
        Number of phase-grid points.
    step : float or ndarray, optional
        Finite-difference step.
    validate : bool, optional
        Perform posterior input checks.
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
    """
    Compute drift-subtracted MSD from unwrapped phase trajectories.

    Parameters
    ----------
    theta_samples : ndarray, shape (S, N)
        Sampled wrapped phase trajectories.
    lags : int or array_like
        Positive integer lags smaller than ``N``.
    omega : float
        Angular frequency removed from increments.
    dt : float
        Sampling interval.

    Returns
    -------
    msd : float or ndarray
        Mean-squared unwrapped increments; scalar when ``lags`` is scalar.

    Notes
    -----
    This preserves multi-cycle displacement and is preferred for long-lag
    diffusion. Use :func:`phase_msd_from_samples` for circular increments.
    """
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
