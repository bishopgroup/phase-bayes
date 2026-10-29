import numpy as np
import warnings
from scipy.optimize import minimize
from scipy.linalg import solve_banded
from typing import Literal, Tuple


def simulate_data(param, N, dt, rng, state_priors, *, validate=False):
    """
    Simulate latent phase θ_k and 2D nonlinear measurements y_k for k = 1..N.

    Model
    -----
    Prior:
        θ_1 ~ N(μ_θ, σ_θ^2)

    Process:
        θ_k = θ_{k-1} + ω Δt + ε_{k-1},     ε_{k-1} ~ N(0, Q),
        Q = 2 D_θ Δt,                       k = 2..N

    Measurement (2D, nonlinear, isotropic):
        y_k = h(θ_k) + η_k,                 η_k ~ N(0, R I_2),   k = 1..N
        h(θ) = [cos θ, sin θ]^T

    Indexing convention:
        k = 1..N,  t_k = k * dt.
        Python element i corresponds to k = i+1.

    Parameters
    ----------
    param : tuple[float, float, float]
        (omega, Dtheta, R)

    N : int
        Number of samples.

    dt : float
        Time step (> 0).

    rng : numpy.random.Generator
        Random number generator.

    state_priors : tuple[float, float]
        (mu_theta, sigma_theta) defining the prior for θ_1.

    validate : bool, optional
        Basic sanity checks.

    Returns
    -------
    tk : ndarray, shape (N,)
        Times t_k = dt * k for k=1..N.

    thetak : ndarray, shape (N,)
        Latent phase θ_k for k=1..N.

    yk : ndarray, shape (N, 2)
        2D measurements y_k = [y_{1k}, y_{2k}] for k=1..N.
    """
    omega, Dtheta, R = param
    mu_theta, sigma_theta = state_priors

    if validate:
        if not (isinstance(N, (int, np.integer)) and N > 0):
            raise ValueError(f"N must be a positive integer; got {N!r}")
        if dt <= 0:
            raise ValueError(f"dt must be > 0; got {dt!r}")
        if sigma_theta < 0:
            raise ValueError(f"sigma_theta must be >= 0; got {sigma_theta!r}")
        if Dtheta < 0:
            raise ValueError(f"Dtheta must be >= 0; got {Dtheta!r}")
        if R < 0:
            raise ValueError(f"R must be >= 0; got {R!r}")

    tk = dt * np.arange(1, N + 1, dtype=float)

    Q = 2.0 * Dtheta * dt
    wt = omega * dt

    thetak = np.empty(N, dtype=float)

    # θ_1 ~ N(μ_θ, σ_θ^2)
    if sigma_theta == 0.0:
        thetak[0] = float(mu_theta)
    else:
        thetak[0] = float(mu_theta) + float(sigma_theta) * rng.standard_normal()

    # Propagate for k = 2..N: θ_k = θ_{k-1} + ω dt + ε_{k-1}
    if N > 1:
        proc_std = np.sqrt(Q)
        thetak[1:] = thetak[0] + wt * np.arange(1, N, dtype=float) \
            + proc_std * np.cumsum(rng.standard_normal(N - 1))

    # Nonlinear measurements: y_k = [cos θ_k, sin θ_k]^T + η_k,  η_k ~ N(0, R I_2)
    yk_det = np.column_stack((np.cos(thetak), np.sin(thetak)))  # shape (N, 2)

    if R == 0.0:
        yk = yk_det
    else:
        yk = yk_det + np.sqrt(R) * rng.standard_normal((N, 2))

    return tk, thetak, yk


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
    start_index=0,
    validate=False,
    nsigma_check=2.0,
):
    """
    Estimate initial relative phase phi from a short window of 2D harmonic measurements,
    assuming omega_hat is known and phase diffusion is negligible over the window.

    Model
    -----
    theta_k ≈ omega_hat * t_k + phi

    y_k = h(theta_k) + eta_k,   eta_k ~ N(0, R I_2)
    h(theta) = [cos(theta), sin(theta)]^T

    Writing
        A = cos(phi),  B = sin(phi),
    the measurement model becomes linear in (A, B):
        y1_k =  A cos(omega_hat t_k) - B sin(omega_hat t_k) + noise
        y2_k =  B cos(omega_hat t_k) + A sin(omega_hat t_k) + noise

    Parameters
    ----------
    data : tuple
        (tk, yk), where
          tk : ndarray, shape (N,)
          yk : ndarray, shape (N, 2)

    omega_hat : float
        Estimated angular frequency (rad / unit time).

    window_size : int
        Number of points in the initial fitting window.

    R : float
        Known per-component measurement-noise variance.

    start_index : int, optional
        First sample to include in the window. Default is 0.

    validate : bool, optional
        If True, perform basic sanity checks.

    nsigma_check : float, optional
        Number of standard deviations used when checking whether
        sqrt(A^2 + B^2) is consistent with 1. Default is 2.

    Returns
    -------
    result : dict
        Dictionary with fields:
          phi_mean : float
              Estimated phase phi = atan2(B, A), wrapped to (-pi, pi].

          phi_std : float
              Delta-method standard deviation of phi.

          A_mean, B_mean : float
              Estimated A = cos(phi), B = sin(phi)

          cov_AB : ndarray, shape (2, 2)
              Covariance matrix of [A, B]

          radius_mean : float
              Estimated radius sqrt(A^2 + B^2)

          radius_std : float
              Delta-method standard deviation of radius

          radius_zscore : float
              (radius_mean - 1) / radius_std

          radius_consistent : bool
              Whether |radius_mean - 1| <= nsigma_check * radius_std

          fit_indices : ndarray
              Indices used in the fit

          y_fit : ndarray, shape (m, 2)
              Fitted mean measurements over the fitting window
    """
    tk, yk = data
    tk = np.asarray(tk, dtype=float)
    yk = np.asarray(yk, dtype=float)

    if validate:
        if yk.ndim != 2 or yk.shape[1] != 2:
            raise ValueError(f"yk must have shape (N, 2); got {yk.shape}")
        if tk.ndim != 1 or tk.shape[0] != yk.shape[0]:
            raise ValueError(f"tk must have shape (N,) with N={yk.shape[0]}; got {tk.shape}")
        if not np.isscalar(window_size) or int(window_size) != window_size or window_size < 2:
            raise ValueError("window_size must be an integer >= 2")
        if not np.isscalar(start_index) or int(start_index) != start_index or start_index < 0:
            raise ValueError("start_index must be a nonnegative integer")
        if start_index + window_size > len(tk):
            raise ValueError("start_index + window_size exceeds number of samples")
        if R <= 0:
            raise ValueError(f"R must be > 0; got {R!r}")

    m = int(window_size)
    i0 = int(start_index)
    idx = np.arange(i0, i0 + m)

    t = tk[idx]
    y = yk[idx]

    c = np.cos(omega_hat * t)
    s = np.sin(omega_hat * t)

    # Build stacked linear system Z beta + noise, beta = [A, B]^T
    #
    # y1_k = [ c_k, -s_k ] [A, B]^T + noise
    # y2_k = [ s_k,  c_k ] [A, B]^T + noise
    #
    Z = np.zeros((2 * m, 2), dtype=float)
    z = np.zeros(2 * m, dtype=float)

    Z[0::2, 0] = c
    Z[0::2, 1] = -s
    Z[1::2, 0] = s
    Z[1::2, 1] = c

    z[0::2] = y[:, 0]
    z[1::2] = y[:, 1]

    # GLS = OLS here because covariance is R * I on the stacked observations
    JTJ = Z.T @ Z
    Jtz = Z.T @ z

    beta_hat = np.linalg.solve(JTJ, Jtz)
    A_hat, B_hat = beta_hat

    cov_AB = R * np.linalg.inv(JTJ)

    # Phase estimate: phi = atan2(B, A)
    phi_mean = np.arctan2(B_hat, A_hat)

    # Delta-method variance for phi
    r2 = A_hat**2 + B_hat**2
    if r2 <= 0:
        raise RuntimeError("Degenerate fit: A_hat^2 + B_hat^2 <= 0")

    grad_phi = np.array([-B_hat / r2, A_hat / r2], dtype=float)
    phi_var = float(grad_phi @ cov_AB @ grad_phi)
    phi_std = np.sqrt(max(phi_var, 0.0))

    # Radius check: r = sqrt(A^2 + B^2) should be 1
    r = np.sqrt(r2)
    grad_r = np.array([A_hat / r, B_hat / r], dtype=float)
    r_var = float(grad_r @ cov_AB @ grad_r)
    r_std = np.sqrt(max(r_var, 0.0))

    if r_std > 0:
        r_z = (r - 1.0) / r_std
        radius_consistent = (abs(r - 1.0) <= nsigma_check * r_std)
    else:
        r_z = np.inf if r != 1.0 else 0.0
        radius_consistent = (r == 1.0)

    # Fitted measurements over the window
    y_fit = np.column_stack([
        A_hat * c - B_hat * s,
        B_hat * c + A_hat * s,
    ])

    return {
        "phi_mean": phi_mean,
        "phi_std": phi_std,
        "A_mean": A_hat,
        "B_mean": B_hat,
        "cov_AB": cov_AB,
        "radius_mean": r,
        "radius_std": r_std,
        "radius_zscore": r_z,
        "radius_consistent": radius_consistent,
        "fit_indices": idx,
        "y_fit": y_fit,
    }

def kalman_filter(param, data, state_priors, *, validate=False):
    """
    Extended Kalman filter (EKF) for the drifting–diffusing phase with nonlinear 2D measurements.

    Model (k = 1..N):
        Prior:      θ_1 ~ N(μ_θ, σ_θ^2)
        Process:    θ_k = θ_{k-1} + ω Δt + ε_{k-1},   ε_{k-1} ~ N(0, Q),  Q = 2 D_θ Δt,  k=2..N
        Measurement y_k = h(θ_k) + η_k,              η_k ~ N(0, R I_2),   k=1..N
                    h(θ) = [cos θ, sin θ]^T

    Linearization (single-pass EKF, stored for downstream use):
        h(θ_k) ≈ h(θ̂_{k|k-1}) + H_k (θ_k - θ̂_{k|k-1})
              = c_k + H_k θ_k
        where
            H_k = dh/dθ |_{θ̂_{k|k-1}}  ∈ R^{2×1},
            c_k = h(θ̂_{k|k-1}) - H_k θ̂_{k|k-1} ∈ R^2.

    Indexing convention:
        - Python arrays are 0..N-1, but element i corresponds to SI index k=i+1.
        - tk[i] should equal (i+1)*dt (uniform spacing assumed).

    Inputs
    ------
    param : (omega, Dtheta, R)
        omega  : drift rate (angular frequency)
        Dtheta : phase diffusivity (>=0)
        R      : per-component measurement-noise variance (>=0), covariance = R I_2

    data : (tk, yk)
        tk : times, shape (N,), uniformly spaced
        yk : measurements, shape (N,2)

    state_priors : (mu_theta, sigma_theta, ...)
        mu_theta    : prior mean for θ_1
        sigma_theta : prior std for θ_1
        Remaining entries are ignored by this function.

    validate : bool (default False)
        If True, do basic sanity checks (helpful while debugging).

    Returns
    -------
    thetakk   : ndarray, shape (N,)
        θ̂_{k|k} filtered posterior mean.

    Pkk       : ndarray, shape (N,)
        P_{k|k} filtered posterior variance.

    thetakk_1 : ndarray, shape (N,)
        θ̂_{k|k-1} predicted mean.

    Pkk_1     : ndarray, shape (N,)
        P_{k|k-1} predicted variance.

    ypred     : ndarray, shape (N,2)
        Predicted measurement h(θ̂_{k|k-1}).

    Hk        : ndarray, shape (N,2)
        Linearized measurement Jacobian entries stacked as [H1k, H2k],
        i.e. H_k = [[H1k],[H2k]] = [-sin θ̂_{k|k-1}, cos θ̂_{k|k-1}]^T.

    ck        : ndarray, shape (N,2)
        Affine offset c_k = h(θ̂_{k|k-1}) - H_k θ̂_{k|k-1}.
        With Hk stored as (N,2), interpret c_k componentwise.
    """
    omega, Dtheta, R = param
    tk, yk = data
    mu_theta, sigma_theta = state_priors

    yk = np.asarray(yk)
    tk = np.asarray(tk)

    N = yk.shape[0]
    if N == 0:
        empty0 = np.empty((0,), dtype=float)
        empty2 = np.empty((0, 2), dtype=float)
        return empty0, empty0, empty0, empty0, empty2, empty2, empty2

    # Uniform time step (SI uses t_k = k Δt for k=1..N)
    dt = (tk[1] - tk[0]) if N > 1 else 0.0

    if validate:
        if not (isinstance(N, (int, np.integer)) and N > 0):
            raise ValueError(f"N must be a positive integer; got {N!r}")
        if yk.ndim != 2 or yk.shape[1] != 2:
            raise ValueError(f"yk must have shape (N,2); got {yk.shape}")
        if tk.shape != (N,):
            raise ValueError(f"tk must have shape (N,); got {tk.shape}")
        if N > 2 and not np.allclose(np.diff(tk), dt):
            raise ValueError("tk must be uniformly spaced (constant dt).")
        if dt <= 0 and N > 1:
            raise ValueError(f"dt must be > 0; got {dt!r}")
        if Dtheta < 0:
            raise ValueError(f"Dtheta must be >= 0; got {Dtheta!r}")
        if R < 0:
            raise ValueError(f"R must be >= 0; got {R!r}")
        if sigma_theta < 0:
            raise ValueError(f"sigma_theta must be >= 0; got {sigma_theta!r}")

    # Helper scalars
    Q  = 2.0 * Dtheta * dt          # process noise variance per step (k-1 -> k)
    wt = omega * dt                 # deterministic drift per step

    # Allocate outputs
    thetakk   = np.empty(N, dtype=float)
    Pkk       = np.empty(N, dtype=float)
    thetakk_1 = np.empty(N, dtype=float)
    Pkk_1     = np.empty(N, dtype=float)

    # Linearized measurement model storage
    ypred = np.empty((N, 2), dtype=float)
    Hk    = np.empty((N, 2), dtype=float)  # store entries of 2×1 Jacobian as length-2 row
    ck    = np.empty((N, 2), dtype=float)

    def h(theta):
        return np.array([np.cos(theta), np.sin(theta)], dtype=float)

    # dh/dθ evaluated at theta: returns length-2 vector corresponding to 2×1 Jacobian
    def dh(theta):
        return np.array([-np.sin(theta), np.cos(theta)], dtype=float)

    # ----- k = 1 (python index 0): prediction is prior -----
    thetakk_1[0] = float(mu_theta)
    Pkk_1[0]     = float(sigma_theta)**2

    # Linearize measurement about θ̂_{1|0} = θ̂_{1|1-1}
    th_pred = thetakk_1[0]
    ypred[0] = h(th_pred)
    Hk[0]    = dh(th_pred)
    ck[0]    = ypred[0] - Hk[0] * th_pred  # componentwise since Hk is stored as (2,)

    # EKF update at k=1
    # S = H P H^T + R I2  (2×2),  K = P H^T S^{-1}  (1×2)
    H = Hk[0].reshape(2, 1)               # 2×1
    Ppred = Pkk_1[0]                      # scalar
    S = (Ppred * (H @ H.T)) + (R * np.eye(2))  # 2×2
    Sinv = np.linalg.pinv(S) if R == 0.0 else np.linalg.inv(S)
    K = (Ppred * (H.T @ Sinv)).reshape(2,)     # 1×2 -> store as length-2

    innov = yk[0] - ypred[0]               # 2,
    thetakk[0] = th_pred + float(K @ innov)

    # Joseph-stabilized covariance update (scalar state)
    KH = float(K @ Hk[0])                  # (1×2)(2×1) = scalar
    P_post = (1.0 - KH) * Ppred * (1.0 - KH) + (R * float(K @ K))
    Pkk[0] = P_post

    # ----- k = 2..N -----
    for k in range(1, N):
        # Predict
        thetakk_1[k] = thetakk[k - 1] + wt
        Pkk_1[k]     = Pkk[k - 1] + Q

        # Linearize measurement about θ̂_{k|k-1}
        th_pred = thetakk_1[k]
        ypred[k] = h(th_pred)
        Hk[k]    = dh(th_pred)
        ck[k]    = ypred[k] - Hk[k] * th_pred

        # Update
        H = Hk[k].reshape(2, 1)                 # 2×1
        Ppred = Pkk_1[k]                         # scalar
        S = (Ppred * (H @ H.T)) + (R * np.eye(2))
        Sinv = np.linalg.pinv(S) if R == 0.0 else np.linalg.inv(S)
        K = (Ppred * (H.T @ Sinv)).reshape(2,)   # length-2

        innov = yk[k] - ypred[k]
        thetakk[k] = th_pred + float(K @ innov)

        KH = float(K @ Hk[k])
        P_post = (1.0 - KH) * Ppred * (1.0 - KH) + (R * float(K @ K))
        Pkk[k] = P_post

    return thetakk, Pkk, thetakk_1, Pkk_1, ypred, Hk, ck


def neg_log_marginal_likelihood(param, data, state_priors, *, param_mode="log", validate=False):
    """
    Negative log marginal likelihood for the nonlinear 2D measurement model
    using the EKF (single-pass linearization):
        nll = -log p(y_{1:N} | omega, Dtheta, R)   (approx.)

    Model (k = 1..N):
        Prior:
            θ_1 ~ N(μ_θ, σ_θ^2)

        State evolution (k = 2..N):
            θ_k = θ_{k-1} + ω Δt + ε_{k-1},   ε_{k-1} ~ N(0, Q),  Q = 2 Dθ Δt

        Measurement (k = 1..N):
            y_k = h(θ_k) + η_k,              η_k ~ N(0, R I_2)
            h(θ) = [cos θ, sin θ]^T

    The marginal likelihood is approximated by the EKF innovations:
        p(y_k | y_{1:k-1}) ≈ N(y_k; h(θ̂_{k|k-1}), S_k),
        S_k = H_k P_{k|k-1} H_k^T + R I_2,
        H_k = dh/dθ evaluated at θ̂_{k|k-1}.

    Indexing convention:
        Python index i corresponds to SI index k=i+1.

    Parameters
    ----------
    param : tuple[float, float, float]
        If param_mode == "natural":
            (omega, Dtheta, R)
        If param_mode == "log":
            (omega, lnDtheta, lnR) with Dtheta = exp(lnDtheta), R = exp(lnR)

    data : tuple[ndarray, ndarray]
        (tk, yk) with tk uniformly spaced, yk shape (N,2).

    state_priors : tuple
        (mu_theta, sigma_theta, ...) only first two entries used.

    param_mode : {"natural", "log"}, optional
        Choose whether to interpret param as (omega, Dtheta, R) or (omega, lnDtheta, lnR).

    validate : bool, optional
        If True, run basic checks (uniform dt, positivity constraints).

    Returns
    -------
    nll : float
        Approximate negative log marginal likelihood (EKF).
    """
    tk, yk = data
    mu_theta, sigma_theta = state_priors

    tk = np.asarray(tk)
    yk = np.asarray(yk)

    # Unpack parameters (no priors on omega, Dtheta, R included here)
    if param_mode == "natural":
        omega, Dtheta, R = param
    elif param_mode == "log":
        omega, lnDtheta, lnR = param
        Dtheta = np.exp(lnDtheta)
        R = np.exp(lnR)
    else:
        raise ValueError(f"param_mode must be 'natural' or 'log'; got {param_mode!r}")

    N = yk.shape[0]
    if N == 0:
        return 0.0

    # Uniform time step
    dt = (tk[1] - tk[0]) if N > 1 else 0.0

    if validate:
        if yk.ndim != 2 or yk.shape[1] != 2:
            raise ValueError(f"yk must have shape (N,2); got {yk.shape}")
        if tk.shape != (N,):
            raise ValueError(f"tk must have shape (N,); got {tk.shape}")
        if N > 2 and not np.allclose(np.diff(tk), dt):
            raise ValueError("tk must be uniformly spaced (constant dt).")
        if sigma_theta <= 0:
            raise ValueError("sigma_theta must be > 0.")
        if dt <= 0 and N > 1:
            raise ValueError("dt must be > 0 for N>1.")
        if Dtheta < 0:
            raise ValueError("Dtheta must be >= 0.")
        if R < 0:
            raise ValueError("R must be >= 0.")

    Q  = 2.0 * Dtheta * dt
    wt = omega * dt
    log2pi = np.log(2.0 * np.pi)

    def h(theta):
        return np.array([np.cos(theta), np.sin(theta)], dtype=float)

    def dh(theta):
        return np.array([-np.sin(theta), np.cos(theta)], dtype=float)  # length-2 (2×1 Jacobian)

    # ---- k = 1: predicted state is the prior on θ_1 ----
    theta_pred = float(mu_theta)
    P_pred = float(sigma_theta)**2

    nll = 0.0

    # ---- EKF innovation likelihood contributions ----
    for k in range(N):
        # Predicted measurement and Jacobian at θ̂_{k|k-1}
        y_pred = h(theta_pred)            # (2,)
        Hk = dh(theta_pred).reshape(2, 1) # (2,1)

        # Innovation covariance S_k = H P H^T + R I2  (2×2)
        S = (P_pred * (Hk @ Hk.T)) + (R * np.eye(2))

        # Innovation
        innov = (yk[k] - y_pred).reshape(2, 1)

        # Handle degenerate S robustly
        # (In practice, keep sigma_theta>0 and R>0 to avoid this.)
        try:
            sign, logdetS = np.linalg.slogdet(S)
            if sign <= 0:
                return np.inf
            Sinv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return np.inf

        # Accumulate -log p(y_k | y_{1:k-1})
        quad = float((innov.T @ Sinv @ innov).squeeze())
        nll += 0.5 * (2.0 * log2pi + logdetS + quad)

        # EKF update:
        # K = P H^T S^{-1}  (1×2)
        K = (P_pred * (Hk.T @ Sinv)).reshape(2,)  # length-2
        theta_filt = theta_pred + float(K @ innov.ravel())

        # Joseph-stabilized covariance update for scalar state
        KH = float(K @ Hk.ravel())                # scalar
        P_filt = (1.0 - KH) * P_pred * (1.0 - KH) + (R * float(K @ K))

        # Predict next step (except after last)
        if k < N - 1:
            theta_pred = theta_filt + wt
            P_pred = P_filt + Q

    return float(nll)


def neg_log_prior(param, param_priors, *, param_mode="log", validate=False):
    """
    Negative log prior for model parameters (omega, Dtheta, R).

    Priors:
        omega  ~ Normal(mu_omega, sigma_omega^2)
        Dtheta ~ Inverse-Gamma(alpha_D, beta_D)   (shape α, scale β)
        R      ~ Inverse-Gamma(alpha_R, beta_R)

    Notes:
        - Normalization constants are omitted.
        - If param_mode="log", Jacobian terms are included so this is the
          negative log density in log-parameter space.

    Parameters
    ----------
    param : tuple
        If param_mode == "log":
            (omega, lnDtheta, lnR)
        If param_mode == "natural":
            (omega, Dtheta, R)

    param_priors : tuple
        (mu_omega, sigma_omega,
         alpha_D, beta_D,
         alpha_R, beta_R)

    param_mode : {"log", "natural"}, default "log"

    validate : bool, optional
        If True, perform basic positivity checks.

    Returns
    -------
    nlp : float
        Negative log prior (unnormalized).
    """
    mu_omega, sigma_omega, alpha_D, beta_D, alpha_R, beta_R = param_priors

    if param_mode == "log":
        omega, lnDtheta, lnR = param
        Dtheta = np.exp(lnDtheta)
        R = np.exp(lnR)
    elif param_mode == "natural":
        omega, Dtheta, R = param
    else:
        raise ValueError("param_mode must be 'log' or 'natural'")

    if validate:
        if sigma_omega <= 0:
            raise ValueError("sigma_omega must be > 0")
        if alpha_D <= 0 or beta_D <= 0:
            raise ValueError("alpha_D and beta_D must be > 0")
        if alpha_R <= 0 or beta_R <= 0:
            raise ValueError("alpha_R and beta_R must be > 0")
        if param_mode == "natural" and (Dtheta <= 0 or R <= 0):
            raise ValueError("Dtheta and R must be > 0")

    nlp = 0.0

    # ---- Normal prior on omega (omit constants) ----
    nlp += 0.5 * ((omega - mu_omega) / sigma_omega) ** 2

    # ---- Inverse-Gamma priors (omit constants) ----
    # IG(α, β): p(x) ∝ x^(-α-1) exp(-β/x)
    # => -log p(x) = (α+1) log x + β/x  (up to additive constant)

    nlp += (alpha_D + 1.0) * np.log(Dtheta) + beta_D / Dtheta
    nlp += (alpha_R + 1.0) * np.log(R) + beta_R / R

    # ---- Jacobian correction for log-parameterization ----
    if param_mode == "log":
        # -log p(ln x) = -log p(x) - ln x
        nlp -= lnDtheta
        nlp -= lnR

    return nlp


def neg_log_posterior(param, data, priors, *, param_mode="log", validate=False):
    """
    Negative log posterior (up to an additive constant):


        nlp = -log p(y_{1:N} | omega, Dtheta, R) - log p(omega, Dtheta, R)


    This combines:
      1) neg_log_marginal_likelihood(...)  [no priors on omega, Dtheta, R]
      2) neg_log_prior(...)                [priors on omega, Dtheta, R only; unnormalized]


    Parameters
    ----------
    param : tuple
        If param_mode == "log":
            (omega, lnDtheta, lnR)
        If param_mode == "natural":
            (omega, Dtheta, R)


    data : (tk, yk)
        See neg_log_marginal_likelihood.


    priors : tuple
        (mu_theta, sigma_theta, mu_omega, sigma_omega, alpha_D, beta_D, alpha_R, beta_R)


        Note: This function splits priors internally into:
            state_priors = (mu_theta, sigma_theta, ...)
            param_priors = (mu_omega, sigma_omega, alpha_D, beta_D, alpha_R, beta_R)


    param_mode : {"log", "natural"}, default "log"
    validate : bool, optional
        If True, passes validation flags to the component functions.


    Returns
    -------
    nlp : float
        Negative log posterior (unnormalized).
    """
    # Split priors
    mu_theta, sigma_theta, mu_omega, sigma_omega, alpha_D, beta_D, alpha_R, beta_R = priors
    state_priors = (mu_theta, sigma_theta)
    param_priors = (mu_omega, sigma_omega, alpha_D, beta_D, alpha_R, beta_R)


    nll = neg_log_marginal_likelihood(
        param, data, state_priors, param_mode=param_mode, validate=validate
    )
    nlp = neg_log_prior(
        param, param_priors, param_mode=param_mode, validate=validate
    )
    return nll + nlp


def rts_smoother(param, data, kalman_phases, *, validate=False):
    """
    Rauch–Tung–Striebel (RTS) smoother consistent with the updated filter.

    Notes:
      - The RTS recursions depend only on the *state* transition model and the
        filtered/predicted state moments. They do not require the measurement
        model or the EKF linearization arrays (ypred, Hk, ck). Therefore, if
        kalman_phases includes those extras, they are simply ignored.

    Model / indexing:
      - Arrays are length N; Python index i corresponds to SI index k=i+1.
      - Prior is on θ_1; process noise Q applies only to transitions k>=2.

    Inputs
    ------
    param : (omega, Dtheta, R)
    data : (tk, yk)
    kalman_phases :
        Either (thetakk, Pkk, thetakk_1, Pkk_1)
        or     (thetakk, Pkk, thetakk_1, Pkk_1, ...)  # extra entries ignored

        outputs of kalman_filter:
          thetakk   : θ̂_{k|k}
          Pkk       : P_{k|k}
          thetakk_1 : θ̂_{k|k-1}
          Pkk_1     : P_{k|k-1}

    Returns
    -------
    thetakN : θ̂_{k|N}     smoothed mean, shape (N,)
    PkN     : P_{k|N}      smoothed variance, shape (N,)
    Sigmakk_1 : Σ_{k,k-1|N} lag-one smoothed covariance (lower diagonal), shape (N,)
        Convention: Sigmakk_1[i] corresponds to Σ_{k=i+1, k-1=i|N}.
        Entry Sigmakk_1[0] (k=1) is meaningless and set to 0.
    """
    omega, Dtheta, R = param
    tk, yk = data

    # Accept either 4-tuple (old) or longer tuple (e.g., EKF with stored linearization)
    if len(kalman_phases) < 4:
        raise ValueError("kalman_phases must contain at least (thetakk, Pkk, thetakk_1, Pkk_1).")
    thetakk, Pkk, thetakk_1, Pkk_1 = kalman_phases[:4]

    N = yk.shape[0]
    if N == 0:
        empty = np.empty((0,), dtype=float)
        return empty, empty, empty

    if validate:
        if N != thetakk.shape[0] or N != Pkk.shape[0] or N != thetakk_1.shape[0] or N != Pkk_1.shape[0]:
            raise ValueError("kalman_phases arrays must all have length N.")
        if N > 2:
            dt = tk[1] - tk[0]
            if not np.allclose(np.diff(tk), dt):
                raise ValueError("tk must be uniformly spaced.")

    # Allocate outputs
    thetakN = np.empty(N, dtype=float)        # θ̂_{k|N}
    PkN = np.empty(N, dtype=float)            # P_{k|N}
    Sigmakk_1 = np.zeros(N, dtype=float)      # Σ_{k,k-1|N}; Sigmakk_1[0] unused

    # Initialize at final time: k=N (python index N-1)
    thetakN[-1] = thetakk[-1]
    PkN[-1] = Pkk[-1]

    # Backward pass for k = N-1 ... 1 (python indices N-2 ... 0)
    for k in range(N - 2, -1, -1):
        # RTS gain: C_k = P_{k|k} / P_{k+1|k}  (state transition A=1)
        denom = Pkk_1[k + 1]
        Ck = Pkk[k] / denom

        # Smoothed mean/variance
        thetakN[k] = thetakk[k] + Ck * (thetakN[k + 1] - thetakk_1[k + 1])
        PkN[k] = Pkk[k] + (Ck * Ck) * (PkN[k + 1] - Pkk_1[k + 1])

        # Lag-one smoothed covariance (lower diagonal):
        # Σ_{k+1, k|N} = C_k * P_{k+1|N}
        Sigmakk_1[k + 1] = Ck * PkN[k + 1]

    return thetakN, PkN, Sigmakk_1


def theta_covariance(param, N, dt, state_priors, *, validate=False):
    """
    Approximate posterior covariance of θ_{1:N} | y_{1:N} for the EKF-linearized model.

    Model:
        θ_1 ~ N(μ_θ, σ_θ^2)
        θ_k = θ_{k-1} + ω Δt + ε_{k-1},   ε ~ N(0, Q),   Q = 2 Dθ Δt
        y_k ≈ c_k + H_k θ_k + η_k,        η_k ~ N(0, R I_2)

    where:
        - θ_k is scalar
        - y_k is 2D
        - H_k = [-sin θ̂_{k|k-1}, cos θ̂_{k|k-1}]^T

    Since H_k^T H_k = 1 for all k, the linearized measurement contributes
    scalar precision

        J_k = H_k^T (R I_2)^{-1} H_k = 1 / R,

    so the posterior precision matrix is identical to that in Example 1a.
    The posterior precision A is tridiagonal, and this function returns
    Σ = A^{-1}.

    Parameters
    ----------
    param : (omega, Dtheta, R)
        omega : unused here (covariance does not depend on drift)
        Dtheta : phase diffusivity (> 0)
        R : per-component measurement-noise variance (> 0), so covariance = R I_2

    N : int
        Number of time steps.

    dt : float
        Time step.

    state_priors : (mu_theta, sigma_theta)
        Only sigma_theta affects the covariance.

    validate : bool, default False
        If True, perform basic sanity checks.

    Returns
    -------
    Sigma : ndarray, shape (N, N)
        Approximate posterior covariance matrix of θ_{1:N}.
    """
    _, Dtheta, R = param
    _, sigma_theta = state_priors

    if N == 0:
        return np.empty((0, 0), dtype=float)

    Q = 2.0 * Dtheta * dt

    if validate:
        if not (isinstance(N, (int, np.integer)) and N >= 0):
            raise ValueError(f"N must be a nonnegative integer; got {N!r}")
        if dt <= 0:
            raise ValueError(f"dt must be > 0; got {dt!r}")
        if Dtheta <= 0:
            raise ValueError(f"Dtheta must be > 0; got {Dtheta!r}")
        if R <= 0:
            raise ValueError(f"R must be > 0; got {R!r}")
        if sigma_theta <= 0:
            raise ValueError(f"sigma_theta must be > 0; got {sigma_theta!r}")

    invQ = 1.0 / Q
    invR = 1.0 / R
    invSig2 = 1.0 / (sigma_theta ** 2)

    # Tridiagonal posterior precision matrix A in banded form
    ab = np.zeros((3, N), dtype=float)

    # Off-diagonals
    ab[0, 1:] = -invQ
    ab[2, :-1] = -invQ

    # Diagonal
    ab[1, :] = invR + 2.0 * invQ
    ab[1, 0] = invR + invQ + invSig2
    ab[1, -1] = invR + invQ

    # Solve A Σ = I
    Sigma = solve_banded((1, 1), ab, np.eye(N))
    return Sigma
    

def neg_log_marginal_likelihood_frozen(
    param,
    data,
    state_priors,
    Hk,
    ck,
    *,
    param_mode="log",
    validate=False,
):
    """
    Negative log marginal likelihood for the frozen linearized model:
        nll = -log p(y_{1:N} | omega, Dtheta, R ; Hk, ck)

    Frozen linearized model
    -----------------------
        θ_1 ~ N(μ_θ, σ_θ^2)

        θ_k = θ_{k-1} + ω Δt + ε_{k-1},
        ε_{k-1} ~ N(0, Q),   Q = 2 Dθ Δt

        y_k = c_k + H_k θ_k + η_k,
        η_k ~ N(0, R I_2)

    where H_k and c_k are treated as fixed inputs, typically obtained by
    freezing the EKF linearization at a reference parameter estimate.

    Parameters
    ----------
    param : tuple
        If param_mode == "natural":
            (omega, Dtheta, R)
        If param_mode == "log":
            (omega, lnDtheta, lnR) with Dtheta = exp(lnDtheta), R = exp(lnR)

    data : tuple
        (tk, yk), with yk shape (N, 2)

    state_priors : tuple
        (mu_theta, sigma_theta)

    Hk : ndarray, shape (N, 2)
        Frozen Jacobian rows, with Hk[k] the length-2 representation of the
        2x1 matrix H_k.

    ck : ndarray, shape (N, 2)
        Frozen affine offsets.

    param_mode : {"natural", "log"}, optional

    validate : bool, optional

    Returns
    -------
    nll : float
        Negative log marginal likelihood for the frozen linear-Gaussian model.
    """
    tk, yk = data
    mu_theta, sigma_theta = state_priors

    tk = np.asarray(tk, dtype=float).ravel()
    yk = np.asarray(yk, dtype=float)
    Hk = np.asarray(Hk, dtype=float)
    ck = np.asarray(ck, dtype=float)

    if param_mode == "natural":
        omega, Dtheta, R = param
    elif param_mode == "log":
        omega, lnDtheta, lnR = param
        Dtheta = np.exp(lnDtheta)
        R = np.exp(lnR)
    else:
        raise ValueError(f"param_mode must be 'natural' or 'log'; got {param_mode!r}")

    N = yk.shape[0]
    if N == 0:
        return 0.0

    dt = (tk[1] - tk[0]) if N > 1 else 0.0

    if validate:
        if yk.ndim != 2 or yk.shape[1] != 2:
            raise ValueError(f"yk must have shape (N,2); got {yk.shape}")
        if tk.shape != (N,):
            raise ValueError(f"tk must have shape (N,); got {tk.shape}")
        if Hk.shape != (N, 2):
            raise ValueError(f"Hk must have shape (N,2); got {Hk.shape}")
        if ck.shape != (N, 2):
            raise ValueError(f"ck must have shape (N,2); got {ck.shape}")
        if N > 2 and not np.allclose(np.diff(tk), dt):
            raise ValueError("tk must be uniformly spaced (constant dt).")
        if sigma_theta <= 0:
            raise ValueError("sigma_theta must be > 0.")
        if dt <= 0 and N > 1:
            raise ValueError("dt must be > 0 for N>1.")
        if Dtheta <= 0:
            raise ValueError("Dtheta must be > 0.")
        if R <= 0:
            raise ValueError("R must be > 0.")

    Q = 2.0 * Dtheta * dt
    wt = omega * dt
    log2pi = np.log(2.0 * np.pi)

    theta_pred = float(mu_theta)
    P_pred = float(sigma_theta) ** 2
    nll = 0.0

    for k in range(N):
        H = Hk[k].reshape(2, 1)  # 2x1
        y_pred = ck[k] + Hk[k] * theta_pred
        innov = (yk[k] - y_pred).reshape(2, 1)

        S = P_pred * (H @ H.T) + R * np.eye(2)

        try:
            sign, logdetS = np.linalg.slogdet(S)
            if sign <= 0:
                return np.inf
            Sinv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return np.inf

        quad = float((innov.T @ Sinv @ innov).squeeze())
        nll += 0.5 * (2.0 * log2pi + logdetS + quad)

        # Kalman update for frozen linear model
        K = (P_pred * (H.T @ Sinv)).reshape(2,)
        theta_filt = theta_pred + float(K @ innov.ravel())

        KH = float(K @ Hk[k])
        P_filt = (1.0 - KH) * P_pred * (1.0 - KH) + R * float(K @ K)

        if k < N - 1:
            theta_pred = theta_filt + wt
            P_pred = P_filt + Q

    return float(nll)


def neg_log_posterior_frozen(
    param,
    data,
    priors,
    Hk,
    ck,
    *,
    param_mode="log",
    validate=False,
):
    """
    Negative log posterior for the frozen linearized model:

        nlp = -log p(y_{1:N} | omega, Dtheta, R ; Hk, ck)
              -log p(omega, Dtheta, R)

    using the same prior convention as `neg_log_prior(...)`.

    Parameters
    ----------
    param : tuple
        If param_mode == "log":
            (omega, lnDtheta, lnR)
        If param_mode == "natural":
            (omega, Dtheta, R)

    data : tuple
        (tk, yk)

    priors : tuple
        (mu_theta, sigma_theta, mu_omega, sigma_omega, alpha_D, beta_D, alpha_R, beta_R)

    Hk : ndarray, shape (N, 2)
        Frozen Jacobians.

    ck : ndarray, shape (N, 2)
        Frozen affine offsets.

    param_mode : {"log", "natural"}, default "log"

    validate : bool, optional

    Returns
    -------
    nlp : float
        Negative log posterior for the frozen linear-Gaussian surrogate model.
    """
    mu_theta, sigma_theta, mu_omega, sigma_omega, alpha_D, beta_D, alpha_R, beta_R = priors
    state_priors = (mu_theta, sigma_theta)
    param_priors = (mu_omega, sigma_omega, alpha_D, beta_D, alpha_R, beta_R)

    nll = neg_log_marginal_likelihood_frozen(
        param,
        data,
        state_priors,
        Hk,
        ck,
        param_mode=param_mode,
        validate=validate,
    )

    nlp = neg_log_prior(
        param,
        param_priors,
        param_mode=param_mode,
        validate=validate,
    )

    return nll + nlp


def freeze_linearization(param, data, state_priors, *, param_mode="log", validate=False):
    """
    Run the EKF once at a reference parameter estimate and return the frozen
    linearization (Hk, ck) for downstream use.


    Returns
    -------
    Hk : ndarray, shape (N, 2)
    ck : ndarray, shape (N, 2)
    kalman_phases : tuple
        Full output of kalman_filter(...)
    """
    if param_mode == "log":
        omega, lnDtheta, lnR = param
        param_natural = (omega, np.exp(lnDtheta), np.exp(lnR))
    elif param_mode == "natural":
        param_natural = param
    else:
        raise ValueError(f"param_mode must be 'natural' or 'log'; got {param_mode!r}")


    kalman_phases = kalman_filter(param_natural, data, state_priors, validate=validate)
    _, _, _, _, _, Hk, ck = kalman_phases
    return Hk, ck, kalman_phases


def grad_neg_log_posterior_frozen(
    param: Tuple[float, float, float],
    data,
    priors,
    Hk,
    ck,
    *,
    param_mode: Literal["log", "natural"] = "log",
    validate: bool = False,
    return_aux: bool = False,
):
    """
    Gradient of the *negative* log posterior for the FROZEN linearized model
    associated with Example 1b.

    This function is intended to match finite differences of
    `neg_log_posterior_frozen(...)`, not the fully relinearized EKF posterior.

    Frozen linearized model
    -----------------------
        θ_1 ~ N(μ_θ, σ_θ^2)

        θ_k = θ_{k-1} + ω Δt + ε_{k-1},
        ε_{k-1} ~ N(0, Q),   Q = 2 Dθ Δt,                      k=2..N

        y_k = c_k + H_k θ_k + η_k,
        η_k ~ N(0, R I_2),                                    k=1..N

    where H_k and c_k are held fixed.

    Because ||H_k|| = 1 for this model, the latent-state posterior depends on
    the scalar pseudo-observations
        z_k = H_k^T (y_k - c_k) = θ_k + ξ_k,   ξ_k ~ N(0, R),
    so the smoothing distribution can be computed with a scalar Kalman filter
    and RTS smoother.

    Coordinates / parameterization
    ------------------------------
    If param_mode == "log":
        param = (omega, lnDtheta, lnR), with Dtheta = exp(lnDtheta), R = exp(lnR)
        Returns: ∇_{(omega, lnDtheta, lnR)} nlp_log, where nlp_log is the
        negative log posterior for the transformed density
            p(omega, ln Dtheta, ln R | y; Hk, ck).

    If param_mode == "natural":
        param = (omega, Dtheta, R)
        Returns: ∇_{(omega, Dtheta, R)} nlp_nat, where nlp_nat is the
        negative log posterior for the density
            p(omega, Dtheta, R | y; Hk, ck).

    Important convention
    --------------------
    The "log" and "natural" objectives differ by the Jacobian of the
    transformation (Dtheta, R) <-> (lnDtheta, lnR):
        nlp_log = nlp_nat - lnDtheta - lnR.

    Consequently, conversion from log-coordinate gradients to natural-coordinate
    gradients includes both the chain rule and the Jacobian correction:
        ∂ nlp_nat / ∂ Dtheta = (∂ nlp_log / ∂ lnDtheta + 1) / Dtheta
        ∂ nlp_nat / ∂ R      = (∂ nlp_log / ∂ lnR      + 1) / R

    Priors
    ------
        omega  ~ Normal(mu_omega, sigma_omega^2)
        Dtheta ~ Inverse-Gamma(alpha_D, beta_D)
        R      ~ Inverse-Gamma(alpha_R, beta_R)

    Assumed layout:
        priors = (
            mu_theta, sigma_theta,
            mu_omega, sigma_omega,
            alpha_D, beta_D,
            alpha_R, beta_R
        )

    Parameters
    ----------
    Hk : ndarray, shape (N, 2)
        Frozen Jacobians from the EKF pass at a reference parameter estimate.

    ck : ndarray, shape (N, 2)
        Frozen affine offsets from the EKF pass at a reference parameter estimate.

    Returns
    -------
    grad : ndarray, shape (3,)
        Gradient of the negative log posterior in the requested coordinates.

    aux : dict, optional
        Returned if return_aux=True.
    """
    tk, yk = data
    tk = np.asarray(tk, dtype=float).ravel()
    yk = np.asarray(yk, dtype=float)
    Hk = np.asarray(Hk, dtype=float)
    ck = np.asarray(ck, dtype=float)

    if yk.ndim != 2 or yk.shape[1] != 2:
        raise ValueError(f"yk must have shape (N, 2); got {yk.shape}")

    N = yk.shape[0]
    if N < 2:
        raise ValueError("Need N >= 2 for the increment statistics in the Louis score.")

    if Hk.shape != (N, 2):
        raise ValueError(f"Hk must have shape (N, 2); got {Hk.shape}")
    if ck.shape != (N, 2):
        raise ValueError(f"ck must have shape (N, 2); got {ck.shape}")

    # unpack priors
    mu_theta, sigma_theta, mu_omega, sigma_omega, alpha_D, beta_D, alpha_R, beta_R = priors
    state_priors = (mu_theta, sigma_theta)

    # interpret parameters
    if param_mode == "log":
        omega, lnDtheta, lnR = param
        Dtheta = float(np.exp(lnDtheta))
        R = float(np.exp(lnR))
    elif param_mode == "natural":
        omega, Dtheta, R = param
        Dtheta = float(Dtheta)
        R = float(R)
        if validate and (Dtheta <= 0 or R <= 0):
            raise ValueError("For param_mode='natural', require Dtheta > 0 and R > 0.")
        lnDtheta = float(np.log(Dtheta))
        lnR = float(np.log(R))
    else:
        raise ValueError(f"param_mode must be 'log' or 'natural'; got {param_mode!r}")

    dt = float(tk[1] - tk[0])

    if validate:
        if tk.shape != (N,):
            raise ValueError(f"tk must have shape ({N},); got {tk.shape}")
        if N > 2 and not np.allclose(np.diff(tk), dt):
            raise ValueError("tk must be uniformly spaced (constant dt).")
        if dt <= 0:
            raise ValueError("dt must be > 0.")
        if sigma_theta < 0:
            raise ValueError("sigma_theta must be >= 0.")
        if sigma_omega <= 0:
            raise ValueError("sigma_omega must be > 0.")
        if alpha_D <= 0 or beta_D <= 0 or alpha_R <= 0 or beta_R <= 0:
            raise ValueError("Inverse-gamma hyperparameters must be > 0.")
        if Dtheta <= 0 or R <= 0:
            raise ValueError("Require Dtheta > 0 and R > 0.")
        if not np.allclose(np.sum(Hk**2, axis=1), 1.0, atol=1e-8, rtol=1e-8):
            raise ValueError("Each frozen Hk row should have unit norm for this model.")

    # ------------------------------------------------------------------
    # 1) Construct scalar pseudo-observations for the frozen linear model
    # ------------------------------------------------------------------
    # z_k = H_k^T (y_k - c_k) = theta_k + scalar Gaussian noise with variance R
    a = yk - ck
    zk = np.sum(Hk * a, axis=1)

    # also keep orthogonal residual part for ESy
    # ||a_k - H_k theta||^2 = (z_k - theta)^2 + ||a_k||^2 - z_k^2
    orth_sq = np.sum(a**2, axis=1) - zk**2

    # ------------------------------------------------------------------
    # 2) Scalar Kalman filter for z_k = theta_k + noise
    # ------------------------------------------------------------------
    Q = 2.0 * Dtheta * dt
    wt = omega * dt

    thetakk = np.empty(N, dtype=float)
    Pkk = np.empty(N, dtype=float)
    thetakk_1 = np.empty(N, dtype=float)
    Pkk_1 = np.empty(N, dtype=float)

    # k = 1
    thetakk_1[0] = float(mu_theta)
    Pkk_1[0] = float(sigma_theta) ** 2

    S0 = Pkk_1[0] + R
    K0 = Pkk_1[0] / S0
    thetakk[0] = thetakk_1[0] + K0 * (zk[0] - thetakk_1[0])
    Pkk[0] = (1.0 - K0) * Pkk_1[0]

    for k in range(1, N):
        thetakk_1[k] = thetakk[k - 1] + wt
        Pkk_1[k] = Pkk[k - 1] + Q

        Sk = Pkk_1[k] + R
        Kk = Pkk_1[k] / Sk
        thetakk[k] = thetakk_1[k] + Kk * (zk[k] - thetakk_1[k])
        Pkk[k] = (1.0 - Kk) * Pkk_1[k]

    # ------------------------------------------------------------------
    # 3) RTS smoother for the scalar frozen model
    # ------------------------------------------------------------------
    thetakN = np.empty(N, dtype=float)
    PkN = np.empty(N, dtype=float)

    thetakN[-1] = thetakk[-1]
    PkN[-1] = Pkk[-1]

    for k in range(N - 2, -1, -1):
        A = Pkk[k] / Pkk_1[k + 1]
        thetakN[k] = thetakk[k] + A * (thetakN[k + 1] - thetakk_1[k + 1])
        PkN[k] = Pkk[k] + A**2 * (PkN[k + 1] - Pkk_1[k + 1])

    # ------------------------------------------------------------------
    # 4) Lag-one covariance from full posterior covariance matrix
    #    (valid here because the frozen model reduces to Example 1a)
    # ------------------------------------------------------------------
    Sigma = theta_covariance((omega, Dtheta, R), N, dt, state_priors, validate=validate)
    Sigmakk_1 = np.zeros(N, dtype=float)
    if N > 1:
        Sigmakk_1[1:] = np.diag(Sigma, k=-1)

    # ------------------------------------------------------------------
    # 5) Expected sufficient statistics under p(theta | y, lambda; Hk, ck)
    # ------------------------------------------------------------------
    dtheta_mean = thetakN[1:] - thetakN[:-1]
    ES0 = np.sum(dtheta_mean**2 + PkN[1:] + PkN[:-1] - 2.0 * Sigmakk_1[1:])

    ES1 = float(thetakN[-1] - thetakN[0])

    # Full 2D measurement statistic for the frozen model
    #   ESy = sum_k E[ || y_k - c_k - H_k theta_k ||^2 ]
    #       = sum_k [ E[(z_k - theta_k)^2] + orth_sq_k ]
    ESy = float(np.sum((zk - thetakN) ** 2 + PkN + orth_sq))

    wdt = omega * dt
    M = N - 1

    # ------------------------------------------------------------------
    # 6) Score of the transformed log posterior in log-coordinates
    # ------------------------------------------------------------------
    score_log = np.zeros(3, dtype=float)

    score_log[0] = (
        -(omega - mu_omega) / (sigma_omega**2)
        + (ES1 - M * wdt) / (2.0 * Dtheta)
    )

    score_log[1] = (
        -alpha_D + beta_D / Dtheta
        - 0.5 * M
        + (ES0 - 2.0 * ES1 * wdt + M * (wdt**2)) / (4.0 * Dtheta * dt)
    )

    score_log[2] = (
        -alpha_R + beta_R / R
        - N
        + ESy / (2.0 * R)
    )

    grad_log = -score_log

    # ------------------------------------------------------------------
    # 7) Convert to requested coordinates
    # ------------------------------------------------------------------
    if param_mode == "log":
        grad = grad_log
    else:
        grad = np.zeros(3, dtype=float)
        grad[0] = grad_log[0]
        grad[1] = (grad_log[1] + 1.0) / Dtheta
        grad[2] = (grad_log[2] + 1.0) / R

    if not return_aux:
        return grad

    aux = dict(
        dt=dt,
        zk=zk,
        orth_sq=orth_sq,
        filtered_mean=thetakk,
        filtered_var=Pkk,
        predicted_mean=thetakk_1,
        predicted_var=Pkk_1,
        smoothed_mean=thetakN,
        smoothed_var=PkN,
        smoothed_lag1_cov=Sigmakk_1,
        Sigma=Sigma,
        Hk=Hk,
        ck=ck,
        expected_stats=dict(S0=ES0, S1=ES1, Sy=ESy),
        score_log=score_log,
        grad_log=grad_log,
        grad=grad,
        param_mode=param_mode,
        param_natural=(omega, Dtheta, R),
        param_log=(omega, lnDtheta, lnR),
    )
    return grad, aux


def em_update(
    old_param,
    data,
    priors,
    param_mode: Literal["log", "natural"] = "log",
    *,
    validate: bool = False,
    return_aux: bool = False,
):
    """
    Perform one generalized-EM update for Example 1b using a frozen EKF linearization.

    At the current parameter estimate, this routine:
      1) runs the EKF to obtain frozen linearization terms H_k and c_k,
      2) converts the 2D frozen model to a scalar pseudo-measurement model,
      3) runs scalar Kalman smoothing,
      4) performs closed-form MAP updates for omega, Dtheta, and R.

    Frozen linearized measurement model
    -----------------------------------
        y_k ≈ c_k + H_k θ_k + η_k,     η_k ~ N(0, R I_2)

    Because ||H_k|| = 1 for this model, define the scalar pseudo-observation
        z_k = H_k^T (y_k - c_k) = θ_k + ξ_k,     ξ_k ~ N(0, R),
    which yields the same latent-state posterior as the frozen 2D model.

    Parameters
    ----------
    old_param : array_like
        Current parameter estimate.

        If param_mode == "natural":
            [omega, Dtheta, R]

        If param_mode == "log":
            [omega, lnDtheta, lnR]

    data : tuple
        (tk, yk)
        tk : time points, shape (N,)
        yk : nonlinear 2D observations, shape (N, 2)

    priors : tuple
        (mu_theta, sigma_theta,
         mu_omega, sigma_omega,
         alpha_D, beta_D,
         alpha_R, beta_R)

    param_mode : {"log", "natural"}, default="log"
        Determines parameterization of input/output.

    validate : bool, default False
        Run basic sanity checks.

    return_aux : bool, default False
        If True, also return a dictionary of intermediate quantities.

    Returns
    -------
    new_param : list
        Updated parameters in the same parameterization as `param_mode`.

    aux : dict, optional
        Returned only if return_aux=True.
    """
    tk, yk = data
    tk = np.asarray(tk, dtype=float).ravel()
    yk = np.asarray(yk, dtype=float)

    if yk.ndim != 2 or yk.shape[1] != 2:
        raise ValueError(f"yk must have shape (N, 2); got {yk.shape}")

    # --------------------------------------------------
    # Unpack priors
    # --------------------------------------------------
    (
        mu_theta, sigma_theta,
        mu_omega, sigma_omega,
        alpha_D, beta_D,
        alpha_R, beta_R
    ) = priors

    state_priors = (mu_theta, sigma_theta)

    # --------------------------------------------------
    # Convert parameters to natural scale if needed
    # --------------------------------------------------
    if param_mode == "natural":
        omega, Dtheta, R = old_param
    elif param_mode == "log":
        omega, lnDtheta, lnR = old_param
        Dtheta = float(np.exp(lnDtheta))
        R = float(np.exp(lnR))
    else:
        raise ValueError("param_mode must be 'natural' or 'log'")

    N = yk.shape[0]
    if N < 2:
        raise ValueError("Need N >= 2 for EM updates of drift/diffusion parameters.")

    dt = float(tk[1] - tk[0])

    if validate:
        if tk.shape != (N,):
            raise ValueError(f"tk must have shape ({N},); got {tk.shape}")
        if N > 2 and not np.allclose(np.diff(tk), dt):
            raise ValueError("tk must be uniformly spaced (constant dt).")
        if dt <= 0:
            raise ValueError("dt must be > 0.")
        if sigma_theta < 0:
            raise ValueError("sigma_theta must be >= 0.")
        if sigma_omega <= 0:
            raise ValueError("sigma_omega must be > 0.")
        if alpha_D <= 0 or beta_D <= 0 or alpha_R <= 0 or beta_R <= 0:
            raise ValueError("Inverse-gamma hyperparameters must be > 0.")
        if Dtheta <= 0 or R <= 0:
            raise ValueError("Require Dtheta > 0 and R > 0.")

    # --------------------------------------------------
    # E-step part 1: freeze nonlinear measurement model
    # --------------------------------------------------
    kalman_phases = kalman_filter((omega, Dtheta, R), (tk, yk), state_priors, validate=validate)
    _, _, _, _, _, Hk, ck = kalman_phases

    if validate:
        hn = np.sum(Hk**2, axis=1)
        if not np.allclose(hn, 1.0, atol=1e-8, rtol=1e-8):
            raise ValueError("Each Hk row should have unit norm for this model.")

    # Pseudo-measurements z_k = H_k^T (y_k - c_k)
    a = yk - ck
    zk = np.sum(Hk * a, axis=1)

    # Orthogonal residual energy:
    # ||a_k - H_k theta||^2 = (z_k - theta)^2 + (||a_k||^2 - z_k^2)
    orth_sq = np.sum(a**2, axis=1) - zk**2

    # --------------------------------------------------
    # E-step part 2: scalar Kalman filter + RTS smoother on z_k
    # --------------------------------------------------
    Q = 2.0 * Dtheta * dt
    wt = omega * dt

    thetakk = np.empty(N, dtype=float)
    Pkk = np.empty(N, dtype=float)
    thetakk_1 = np.empty(N, dtype=float)
    Pkk_1 = np.empty(N, dtype=float)

    # k = 1
    thetakk_1[0] = float(mu_theta)
    Pkk_1[0] = float(sigma_theta) ** 2

    S = Pkk_1[0] + R
    K = Pkk_1[0] / S
    thetakk[0] = thetakk_1[0] + K * (zk[0] - thetakk_1[0])
    Pkk[0] = (1.0 - K) * Pkk_1[0]

    # k = 2..N
    for k in range(1, N):
        thetakk_1[k] = thetakk[k - 1] + wt
        Pkk_1[k] = Pkk[k - 1] + Q

        S = Pkk_1[k] + R
        K = Pkk_1[k] / S
        thetakk[k] = thetakk_1[k] + K * (zk[k] - thetakk_1[k])
        Pkk[k] = (1.0 - K) * Pkk_1[k]

    # RTS smoother
    thetakN = np.empty(N, dtype=float)
    PkN = np.empty(N, dtype=float)

    thetakN[-1] = thetakk[-1]
    PkN[-1] = Pkk[-1]

    for k in range(N - 2, -1, -1):
        A = Pkk[k] / Pkk_1[k + 1]
        thetakN[k] = thetakk[k] + A * (thetakN[k + 1] - thetakk_1[k + 1])
        PkN[k] = Pkk[k] + A**2 * (PkN[k + 1] - Pkk_1[k + 1])

    # Lag-one covariance from full covariance matrix of the scalar frozen model
    Sigma = theta_covariance((omega, Dtheta, R), N, dt, state_priors, validate=validate)
    Sigmakk_1 = np.zeros(N, dtype=float)
    Sigmakk_1[1:] = np.diag(Sigma, k=-1)

    # --------------------------------------------------
    # Compute sufficient statistics
    # --------------------------------------------------
    M = N - 1

    ES0 = np.sum(
        (thetakN[1:] - thetakN[:-1])**2
        + PkN[1:] + PkN[:-1] - 2.0 * Sigmakk_1[1:]
    )

    ES1 = float(thetakN[-1] - thetakN[0])

    # 2D frozen-model measurement residual energy
    ESy = float(np.sum((zk - thetakN)**2 + PkN + orth_sq))

    # --------------------------------------------------
    # M-step updates (MAP under conjugate priors)
    # --------------------------------------------------

    # ---- Update omega (Gaussian prior) ----
    omega_new = (
        ES1 / (2.0 * Dtheta) + mu_omega / sigma_omega**2
    ) / (
        M * dt / (2.0 * Dtheta) + 1.0 / sigma_omega**2
    )

    # ---- Update Dtheta (Inverse-Gamma prior) ----
    wdt = omega_new * dt
    B = ES0 - 2.0 * ES1 * wdt + M * (wdt**2)

    if param_mode == "natural":
        Dtheta_new = (B + 4.0 * beta_D * dt) / (2.0 * (M + 2.0 * (alpha_D + 1.0)) * dt)

        # 2D isotropic measurement model:
        # natural-space MAP update
        R_new = (ESy + 2.0 * beta_R) / (2.0 * (N + alpha_R + 1.0))

        new_param = [omega_new, Dtheta_new, R_new]

    else:
        # log-parameterization:
        # Jacobian shifts alpha by 1 for each positive scale parameter
        Dtheta_new = (B + 4.0 * beta_D * dt) / (2.0 * (M + 2.0 * alpha_D) * dt)

        # 2D isotropic measurement model in log-space
        R_new = (ESy + 2.0 * beta_R) / (2.0 * (N + alpha_R))

        new_param = [omega_new, np.log(Dtheta_new), np.log(R_new)]

    if not return_aux:
        return new_param

    aux = dict(
        Hk=Hk,
        ck=ck,
        zk=zk,
        orth_sq=orth_sq,
        filtered_mean=thetakk,
        filtered_var=Pkk,
        predicted_mean=thetakk_1,
        predicted_var=Pkk_1,
        smoothed_mean=thetakN,
        smoothed_var=PkN,
        smoothed_lag1_cov=Sigmakk_1,
        expected_stats=dict(S0=ES0, S1=ES1, Sy=ESy),
        param_old_natural=(omega, Dtheta, R),
        param_new=new_param,
    )
    return new_param, aux


def em_map_parameters(
    initial_params,
    data,
    priors,
    max_iter: int = 500,
    tol: float = 1e-5,
    verbose: bool = True,
    print_every: int = 10,
    param_mode: Literal["log", "natural"] = "log",
    *,
    validate: bool = False,
):
    """
    Estimate MAP parameters for Example 1b using generalized EM with
    frozen EKF linearization.

    Each iteration:
      1) linearizes the nonlinear measurement model at the current parameter,
      2) freezes that linearization,
      3) performs a closed-form EM/MAP update for the resulting linear-Gaussian surrogate.

    This is a generalized EM scheme for the nonlinear model, not exact EM.

    Parameters
    ----------
    initial_params : array_like, shape (3,)
        Initial guess.

        natural mode:
            [omega, Dtheta, R]

        log mode:
            [omega, lnDtheta, lnR]

    data : tuple
        (tk, yk) with yk shape (N, 2)

    priors : tuple
        Prior hyperparameters in standard ordering.

    max_iter : int
        Maximum EM iterations.

    tol : float
        Convergence tolerance (max relative change).

    verbose : bool
        Print progress if True.

    print_every : int
        Print every `print_every` iterations.

    param_mode : {"log", "natural"}
        Parameterization of iterates and return value.

    validate : bool, default False
        If True, validate inputs during iterations.

    Returns
    -------
    lambda_iter : ndarray, shape (K,3)
        Parameter trajectory, truncated at convergence.
    """
    lambda_iter = np.zeros((max_iter, 3), dtype=float)
    lambda_iter[0, :] = np.asarray(initial_params, dtype=float)

    if verbose:
        if param_mode == "natural":
            print("iter\tomega\t\tDtheta\t\tR")
        else:
            print("iter\tomega\t\tlnDtheta\tlnR")

    converged = False
    rel_change = np.nan

    for i in range(1, max_iter):
        lambda_iter[i] = em_update(
            lambda_iter[i - 1],
            data,
            priors,
            param_mode=param_mode,
            validate=validate,
            return_aux=False,
        )

        denom = np.maximum(np.abs(lambda_iter[i - 1]), 1e-12)
        rel_change = np.max(np.abs(lambda_iter[i] - lambda_iter[i - 1]) / denom)

        if verbose and (i % print_every == 0 or i == 1):
            print(
                "%d\t%.5e\t%.5e\t%.5e"
                % (
                    i,
                    lambda_iter[i, 0],
                    lambda_iter[i, 1],
                    lambda_iter[i, 2],
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
            f"Generalized-EM MAP inference did not converge within {max_iter} iterations "
            f"(last relative change = {rel_change:.2e}).",
            RuntimeWarning,
        )

    return lambda_iter[: i + 1]


def prior_information(lambda_map, priors, param_type="natural"):
    """
    Prior information matrix defined as the negative Hessian of the log prior,
    evaluated at lambda_map.

    Parameters
    ----------
    lambda_map : tuple
        If param_type == "natural": (omega, Dtheta, R)
        If param_type == "log":     (omega, logDtheta, logR)

    param_priors : tuple
        (mu_omega, sigma_omega, alpha_D, beta_D, alpha_R, beta_R)

    param_type : str
        "natural" or "log"

    Notes
    -----
    - Omega prior: omega ~ N(mu_omega, sigma_omega^2)
      => -d^2/domega^2 log p(omega) = 1/sigma_omega^2

    - Dtheta, R priors (inverse-gamma in natural space):
        log p(D) = -(alpha+1) log D - beta/D + const
      Natural-space curvature:
        -d^2/dD^2 log p(D) = -(alpha+1)/D^2 + 2 beta / D^3

      Log-space curvature uses the Jacobian-adjusted density in phi=log D:
        log p(phi) = log p(e^phi) + phi = -alpha phi - beta e^{-phi} + const
        -d^2/dphi^2 log p(phi) = beta e^{-phi} = beta / D
    """
    _, _, mu_omega, sigma_omega, alpha_D, beta_D, alpha_R, beta_R = priors

    I_prior = np.zeros((3, 3), dtype=float)
    I_prior[0, 0] = 1.0 / (sigma_omega ** 2)

    if param_type == "natural":
        omega, Dtheta, R = lambda_map

        I_prior[1, 1] = -(alpha_D + 1.0) / (Dtheta ** 2) + (2.0 * beta_D) / (Dtheta ** 3)
        I_prior[2, 2] = -(alpha_R + 1.0) / (R ** 2)     + (2.0 * beta_R) / (R ** 3)

    elif param_type == "log":
        omega, logDtheta, logR = lambda_map
        Dtheta = np.exp(logDtheta)
        R = np.exp(logR)

        I_prior[1, 1] = beta_D / Dtheta
        I_prior[2, 2] = beta_R / R

    else:
        raise ValueError("param_type must be 'natural' or 'log'")

    return I_prior


def smoothing_statistics_frozen(param_eval, data, priors, Hk, ck, param_type="log", *, validate=False):
    """
    Run smoothing for the frozen linearized model and return reusable statistics
    for information / uncertainty calculations.

    This function evaluates the frozen surrogate model at `param_eval`, while
    keeping the linearization terms Hk and ck fixed (typically obtained from
    `freeze_linearization(...)` at a different parameter value).

    Frozen linearized model
    -----------------------
        y_k ≈ c_k + H_k theta_k + eta_k,    eta_k ~ N(0, R I_2)

    With ||H_k|| = 1, define scalar pseudo-observations
        z_k = H_k^T (y_k - c_k) = theta_k + xi_k,   xi_k ~ N(0, R)

    Parameters
    ----------
    param_eval : array_like, shape (3,)
        Parameter value at which to evaluate the frozen model.

        If param_type == "natural":
            (omega, Dtheta, R)

        If param_type == "log":
            (omega, lnDtheta, lnR)

    data : tuple
        (tk, yk), with yk shape (N, 2)

    priors : tuple
        (mu_theta, sigma_theta,
         mu_omega, sigma_omega,
         alpha_D, beta_D,
         alpha_R, beta_R)

    Hk : ndarray, shape (N, 2)
        Frozen Jacobians.

    ck : ndarray, shape (N, 2)
        Frozen affine offsets.

    param_type : {"log", "natural"}, default "log"
        Interpretation of `param_eval`.

    validate : bool, default False
        If True, run basic checks.

    Returns
    -------
    stats : dict
        Reusable smoothing statistics for the frozen model evaluated at
        `param_eval`.
    """
    tk, yk = data
    tk = np.asarray(tk, dtype=float).ravel()
    yk = np.asarray(yk, dtype=float)
    Hk = np.asarray(Hk, dtype=float)
    ck = np.asarray(ck, dtype=float)

    if yk.ndim != 2 or yk.shape[1] != 2:
        raise ValueError(f"yk must have shape (N, 2); got {yk.shape}")

    N = yk.shape[0]
    dt = float(tk[1] - tk[0])

    if Hk.shape != (N, 2):
        raise ValueError(f"Hk must have shape (N, 2); got {Hk.shape}")
    if ck.shape != (N, 2):
        raise ValueError(f"ck must have shape (N, 2); got {ck.shape}")

    (
        mu_theta, sigma_theta,
        mu_omega, sigma_omega,
        alpha_D, beta_D,
        alpha_R, beta_R
    ) = priors

    # ------------------------------------------------------------
    # Map input -> natural parameters
    # ------------------------------------------------------------
    if param_type == "natural":
        omega, Dtheta, R = param_eval
        lnDtheta = float(np.log(Dtheta))
        lnR = float(np.log(R))
    elif param_type == "log":
        omega, lnDtheta, lnR = param_eval
        Dtheta = float(np.exp(lnDtheta))
        R = float(np.exp(lnR))
    else:
        raise ValueError("param_type must be 'natural' or 'log'")

    lambda_nat = (omega, Dtheta, R)
    lambda_log = (omega, lnDtheta, lnR)

    if validate:
        if tk.shape != (N,):
            raise ValueError(f"tk must have shape ({N},); got {tk.shape}")
        if N > 2 and not np.allclose(np.diff(tk), dt):
            raise ValueError("tk must be uniformly spaced.")
        if dt <= 0:
            raise ValueError("dt must be > 0.")
        if Dtheta <= 0 or R <= 0:
            raise ValueError("Require Dtheta > 0 and R > 0.")
        if sigma_theta < 0:
            raise ValueError("sigma_theta must be >= 0.")
        hn = np.sum(Hk**2, axis=1)
        if not np.allclose(hn, 1.0, atol=1e-8, rtol=1e-8):
            raise ValueError("Each Hk row should have unit norm for this model.")

    # ------------------------------------------------------------
    # Scalar pseudo-observations for frozen model
    # ------------------------------------------------------------
    a = yk - ck
    zk = np.sum(Hk * a, axis=1)

    # ||a_k - H_k theta||^2 = (z_k - theta)^2 + orth_sq_k
    orth_sq = np.sum(a**2, axis=1) - zk**2

    # ------------------------------------------------------------
    # Scalar KF for z_k = theta_k + noise
    # ------------------------------------------------------------
    Q = 2.0 * Dtheta * dt
    wt = omega * dt

    thetakk = np.empty(N, dtype=float)
    Pkk = np.empty(N, dtype=float)
    thetakk_1 = np.empty(N, dtype=float)
    Pkk_1 = np.empty(N, dtype=float)

    thetakk_1[0] = float(mu_theta)
    Pkk_1[0] = float(sigma_theta) ** 2

    S = Pkk_1[0] + R
    K = Pkk_1[0] / S
    thetakk[0] = thetakk_1[0] + K * (zk[0] - thetakk_1[0])
    Pkk[0] = (1.0 - K) * Pkk_1[0]

    for k in range(1, N):
        thetakk_1[k] = thetakk[k - 1] + wt
        Pkk_1[k] = Pkk[k - 1] + Q

        S = Pkk_1[k] + R
        K = Pkk_1[k] / S
        thetakk[k] = thetakk_1[k] + K * (zk[k] - thetakk_1[k])
        Pkk[k] = (1.0 - K) * Pkk_1[k]

    # ------------------------------------------------------------
    # RTS smoother
    # ------------------------------------------------------------
    thetakN = np.empty(N, dtype=float)
    PkN = np.empty(N, dtype=float)

    thetakN[-1] = thetakk[-1]
    PkN[-1] = Pkk[-1]

    for k in range(N - 2, -1, -1):
        A = Pkk[k] / Pkk_1[k + 1]
        thetakN[k] = thetakk[k] + A * (thetakN[k + 1] - thetakk_1[k + 1])
        PkN[k] = Pkk[k] + A**2 * (PkN[k + 1] - Pkk_1[k + 1])

    # ------------------------------------------------------------
    # Lag-one covariance via full covariance matrix
    # ------------------------------------------------------------
    state_priors = (mu_theta, sigma_theta)
    Sigma = theta_covariance(lambda_nat, N, dt, state_priors, validate=validate)
    Sigmakk_1 = np.zeros(N, dtype=float)
    if N > 1:
        Sigmakk_1[1:] = np.diag(Sigma, k=-1)

    # ------------------------------------------------------------
    # Expected sufficient statistics
    # ------------------------------------------------------------
    ES0 = float(np.sum(
        (thetakN[1:] - thetakN[:-1])**2
        + PkN[1:] + PkN[:-1]
        - 2.0 * Sigmakk_1[1:]
    ))
    ES1 = float(thetakN[-1] - thetakN[0])
    ESy = float(np.sum((zk - thetakN)**2 + PkN + orth_sq))

    return {
        "tk": tk,
        "yk": yk,
        "dt": dt,
        "N": N,
        "zk": zk,
        "orth_sq": orth_sq,
        "m": thetakN,
        "PkN": PkN,
        "Sigmakk_1": Sigmakk_1,
        "Sigma": Sigma,
        "Hk": Hk,
        "ck": ck,
        "filtered_mean": thetakk,
        "filtered_var": Pkk,
        "predicted_mean": thetakk_1,
        "predicted_var": Pkk_1,
        "lambda_natural": lambda_nat,
        "lambda_log": lambda_log,
        "ES0": ES0,
        "ES1": ES1,
        "ESy": ESy,
        "sigma_theta2": float(sigma_theta)**2,
    }


def complete_data_information_frozen(statistics, param_type="log"):
    """
    Complete-data information matrix for the frozen linearized model:

        I_comp = -E[ ∇^2 ell_c | y, lambda ]

    computed from frozen-model smoothing statistics.
    """
    omega, Dtheta, R = statistics["lambda_natural"]
    N = statistics["N"]
    dt = statistics["dt"]
    ES0 = statistics["ES0"]
    ES1 = statistics["ES1"]
    ESy = statistics["ESy"]

    M = N - 1
    wdt = omega * dt
    EA = ES0 - 2.0 * ES1 * wdt + M * (wdt ** 2)

    I_comp = np.zeros((3, 3), dtype=float)

    if param_type == "log":
        I_comp[0, 0] = M * dt / (2.0 * Dtheta)
        I_comp[0, 1] = (ES1 - M * wdt) / (2.0 * Dtheta)
        I_comp[1, 1] = EA / (4.0 * Dtheta * dt)
        I_comp[2, 2] = ESy / (2.0 * R)
        I_comp[1, 0] = I_comp[0, 1]

    elif param_type == "natural":
        I_comp[0, 0] = M * dt / (2.0 * Dtheta)
        I_comp[0, 1] = (ES1 - M * wdt) / (2.0 * Dtheta ** 2)
        I_comp[1, 1] = -M / (2.0 * Dtheta ** 2) + EA / (2.0 * Dtheta ** 3 * dt)
        I_comp[2, 2] = -N / (R ** 2) + ESy / (R ** 3)
        I_comp[1, 0] = I_comp[0, 1]

    else:
        raise ValueError("param_type must be 'natural' or 'log'")

    return I_comp


def missing_information_frozen(statistics, priors=None, param_type="log") -> np.ndarray:
    """
    Exact missing information matrix for the frozen linearized model,

        I_miss = Cov(score_c | y, lambda),

    computed using banded solves and streaming trace identities.
    """
    z = np.asarray(statistics["zk"], dtype=float).ravel()
    m = np.asarray(statistics["m"], dtype=float).ravel()
    dt = float(statistics["dt"])
    N = int(statistics["N"])
    sigma_theta2 = float(statistics["sigma_theta2"])

    omega, D, R = statistics["lambda_natural"]

    if D <= 0 or R <= 0 or dt <= 0 or sigma_theta2 <= 0:
        raise ValueError("Require D>0, R>0, dt>0, sigma_theta2>0.")
    if N < 2:
        raise ValueError("Need N>=2.")
    if z.shape != (N,):
        raise ValueError(f"statistics['zk'] must have shape ({N},); got {z.shape}")
    if m.shape != (N,):
        raise ValueError(f"statistics['m'] must have shape ({N},); got {m.shape}")

    M = N - 1

    q = 2.0 * D * dt
    invR = 1.0 / R
    invq = 1.0 / q
    invsig = 1.0 / sigma_theta2

    d = np.empty(N, dtype=float)
    u = np.full(N - 1, -invq, dtype=float)
    l = np.full(N - 1, -invq, dtype=float)

    d[0] = invR + invq + invsig
    if N > 2:
        d[1:N - 1] = invR + 2.0 * invq
    d[N - 1] = invR + invq

    ab = np.zeros((3, N), dtype=float)
    ab[0, 1:] = u
    ab[1, :] = d
    ab[2, :-1] = l

    def apply_sigma(b: np.ndarray) -> np.ndarray:
        b = np.asarray(b, dtype=float).ravel()
        return solve_banded((1, 1), ab, b)

    def apply_A0(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float).ravel()
        out = np.empty_like(x)
        if N == 2:
            out[0] = x[0] - x[1]
            out[1] = -x[0] + x[1]
            return out
        out[0] = x[0] - x[1]
        out[1:N - 1] = -x[0:N - 2] + 2.0 * x[1:N - 1] - x[2:N]
        out[N - 1] = -x[N - 2] + x[N - 1]
        return out

    def apply_B(theta: np.ndarray) -> np.ndarray:
        theta = np.asarray(theta, dtype=float).ravel()
        return theta[1:] - theta[:-1]

    def apply_Bt(v: np.ndarray) -> np.ndarray:
        v = np.asarray(v, dtype=float).ravel()
        w = np.zeros(N, dtype=float)
        w[0] = -v[0]
        if N > 2:
            w[1:N - 1] = v[0:M - 1] - v[1:M]
        w[N - 1] = v[M - 1]
        return w

    def apply_sigma_u(v: np.ndarray) -> np.ndarray:
        return apply_B(apply_sigma(apply_Bt(v)))

    r = z - m
    c = np.zeros(N, dtype=float)
    c[0] = -1.0
    c[-1] = +1.0
    onesM = np.ones(M, dtype=float)
    mu_u = apply_B(m)

    Sig_c = apply_sigma(c)
    V_S1 = float(c @ Sig_c)

    Sig_r = apply_sigma(r)
    C_S1_Sy = float(-2.0 * (c @ Sig_r))

    SigU_mu = apply_sigma_u(mu_u)
    C_S1_S0 = float(2.0 * (onesM @ SigU_mu))
    mu_SigU_mu = float(mu_u @ SigU_mu)

    tr_Sig2 = 0.0
    tr_A0Sig2 = 0.0
    e = np.zeros(N, dtype=float)
    for i in range(N):
        e.fill(0.0)
        e[i] = 1.0
        x = solve_banded((1, 1), ab, e)
        tr_Sig2 += float(x @ x)
        tr_A0Sig2 += float(x @ apply_A0(x))

    tr_SigU2 = 0.0
    ej = np.zeros(M, dtype=float)
    for j in range(M):
        ej.fill(0.0)
        ej[j] = 1.0
        zvec = apply_sigma_u(ej)
        tr_SigU2 += float(zvec @ zvec)

    Var_S0 = 2.0 * tr_SigU2 + 4.0 * mu_SigU_mu
    r_Sig_r = float(r @ Sig_r)
    Var_Sy = 2.0 * tr_Sig2 + 4.0 * r_Sig_r

    Cov_S1_Q = C_S1_S0 - 2.0 * omega * dt * V_S1
    Var_Q = Var_S0 + 4.0 * (omega * dt) ** 2 * V_S1 - 4.0 * omega * dt * C_S1_S0

    Sig_m = apply_sigma(m)
    term_m_A0_Sig_m = float(m @ apply_A0(Sig_m))
    A0m = apply_A0(m)
    Sig_A0m = apply_sigma(A0m)
    term_z_Sig_A0m = float(z @ Sig_A0m)
    term_c_Sig_m = float(c @ Sig_m)
    Sig_z = apply_sigma(z)
    term_c_Sig_z = float(c @ Sig_z)

    Cov_Q_Sy = (
        2.0 * tr_A0Sig2
        + 4.0 * term_m_A0_Sig_m
        - 4.0 * term_z_Sig_A0m
        - 4.0 * omega * dt * term_c_Sig_m
        + 4.0 * omega * dt * term_c_Sig_z
    )

    I = np.zeros((3, 3), dtype=float)

    if param_type == "log":
        I[0, 0] = V_S1 / (4.0 * D * D)
        I[0, 1] = Cov_S1_Q / (8.0 * D * D * dt)
        I[1, 0] = I[0, 1]
        I[1, 1] = Var_Q / ((4.0 * D * dt) ** 2)
        I[2, 2] = Var_Sy / (4.0 * R * R)
        I[0, 2] = C_S1_Sy / (4.0 * D * R)
        I[2, 0] = I[0, 2]
        I[1, 2] = Cov_Q_Sy / (8.0 * D * R * dt)
        I[2, 1] = I[1, 2]

    elif param_type == "natural":
        I[0, 0] = V_S1 / (4.0 * D * D)
        I[0, 1] = Cov_S1_Q / (8.0 * D**3 * dt)
        I[1, 0] = I[0, 1]
        I[1, 1] = Var_Q / (16.0 * D**4 * dt**2)
        I[2, 2] = Var_Sy / (4.0 * R**4)
        I[0, 2] = C_S1_Sy / (4.0 * D * R**2)
        I[2, 0] = I[0, 2]
        I[1, 2] = Cov_Q_Sy / (8.0 * D**2 * R**2 * dt)
        I[2, 1] = I[1, 2]

    else:
        raise ValueError("param_type must be 'natural' or 'log'")

    return I


def posterior_information_frozen(param_eval, data, priors, Hk, ck, param_type="log"):
    """
    Compute Laplace posterior information for the frozen linearized model:

        I_obs = I_prior + I_comp - I_miss

    where Hk and ck are fixed externally, typically from `freeze_linearization`
    at the nonlinear MAP, and the information is evaluated at `param_eval`
    (typically the MAP of the frozen surrogate model).
    """
    stats = smoothing_statistics_frozen(
        param_eval, data, priors, Hk, ck, param_type=param_type
    )

    I_prior = prior_information(param_eval, priors, param_type=param_type)
    I_comp = complete_data_information_frozen(stats, param_type=param_type)
    I_miss = missing_information_frozen(stats, priors, param_type=param_type)

    I_obs = I_prior + I_comp - I_miss
    I_obs = 0.5 * (I_obs + I_obs.T)

    cov = np.linalg.inv(I_obs)

    A = I_prior + I_comp
    A = 0.5 * (A + A.T)

    M = I_miss @ np.linalg.inv(A)
    vals = np.linalg.eigvals(M)
    rho = float(np.max(np.real(vals)))
    em_rate = float(-np.log(rho)) if rho > 0 else np.inf

    return {
        "I_prior": I_prior,
        "I_comp": I_comp,
        "I_miss": I_miss,
        "I_obs": I_obs,
        "cov": cov,
        "rho": rho,
        "em_rate": em_rate,
        "stats": stats,
    }
    
def _to_log_params(param_natural):
    omega, D, R = param_natural
    return np.array([omega, np.log(D), np.log(R)], dtype=float)

def _to_natural_params(param_log):
    omega, lnD, lnR = param_log
    return np.array([omega, np.exp(lnD), np.exp(lnR)], dtype=float)


def simulate_and_infer(
    param_true_natural,
    N,
    dt,
    rng,
    priors,
    *,
    param_type: str = "log",
    nonlinear_inference: str = "optimize",
    nonlinear_method: str = "Powell",
    frozen_method: str = "L-BFGS-B",
    max_iter: int = 500,
    tol: float = 1e-6,
    return_stats: bool = False,
    initial_phase_method: str = "data_informed",
    initial_phase_window_size: int = 6,
    initial_phase_sigma_inflation: float = 2.0,
    initial_phase_R_strategy: str = "prior_mode",
):
    """
    Simulate data for the nonlinear 2D measurement model, infer MAP parameters
    for the nonlinear EKF posterior, then freeze the linearization and compute
    Laplace uncertainty using the MAP of the frozen surrogate posterior.

    IMPORTANT
    ---------
    `param_true_natural` is ALWAYS interpreted as natural parameters
        (omega, Dtheta, R),
    regardless of `param_type`.

    Workflow
    --------
    1. Simulate synthetic data from the nonlinear model.
    2. Optionally replace the nominal initial-phase prior (mu_theta, sigma_theta)
       with a data-informed prior estimated from the simulated observations.
    3. Infer nonlinear MAP parameters using one of two methods:
           a) direct numerical optimization of neg_log_posterior(...)
           b) generalized EM via em_map_parameters(...)
    4. Freeze the EKF linearization at the resulting nonlinear estimate:
           (Hk, ck) = freeze_linearization(lambda_map, ...)
    5. Optimize the frozen surrogate posterior:
           lambda_map_frozen = argmin neg_log_posterior_frozen(..., Hk, ck)
    6. Compute posterior information / Laplace covariance for the frozen model
       at lambda_map_frozen, while keeping Hk and ck fixed.

    Parameters
    ----------
    param_true_natural : array_like, shape (3,)
        True natural parameters (omega, Dtheta, R) used to simulate the data.
    N : int
        Number of observations.
    dt : float
        Sampling interval.
    rng : np.random.Generator
        Random number generator.
    priors : tuple
        Prior hyperparameters:
            (mu_theta, sigma_theta,
             mu_omega, sigma_omega,
             alpha_D, beta_D,
             alpha_R, beta_R)
    param_type : {"log", "natural"}, default="log"
        Coordinate system used in the optimization/information calculations.
    nonlinear_inference : {"optimize", "em"}, default="optimize"
        Method used for section (1), i.e. inference of the nonlinear-model
        parameter estimate before freezing the linearization.
    nonlinear_method : str, default="Powell"
        Optimizer for the nonlinear EKF posterior when
        nonlinear_inference="optimize".
    frozen_method : str, default="L-BFGS-B"
        Optimizer for the frozen surrogate posterior.
    max_iter : int, default=500
        Maximum optimizer / EM iterations.
    tol : float, default=1e-6
        Convergence tolerance.
    return_stats : bool, default=False
        If True, include `post["stats"]` in the returned dictionary.
    initial_phase_method : {"data_informed", "prior"}, default="data_informed"
        How to set the initial-state prior.
        - "prior": use the supplied (mu_theta, sigma_theta)
        - "data_informed": estimate (mu_theta, sigma_theta) from the data
    initial_phase_window_size : int, default=6
        Window size passed to `estimate_initial_phase_regression(...)`.
    initial_phase_sigma_inflation : float, default=2.0
        Inflation factor applied to the estimated phase std:
            sigma_theta = inflation * phi_std
    initial_phase_R_strategy : {"prior_mode", "prior_mean"}, default="prior_mode"
        How to estimate R for the initial phase regression.

    Returns
    -------
    out : dict
        Dictionary containing truth, data, initialization, optimization
        diagnostics, MAP estimates, frozen linearization, posterior
        information, covariance summaries, and optional statistics.

    Added benchmark-friendly outputs
    --------------------------------
      - stderr (sqrt diag covariance) in chosen parameterization
      - zscore_true = (lambda_map_frozen - lambda_true_in_same_coords)/stderr
    """
    if param_type not in ("log", "natural"):
        raise ValueError("param_type must be 'log' or 'natural'")

    if nonlinear_inference not in ("optimize", "em"):
        raise ValueError("nonlinear_inference must be 'optimize' or 'em'")

    if initial_phase_method not in ("data_informed", "prior"):
        raise ValueError("initial_phase_method must be 'data_informed' or 'prior'")

    if initial_phase_R_strategy not in ("prior_mode", "prior_mean"):
        raise ValueError("initial_phase_R_strategy must be 'prior_mode' or 'prior_mean'")

    if initial_phase_window_size < 1:
        raise ValueError("initial_phase_window_size must be >= 1")

    if initial_phase_sigma_inflation <= 0:
        raise ValueError("initial_phase_sigma_inflation must be > 0")

    # ------------------------------------------------------------------
    # Unpack priors
    # ------------------------------------------------------------------
    (
        mu_theta_prior,
        sigma_theta_prior,
        mu_omega,
        sigma_omega,
        alpha_D,
        beta_D,
        alpha_R,
        beta_R,
    ) = priors

    if sigma_theta_prior <= 0:
        raise ValueError("sigma_theta must be > 0 in the supplied priors.")
    if alpha_D <= 0 or beta_D <= 0 or alpha_R <= 0 or beta_R <= 0:
        raise ValueError("Inverse-gamma hyperparameters must satisfy alpha,beta > 0.")

    state_priors_nominal = (mu_theta_prior, sigma_theta_prior)
    param_priors = (mu_omega, sigma_omega, alpha_D, beta_D, alpha_R, beta_R)

    # ------------------------------------------------------------------
    # Truth (always natural)
    # ------------------------------------------------------------------
    param_true_natural = np.asarray(param_true_natural, dtype=float)
    if param_true_natural.shape != (3,):
        raise ValueError("param_true_natural must have shape (3,)")

    omega_true, D_true, R_true = param_true_natural
    if D_true <= 0 or R_true <= 0:
        raise ValueError("Truth must satisfy Dtheta > 0 and R > 0.")

    param_true_log = _to_log_params(param_true_natural)

    # ------------------------------------------------------------------
    # Simulate from nonlinear 2D model
    # ------------------------------------------------------------------
    tk, thetak_true, yk = simulate_data(
        tuple(param_true_natural), N, dt, rng, state_priors_nominal
    )
    data = (tk, yk)

    # ------------------------------------------------------------------
    # Initial guess for parameter inference from parameter priors
    # ------------------------------------------------------------------
    omega0 = rng.normal(mu_omega, sigma_omega)

    # Inverse-gamma prior sampling using 1/X ~ Gamma(alpha, scale=1/beta)
    D0 = 1.0 / rng.gamma(shape=alpha_D, scale=1.0 / beta_D)
    R0 = 1.0 / rng.gamma(shape=alpha_R, scale=1.0 / beta_R)

    if param_type == "log":
        x0 = np.array([omega0, np.log(D0), np.log(R0)], dtype=float)
        bounds = None
    else:
        x0 = np.array([omega0, D0, R0], dtype=float)
        bounds = [(None, None), (1e-15, None), (1e-15, None)]

    # ------------------------------------------------------------------
    # Data-informed prior for the initial phase
    # ------------------------------------------------------------------
    phase_init = None
    omega_init = None
    R_init = None

    if initial_phase_method == "data_informed":
        omega_init, _, _ = estimate_frequency_fft(data)

        if initial_phase_R_strategy == "prior_mode":
            if param_type == "log":
                # mode in log-R coordinates
                R_init = beta_R / alpha_R
            else:
                # mode in natural-R coordinates
                R_init = beta_R / (alpha_R + 1.0)
        else:
            # IG(alpha, beta): mean = beta / (alpha - 1), requires alpha > 1
            if alpha_R <= 1:
                raise ValueError(
                    "initial_phase_R_strategy='prior_mean' requires alpha_R > 1."
                )
            R_init = beta_R / (alpha_R - 1.0)

        phase_init = estimate_initial_phase_regression(
            data,
            omega_init,
            initial_phase_window_size,
            R_init,
        )

        mu_theta_used = phase_init["phi_mean"] + omega_init * tk[0]
        sigma_theta_used = initial_phase_sigma_inflation * phase_init["phi_std"]

        if not np.isfinite(mu_theta_used):
            raise ValueError("Estimated mu_theta is not finite.")
        if (not np.isfinite(sigma_theta_used)) or sigma_theta_used <= 0:
            raise ValueError("Estimated sigma_theta must be finite and > 0.")

        state_priors_used = (mu_theta_used, sigma_theta_used)
    else:
        state_priors_used = state_priors_nominal

    priors_used = state_priors_used + param_priors

    # ------------------------------------------------------------------
    # 1) Nonlinear MAP via either direct optimization or generalized EM
    # ------------------------------------------------------------------
    if nonlinear_inference == "optimize":
        res_nonlinear = minimize(
            fun=lambda x: neg_log_posterior(
                x, data, priors_used, param_mode=param_type
            ),
            x0=x0,
            method=nonlinear_method,
            bounds=bounds if nonlinear_method in ("L-BFGS-B", "Powell", "TNC", "SLSQP") else None,
            options={"maxiter": max_iter, "ftol": tol},
        )

        x_map_nonlinear = np.asarray(res_nonlinear.x, dtype=float)

    else:
        lambda_iter = em_map_parameters(
            x0,
            data,
            priors_used,
            max_iter=max_iter,
            tol=tol,
            verbose=False,
            print_every=10,
            param_mode=param_type,
            validate=False,
        )

        x_map_nonlinear = np.asarray(lambda_iter[-1], dtype=float)

        # Build a lightweight result object with the same attribute interface
        # used below for the direct-optimization branch.
        res_nonlinear = type("EMResult", (), {})()
        res_nonlinear.x = x_map_nonlinear
        res_nonlinear.success = bool(lambda_iter.shape[0] < max_iter)
        res_nonlinear.message = (
            "Generalized EM converged"
            if res_nonlinear.success
            else "Generalized EM reached max_iter without convergence"
        )
        res_nonlinear.status = 0 if res_nonlinear.success else 1
        res_nonlinear.nit = int(lambda_iter.shape[0] - 1)
        res_nonlinear.nfev = None
        res_nonlinear.njev = None
        res_nonlinear.fun = float(
            neg_log_posterior(x_map_nonlinear, data, priors_used, param_mode=param_type)
        )
        res_nonlinear.lambda_iter = lambda_iter

    # Nonlinear MAP in both coordinate systems
    if param_type == "log":
        lambda_map_nonlinear_log = x_map_nonlinear
        lambda_map_nonlinear_natural = _to_natural_params(lambda_map_nonlinear_log)
    else:
        lambda_map_nonlinear_natural = x_map_nonlinear
        lambda_map_nonlinear_log = _to_log_params(lambda_map_nonlinear_natural)

    # ------------------------------------------------------------------
    # 2) Freeze linearization at nonlinear estimate
    # ------------------------------------------------------------------
    Hk, ck, kalman_phases = freeze_linearization(
        tuple(x_map_nonlinear),
        data,
        state_priors_used,
        param_mode=param_type,
    )

    # ------------------------------------------------------------------
    # 3) Frozen-model MAP using consistent frozen objective/gradient
    # ------------------------------------------------------------------
    res_frozen = minimize(
        fun=lambda x: neg_log_posterior_frozen(
            x, data, priors_used, Hk, ck, param_mode=param_type
        ),
        x0=x_map_nonlinear,
        jac=lambda x: grad_neg_log_posterior_frozen(
            x, data, priors_used, Hk, ck, param_mode=param_type
        ),
        method=frozen_method,
        bounds=bounds if frozen_method in ("L-BFGS-B", "Powell", "TNC", "SLSQP") else None,
        options={"maxiter": max_iter, "ftol": tol},
    )

    x_map_frozen = np.asarray(res_frozen.x, dtype=float)

    # Frozen MAP in both coordinate systems
    if param_type == "log":
        lambda_map_frozen_log = x_map_frozen
        lambda_map_frozen_natural = _to_natural_params(lambda_map_frozen_log)
        lambda_true_same = param_true_log
    else:
        lambda_map_frozen_natural = x_map_frozen
        lambda_map_frozen_log = _to_log_params(lambda_map_frozen_natural)
        lambda_true_same = param_true_natural

    # ------------------------------------------------------------------
    # 4) Posterior information for frozen model
    #    evaluated at frozen MAP, but with Hk, ck frozen at nonlinear MAP
    # ------------------------------------------------------------------
    post = posterior_information_frozen(
        tuple(x_map_frozen), data, priors_used, Hk, ck, param_type=param_type
    )

    cov = post["cov"]
    stderr = np.sqrt(np.diag(cov))

    with np.errstate(divide="ignore", invalid="ignore"):
        zscore_true = (x_map_frozen - lambda_true_same) / stderr

    out = {
        # meta
        "param_type": param_type,
        "nonlinear_inference": nonlinear_inference,
        "nonlinear_method": nonlinear_method,
        "frozen_method": frozen_method,

        # truth + data
        "param_true_natural": param_true_natural,
        "param_true_log": param_true_log,
        "tk": tk,
        "thetak_true": thetak_true,
        "yk": yk,
        "data": data,

        # priors
        "priors_input": priors,
        "state_priors_nominal": state_priors_nominal,
        "state_priors_used": state_priors_used,
        "priors_used": priors_used,

        # initialization
        "lambda_init": x0,
        "omega_init_fft": omega_init,
        "R_init_phase": R_init,
        "phase_init": phase_init,
        "initial_phase_method": initial_phase_method,
        "initial_phase_window_size": initial_phase_window_size,
        "initial_phase_sigma_inflation": initial_phase_sigma_inflation,
        "initial_phase_R_strategy": initial_phase_R_strategy,

        # nonlinear inference diagnostics
        "opt_result_nonlinear": res_nonlinear,
        "success_nonlinear": bool(res_nonlinear.success),
        "message_nonlinear": getattr(res_nonlinear, "message", None),
        "status_nonlinear": getattr(res_nonlinear, "status", None),
        "nit_nonlinear": getattr(res_nonlinear, "nit", None),
        "nfev_nonlinear": getattr(res_nonlinear, "nfev", None),
        "njev_nonlinear": getattr(res_nonlinear, "njev", None),
        "fun_nonlinear": float(res_nonlinear.fun),

        # frozen optimization diagnostics
        "opt_result_frozen": res_frozen,
        "success_frozen": bool(res_frozen.success),
        "message_frozen": getattr(res_frozen, "message", None),
        "status_frozen": getattr(res_frozen, "status", None),
        "nit_frozen": getattr(res_frozen, "nit", None),
        "nfev_frozen": getattr(res_frozen, "nfev", None),
        "njev_frozen": getattr(res_frozen, "njev", None),
        "fun_frozen": float(res_frozen.fun),

        # MAP estimates: nonlinear EKF posterior / EM estimate
        "lambda_map_nonlinear": x_map_nonlinear,
        "lambda_map_nonlinear_natural": lambda_map_nonlinear_natural,
        "lambda_map_nonlinear_log": lambda_map_nonlinear_log,

        # MAP estimates: frozen surrogate posterior
        "lambda_map": x_map_frozen,
        "lambda_map_frozen": x_map_frozen,
        "lambda_map_frozen_natural": lambda_map_frozen_natural,
        "lambda_map_frozen_log": lambda_map_frozen_log,

        # frozen linearization used for uncertainty calculations
        "Hk": Hk,
        "ck": ck,
        "kalman_phases_freeze": kalman_phases,

        # information matrices (in param_type coords)
        "I_prior": post["I_prior"],
        "I_comp": post["I_comp"],
        "I_miss": post["I_miss"],
        "I_obs": post["I_obs"],

        # Laplace covariance + summaries
        "cov": cov,
        "stderr": stderr,
        "zscore_true": zscore_true,

        # EM/local-rate diagnostic for frozen surrogate
        "rho": post["rho"],
        "em_rate": post["em_rate"],
    }

    if return_stats:
        out["stats"] = post["stats"]

    return out