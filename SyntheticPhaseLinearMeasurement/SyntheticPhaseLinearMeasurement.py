import numpy as np
import warnings
from scipy.optimize import minimize
from scipy.linalg import solve_banded
from typing import Literal, Tuple


def simulate_data(param, N, dt, rng, state_priors, *, validate=False):
    """
    Simulate latent phase θ_k and measurements y_k for k = 1..N, consistent with SI.

    Model:
        Prior:      θ_1 ~ N(μ_θ, σ_θ^2)
        Process:    θ_k = θ_{k-1} + ω Δt + ε_{k-1},   ε_{k-1} ~ N(0, Q),  Q = 2 Dθ Δt,  k=2..N
        Measurement y_k = θ_k + η_k,                 η_k ~ N(0, R),      k=1..N

    Indexing convention:
        k = 1..N, t_k = k * dt. Python element i corresponds to k=i+1.

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

    yk : ndarray, shape (N,)
        Measurements y_k for k=1..N.
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

    thetak = np.empty(N)

    # θ_1 ~ N(μ_θ, σ_θ^2)
    if sigma_theta == 0.0:
        thetak[0] = mu_theta
    else:
        thetak[0] = mu_theta + sigma_theta * rng.standard_normal()

    # Propagate for k = 2..N: θ_k = θ_{k-1} + ω dt + ε_{k-1}
    if N > 1:
        proc_std = np.sqrt(Q)
        # one noise draw per transition (N-1 transitions)
        thetak[1:] = thetak[0] + wt * np.arange(1, N, dtype=float) \
            + proc_std * np.cumsum(rng.standard_normal(N - 1))

    # Measurements: y_k = θ_k + η_k
    if R == 0.0:
        yk = thetak.copy()
    else:
        yk = thetak + np.sqrt(R) * rng.standard_normal(N)

    return tk, thetak, yk


def kalman_filter(param, data, state_priors, *, validate=False):
    """
    Scalar Kalman filter for the linear phase model in the SI (Example 1a).

    Model (k = 1..N):
        Prior:      θ_1 ~ N(μ_θ, σ_θ^2)
        Process:    θ_k = θ_{k-1} + ω Δt + ε_{k-1},   ε_{k-1} ~ N(0, Q),  Q = 2 D_θ Δt,  k=2..N
        Measurement y_k = θ_k + η_k,                 η_k ~ N(0, R),      k=1..N

    Indexing convention:
        - Python arrays are 0..N-1, but element i corresponds to SI index k=i+1.
        - tk[i] should equal (i+1)*dt (uniform spacing assumed).

    Inputs
    ------
    param  : (omega, Dtheta, R)
        omega  : drift rate (angular frequency)
        Dtheta : phase diffusivity (>=0)
        R      : measurement-noise variance (>=0)

    data   : (tk, yk)
        tk : times, shape (N,), uniformly spaced
        yk : measurements, shape (N,)

    priors : (mu_theta, sigma_theta, ...)
        mu_theta    : prior mean for θ_1
        sigma_theta : prior std for θ_1
        Remaining entries are ignored by this function.

    validate : bool (default False)
        If True, do basic sanity checks (helpful while debugging).

    Outputs
    -------
    thetakk   : θ̂_{k|k}    filtered posterior mean, shape (N,)
    Pkk       : P_{k|k}    filtered posterior variance, shape (N,)
    thetakk_1 : θ̂_{k|k-1}  predicted mean, shape (N,)
    Pkk_1     : P_{k|k-1}  predicted variance, shape (N,)
    """
    omega, Dtheta, R = param
    tk, yk = data
    mu_theta, sigma_theta = state_priors

    N = yk.shape[0]
    if N == 0:
        empty = np.empty((0,))
        return empty, empty, empty, empty

    # Uniform time step (SI uses t_k = k Δt for k=1..N)
    dt = (tk[1] - tk[0]) if N > 1 else 0.0

    if validate:
        if not (isinstance(N, (int, np.integer)) and N > 0):
            raise ValueError(f"N must be a positive integer; got {N!r}")
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
    invR = 0.0 if R == 0.0 else 1.0 / R

    # Allocate outputs (np.empty is slightly faster than zeros)
    thetakk   = np.empty(N)  # θ̂_{k|k}
    Pkk       = np.empty(N)  # P_{k|k}
    thetakk_1 = np.empty(N)  # θ̂_{k|k-1}
    Pkk_1     = np.empty(N)  # P_{k|k-1}

    # --- k = 1 (python index 0) ---
    # Prediction for θ_1 is just the prior p(θ_1): no Q, no +wt here.
    thetakk_1[0] = mu_theta
    Pkk_1[0]     = sigma_theta**2

    # Update with y_1
    if R == 0.0:
        # Perfect measurement: θ_1 is observed exactly.
        thetakk[0] = yk[0]
        Pkk[0]     = 0.0
    else:
        # Kalman gain form (numerically stable and avoids repeated reciprocals)
        S = Pkk_1[0] + R           # innovation variance
        K = Pkk_1[0] / S
        innov = yk[0] - thetakk_1[0]
        thetakk[0] = thetakk_1[0] + K * innov
        Pkk[0]     = (1.0 - K) * Pkk_1[0]

    # --- k = 2..N (python indices 1..N-1) ---
    for k in range(1, N):
        # Predict: θ̂_{k|k-1} = θ̂_{k-1|k-1} + ω dt
        thetakk_1[k] = thetakk[k - 1] + wt
        Pkk_1[k]     = Pkk[k - 1] + Q

        # Update with y_k
        if R == 0.0:
            thetakk[k] = yk[k]
            Pkk[k]     = 0.0
        else:
            S = Pkk_1[k] + R
            K = Pkk_1[k] / S
            innov = yk[k] - thetakk_1[k]
            thetakk[k] = thetakk_1[k] + K * innov
            Pkk[k]     = (1.0 - K) * Pkk_1[k]

    return thetakk, Pkk, thetakk_1, Pkk_1


def neg_log_marginal_likelihood(param, data, state_priors, *, param_mode="log", validate=False):
    """
    Negative log marginal likelihood for the scalar linear Gaussian phase model:
        nll = -log p(y_{1:N} | omega, Dtheta, R)

    Model (k = 1..N):
        Prior on initial state:
            θ_1 ~ N(μ_θ, σ_θ^2)

        State evolution (k = 2..N):
            θ_k = θ_{k-1} + ω Δt + ε_{k-1},   ε_{k-1} ~ N(0, Q),  Q = 2 Dθ Δt

        Measurement (k = 1..N):
            y_k = θ_k + η_k,                 η_k ~ N(0, R)

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
        (tk, yk) with tk uniformly spaced.

    priors : tuple
        (mu_theta, sigma_theta, ...) only first two entries used.

    param_mode : {"natural", "log"}, optional
        Choose whether to interpret param as (omega, Dtheta, R) or (omega, lnDtheta, lnR).

    validate : bool, optional
        If True, run basic checks (uniform dt, positivity constraints).

    Returns
    -------
    nll : float
        Negative log marginal likelihood.
    """
    tk, yk = data
    mu_theta, sigma_theta = state_priors

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

    # ---- k = 1: predicted state is the prior on θ_1 ----
    theta_pred = mu_theta
    P_pred = sigma_theta**2

    nll = 0.0

    # Update / likelihood contribution at k=1
    S = P_pred + R  # innovation variance
    innov = yk[0] - theta_pred

    # Handle degenerate S (can occur if sigma_theta=0 and R=0)
    if S <= 0.0:
        return 0.0 if innov == 0.0 else np.inf

    nll += 0.5 * (log2pi + np.log(S) + (innov * innov) / S)

    # Kalman update
    K = P_pred / S
    theta_filt = theta_pred + K * innov
    P_filt = (1.0 - K) * P_pred

    # ---- k = 2..N ----
    for k in range(1, N):
        # Predict
        theta_pred = theta_filt + wt
        P_pred = P_filt + Q

        S = P_pred + R
        innov = yk[k] - theta_pred

        if S <= 0.0:
            return np.inf

        # Accumulate -log likelihood
        nll += 0.5 * (log2pi + np.log(S) + (innov * innov) / S)

        # Update
        K = P_pred / S
        theta_filt = theta_pred + K * innov
        P_filt = (1.0 - K) * P_pred

    return nll


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

    Model / indexing:
      - Arrays are length N; Python index i corresponds to SI index k=i+1.
      - Prior is on θ_1; process noise Q applies only to transitions k>=2.

    Inputs
    ------
    param : (omega, Dtheta, R)
    data : (tk, yk)
    kalman_phases : (thetakk, Pkk, thetakk_1, Pkk_1)
        outputs of kalman_filter:
          thetakk   : θ̂_{k|k}
          Pkk       : P_{k|k}
          thetakk_1 : θ̂_{k|k-1}
          Pkk_1     : P_{k|k-1}

    Returns
    -------
    thetakN : θ̂_{k|N}     smoothed mean
    PkN     : P_{k|N}      smoothed variance
    Sigmakk_1 : Σ_{k,k-1|N} lag-one smoothed covariance (lower diagonal)
        Convention: Sigmakk_1[i] corresponds to Σ_{k=i+1, k-1=i|N}.
        Entries Sigmakk_1[0] (k=1) is meaningless and set to 0.
    """
    omega, Dtheta, R = param
    tk, yk = data
    thetakk, Pkk, thetakk_1, Pkk_1 = kalman_phases

    N = yk.shape[0]
    if N == 0:
        empty = np.empty((0,))
        return empty, empty, empty

    if validate:
        if N != thetakk.shape[0]:
            raise ValueError("kalman_phases arrays must all have length N.")
        if N > 2:
            dt = tk[1] - tk[0]
            if not np.allclose(np.diff(tk), dt):
                raise ValueError("tk must be uniformly spaced.")

    # Allocate outputs
    thetakN = np.empty(N)        # θ̂_{k|N}
    PkN = np.empty(N)            # P_{k|N}
    Sigmakk_1 = np.zeros(N)      # Σ_{k,k-1|N}; Sigmakk_1[0] unused

    # Initialize at final time: k=N (python index N-1)
    thetakN[-1] = thetakk[-1]
    PkN[-1] = Pkk[-1]

    # Backward pass for k = N-1 ... 1 (python indices N-2 ... 0)
    for k in range(N - 2, -1, -1):
        # RTS gain: C_k = P_{k|k} / P_{k+1|k}
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
    Posterior covariance of θ_{1:N} | y_{1:N}.

    Model:
        θ_1 ~ N(μ_θ, σ_θ^2)
        θ_k = θ_{k-1} + ε_{k-1},   ε ~ N(0, Q),   Q = 2 Dθ Δt
        y_k = θ_k + η_k,          η ~ N(0, R)

    The posterior precision matrix A is tridiagonal.
    This function returns Σ = A^{-1}.

    Parameters
    ----------
    param : (omega, Dtheta, R)
        omega is unused (covariance does not depend on drift).

    N : int
        Number of time steps.

    dt : float
        Time step.

    state_priors : (mu_theta, sigma_theta)
        Only sigma_theta affects the covariance.

    Returns
    -------
    Sigma : ndarray, shape (N, N)
        Posterior covariance matrix of θ_{1:N}.
    """
    _, Dtheta, R = param
    _, sigma_theta = state_priors

    if N == 0:
        return np.empty((0, 0))

    Q = 2.0 * Dtheta * dt

    if validate:
        if Q <= 0:
            raise ValueError("Need Dtheta > 0 and dt > 0.")
        if R <= 0:
            raise ValueError("R must be > 0.")
        if sigma_theta <= 0:
            raise ValueError("sigma_theta must be > 0.")

    invQ = 1.0 / Q
    invR = 1.0 / R
    invSig2 = 1.0 / (sigma_theta ** 2)

    # Banded precision matrix A (tridiagonal)
    ab = np.zeros((3, N))

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


def grad_neg_log_posterior(
    param: Tuple[float, float, float],
    data,
    priors,
    *,
    param_mode: Literal["log", "natural"] = "log",
    validate: bool = False,
    return_aux: bool = False,
):
    """
    Gradient of the *negative* log posterior for the scalar linear phase model,
    computed via the Louis identity (expected complete-data score under the RTS
    smoothing distribution).

    This function is designed to be used as `jac` in `scipy.optimize.minimize`
    when minimizing `neg_log_posterior(...)`.

    Model (SI Example 1a)
    ---------------------
        θ_1 ~ N(μ_θ, σ_θ^2)
        θ_k = θ_{k-1} + ω Δt + ε_{k-1},     ε_{k-1} ~ N(0, Q),   Q = 2 Dθ Δt      (k=2..N)
        y_k = θ_k + η_k,                   η_k ~ N(0, R)                         (k=1..N)

    Posterior objective (up to an additive constant)
    ------------------------------------------------
        L(λ) = log p(λ) + log p(y | λ)
        nlp(λ) = -L(λ)

    Louis identity (SI)
    -------------------
        ∇_λ log p(y|λ) = E[ ∇_λ ℓ_c(λ; θ, y) | y, λ ]
    where ℓ_c = log p(θ, y | λ).

    Coordinates / parameterization
    ------------------------------
    If param_mode == "log":
        param = (omega, lnDtheta, lnR), with Dtheta = exp(lnDtheta), R = exp(lnR)
        Returns: ∇_{(omega, lnDtheta, lnR)} nlp

    If param_mode == "natural":
        param = (omega, Dtheta, R)
        Returns: ∇_{(omega, Dtheta, R)} nlp

    Priors on parameters (as in your code)
    --------------------------------------
        omega  ~ Normal(mu_omega, sigma_omega^2)
        Dtheta ~ Inverse-Gamma(alpha_D, beta_D)     (shape α, scale β)
        R      ~ Inverse-Gamma(alpha_R, beta_R)

    Note: we assume `priors` has the same layout used elsewhere in this file:
        priors = (mu_theta, sigma_theta, mu_omega, sigma_omega, alpha_D, beta_D, alpha_R, beta_R)

    Returns
    -------
    grad : ndarray, shape (3,)
        Gradient of negative log posterior in the requested coordinates.

    aux : dict (optional)
        If return_aux=True, also returns intermediate quantities:
        smoothed means/variances, expected sufficient stats, and the score.

    Dependencies
    ------------
    Requires: kalman_filter(...), rts_smoother(...)
    which are defined in this module. :contentReference[oaicite:1]{index=1}
    """
    tk, yk = data
    yk = np.asarray(yk, dtype=float).ravel()
    tk = np.asarray(tk, dtype=float).ravel()
    N = yk.size
    if N < 2:
        raise ValueError("Need N >= 2 for the increment statistics in the Louis score.")

    # unpack priors (same ordering used by neg_log_posterior in this file)
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
        # for convenience downstream, define their logs too
        lnDtheta = float(np.log(Dtheta))
        lnR = float(np.log(R))
    else:
        raise ValueError(f"param_mode must be 'log' or 'natural'; got {param_mode!r}")

    # basic time step
    dt = float(tk[1] - tk[0])

    if validate:
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

    # ------------------------------------------------------------
    # 1) Run filter + RTS smoother to get p(theta | y, lambda)
    # ------------------------------------------------------------
    kalman_phases = kalman_filter((omega, Dtheta, R), (tk, yk), state_priors, validate=validate)
    thetakN, PkN, Sigmakk_1 = rts_smoother((omega, Dtheta, R), (tk, yk), kalman_phases, validate=validate)

    # ------------------------------------------------------------
    # 2) Expected sufficient statistics under smoothing distribution
    #    (matches your SI and your existing neg_score)
    # ------------------------------------------------------------
    # E[S0 | y,lambda] = sum_k E[(Δθ_k)^2] where Δθ_k = θ_k - θ_{k-1}
    dtheta_mean = thetakN[1:] - thetakN[:-1]
    ES0 = np.sum(dtheta_mean**2 + PkN[1:] + PkN[:-1] - 2.0 * Sigmakk_1[1:])

    # E[S1 | y,lambda] = E[θ_N - θ_1]
    ES1 = float(thetakN[-1] - thetakN[0])

    # E[Sy | y,lambda] = sum_k E[(y_k - θ_k)^2]
    r = yk - thetakN
    ESy = np.sum(r**2 + PkN)

    wdt = omega * dt
    M = N - 1  # number of increments

    # ------------------------------------------------------------
    # 3) Score (gradient of log posterior) in log-coordinates:
    #    (omega, lnDtheta, lnR)
    # ------------------------------------------------------------
    score_log = np.zeros(3, dtype=float)

    # omega: Normal prior + Louis likelihood term
    score_log[0] = (
        -(omega - mu_omega) / (sigma_omega**2)
        + (ES1 - M * wdt) / (2.0 * Dtheta)
    )

    # lnDtheta: IG prior in ln-space + Louis likelihood term
    # Prior: D * d/dD log p(D) = -(alpha+1) + beta/D
    score_log[1] = (
        -alpha_D + beta_D / Dtheta
        - 0.5 * M
        + (ES0 - 2.0 * ES1 * wdt + M * (wdt**2)) / (4.0 * Dtheta * dt)
    )

    # lnR: IG prior in ln-space + Louis likelihood term
    score_log[2] = (
        -alpha_R + beta_R / R
        - 0.5 * N
        + ESy / (2.0 * R)
    )

    # Gradient of *negative* log posterior:
    # nlp = -log posterior  =>  ∇ nlp = - score
    grad_log = -score_log

    # ------------------------------------------------------------
    # 4) Convert to requested coordinates if needed
    # ------------------------------------------------------------
    if param_mode == "log":
        grad = grad_log
    else:
        # Natural coordinates: (omega, Dtheta, R)
        # Chain rule: d/dD = (1/D) d/d(ln D), similarly for R
        grad = np.zeros(3, dtype=float)
        grad[0] = grad_log[0]
        grad[1] = (grad_log[1] + 1.0) / Dtheta
        grad[2] = (grad_log[2] + 1.0) / R

    if not return_aux:
        return grad

    aux = dict(
        dt=dt,
        smoothed_mean=thetakN,
        smoothed_var=PkN,
        smoothed_lag1_cov=Sigmakk_1,
        expected_stats=dict(S0=ES0, S1=ES1, Sy=ESy),
        score_log=score_log,
        grad_log=grad_log,
        grad=grad,
        param_mode=param_mode,
        param_natural=(omega, Dtheta, R),
        param_log=(omega, lnDtheta, lnR),
    )
    return grad, aux


def em_update(old_param, data, priors,
              param_mode: Literal["log", "natural"] = "log"):
    """
    Perform one EM update for the phase-drift-diffusion model.

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
        tk : time points (N,)
        yk : noisy phase observations (N,)

    priors : tuple
        (mu_theta, sigma_theta,
         mu_omega, sigma_omega,
         alpha_D, beta_D,
         alpha_R, beta_R)

        Priors:
            theta0 ~ N(mu_theta, sigma_theta^2)
            omega  ~ N(mu_omega, sigma_omega^2)
            Dtheta ~ InvGamma(alpha_D, beta_D)
            R      ~ InvGamma(alpha_R, beta_R)

    param_mode : {"log", "natural"}, default="log"
        Determines parameterization of input/output.

    Returns
    -------
    new_param : list
        Updated parameters in the same parameterization
        as `param_mode`.
    """

    tk, yk = data

    # --------------------------------------------------
    # Unpack priors
    # --------------------------------------------------
    mu_theta, sigma_theta, \
    mu_omega, sigma_omega, \
    alpha_D, beta_D, \
    alpha_R, beta_R = priors

    state_priors = (mu_theta, sigma_theta)

    # --------------------------------------------------
    # Convert parameters to natural scale if needed
    # --------------------------------------------------
    if param_mode == "natural":
        omega, Dtheta, R = old_param
    elif param_mode == "log":
        omega, lnDtheta, lnR = old_param
        Dtheta = np.exp(lnDtheta)
        R = np.exp(lnR)
    else:
        raise ValueError("param_mode must be 'natural' or 'log'")

    # --------------------------------------------------
    # E-step: Kalman filter + RTS smoother
    # --------------------------------------------------
    kalman_phases = kalman_filter((omega, Dtheta, R), data, state_priors)
    thetakN, PkN, Sigmakk_1 = rts_smoother((omega, Dtheta, R), data, kalman_phases)

    # --------------------------------------------------
    # Compute sufficient statistics
    # --------------------------------------------------
    N = yk.shape[0]
    dt = tk[1] - tk[0]

    # Expected squared increments of latent phase
    ES0 = np.sum(
        (thetakN[1:] - thetakN[:-1])**2
        + PkN[1:] + PkN[:-1] - 2 * Sigmakk_1[1:]
    )

    # Total drift
    ES1 = thetakN[-1] - thetakN[0]

    # Expected measurement residual energy
    ESy = np.sum((thetakN - yk)**2 + PkN)

    # --------------------------------------------------
    # M-step updates (MAP under conjugate priors)
    # --------------------------------------------------

    # ---- Update omega (Gaussian prior)
    omega_new = (ES1 / (2 * Dtheta) + mu_omega / sigma_omega**2) \
              / ((N - 1) * dt / (2 * Dtheta) + 1 / sigma_omega**2)

    # ---- Update Dtheta (Inverse-Gamma prior)
    wdt = omega_new * dt

    if param_mode == "natural":
        # Standard conjugate update
        Dtheta_new = (ES0 - 2 * ES1 * wdt + (N - 1) * wdt**2 + 4 * beta_D * dt) \
                   / (2 * (N - 1 + 2 * (alpha_D + 1)) * dt)

        R_new = (2 * beta_R + ESy) / (N + 2 * (alpha_R + 1))

        return [omega_new, Dtheta_new, R_new]

    else:
        # Log-parameterization:
        # Jacobian contributes a -log(Dtheta) term,
        # shifting alpha by 1 relative to natural form.

        Dtheta_new = (ES0 - 2 * ES1 * wdt + (N - 1) * wdt**2 + 4 * beta_D * dt) \
                   / (2 * (N - 1 + 2 * alpha_D) * dt)

        R_new = (2 * beta_R + ESy) / (N + 2 * alpha_R)

        return [omega_new, np.log(Dtheta_new), np.log(R_new)]


def em_map_parameters(
    initial_params,
    data,
    priors,
    max_iter: int = 500,
    tol: float = 1e-5,
    verbose: bool = True,
    print_every: int = 10,
    param_mode: Literal["log", "natural"] = "log",
):
    """
    Estimate MAP parameters using EM (fixed-point iteration).

    This routine repeatedly applies the EM update until convergence
    in relative parameter change.

    Parameters
    ----------
    initial_params : array_like, shape (3,)
        Initial guess.

        natural mode:
            [omega, Dtheta, R]

        log mode:
            [omega, lnDtheta, lnR]

    data : tuple
        (tk, yk)

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

    Returns
    -------
    lambda_iter : ndarray, shape (K,3)
        Parameter trajectory (truncated to convergence).
    """

    lambda_iter = np.zeros((max_iter, 3), dtype=float)
    lambda_iter[0, :] = np.asarray(initial_params, dtype=float)

    if verbose:
        if param_mode == "natural":
            print("iter\tomega\t\tDtheta\t\tR")
        else:
            print("iter\tomega\t\tlnDtheta\tlnR")

    converged = False

    for i in range(1, max_iter):

        # --- EM update ---
        lambda_iter[i] = em_update(
            lambda_iter[i-1],
            data,
            priors,
            param_mode=param_mode
        )

        # --- Relative change ---
        denom = np.maximum(np.abs(lambda_iter[i-1]), 1e-12)
        rel_change = np.max(
            np.abs(lambda_iter[i] - lambda_iter[i-1]) / denom
        )

        # --- Progress print ---
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

        # --- Convergence ---
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
            f"EM MAP inference did not converge within {max_iter} iterations "
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


def complete_data_information(lambda_map, statistics, param_type="log"):
    """
    I_comp = -E[ ∇^2 ell_c | y, lambda ] computed from smoothing statistics.
    Returns in coordinates specified by param_type.
    """
    omega, Dtheta, R = statistics["lambda_natural"]
    N = statistics["N"]
    dt = statistics["dt"]
    ES0 = statistics["ES0"]
    ES1 = statistics["ES1"]
    ESy = statistics["ESy"]

    wdt = omega * dt
    EA = ES0 - 2.0 * ES1 * wdt + (N - 1) * (wdt**2)

    I_comp = np.zeros((3, 3), dtype=float)

    if param_type == "log":
        # coords: (omega, lnDtheta, lnR)
        I_comp[0, 0] = (N - 1) * dt / (2.0 * Dtheta)
        I_comp[0, 1] = (ES1 - (N - 1) * wdt) / (2.0 * Dtheta)
        I_comp[1, 1] = EA / (4.0 * Dtheta * dt)
        I_comp[2, 2] = ESy / (2.0 * R)
        I_comp[1, 0] = I_comp[0, 1]

    elif param_type == "natural":
        # coords: (omega, Dtheta, R)
        I_comp[0, 0] = (N - 1) * dt / (2.0 * Dtheta)
        I_comp[0, 1] = (ES1 - (N - 1) * wdt) / (2.0 * Dtheta**2)
        I_comp[1, 1] = -(N - 1) / (2.0 * Dtheta**2) + EA / (2.0 * Dtheta**3 * dt)
        I_comp[2, 2] = -N / (2.0 * R**2) + ESy / (R**3)
        I_comp[1, 0] = I_comp[0, 1]

    else:
        raise ValueError("param_type must be 'natural' or 'log'")

    return I_comp


def missing_information(lambda_map, statistics, priors, param_type="log") -> np.ndarray:
    """
    Exact I_miss = Cov(score_c | y, lambda), computed using banded solves
    and streaming trace identities, using precomputed smoother mean m.

    Returns in coordinates specified by param_type:
      - log:     [omega, ln Dtheta, ln R]
      - natural: [omega, Dtheta, R]
    """
    y = statistics["yk"]
    m = statistics["m"]
    dt = statistics["dt"]
    N = statistics["N"]
    sigma_theta2 = statistics["sigma_theta2"]

    omega, D, R = statistics["lambda_natural"]

    if D <= 0 or R <= 0 or dt <= 0 or sigma_theta2 <= 0:
        raise ValueError("Require D>0, R>0, dt>0, sigma_theta2>0.")
    if N < 2:
        raise ValueError("Need N>=2.")
    M = N - 1

    # --- Build banded posterior precision J for theta_1..theta_N ---
    q = 2.0 * D * dt
    invR = 1.0 / R
    invq = 1.0 / q
    invsig = 1.0 / sigma_theta2

    d = np.empty(N, dtype=float)
    u = np.full(N - 1, -invq, dtype=float)
    l = np.full(N - 1, -invq, dtype=float)

    d[0] = invR + invq + invsig
    if N > 2:
        d[1:N-1] = invR + 2.0 * invq
    d[N-1] = invR + invq

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
        out[1:N-1] = -x[0:N-2] + 2.0 * x[1:N-1] - x[2:N]
        out[N-1] = -x[N-2] + x[N-1]
        return out

    def apply_B(theta: np.ndarray) -> np.ndarray:
        theta = np.asarray(theta, dtype=float).ravel()
        return theta[1:] - theta[:-1]  # length M

    def apply_Bt(v: np.ndarray) -> np.ndarray:
        v = np.asarray(v, dtype=float).ravel()
        w = np.zeros(N, dtype=float)
        w[0] = -v[0]
        if N > 2:
            w[1:N-1] = v[0:M-1] - v[1:M]
        w[N-1] = v[M-1]
        return w

    def apply_sigma_u(v: np.ndarray) -> np.ndarray:
        return apply_B(apply_sigma(apply_Bt(v)))

    # --- Common vectors ---
    r = y - m
    c = np.zeros(N, dtype=float)
    c[0] = -1.0
    c[-1] = +1.0
    onesM = np.ones(M, dtype=float)
    mu_u = apply_B(m)

    # Var(S1) with S1 = theta_N - theta_1
    Sig_c = apply_sigma(c)
    V_S1 = float(c @ Sig_c)

    # Cov(S1, Sy) where Sy = sum (y-theta)^2
    Sig_r = apply_sigma(r)
    C_S1_Sy = float(-2.0 * (c @ Sig_r))

    # Cov(S1, S0) where S0 = sum (dtheta)^2
    SigU_mu = apply_sigma_u(mu_u)
    C_S1_S0 = float(2.0 * (onesM @ SigU_mu))
    mu_SigU_mu = float(mu_u @ SigU_mu)

    # trace(Sigma^2) and trace(A0 Sigma^2)
    tr_Sig2 = 0.0
    tr_A0Sig2 = 0.0
    e = np.zeros(N, dtype=float)
    for i in range(N):
        e.fill(0.0)
        e[i] = 1.0
        x = solve_banded((1, 1), ab, e)
        tr_Sig2 += float(x @ x)
        tr_A0Sig2 += float(x @ apply_A0(x))

    # trace(Sigma_u^2)
    tr_SigU2 = 0.0
    ej = np.zeros(M, dtype=float)
    for j in range(M):
        ej.fill(0.0)
        ej[j] = 1.0
        z = apply_sigma_u(ej)
        tr_SigU2 += float(z @ z)

    Var_S0 = 2.0 * tr_SigU2 + 4.0 * mu_SigU_mu
    r_Sig_r = float(r @ Sig_r)
    Var_Sy = 2.0 * tr_Sig2 + 4.0 * r_Sig_r

    # Q = S0 - 2 omega dt S1 + const
    Cov_S1_Q = C_S1_S0 - 2.0 * omega * dt * V_S1
    Var_Q = Var_S0 + 4.0 * (omega * dt) ** 2 * V_S1 - 4.0 * omega * dt * C_S1_S0

    # Cov(Q, Sy)
    Sig_m = apply_sigma(m)
    term_m_A0_Sig_m = float(m @ apply_A0(Sig_m))
    A0m = apply_A0(m)
    Sig_A0m = apply_sigma(A0m)
    term_y_Sig_A0m = float(y @ Sig_A0m)
    term_c_Sig_m = float(c @ Sig_m)
    Sig_y = apply_sigma(y)
    term_c_Sig_y = float(c @ Sig_y)

    Cov_Q_Sy = (
        2.0 * tr_A0Sig2
        + 4.0 * term_m_A0_Sig_m
        - 4.0 * term_y_Sig_A0m
        - 4.0 * omega * dt * term_c_Sig_m
        + 4.0 * omega * dt * term_c_Sig_y
    )

    # --- Assemble I_miss in requested coordinates ---
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
    

def smoothing_statistics(lambda_map, data, priors, param_type="log"):
    """
    Run KF + RTS once (always in natural params) and return a dict of
    reusable statistics for information calculations.
    """
    tk, yk = data
    yk = np.asarray(yk, dtype=float).ravel()
    dt = float(tk[1] - tk[0])
    N = yk.size

    (mu_theta, sigma_theta,
     mu_omega, sigma_omega,
     alpha_D, beta_D,
     alpha_R, beta_R) = priors

    # Map input -> natural params for filter/smoother
    if param_type == "natural":
        omega, Dtheta, R = lambda_map
    elif param_type == "log":
        omega, lnDtheta, lnR = lambda_map
        Dtheta = float(np.exp(lnDtheta))
        R = float(np.exp(lnR))
    else:
        raise ValueError("param_type must be 'natural' or 'log'")

    lambda_nat = (omega, Dtheta, R)

    state_priors = (mu_theta, sigma_theta)

    # Filter + Smooth (assumed to take natural params)
    kalman_phases = kalman_filter(lambda_nat, data, state_priors)
    thetakN, PkN, Sigmakk_1 = rts_smoother(lambda_nat, data, kalman_phases)

    thetakN = np.asarray(thetakN, dtype=float).ravel()
    PkN = np.asarray(PkN, dtype=float).ravel()
    Sigmakk_1 = np.asarray(Sigmakk_1, dtype=float).ravel()

    # Expected sufficient statistics under smoothing distribution
    ES0 = float(np.sum(
        (thetakN[1:] - thetakN[:-1])**2
        + PkN[1:] + PkN[:-1]
        - 2.0 * Sigmakk_1[1:]
    ))
    ES1 = float(thetakN[-1] - thetakN[0])
    ESy = float(np.sum((yk - thetakN)**2 + PkN))

    return {
        "tk": tk,
        "yk": yk,
        "dt": dt,
        "N": N,
        "m": thetakN,                 # smoother mean
        "PkN": PkN,                   # marginal variances (may be unused)
        "Sigmakk_1": Sigmakk_1,       # lag-1 cov (may be unused)
        "lambda_natural": lambda_nat, # (omega, Dtheta, R)
        "ES0": ES0,
        "ES1": ES1,
        "ESy": ESy,
        "sigma_theta2": float(sigma_theta)**2,
    }


def posterior_information(lambda_map, data, priors, param_type="log"):
    """
    Compute Laplace posterior information at MAP:
        I_obs = I_prior + I_comp - I_miss
    returning all components plus covariance and EM-rate diagnostic.

    Parameters
    ----------
    lambda_map : tuple
        MAP parameters in coordinates specified by param_type.

    data : tuple
        (tk, yk)

    priors : tuple
        Prior hyperparameters (see above).

    param_type : str
        "log" (default) or "natural"

    Returns
    -------
    out : dict
        {
          "I_prior": ...,
          "I_comp":  ...,
          "I_miss":  ...,
          "I_obs":   ...,
          "cov":     inv(I_obs),
          "rho":     spectral radius of (I_miss @ inv(I_prior + I_comp)),
          "em_rate": -log(rho),
          "stats":   smoothing statistics dict (optional but useful)
        }
    """
    # Run smoother once + compute reusable expectations
    stats = smoothing_statistics(lambda_map, data, priors, param_type=param_type)

    # Prior information (your corrected natural/log Hessians)
    I_prior = prior_information(lambda_map, priors, param_type=param_type)

    # Complete-data and missing information (no additional smoothing)
    I_comp = complete_data_information(lambda_map, stats, param_type=param_type)
    I_miss = missing_information(lambda_map, stats, priors, param_type=param_type)

    # Observed posterior information
    I_obs = I_prior + I_comp - I_miss

    # Laplace covariance
    cov = np.linalg.inv(I_obs)

    # EM convergence diagnostic: rho = max eig( I_miss (I_prior+I_comp)^{-1} )
    A = I_prior + I_comp
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
    max_iter: int = 500,
    tol: float = 1e-6,
    return_stats: bool = False,
):
    """
    Simulate data and infer MAP parameters via gradient-based optimization (L-BFGS-B),
    then compute Laplace covariance from posterior information.

    IMPORTANT: param_true_natural is ALWAYS interpreted as natural parameters
               (omega, Dtheta, R), regardless of param_type.

    Returns a dictionary with truth, MAP, uncertainty, and diagnostics.

    Added benchmark-friendly outputs:
      - stderr (sqrt diag covariance) in chosen parameterization
      - zscore_true = (lambda_map - lambda_true_in_same_coords)/stderr
    """
    if param_type not in ("log", "natural"):
        raise ValueError("param_type must be 'log' or 'natural'")

    # --- Unpack priors ---
    mu_theta, sigma_theta, \
    mu_omega, sigma_omega, \
    alpha_D, beta_D, \
    alpha_R, beta_R = priors

    state_priors = (mu_theta, sigma_theta)
    
    # --- Truth (always natural) ---
    param_true_natural = np.array(param_true_natural, dtype=float)
    omega_true, D_true, R_true = param_true_natural
    if D_true <= 0 or R_true <= 0:
        raise ValueError("Truth must satisfy Dtheta>0 and R>0.")
    param_true_log = _to_log_params(param_true_natural)

    # --- Simulate (simulate_data assumed natural) ---
    tk, thetak_true, yk = simulate_data(tuple(param_true_natural), N, dt, rng, state_priors)
    data = (tk, yk)

    # --- Initial guess for optimizer: prior sample (in chosen coordinates) ---
    omega0 = rng.normal(mu_omega, sigma_omega)

    # Inverse-gamma sampling via Gamma(shape=α, scale=1/β) on 1/D
    D0 = 1.0 / rng.gamma(shape=alpha_D, scale=1.0 / beta_D)
    R0 = 1.0 / rng.gamma(shape=alpha_R, scale=1.0 / beta_R)

    if param_type == "log":
        x0 = np.array([omega0, np.log(D0), np.log(R0)], dtype=float)
        bounds = [(None, None), (None, None), (None, None)]  # unconstrained in log
    else:
        x0 = np.array([omega0, D0, R0], dtype=float)
        bounds = [(None, None), (1e-15, None), (1e-15, None)]  # enforce positivity
    
    # --- MAP via gradient-based optimization ---
    res = minimize(
        fun=lambda x: neg_log_posterior(x, data, priors, param_mode=param_type),
        x0=x0,
        jac=lambda x: grad_neg_log_posterior(x, data, priors, param_mode=param_type),
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": max_iter, "ftol": tol},
    )

    x_map = np.array(res.x, dtype=float)  # MAP in chosen parameterization

    # --- MAP in both coordinate systems ---
    if param_type == "log":
        lambda_map_log = x_map
        lambda_map_natural = _to_natural_params(lambda_map_log)
        lambda_true_same = param_true_log
    else:
        lambda_map_natural = x_map
        lambda_map_log = _to_log_params(lambda_map_natural)
        lambda_true_same = param_true_natural

    # --- Posterior information (Laplace) in chosen coordinates ---
    post = posterior_information(tuple(x_map), data, priors, param_type=param_type)

    cov = post["cov"]
    stderr = np.sqrt(np.diag(cov))

    # Avoid divide-by-zero warnings if any stderr ~ 0
    with np.errstate(divide="ignore", invalid="ignore"):
        zscore_true = (x_map - lambda_true_same) / stderr

    out = {
        # meta
        "param_type": param_type,

        # truth + data
        "param_true_natural": param_true_natural,
        "param_true_log": param_true_log,
        "tk": tk,
        "thetak_true": thetak_true,
        "yk": yk,
        "data": data,

        # optimizer diagnostics
        "opt_result": res,
        "success": bool(res.success),
        "message": getattr(res, "message", None),
        "status": getattr(res, "status", None),
        "nit": getattr(res, "nit", None),
        "nfev": getattr(res, "nfev", None),
        "njev": getattr(res, "njev", None),
        "fun": float(res.fun),

        # MAP estimates
        "lambda_map": x_map,  # in param_type coords
        "lambda_map_natural": lambda_map_natural,
        "lambda_map_log": lambda_map_log,

        # information matrices (in param_type coords)
        "I_prior": post["I_prior"],
        "I_comp": post["I_comp"],
        "I_miss": post["I_miss"],
        "I_obs": post["I_obs"],

        # Laplace covariance + convenient summaries
        "cov": cov,
        "stderr": stderr,
        "zscore_true": zscore_true,

        # EM diagnostic (useful even if you didn't run EM)
        "rho": post["rho"],
        "em_rate": post["em_rate"],

        # prior 
        "lambda_init": x0
    }

    if return_stats:
        out["stats"] = post["stats"]

    return out


def sample_phases(param, data, priors, rng, N_samples=1000):
    #### Kalman Filter
    thetakk, Pkk, thetakk_1, Pkk_1 = kalman_filter(param, data, priors)
    
    #### Forward-Filter Backward-Sample (FFBS)
    N = thetakk.shape[0]
    theta_samples = np.zeros((N_samples,N))

    # initialize for k = N
    k = N-1
    theta_samples[:,k] = rng.normal(loc=thetakk[k], scale=np.sqrt(Pkk[k]), size=N_samples)
    
    # iterate backward   
    Jk = Pkk[:-1] / Pkk_1[1:]
    for k in range(N-1,0,-1):
        # mean
        mean = thetakk[k-1] + Jk[k-1]*(theta_samples[:,k] - thetakk_1[k])
        # variance
        var = Pkk[k-1] - Jk[k-1] * Pkk_1[k] * Jk[k-1]
        # sample
        theta_samples[:,k-1] = rng.normal(loc=mean, scale=np.sqrt(var))

    return theta_samples
