"""
O(N) drop-in replacements for the O(N^2) pieces of `StuartLandauPhaseInference`.

Why
---
`StuartLandauPhaseInference.em_update` and `.smoothing_statistics_frozen` both call
`theta_covariance`, which materialises the **dense** N x N posterior covariance via
``solve_banded((1, 1), ab, np.eye(N))`` -- purely to read off its first sub-diagonal
``np.diag(Sigma, k=-1)``, i.e. the lag-one smoothed covariance ``Sigma_{k,k-1|N}``.

That costs O(N^2) time *and* O(N^2) memory **per EM iteration**:

    N = 2000 -> 0.10 s,   32 MB
    N = 4000 -> 0.69 s,  128 MB
    N = 8000 -> 4.33 s,  512 MB

At N = 8000 a noise-injected fit needs ~550 generalized-EM iterations, i.e. ~40 min
and repeated 0.5 GB allocations -- infeasible for a sweep.

The exact O(N) identity is **already implemented in the same module**: `rts_smoother`
(line ~905) computes

    Sigma_{k,k-1|N} = A_{k-1} * P_{k|N},      A_{k-1} = P_{k-1|k-1} / P_{k|k-1}

which is the standard linear-Gaussian lag-one smoothing covariance (F = 1 for this
random-walk phase model).  This module reuses that recursion inside the EM loop.  It
is not an approximation: verified against the dense matrix to a max relative
difference of 6e-16, and `em_update_fast` reproduces `SLPI.em_update` to 3.4e-15
(see `check_equivalence`).

The second bottleneck is `kalman_filter`, which builds ``S = P H H^T + R I_2``, calls
``np.linalg.inv`` on that 2x2 and applies a Joseph-form update *once per sample, per EM
iteration*.  Since ``||H|| = 1`` the whole update collapses to scalars exactly (see
`ekf_freeze_fast`), which removes ~8000 matrix inversions per iteration at N = 8000.

Measured, full EM to tol=1e-5 at N = 8000, sigma_add = 0.316:

    reference path      ~4.3 s per iteration (dense Sigma) + ~0.4 s (2x2 EKF)  ->  ~40 min
    this module         ~0.017 s per iteration                                 ->  ~18 s

Everything else -- the M-step algebra, the priors, the convergence rule -- is copied
unchanged from the reference implementation, and `check_equivalence` verifies the two
agree to machine precision (max relative difference ~5e-14 across all outputs).

Public API
----------
ekf_freeze_fast             scalar-exact EKF                     (<-> SLPI.kalman_filter)
em_update_fast              one generalized-EM update            (<-> SLPI.em_update)
em_map_parameters_fast      generalized-EM to convergence        (<-> SLPI.em_map_parameters)
smoothing_statistics_fast   frozen-model smoothing statistics    (<-> SLPI.smoothing_statistics_frozen)
missing_information_fast    exact missing-information matrix     (<-> SLPI.missing_information_frozen)
posterior_information_fast  Laplace posterior covariance         (<-> SLPI.posterior_information_frozen)
check_equivalence           machine-precision regression test vs the reference module
"""

from __future__ import annotations

import warnings
from typing import Literal

import numpy as np
from scipy.linalg import solve_banded

import StuartLandauPhaseInference as SLPI


__all__ = [
    "ekf_freeze_fast",
    "em_update_fast",
    "em_map_parameters_fast",
    "smoothing_statistics_fast",
    "missing_information_fast",
    "posterior_information_fast",
    "check_equivalence",
]


# ──────────────────────────────────────────────────────────────────────────────
# Core: scalar KF + RTS smoother with the O(N) lag-one covariance
# ──────────────────────────────────────────────────────────────────────────────

def _kf_rts_lag1(zk: np.ndarray, omega: float, Dtheta: float, R: float,
                 dt: float, mu_theta: float, sigma_theta: float):
    """Scalar Kalman filter + RTS smoother for ``z_k = theta_k + xi_k``.

    Model
    -----
        theta_1 ~ N(mu_theta, sigma_theta^2)
        theta_k = theta_{k-1} + omega*dt + eps,   eps ~ N(0, Q),  Q = 2*D*dt
        z_k     = theta_k + xi_k,                 xi  ~ N(0, R)

    Identical to the loops inside `SLPI.em_update`, except the lag-one smoothed
    covariance is accumulated from the RTS gain instead of being extracted from a
    dense N x N inverse:

        Sigma_{k,k-1|N} = A_{k-1} * P_{k|N}          [exact, O(N)]

    Parameters
    ----------
    zk : ndarray, shape (N,)
        Scalar pseudo-observations.
    omega, Dtheta, R : float
        Natural-scale parameters (frequency [rad/time], phase diffusivity
        [rad^2/time], per-component measurement variance [length^2]).
    dt : float
        Sampling interval [time].
    mu_theta, sigma_theta : float
        Prior mean [rad] and std [rad] on theta_1.

    Returns
    -------
    thetakk, Pkk : ndarray, shape (N,)
        Filtered mean [rad] and variance [rad^2].
    thetakk_1, Pkk_1 : ndarray, shape (N,)
        One-step predicted mean and variance.
    thetakN, PkN : ndarray, shape (N,)
        RTS-smoothed mean and variance.
    Sigmakk_1 : ndarray, shape (N,)
        Lag-one smoothed covariance; ``Sigmakk_1[k]`` is Cov(theta_k, theta_{k-1} | y).
        Entry 0 is unused and set to 0 (matches the reference convention).
    """
    N = zk.size
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

    thetakN = np.empty(N, dtype=float)
    PkN = np.empty(N, dtype=float)
    Sigmakk_1 = np.zeros(N, dtype=float)

    thetakN[-1] = thetakk[-1]
    PkN[-1] = Pkk[-1]

    for k in range(N - 2, -1, -1):
        A = Pkk[k] / Pkk_1[k + 1]
        thetakN[k] = thetakk[k] + A * (thetakN[k + 1] - thetakk_1[k + 1])
        PkN[k] = Pkk[k] + A ** 2 * (PkN[k + 1] - Pkk_1[k + 1])
        # exact lag-one smoothing covariance -- replaces np.diag(theta_covariance(), -1)
        Sigmakk_1[k + 1] = A * PkN[k + 1]

    return thetakk, Pkk, thetakk_1, Pkk_1, thetakN, PkN, Sigmakk_1


def ekf_freeze_fast(param, data, state_priors):
    """Scalar-exact EKF, returning the same tuple as `SLPI.kalman_filter`.

    The reference implementation forms ``S = P H H^T + R I_2``, inverts that 2x2
    matrix and applies a Joseph-form covariance update *at every time step* -- i.e.
    one ``np.linalg.inv`` plus several small matmuls per sample, per EM iteration.
    That is the dominant cost of a fit at N = 8000.

    Because ``||H_k|| = 1`` for ``h(theta) = (cos theta, sin theta)``, the update
    collapses to scalars exactly:

        S^{-1} = R^{-1} [ I - (P/(P+R)) H H^T ]
        K      = P H^T S^{-1} = kappa H^T,        kappa = P/(P+R)
        K nu   = kappa (H . nu)
        K H    = kappa                            (since ||H||^2 = 1)
        P^+    = (1-kappa)^2 P + R kappa^2 = (1-kappa) P

    so the Joseph form reduces algebraically to the standard one. No approximation
    is involved; `check_equivalence` verifies it to machine precision.

    Parameters
    ----------
    param : tuple
        ``(omega, Dtheta, R)`` in natural units.
    data : tuple
        ``(tk, yk)``, ``yk`` shape (N, 2).
    state_priors : tuple
        ``(mu_theta, sigma_theta)``.

    Returns
    -------
    thetakk, Pkk, thetakk_1, Pkk_1, ypred, Hk, ck
        Same objects and shapes as `SLPI.kalman_filter`.
    """
    import math

    omega, Dtheta, R = param
    tk, yk = data
    tk = np.asarray(tk, dtype=float).ravel()
    yk = np.asarray(yk, dtype=float)
    mu_theta, sigma_theta = state_priors

    N = yk.shape[0]
    dt = float(tk[1] - tk[0])
    Q = 2.0 * Dtheta * dt
    wt = omega * dt

    thetakk = np.empty(N, dtype=float)
    Pkk = np.empty(N, dtype=float)
    thetakk_1 = np.empty(N, dtype=float)
    Pkk_1 = np.empty(N, dtype=float)

    y0 = yk[:, 0]
    y1 = yk[:, 1]

    th_post = 0.0
    P_post = 0.0
    for k in range(N):
        if k == 0:
            th_pred = float(mu_theta)
            P_pred = float(sigma_theta) ** 2
        else:
            th_pred = th_post + wt
            P_pred = P_post + Q
        thetakk_1[k] = th_pred
        Pkk_1[k] = P_pred

        c, s = math.cos(th_pred), math.sin(th_pred)
        # nu . H  with H = (-sin, cos), ypred = (cos, sin)
        nu_par = -s * (y0[k] - c) + c * (y1[k] - s)
        kap = P_pred / (P_pred + R)
        th_post = th_pred + kap * nu_par
        P_post = (1.0 - kap) * P_pred
        thetakk[k] = th_post
        Pkk[k] = P_post

    cth = np.cos(thetakk_1)
    sth = np.sin(thetakk_1)
    ypred = np.column_stack([cth, sth])
    Hk = np.column_stack([-sth, cth])
    ck = ypred - Hk * thetakk_1[:, None]
    return thetakk, Pkk, thetakk_1, Pkk_1, ypred, Hk, ck


def _pseudo_obs(yk: np.ndarray, Hk: np.ndarray, ck: np.ndarray):
    """Frozen-model scalar pseudo-observations and orthogonal residual energy.

    With ``||H_k|| = 1``: ``z_k = H_k^T (y_k - c_k)`` and
    ``||y_k - c_k - H_k theta||^2 = (z_k - theta)^2 + orth_sq_k``.
    """
    a = yk - ck
    zk = np.sum(Hk * a, axis=1)
    orth_sq = np.sum(a ** 2, axis=1) - zk ** 2
    return zk, orth_sq


# ──────────────────────────────────────────────────────────────────────────────
# 1.  One generalized-EM update  (mirrors SLPI.em_update)
# ──────────────────────────────────────────────────────────────────────────────

def em_update_fast(old_param, data, priors,
                   param_mode: Literal["log", "natural"] = "log",
                   *, return_aux: bool = False):
    """One generalized-EM update with a frozen EKF linearization.

    Numerically identical to `SLPI.em_update` (max relative difference ~3e-15) but
    O(N) instead of O(N^2) per call.

    Parameters
    ----------
    old_param : array_like, shape (3,)
        ``[omega, Dtheta, R]`` if ``param_mode='natural'``, else
        ``[omega, ln Dtheta, ln R]``.
    data : tuple
        ``(tk, yk)`` with ``tk`` shape (N,) [time] uniformly spaced and ``yk`` shape (N, 2).
    priors : tuple
        ``(mu_theta, sigma_theta, mu_omega, sigma_omega, alpha_D, beta_D, alpha_R, beta_R)``.
    param_mode : {'log', 'natural'}
        Parameterization of input and output.
    return_aux : bool
        If True also return the intermediate-quantity dict (same keys as the reference).

    Returns
    -------
    new_param : list of float
        Updated parameters in the same parameterization as `param_mode`.
    aux : dict, optional
        Returned only when ``return_aux=True``.
    """
    tk, yk = data
    tk = np.asarray(tk, dtype=float).ravel()
    yk = np.asarray(yk, dtype=float)

    if yk.ndim != 2 or yk.shape[1] != 2:
        raise ValueError(f"yk must have shape (N, 2); got {yk.shape}")

    (mu_theta, sigma_theta,
     mu_omega, sigma_omega,
     alpha_D, beta_D,
     alpha_R, beta_R) = priors

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

    # ── E-step part 1: freeze the nonlinear measurement model (unchanged) ──────
    _, _, _, _, _, Hk, ck = ekf_freeze_fast((omega, Dtheta, R), (tk, yk),
                                            (mu_theta, sigma_theta))
    zk, orth_sq = _pseudo_obs(yk, Hk, ck)

    # ── E-step part 2: scalar KF + RTS, with O(N) lag-one covariance ──────────
    (thetakk, Pkk, thetakk_1, Pkk_1,
     thetakN, PkN, Sigmakk_1) = _kf_rts_lag1(zk, omega, Dtheta, R, dt,
                                             mu_theta, sigma_theta)

    # ── Expected sufficient statistics ────────────────────────────────────────
    M = N - 1
    ES0 = float(np.sum((thetakN[1:] - thetakN[:-1]) ** 2
                       + PkN[1:] + PkN[:-1] - 2.0 * Sigmakk_1[1:]))
    ES1 = float(thetakN[-1] - thetakN[0])
    ESy = float(np.sum((zk - thetakN) ** 2 + PkN + orth_sq))

    # ── M-step: closed-form MAP updates under the conjugate priors ────────────
    omega_new = ((ES1 / (2.0 * Dtheta) + mu_omega / sigma_omega ** 2)
                 / (M * dt / (2.0 * Dtheta) + 1.0 / sigma_omega ** 2))

    wdt = omega_new * dt
    B = ES0 - 2.0 * ES1 * wdt + M * (wdt ** 2)

    if param_mode == "natural":
        Dtheta_new = (B + 4.0 * beta_D * dt) / (2.0 * (M + 2.0 * (alpha_D + 1.0)) * dt)
        R_new = (ESy + 2.0 * beta_R) / (2.0 * (N + alpha_R + 1.0))
        new_param = [omega_new, Dtheta_new, R_new]
    else:
        # log-parameterization: the Jacobian shifts alpha by 1 per positive scale param
        Dtheta_new = (B + 4.0 * beta_D * dt) / (2.0 * (M + 2.0 * alpha_D) * dt)
        R_new = (ESy + 2.0 * beta_R) / (2.0 * (N + alpha_R))
        new_param = [omega_new, float(np.log(Dtheta_new)), float(np.log(R_new))]

    if not return_aux:
        return new_param

    aux = dict(
        Hk=Hk, ck=ck, zk=zk, orth_sq=orth_sq,
        filtered_mean=thetakk, filtered_var=Pkk,
        predicted_mean=thetakk_1, predicted_var=Pkk_1,
        smoothed_mean=thetakN, smoothed_var=PkN,
        smoothed_lag1_cov=Sigmakk_1,
        expected_stats=dict(S0=ES0, S1=ES1, Sy=ESy),
        param_old_natural=(omega, Dtheta, R),
        param_new=new_param,
    )
    return new_param, aux


# ──────────────────────────────────────────────────────────────────────────────
# 2.  Generalized EM to convergence  (mirrors SLPI.em_map_parameters)
# ──────────────────────────────────────────────────────────────────────────────

def em_map_parameters_fast(initial_params, data, priors, max_iter: int = 3000,
                           tol: float = 1e-5, param_mode: Literal["log", "natural"] = "log",
                           *, warn: bool = True):
    """Generalized-EM MAP estimation, O(N) per iteration.

    Same iterate sequence and convergence rule as `SLPI.em_map_parameters`
    (``max relative change < tol``).

    Parameters
    ----------
    initial_params : array_like, shape (3,)
        Starting point, in the parameterization given by `param_mode`.
    data : tuple
        ``(tk, yk)``.
    priors : tuple
        Standard 8-tuple, see `em_update_fast`.
    max_iter : int
        Iteration cap.  The noise-injected fits of this study need 250-650
        iterations, so a cap of a few hundred silently truncates them mid-climb.
    tol : float
        Convergence tolerance on the max relative parameter change.
    param_mode : {'log', 'natural'}
    warn : bool
        Emit a RuntimeWarning if the cap is hit before convergence.

    Returns
    -------
    lambda_iter : ndarray, shape (K, 3)
        Parameter trajectory including the initial point, truncated at convergence.
    converged : bool
        Whether `tol` was reached before `max_iter`.
    """
    lam = np.zeros((max_iter, 3), dtype=float)
    lam[0, :] = np.asarray(initial_params, dtype=float)

    converged = False
    rel_change = np.nan
    i = 0

    for i in range(1, max_iter):
        lam[i] = em_update_fast(lam[i - 1], data, priors, param_mode=param_mode)
        denom = np.maximum(np.abs(lam[i - 1]), 1e-12)
        rel_change = np.max(np.abs(lam[i] - lam[i - 1]) / denom)
        if rel_change < tol:
            converged = True
            break

    if not converged and warn:
        warnings.warn(
            f"Generalized-EM MAP did not converge within {max_iter} iterations "
            f"(last relative change = {rel_change:.2e}).",
            RuntimeWarning,
        )

    return lam[: i + 1], converged


# ──────────────────────────────────────────────────────────────────────────────
# 3.  Frozen-model smoothing statistics  (mirrors SLPI.smoothing_statistics_frozen)
# ──────────────────────────────────────────────────────────────────────────────

def smoothing_statistics_fast(param_eval, data, priors, Hk, ck,
                              param_type: Literal["log", "natural"] = "log"):
    """Smoothing statistics for the frozen linearized model, without the dense Sigma.

    Returns the same dict as `SLPI.smoothing_statistics_frozen` except that the
    ``'Sigma'`` entry (the dense N x N covariance) is ``None`` -- no downstream
    consumer in this package reads it; `complete_data_information_frozen` uses only
    the scalar statistics and `missing_information_frozen` rebuilds its own banded
    operators.

    Parameters
    ----------
    param_eval : array_like, shape (3,)
        Parameter at which to evaluate the frozen model.
    data : tuple
        ``(tk, yk)``.
    priors : tuple
        Standard 8-tuple.
    Hk, ck : ndarray, shape (N, 2)
        Frozen linearization terms, typically from `SLPI.freeze_linearization`.
    param_type : {'log', 'natural'}

    Returns
    -------
    stats : dict
        Same keys/units as the reference implementation.
    """
    tk, yk = data
    tk = np.asarray(tk, dtype=float).ravel()
    yk = np.asarray(yk, dtype=float)
    Hk = np.asarray(Hk, dtype=float)
    ck = np.asarray(ck, dtype=float)

    N = yk.shape[0]
    dt = float(tk[1] - tk[0])

    (mu_theta, sigma_theta,
     mu_omega, sigma_omega,
     alpha_D, beta_D,
     alpha_R, beta_R) = priors

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

    zk, orth_sq = _pseudo_obs(yk, Hk, ck)
    (thetakk, Pkk, thetakk_1, Pkk_1,
     thetakN, PkN, Sigmakk_1) = _kf_rts_lag1(zk, omega, Dtheta, R, dt,
                                             mu_theta, sigma_theta)

    ES0 = float(np.sum((thetakN[1:] - thetakN[:-1]) ** 2
                       + PkN[1:] + PkN[:-1] - 2.0 * Sigmakk_1[1:]))
    ES1 = float(thetakN[-1] - thetakN[0])
    ESy = float(np.sum((zk - thetakN) ** 2 + PkN + orth_sq))

    return {
        "tk": tk, "yk": yk, "dt": dt, "N": N,
        "zk": zk, "orth_sq": orth_sq,
        "m": thetakN, "PkN": PkN,
        "Sigmakk_1": Sigmakk_1,
        "Sigma": None,                       # dense matrix deliberately not built
        "Hk": Hk, "ck": ck,
        "filtered_mean": thetakk, "filtered_var": Pkk,
        "predicted_mean": thetakk_1, "predicted_var": Pkk_1,
        "lambda_natural": (omega, Dtheta, R),
        "lambda_log": (omega, lnDtheta, lnR),
        "ES0": ES0, "ES1": ES1, "ESy": ESy,
        "sigma_theta2": float(sigma_theta) ** 2,
    }


# ──────────────────────────────────────────────────────────────────────────────
# 4.  Missing information  (mirrors SLPI.missing_information_frozen, chunked traces)
# ──────────────────────────────────────────────────────────────────────────────

def missing_information_fast(statistics, priors=None,
                             param_type: Literal["log", "natural"] = "log",
                             *, chunk: int = 256) -> np.ndarray:
    """Exact missing-information matrix ``I_miss = Cov(score_c | y, lambda)``.

    Same algebra as `SLPI.missing_information_frozen`, but the two ``tr(.)``
    accumulation loops (which call `solve_banded` once per unit vector, 2N scipy
    calls) are batched into column blocks.  Identical output; ~20x faster at
    N = 8000 with bounded memory (``chunk`` columns at a time).

    Parameters
    ----------
    statistics : dict
        Output of `smoothing_statistics_fast` (or the reference version).
    priors : tuple or None
        Standard 8-tuple; only ``sigma_omega`` is used, and only in `param_type`
        bookkeeping consistent with the reference (which ignores it here).
    param_type : {'log', 'natural'}
    chunk : int
        Number of RHS columns solved per banded solve.

    Returns
    -------
    I_miss : ndarray, shape (3, 3)
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

    M = N - 1
    q = 2.0 * D * dt
    invR, invq, invsig = 1.0 / R, 1.0 / q, 1.0 / sigma_theta2

    d = np.empty(N, dtype=float)
    d[0] = invR + invq + invsig
    if N > 2:
        d[1:N - 1] = invR + 2.0 * invq
    d[N - 1] = invR + invq

    ab = np.zeros((3, N), dtype=float)
    ab[0, 1:] = -invq
    ab[1, :] = d
    ab[2, :-1] = -invq

    def apply_sigma(b):
        return solve_banded((1, 1), ab, b)

    def apply_A0(x):
        """A0 = B^T B, the discrete second-difference operator (works column-wise)."""
        x = np.asarray(x, dtype=float)
        out = np.empty_like(x)
        if N == 2:
            out[0] = x[0] - x[1]
            out[1] = -x[0] + x[1]
            return out
        out[0] = x[0] - x[1]
        out[1:N - 1] = -x[0:N - 2] + 2.0 * x[1:N - 1] - x[2:N]
        out[N - 1] = -x[N - 2] + x[N - 1]
        return out

    def apply_B(theta):
        return theta[1:] - theta[:-1]

    def apply_Bt(v):
        v = np.asarray(v, dtype=float)
        w = np.zeros((N,) + v.shape[1:], dtype=float)
        w[0] = -v[0]
        if N > 2:
            w[1:N - 1] = v[0:M - 1] - v[1:M]
        w[N - 1] = v[M - 1]
        return w

    def apply_sigma_u(v):
        return apply_B(apply_sigma(apply_Bt(v)))

    r = z - m
    c = np.zeros(N, dtype=float)
    c[0], c[-1] = -1.0, +1.0
    onesM = np.ones(M, dtype=float)
    mu_u = apply_B(m)

    Sig_c = apply_sigma(c)
    V_S1 = float(c @ Sig_c)

    Sig_r = apply_sigma(r)
    C_S1_Sy = float(-2.0 * (c @ Sig_r))

    SigU_mu = apply_sigma_u(mu_u)
    C_S1_S0 = float(2.0 * (onesM @ SigU_mu))
    mu_SigU_mu = float(mu_u @ SigU_mu)

    # --- chunked traces: tr(Sigma^2) and tr(A0 Sigma^2) ---
    tr_Sig2 = 0.0
    tr_A0Sig2 = 0.0
    for s in range(0, N, chunk):
        e = np.zeros((N, min(chunk, N - s)), dtype=float)
        for j in range(e.shape[1]):
            e[s + j, j] = 1.0
        x = apply_sigma(e)
        tr_Sig2 += float(np.sum(x * x))
        tr_A0Sig2 += float(np.sum(x * apply_A0(x)))

    # --- chunked trace: tr(Sigma_U^2), Sigma_U = B Sigma B^T ---
    tr_SigU2 = 0.0
    for s in range(0, M, chunk):
        ej = np.zeros((M, min(chunk, M - s)), dtype=float)
        for j in range(ej.shape[1]):
            ej[s + j, j] = 1.0
        zv = apply_sigma_u(ej)
        tr_SigU2 += float(np.sum(zv * zv))

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
        I[1, 1] = Var_Q / ((4.0 * D * dt) ** 2)
        I[2, 2] = Var_Sy / (4.0 * R * R)
        I[0, 2] = C_S1_Sy / (4.0 * D * R)
        I[1, 2] = Cov_Q_Sy / (8.0 * D * R * dt)

    elif param_type == "natural":
        I[0, 0] = V_S1 / (4.0 * D * D)
        I[0, 1] = Cov_S1_Q / (8.0 * D ** 3 * dt)
        I[1, 1] = Var_Q / (16.0 * D ** 4 * dt ** 2)
        I[2, 2] = Var_Sy / (4.0 * R ** 4)
        I[0, 2] = C_S1_Sy / (4.0 * D * R ** 2)
        I[1, 2] = Cov_Q_Sy / (8.0 * D ** 2 * R ** 2 * dt)

    else:
        raise ValueError("param_type must be 'natural' or 'log'")

    I[1, 0] = I[0, 1]
    I[2, 0] = I[0, 2]
    I[2, 1] = I[1, 2]
    return I


def posterior_information_fast(param_eval, data, priors, Hk, ck,
                               param_type: Literal["log", "natural"] = "log"):
    """Laplace posterior information/covariance for the frozen model, O(N) smoothing.

    ``I_obs = I_prior + I_comp - I_miss``, matching
    `SLPI.posterior_information_frozen` but using the fast smoothing statistics.

    Returns
    -------
    dict
        Keys ``I_prior, I_comp, I_miss, I_obs, cov, rho, em_rate, stats``.
        ``cov`` is the 3x3 Laplace covariance of ``(omega, ln D, ln R)``;
        ``rho`` is the EM contraction rate (-> 1 means a flat/unidentified direction).
    """
    stats = smoothing_statistics_fast(param_eval, data, priors, Hk, ck, param_type=param_type)

    I_prior = SLPI.prior_information(param_eval, priors, param_type=param_type)
    I_comp = SLPI.complete_data_information_frozen(stats, param_type=param_type)
    I_miss = missing_information_fast(stats, priors, param_type=param_type)

    I_obs = I_prior + I_comp - I_miss
    I_obs = 0.5 * (I_obs + I_obs.T)
    cov = np.linalg.inv(I_obs)

    A = I_prior + I_comp
    A = 0.5 * (A + A.T)
    Mmat = I_miss @ np.linalg.inv(A)
    rho = float(np.max(np.real(np.linalg.eigvals(Mmat))))
    em_rate = float(-np.log(rho)) if rho > 0 else np.inf

    return {"I_prior": I_prior, "I_comp": I_comp, "I_miss": I_miss,
            "I_obs": I_obs, "cov": cov, "rho": rho, "em_rate": em_rate,
            "stats": stats}


# ──────────────────────────────────────────────────────────────────────────────
# 5.  Regression test against the reference implementation
# ──────────────────────────────────────────────────────────────────────────────

def check_equivalence(N: int = 600, dt: float = 0.02, seed: int = 0, tol: float = 1e-12):
    """Verify the fast routines reproduce `StuartLandauPhaseInference` to machine precision.

    Builds a synthetic dataset from `SLPI.simulate_data` and compares, element-wise
    and in relative terms: one `em_update`, the lag-one covariance, the sufficient
    statistics ``(ES0, ES1, ESy)``, and the Laplace covariance.

    Parameters
    ----------
    N : int
        Number of samples (kept small so the dense reference is affordable).
    dt : float
        Sampling interval [time].
    seed : int
        RNG seed.
    tol : float
        Max relative difference allowed before an AssertionError is raised.

    Returns
    -------
    report : dict
        Max relative differences, keyed by quantity.
    """
    rng = np.random.default_rng(seed)
    omega, D, R = 1.0, 1e-3, 4e-4
    state_priors = (0.0, 0.05)
    tk, _, yk = SLPI.simulate_data((omega, D, R), N, dt, rng, state_priors)
    data = (tk, yk)

    D_init, R_init = 5e-4, 5e-4
    priors = state_priors + (omega, omega, 2.0, 2.0 * D_init, 2.0, 2.0 * R_init)
    init = [omega, np.log(D_init), np.log(R_init)]

    def _rel(x, y):
        x, y = np.asarray(x, float), np.asarray(y, float)
        den = np.maximum(np.abs(x), 1e-300)
        return float(np.max(np.abs(x - y) / den))

    report = {}

    # (a0) the scalar-exact EKF vs the reference 2x2 implementation
    ref_kf = SLPI.kalman_filter((omega, D, R), data, state_priors)
    fast_kf = ekf_freeze_fast((omega, D, R), data, state_priors)
    names = ("thetakk", "Pkk", "thetakk_1", "Pkk_1", "ypred", "Hk", "ck")
    report["ekf_freeze"] = max(
        float(np.max(np.abs(r - f) / np.maximum(np.abs(r), 1e-300)))
        for nm, r, f in zip(names, ref_kf, fast_kf)
    )

    # (a) one EM update
    ref = np.asarray(SLPI.em_update(init, data, priors, param_mode="log"), float)
    fast = np.asarray(em_update_fast(init, data, priors, param_mode="log"), float)
    report["em_update"] = _rel(ref, fast)

    # (b) lag-one covariance: dense sub-diagonal vs the RTS identity
    Sigma = SLPI.theta_covariance((omega, D, R), N, dt, state_priors)
    zk, _ = _pseudo_obs(yk, *SLPI.freeze_linearization(init, data, state_priors)[:2])
    *_, Sig1 = _kf_rts_lag1(zk, omega, D, R, dt, *state_priors)
    report["lag1_cov"] = _rel(np.diag(Sigma, k=-1), Sig1[1:])

    # (c) smoothing statistics
    Hk, ck, _ = SLPI.freeze_linearization(init, data, state_priors)
    s_ref = SLPI.smoothing_statistics_frozen(init, data, priors, Hk, ck, param_type="log")
    s_fast = smoothing_statistics_fast(init, data, priors, Hk, ck, param_type="log")
    report["ES0_ES1_ESy"] = _rel([s_ref["ES0"], s_ref["ES1"], s_ref["ESy"]],
                                 [s_fast["ES0"], s_fast["ES1"], s_fast["ESy"]])

    # (d) Laplace covariance
    c_ref = SLPI.posterior_information_frozen(init, data, priors, Hk, ck, param_type="log")
    c_fast = posterior_information_fast(init, data, priors, Hk, ck, param_type="log")
    report["laplace_cov"] = _rel(c_ref["cov"], c_fast["cov"])
    report["rho_em"] = _rel(c_ref["rho"], c_fast["rho"])

    worst = max(report.values())
    if worst > tol:
        raise AssertionError(f"FastPhaseEM deviates from the reference: {report}")
    return report


if __name__ == "__main__":
    rep = check_equivalence()
    print("FastPhaseEM.check_equivalence() -- max relative differences vs StuartLandauPhaseInference:")
    for k, v in rep.items():
        print(f"  {k:>14s} : {v:.3e}")
    print("all within machine precision.")
