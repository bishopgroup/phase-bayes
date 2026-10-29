"""
Noise-injection convergence study for Stuart-Landau phase inference.

Sweeps the EM-MAP pipeline over (injected noise level) x (number of inference
points) x (independent realization), so that estimator *bias* can be separated
from realization *scatter* -- which a single-trajectory sweep cannot do.

Injection
---------
``y <- y + sigma_add * N(0, I_2)``: isotropic Cartesian noise added to the
measurement.  Note this is isotropic *as injected*, but the measurement error it
produces is not: the pre-existing error is purely radial (the measured angle IS
the true phase at sigma_add = 0), so the total error has radial variance
``sigma_rho^2 + sigma_add^2`` against angular variance ``sigma_add^2 / rbar^2``.

Because ``N_INF`` and the observation window are locked by ``T = N_INF * delta_t_inf``,
the N_grid below is simultaneously a data-quantity axis and a maximum-lag axis
(T = 40 ... 10240).

The heavy lifting uses `FastPhaseEM`, which is machine-precision-equivalent to
`StuartLandauPhaseInference` but O(N) per EM iteration instead of O(N^2)
(see that module's docstring).

Results go to ``data/sweep_results.parquet`` (one row per fit), which
``Figure2_DiffusivityVsAddedNoise.ipynb`` reads.  Two grids are provided::

    python NoiseInjectionSweep.py                    # 'figure' grid, ~4 CPU-h
    python NoiseInjectionSweep.py --grid full        # full grid,     ~99 CPU-h
    python NoiseInjectionSweep.py --no-resume --outdir data_regen   # recompute from scratch
    python NoiseInjectionSweep.py --check-sim        # validate the thinned integrator

Resuming is the default: cells already present in ``--outdir`` are reused, not refit.
"""

from __future__ import annotations

import argparse
import os
import time
import warnings
from dataclasses import dataclass, asdict, replace
from multiprocessing import get_context
from pathlib import Path

import numpy as np

import FastPhaseEM
import StuartLandauPhaseInference as SLPI
from StuartLandauSimulation import simulate_stuart_landau

_HERE = Path(__file__).resolve().parent


# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Config:
    """Physical, simulation and sweep settings (units in comments)."""

    # physics
    a: float = 1.0            # radial relaxation rate            [time^-1]
    b: float = 1.0            # non-isochronicity / shear         [dimensionless]
    Dx: float = 1e-3          # isotropic Cartesian noise strength [length^2/time]

    # simulation
    dt: float = 0.002         # Euler-Maruyama step               [time]
    seed: int = 42            # trajectory RNG seed
    n_rep: int = 8            # independent realizations (M in simulate_stuart_landau)

    # inference window
    delta_t_inf: float = 0.02   # spacing between inference samples [time]
    N_INF_max: int = 512000     # largest inference-point count -> T = 10240

    # EM
    window_size: int = 6       # initial-phase regression window
    max_iter: int = 40000      # EM cap. Most fits converge in <600 iterations, but a handful of
                               # angle-mode fits at large sigma_add need many thousands; a low cap
                               # silently truncates exactly those and corrupts their D_map.
    tol: float = 1e-5

    # sweep grid.  sigma_grid is dense across 0.02-0.1 because the masking-at-tau_r
    # hypothesis predicts an optimum at sqrt(MSD_theta(tau_r)) ~ 0.034, and a plateau
    # there has to be resolved well enough to measure rather than eyeball.
    sigma_grid: tuple = (0.0, 0.01, 0.02, 0.03, 0.045, 0.06, 0.08,
                         0.1, 0.15, 0.2, 0.3, 0.5, 1.0)
    # T = 40 ... 10240.  The simulator draws its Wiener increments *inside* the time
    # loop (M values per step), so lengthening the trajectory leaves every earlier
    # sample bit-identical -- verified with np.array_equal.  Extending N_grid
    # therefore does not invalidate the fits already in sweep_results.parquet, and
    # run_sweep resumes rather than recomputing them.
    N_grid: tuple = (2000, 4000, 8000, 16000, 32000, 64000, 128000, 256000, 512000)
    init_modes: tuple = ("default", "informed")
    noise_seed_base: int = 20260809

    # The Laplace covariance (posterior_information_frozen) costs O(N^2) -- its trace
    # terms dominate the whole fit at N >= 16000 -- while the EM itself is O(N).  It is
    # only needed for the *claimed* error bar; the honest error bar is the spread across
    # replicates, which is free.  So compute it on every replicate up to
    # `cov_all_below`, and only on the first `n_cov_reps` above that.
    # Measured at N=16000: 2.1 s without the covariance, 38.5 s with it.
    #
    # The O(N^2) is not survivable past N=32000: the two trace accumulations in
    # `missing_information_fast` would cost ~16x (N=64000) and ~64x (N=128000) the
    # N=16000 figure, i.e. 10 min and 40 min *per fit*, dwarfing the O(N) EM.  Above
    # `cov_max_N` the Laplace interval is therefore not computed at all; the
    # replicate spread (8 seeds) is the error bar there, and it is the honest one.
    cov_all_below: int = 8000
    n_cov_reps: int = 2
    cov_max_N: int = 32000

    # ── derived ───────────────────────────────────────────────────────────────
    @property
    def stride(self) -> int:
        """Integration steps between successive inference samples."""
        s = int(round(self.delta_t_inf / self.dt))
        if abs(s * self.dt - self.delta_t_inf) > 1e-12:
            raise ValueError("delta_t_inf must be an integer multiple of dt")
        return s

    @property
    def N_steps(self) -> int:
        """Integration steps needed to supply N_INF_max inference samples."""
        return self.N_INF_max * self.stride

    @property
    def tau_r(self) -> float:
        """Radial relaxation time 1/(2a) [time]."""
        return 1.0 / (2.0 * self.a)

    @property
    def D_eff(self) -> float:
        """Long-time effective phase diffusivity Dx(1 + b^2/a^2) [rad^2/time]."""
        return self.Dx * (1.0 + self.b ** 2 / self.a ** 2)

    @property
    def sigma_rho(self) -> float:
        """RMS radial fluctuation sqrt(Dx/2a) [length]."""
        return float(np.sqrt(self.Dx / (2.0 * self.a)))

    # Keys that change the *value* of a fit.  Everything else (N_grid, sigma_grid,
    # init_modes, N_INF_max, and the covariance-budget knobs) only selects which cells
    # are run, so a cache built with a different grid stays valid and can be extended.
    CORE_KEYS = ("a", "b", "Dx", "dt", "seed", "n_rep", "delta_t_inf",
                 "window_size", "max_iter", "tol", "noise_seed_base")

    def signature(self) -> dict:
        """Grid fingerprint, used to invalidate a stale cache."""
        d = asdict(self)
        d["stride"] = self.stride
        d["N_steps"] = self.N_steps
        return d

    def core_signature(self) -> dict:
        """Fingerprint of the settings a cached fit must agree with to be reusable."""
        d = asdict(self)
        return {k: d[k] for k in self.CORE_KEYS} | {"stride": self.stride}


# ──────────────────────────────────────────────────────────────────────────────
# Trajectory
# ──────────────────────────────────────────────────────────────────────────────

def simulate_ensemble(cfg: Config, thin: bool = False, chunk_steps: int = 200_000):
    """Integrate `cfg.n_rep` independent Stuart-Landau realizations.

    Parameters
    ----------
    cfg : Config
    thin : bool
        Keep only the samples the inference actually uses (every ``cfg.stride``-th),
        discarding the integration grid as it is produced.  Nothing downstream reads
        the full-resolution trajectory -- `make_window` is the only consumer and it
        takes ``idx = (arange(N_INF) + start) * stride`` -- so this is lossless for
        this module's purposes and cuts memory by ``stride`` (10x).

        It is what makes the largest budgets runnable at all: at N_INF_max = 512000
        the full form holds six (n_rep, 5.12M) float64 arrays = 1.68 GB resident, and
        `np.unwrap` needs a same-size temporary on top, for a ~3.4 GB peak.  Thinned,
        the same run is 0.17 GB.
    chunk_steps : int
        Integration steps per block.  Only affects peak memory, never the result.

    Returns
    -------
    traj : dict
        ``t`` (N+1,), ``x1``/``x2``/``r``/``theta``/``Phi`` each (n_rep, N+1), plus
        ``stride``: the spacing already applied to those arrays, which is 1 when
        ``thin`` is set and ``cfg.stride`` otherwise.  Consumers index through that
        key rather than assuming, so both forms are interchangeable.
        ``theta`` is the unwrapped polar angle [rad]; ``Phi = theta + (b/a) ln r``.

    Notes
    -----
    The thinned path reproduces `StuartLandauSimulation.simulate_stuart_landau` **bit for
    bit**: it draws ``rng.standard_normal(M)`` for x1 then for x2, once per step, in
    that order, so the Wiener increments are identical draw-for-draw.  The unwrap
    winding number is carried across chunk boundaries rather than restarted, and the
    angle is unwrapped on the integration grid before thinning, so the result does not
    depend on `chunk_steps` or on `stride`.  `check_thinned_simulation` asserts all of
    this; run it before trusting a new budget.
    """
    if not thin:
        t, x1, x2, r, theta, Phi = simulate_stuart_landau(
            a=cfg.a, b=cfg.b, Dx=cfg.Dx, dt=cfg.dt, N=cfg.N_steps,
            M=cfg.n_rep, seed=cfg.seed)
        return {"t": t, "x1": x1, "x2": x2, "r": r, "theta": theta, "Phi": Phi,
                "stride": cfg.stride}

    N, M, s = cfg.N_steps, cfg.n_rep, cfg.stride
    a, b, Dx, dt = cfg.a, cfg.b, cfg.Dx, cfg.dt
    n_keep = N // s + 1

    rng = np.random.default_rng(cfg.seed)
    noise_amp = np.sqrt(2.0 * Dx * dt)

    # state at the current step (theta0 = 0, matching the reference's default)
    x1c = np.cos(np.zeros(M))
    x2c = np.sin(np.zeros(M))

    out_x1 = np.empty((M, n_keep)); out_x2 = np.empty((M, n_keep))
    out_th = np.empty((M, n_keep))
    out_x1[:, 0], out_x2[:, 0] = x1c, x2c
    out_th[:, 0] = np.arctan2(x2c, x1c)

    # Running unwrap state on the INTEGRATION grid, carried across chunks so the
    # result cannot depend on chunk_steps.  This mirrors np.unwrap's *formulation*,
    # not just its output: numpy computes `theta = raw_angle + cumsum(correction)`,
    # where the correction is ~0 except at wraps.  Accumulating the increments
    # instead is algebraically identical but drifts by ~1 ulp of the running total
    # (measured 1.7e-13 rad over 40k steps), which would make the trajectory differ
    # from the cached one.  Adding a near-zero correction to the exact raw angle does
    # not drift.
    prev_raw = np.arctan2(x2c, x1c)
    corr = np.zeros(M)                        # cumsum(correction) at prev_raw

    done = 0
    while done < N:
        n_blk = min(chunk_steps, N - done)
        bx1 = np.empty((M, n_blk)); bx2 = np.empty((M, n_blk))
        for n in range(n_blk):
            r2 = x1c ** 2 + x2c ** 2
            factor = 1.0 + b * (r2 - 1.0)
            F1 = a * (1.0 - r2) * x1c - factor * x2c
            F2 = a * (1.0 - r2) * x2c + factor * x1c
            x1c = x1c + F1 * dt + noise_amp * rng.standard_normal(M)
            x2c = x2c + F2 * dt + noise_amp * rng.standard_normal(M)
            bx1[:, n], bx2[:, n] = x1c, x2c

        # unwrap this block against the carried state, on the integration grid,
        # following np.unwrap step for step (discont = pi)
        raw = np.arctan2(bx2, bx1)
        dd = np.diff(np.concatenate([prev_raw[:, None], raw], axis=1), axis=1)
        ddmod = np.mod(dd + np.pi, 2.0 * np.pi) - np.pi
        np.copyto(ddmod, np.pi, where=(ddmod == -np.pi) & (dd > 0))
        ph = ddmod - dd
        ph[np.abs(dd) < np.pi] = 0.0
        cum = corr[:, None] + np.cumsum(ph, axis=1)
        th_blk = raw + cum
        prev_raw = raw[:, -1].copy()
        corr = cum[:, -1].copy()

        # global step indices this block covers are done+1 .. done+n_blk
        first = done + 1
        k0 = -(-first // s)                       # first multiple of s in the block
        if k0 * s <= done + n_blk:
            cols = np.arange(k0 * s, done + n_blk + 1, s) - first
            slot = np.arange(k0, k0 + cols.size)
            out_x1[:, slot] = bx1[:, cols]
            out_x2[:, slot] = bx2[:, cols]
            out_th[:, slot] = th_blk[:, cols]
        done += n_blk
        del bx1, bx2, raw, dd, ddmod, ph, cum, th_blk

    out_r = np.sqrt(out_x1 ** 2 + out_x2 ** 2)
    out_Phi = out_th + (b / a) * np.log(np.maximum(out_r, 1e-12))
    t = np.arange(n_keep, dtype=float) * (dt * s)
    return {"t": t, "x1": out_x1, "x2": out_x2, "r": out_r,
            "theta": out_th, "Phi": out_Phi, "stride": 1}


def check_thinned_simulation(cfg: Config | None = None, N_INF: int = 4000,
                             chunk_steps: int = 7000, *, fit_check: bool = True) -> dict:
    """Assert the thinned integrator reproduces the reference, to the bar that matters.

    New rows are only comparable with the cached ones if the trajectory is unchanged,
    so this is a precondition for extending the grid, not a nicety.  `chunk_steps`
    deliberately need not divide the step count or the stride, so a boundary bug
    cannot hide.

    Two different tolerances, for a reason:

    - ``x1``, ``x2``, ``r``, ``t`` must match **exactly**.  These are what the
      estimator sees: `make_window` builds ``yk`` from ``x1``/``x2``, and the whole
      EM/EKF/RTS chain reads nothing else.  Exact here means the *fits* are exact.
    - ``theta``, ``Phi`` are allowed a few ulp (bar: 1e-9 rad).  `np.cumsum` over one
      array and the same sum accumulated block-by-block associate differently, which
      costs a few ulp of a running total that reaches ~10^3 rad -- measured 2.8e-14.
      These arrays never enter the fit; inside `fit_window` they are used only for
      ``sigma_psi_emp``, the coverage/RMSE diagnostics and ``omega_*_emp``, where a
      1e-14 rad perturbation is twelve orders of magnitude below the phase noise.

    With ``fit_check`` the claim is verified end to end rather than argued: a full
    `fit_window` is run on both trajectories and the fitted parameters compared.

    Returns
    -------
    dict
        Max absolute difference per array, plus the fitted-parameter differences.
    """
    cfg = Config() if cfg is None else cfg
    c = replace(cfg, N_INF_max=N_INF)

    ref = simulate_ensemble(c, thin=False)
    thin = simulate_ensemble(c, thin=True, chunk_steps=chunk_steps)

    s = c.stride
    report = {}
    for k in ("x1", "x2", "r", "theta", "Phi"):
        report[k] = float(np.max(np.abs(ref[k][:, ::s] - thin[k])))
    report["t"] = float(np.max(np.abs(ref["t"][::s] - thin["t"])))

    exact = ("x1", "x2", "r", "t")
    bad = {k: report[k] for k in exact if report[k] != 0.0}
    if bad:
        raise AssertionError(
            f"thinned simulation differs in what the estimator reads: {bad} "
            "(these must be bit-identical)")
    loose = {k: report[k] for k in ("theta", "Phi") if report[k] > 1e-9}
    if loose:
        raise AssertionError(
            f"thinned simulation's unwrapped phase drifts beyond rounding: {loose}")

    if fit_check:
        sig = 0.1
        seed = noise_seed_for(c, sig, 0)
        rows = []
        for tr in (ref, thin):
            rows.append(fit_window(*make_window(tr, c, sig, N_INF, 0, seed), c,
                                   want_cov=False))
        for k in ("omega_map", "D_map", "R_map", "n_em"):
            report[f"fit_{k}"] = abs(rows[0][k] - rows[1][k])
        if any(report[f"fit_{k}"] != 0.0 for k in ("omega_map", "D_map", "R_map")):
            raise AssertionError(
                f"thinned trajectory changes the FIT: {[(k, report['fit_'+k]) for k in ('omega_map','D_map','R_map')]}")

    return report


# ──────────────────────────────────────────────────────────────────────────────
# Window construction
# ──────────────────────────────────────────────────────────────────────────────

def make_window(traj, cfg: Config, sigma_add: float, N_INF: int,
                m: int, noise_seed: int, *, start: int = 0):
    """Build one inference window: subsample, then add isotropic Cartesian noise.

    Parameters
    ----------
    traj : dict
        Output of `simulate_ensemble`.
    cfg : Config
    sigma_add : float
        Injected noise std [same length units as x].
    N_INF : int
        Number of inference samples.
    m : int
        Realization index.
    noise_seed : int
        RNG seed for the injected noise.
    start : int
        Index of the first inference sample, so that disjoint sub-windows can be
        carved out of one trajectory.

    Returns
    -------
    tk : ndarray (N_INF,)          sample times [time]
    yk : ndarray (N_INF, 2)        measurements handed to the filter
    theta_win : ndarray (N_INF,)   true geometric phase on the grid [rad]
    Phi_win : ndarray (N_INF,)     true asymptotic phase on the grid [rad]
    """
    # The spacing already applied to `traj` -- cfg.stride for a full-resolution
    # trajectory, 1 for a thinned one (see `simulate_ensemble(thin=True)`).  Reading
    # it from the trajectory rather than assuming keeps both forms interchangeable,
    # and this is the only place the trajectory is indexed.
    st = traj.get("stride", cfg.stride)
    idx = (np.arange(N_INF) + start) * st
    tk = traj["t"][idx]
    theta_win = traj["theta"][m, idx]
    Phi_win = traj["Phi"][m, idx]

    yk = np.column_stack([traj["x1"][m, idx], traj["x2"][m, idx]])
    if sigma_add > 0.0:
        rng = np.random.default_rng(noise_seed)
        yk = yk + sigma_add * rng.standard_normal(yk.shape)

    return tk, yk, theta_win, Phi_win


# ──────────────────────────────────────────────────────────────────────────────
# Single fit
# ──────────────────────────────────────────────────────────────────────────────

def wrapped_angle_error(yk, theta_win):
    """Angular measurement error, wrapped to (-pi, pi], with its circular summaries.

    ``np.unwrap`` is meaningless once the per-sample angular error exceeds pi (it
    turns the error sequence into an unbounded random walk), so the error is taken
    as a wrapped difference instead.

    Returns
    -------
    d : ndarray (N,)
        Wrapped error about its circular mean [rad].
    sigma_wrapped : float
        RMS of `d` [rad].  Saturates at pi/sqrt(3) ~ 1.81 for a uniform angle.
    sigma_circ : float
        Wrapped-normal std estimate ``sqrt(-2 ln |<e^{i delta}>|)`` [rad], which
        keeps tracking the true injected sigma past the wrapping point.
    """
    meas = np.arctan2(yk[:, 1], yk[:, 0])
    dz = np.exp(1j * (meas - theta_win))
    mz = dz.mean()
    d = np.angle(dz * np.conj(mz) / max(abs(mz), 1e-300))
    Rbar = min(abs(mz), 1.0 - 1e-15)
    sigma_circ = float(np.sqrt(-2.0 * np.log(Rbar))) if Rbar > 0 else np.inf
    return d, float(np.sqrt(np.mean(d ** 2))), sigma_circ


def estimate_white_angular_variance(tk, yk):
    """Model-free estimate of the white angular-noise variance on the measurement.

    For an observed angle ``psi_k = theta_k + eps_k`` with white ``eps``, the
    time-averaged MSD satisfies ``MSD(m) = MSD_theta(m) + 2 sigma_eps^2``.  Taking
    ``MSD_theta`` locally linear over the first two lags gives the standard
    localization-error estimator

        sigma_eps^2 = (2 MSD(1) - MSD(2)) / 2

    Parameters
    ----------
    tk : ndarray (N,)
    yk : ndarray (N, 2)

    Returns
    -------
    float
        Estimated ``sigma_eps^2`` [rad^2], floored at 0.  Unreliable once the
        per-step angular error approaches pi (unwrapping breaks down), which is
        why callers combine it with the radial heuristic rather than trusting it
        alone.
    """
    psi = np.unwrap(np.arctan2(yk[:, 1], yk[:, 0]))
    m1 = float(np.mean(np.diff(psi, n=1) ** 2))
    m2 = float(np.mean((psi[2:] - psi[:-2]) ** 2))
    return max(0.5 * (2.0 * m1 - m2), 0.0)


def fit_window(tk, yk, theta_win, Phi_win, cfg: Config, *, want_cov: bool = True,
               keep_arrays: bool = False, init_mode: str = "default"):
    """Run the full EKF/RTS + EM-MAP pipeline on one window.

    Same pipeline as `StuartLandauInference.ipynb`
    (FFT frequency seed -> initial-phase regression -> conjugate priors ->
    generalized EM -> Kalman filter -> RTS smoother), with `FastPhaseEM` in place of
    the O(N^2) EM.

    Parameters
    ----------
    tk, yk, theta_win, Phi_win
        Output of `make_window`.
    cfg : Config
    want_cov : bool
        Also compute the Laplace posterior covariance (omega_std, lnD_std, lnR_std,
        rho_em).  Adds a few seconds at N = 8000.
    keep_arrays : bool
        Also return the per-sample arrays needed for the phase/MSD figures.
    init_mode : {'default', 'informed'}
        How the EM is seeded.

        ``default``   the shipped pipeline heuristic: ``R_crude = 0.5 <(|y|-1)^2>``,
                      i.e. the *radial* spread is used to guess the measurement
                      noise.  This is what every existing sweep in this project ran.

        ``informed``  seed R with the *angular* white-noise variance measured
                      directly from the data (`estimate_white_angular_variance`),
                      scaled to Cartesian units by ``<|y|>^2``.

        The distinction matters because R and D are aliased: white noise on the
        observed angle can be explained either as measurement noise (R ~ sigma_psi^2)
        or as process noise (D ~ sigma_psi^2 / dt).  The radial heuristic returns ~0
        whenever the measurement sits on the unit circle, no matter how noisy the
        angle is, which seeds the EM inside the wrong basin.  Running both seeds and
        comparing `nlp` shows which solution the objective actually prefers.

    Returns
    -------
    row : dict
        Scalar results.  Units: ``omega_map`` [rad/time], ``D_map`` [rad^2/time],
        ``R_map`` [length^2], ``sigma_psi_*`` [rad], RMSEs [rad].
    arrays : dict, optional
        Returned only when ``keep_arrays=True``.
    """
    N_INF = yk.shape[0]
    dt_inf = float(tk[1] - tk[0])
    r_inf = np.sqrt(yk[:, 0] ** 2 + yk[:, 1] ** 2)
    rbar = float(np.mean(r_inf))

    # 1) frequency seed
    omega_FFT, _, _ = SLPI.estimate_frequency_fft((tk, yk))

    # 2) crude noise scale + initial-phase prior
    R_crude = max(0.5 * float(np.mean((r_inf - 1.0) ** 2)), 1e-7)
    sigma_psi2_hat = estimate_white_angular_variance(tk, yk)

    if init_mode == "default":
        R_init = R_crude
    elif init_mode == "informed":
        R_init = max(R_crude, sigma_psi2_hat * rbar ** 2, 1e-7)
    else:
        raise ValueError(f"unknown init_mode {init_mode!r}")

    pinit = SLPI.estimate_initial_phase_regression((tk, yk), omega_FFT,
                                                   cfg.window_size, R_init)
    mu_theta = pinit["phi_mean"] + omega_FFT * tk[0]
    sigma_theta = max(2.0 * pinit["phi_std"], 0.01)
    state_priors = (mu_theta, sigma_theta)

    # 3) conjugate priors
    D_init = max(R_init / 20.0, 1e-7)
    priors = state_priors + (omega_FFT, omega_FFT, 2.0, 2.0 * D_init, 2.0, 2.0 * R_init)

    # 4) generalized EM to convergence
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        lam, converged = FastPhaseEM.em_map_parameters_fast(
            [omega_FFT, np.log(D_init), np.log(R_init)], (tk, yk), priors,
            max_iter=cfg.max_iter, tol=cfg.tol, param_mode="log", warn=False,
        )
    lambda_map = lam[-1]
    omega_map = float(lambda_map[0])
    D_map = float(np.exp(lambda_map[1]))
    R_map = float(np.exp(lambda_map[2]))
    param_nat = (omega_map, D_map, R_map)

    # 5) filter + smoother at the MAP
    thetakk, Pkk, thetakk_1, Pkk_1, ypred, Hk, ck = SLPI.kalman_filter(
        param_nat, (tk, yk), state_priors)
    thetakN, PkN, _ = SLPI.rts_smoother(param_nat, (tk, yk),
                                        (thetakk, Pkk, thetakk_1, Pkk_1))

    # innovations in the H-aligned basis: S_k is diagonal there (||H|| = 1)
    innov = yk - ypred
    nu_par = np.sum(Hk * innov, axis=1)
    var_par = Pkk_1 + R_map
    e_par = nu_par / np.sqrt(var_par)
    nu_perp = innov - Hk * nu_par[:, None]
    NIS = nu_par ** 2 / var_par + np.sum(nu_perp ** 2, axis=1) / R_map

    # 6) Laplace covariance
    omega_std = lnD_std = lnR_std = rho_em = np.nan
    if want_cov:
        Hkf, ckf, _ = SLPI.freeze_linearization(lambda_map, (tk, yk), state_priors)
        res = FastPhaseEM.posterior_information_fast(lambda_map, (tk, yk), priors,
                                                 Hkf, ckf, param_type="log")
        cov = res["cov"]
        omega_std = float(np.sqrt(abs(cov[0, 0])))
        lnD_std = float(np.sqrt(abs(cov[1, 1])))
        lnR_std = float(np.sqrt(abs(cov[2, 2])))
        rho_em = float(res["rho"])

    # value of the objective at the converged point, so competing basins can be ranked
    nlp = float(SLPI.neg_log_posterior(lambda_map, (tk, yk), priors, param_mode="log"))

    # ── diagnostics ───────────────────────────────────────────────────────────
    T = float(tk[-1] - tk[0])

    # actual angular error on the measurement (wrapped; unwrap is invalid at large sigma)
    ang_meas = np.unwrap(np.arctan2(yk[:, 1], yk[:, 0]))
    _, sigma_psi_emp, sigma_psi_circ = wrapped_angle_error(yk, theta_win)

    # drift-removed relative phases, all on the same omega_map
    drift = omega_map * (tk - tk[0])
    phi_geo = theta_win - theta_win[0] - drift
    phi_Phi = Phi_win - Phi_win[0] - drift
    phi_inf = thetakN - thetakN[0] - drift
    sd = 1.96 * np.sqrt(PkN)

    e = e_par - e_par.mean()
    v = float(np.dot(e, e))
    acf1 = float(np.dot(e[:-1], e[1:]) / v) if v > 0 else np.nan

    row = dict(
        N_INF=N_INF, dt_inf=dt_inf, T=T,
        omega_map=omega_map, D_map=D_map, R_map=R_map,
        omega_std=omega_std, lnD_std=lnD_std, lnR_std=lnR_std, rho_em=rho_em,
        n_em=int(len(lam) - 1), converged=bool(converged), em_max_iter=cfg.max_iter,
        acf1=acf1, NIS_mean=float(NIS.mean()),
        cov_geo=float(np.mean(np.abs(phi_inf - phi_geo) <= sd)),
        cov_Phi=float(np.mean(np.abs(phi_inf - phi_Phi) <= sd)),
        rmse_geo=float(np.sqrt(np.mean((phi_inf - phi_geo) ** 2))),
        rmse_Phi=float(np.sqrt(np.mean((phi_inf - phi_Phi) ** 2))),
        phase_rms=float(np.std(phi_geo)),
        sigma_psi_emp=sigma_psi_emp, sigma_psi_circ=sigma_psi_circ,
        sigma_psi2_hat=sigma_psi2_hat, rbar=rbar, nlp=nlp, init_mode=init_mode,
        R_crude=R_crude, R_init=R_init, D_init=D_init, omega_FFT=float(omega_FFT),
        omega_theta_emp=float((theta_win[-1] - theta_win[0]) / T),
        omega_Phi_emp=float((Phi_win[-1] - Phi_win[0]) / T),
    )

    if not keep_arrays:
        return row

    arrays = dict(
        tk=tk, theta_win=theta_win, Phi_win=Phi_win,
        thetakN=thetakN, PkN=PkN, e_par=e_par, ang_meas=ang_meas,
        param_nat=np.asarray(param_nat), state_priors=np.asarray(state_priors),
        lambda_iter=lam,
    )
    return row, arrays


# ──────────────────────────────────────────────────────────────────────────────
# Sweep driver
# ──────────────────────────────────────────────────────────────────────────────

_TRAJ = None
_CFG = None
_FITDIR = None


def _init_worker(traj, cfg, fitdir):
    global _TRAJ, _CFG, _FITDIR
    _TRAJ, _CFG, _FITDIR = traj, cfg, fitdir


def _task_key(sigma_add, N_INF, init_mode):
    return f"{init_mode}_sig{sigma_add:g}_N{N_INF}"


def noise_seed_for(cfg: Config, sigma_add: float, m: int) -> int:
    """Deterministic per-cell noise seed.

    Kept as a function so the notebook can rebuild the exact window a given sweep
    row used without having to look the seed up in the results table.
    """
    return cfg.noise_seed_base + 10_007 * m + int(round(sigma_add * 1e6))


def _run_task(task):
    """Worker: build the window, fit it, and (for replicate 0) cache its arrays."""
    sigma_add, N_INF, m, init_mode = task
    cfg, traj = _CFG, _TRAJ

    noise_seed = noise_seed_for(cfg, sigma_add, m)
    tk, yk, theta_win, Phi_win = make_window(traj, cfg, sigma_add, N_INF, m, noise_seed)

    t0 = time.time()
    keep = (m == 0)
    want_cov = (N_INF <= cfg.cov_max_N and
                (N_INF <= cfg.cov_all_below or m < cfg.n_cov_reps))
    out = fit_window(tk, yk, theta_win, Phi_win, cfg, want_cov=want_cov,
                     keep_arrays=keep, init_mode=init_mode)
    row, arrays = out if keep else (out, None)
    row.update(sigma_add=sigma_add, rep=m, noise_seed=noise_seed,
               want_cov=want_cov, wall_s=time.time() - t0)

    if keep and _FITDIR is not None:
        scal = {k: v for k, v in row.items() if isinstance(v, (int, float, bool, str))}
        np.savez_compressed(
            Path(_FITDIR) / f"{_task_key(sigma_add, N_INF, init_mode)}.npz",
            yk=yk, **arrays, **scal)
    return row


def load_cached_rows(outdir: Path, cfg: Config, verbose: bool = True):
    """Return rows from a previous run that are still valid for `cfg`.

    Reusable iff the *core* settings agree (see `Config.CORE_KEYS`): the grid keys
    are deliberately excluded, because lengthening ``N_grid`` / ``N_INF_max`` leaves
    every earlier sample of the trajectory bit-identical (the simulator draws its
    noise per step) and hence leaves earlier fits exactly reproducible.

    Returns
    -------
    rows : list of dict
        Empty if there is no cache, or if the core settings have changed.
    """
    import json

    import pandas as pd

    pq, sig_path = outdir / "sweep_results.parquet", outdir / "grid_signature.json"
    if not (pq.exists() and sig_path.exists()):
        return []

    def _norm(v):
        # json round-trips tuples as lists, so compare on a canonical form
        return json.dumps(list(v) if isinstance(v, (tuple, list)) else v, default=str)

    old = json.loads(sig_path.read_text())
    new = cfg.core_signature()
    bad = {k: (old.get(k), v) for k, v in new.items()
           if _norm(old.get(k)) != _norm(v)}
    if bad:
        if verbose:
            print(f"cache at {pq} ignored: core settings differ {bad}", flush=True)
        return []

    rows = pd.read_parquet(pq).to_dict("records")
    if verbose:
        print(f"reusing {len(rows)} cached fits from {pq}", flush=True)
    return rows


def run_sweep(cfg: Config, outdir: Path, workers: int = 7, verbose: bool = True,
              resume: bool = True, save_fits: bool = False):
    """Run the full (sigma_add x N_INF x replicate x init_mode) sweep.

    Parameters
    ----------
    cfg : Config
    outdir : Path
        Results directory.
    workers : int
        Process-pool size.
    verbose : bool
        Print progress.
    resume : bool
        Reuse any compatible fits already in ``outdir/sweep_results.parquet`` and
        run only the missing cells.  The returned list still covers the whole grid,
        plus any cached rows outside it.
    save_fits : bool
        Also write the per-sample arrays of every replicate-0 fit to ``outdir/fits/``
        (~1.3 GB for the full grid).  Not needed by the figures.

    Returns
    -------
    rows : list of dict
        One entry per fit in the grid (cached + newly computed).
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    fitdir = None
    if save_fits:
        fitdir = outdir / "fits"
        fitdir.mkdir(exist_ok=True)

    cached = load_cached_rows(outdir, cfg, verbose) if resume else []
    done = {(round(float(r["sigma_add"]), 12), int(r["N_INF"]),
             int(r["rep"]), str(r["init_mode"])) for r in cached}

    if verbose:
        print(f"Simulating {cfg.n_rep} realizations, N = {cfg.N_steps} steps "
              f"(T = {cfg.N_steps * cfg.dt:.0f}) at dt = {cfg.dt} ...", flush=True)
    t0 = time.time()
    # thinned: only every stride-th sample is ever read, and at N_INF_max = 512000 the
    # full-resolution form would need ~3.4 GB peak (see simulate_ensemble)
    traj = simulate_ensemble(cfg, thin=True)
    if verbose:
        print(f"  done in {time.time() - t0:.1f} s; "
              f"r in [{traj['r'].min():.3f}, {traj['r'].max():.3f}]", flush=True)

    tasks = [(s, N, m, im)
             for s in cfg.sigma_grid
             for N in cfg.N_grid
             for m in range(cfg.n_rep)
             for im in cfg.init_modes
             if (round(float(s), 12), int(N), int(m), str(im)) not in done]
    # Longest jobs first so the pool drains evenly. Cost is dominated by N, then by
    # whether the O(N^2) Laplace covariance is computed for this replicate.
    tasks.sort(key=lambda t: (-t[1],
                              -(t[1] <= cfg.cov_all_below or t[2] < cfg.n_cov_reps),
                              -t[0]))

    if not tasks:
        if verbose:
            print("nothing to do: every grid cell is already cached", flush=True)
        return cached

    if verbose:
        print(f"{len(tasks)} fits on {workers} workers "
              f"({len(cached)} reused) ...", flush=True)

    t0 = time.time()
    rows = list(cached)
    ctx = get_context("fork")
    with ctx.Pool(workers, initializer=_init_worker, initargs=(traj, cfg, fitdir)) as pool:
        for i, row in enumerate(pool.imap_unordered(_run_task, tasks, chunksize=1), 1):
            rows.append(row)
            # Checkpoint every fit: at N = 512000 a single fit can take 80 minutes,
            # so even a 10-fit interval could put hours of work at risk of one crash.
            # The parquet is ~300 kB, so the write costs nothing next to that.
            _checkpoint(rows, cfg, outdir)
            if verbose and (i % 5 == 0 or i == len(tasks)):
                el = time.time() - t0
                print(f"  {i:>4d}/{len(tasks)}  elapsed {el/60:.1f} min  "
                      f"eta {el/i*(len(tasks)-i)/60:.1f} min", flush=True)

    if verbose:
        print(f"sweep finished in {(time.time() - t0)/60:.1f} min", flush=True)
    return rows


def _signature_for(rows, cfg: Config) -> dict:
    """`cfg.signature()` with the grid keys set to what `rows` actually contain.

    A resumed run merges new fits into a cache that may have been built on a different
    grid, so the run's own grid would misdescribe the file.
    """
    sig = cfg.signature()
    sig["N_grid"] = sorted({int(r["N_INF"]) for r in rows})
    sig["sigma_grid"] = sorted({float(r["sigma_add"]) for r in rows})
    sig["init_modes"] = sorted({str(r["init_mode"]) for r in rows})
    sig["N_INF_max"] = max(sig["N_grid"])
    sig["N_steps"] = sig["N_INF_max"] * cfg.stride
    return sig


def _checkpoint(rows, cfg: Config, outdir: Path):
    """Atomically write the results so far, so a crash loses at most the running fits."""
    import json

    import pandas as pd

    outdir = Path(outdir)
    tmp = outdir / "sweep_results.parquet.tmp"
    pd.DataFrame(rows).to_parquet(tmp)
    tmp.replace(outdir / "sweep_results.parquet")
    (outdir / "grid_signature.json").write_text(
        json.dumps(_signature_for(rows, cfg), default=str))


# ──────────────────────────────────────────────────────────────────────────────
# Command line
# ──────────────────────────────────────────────────────────────────────────────

def figure_config(cfg: Config | None = None) -> Config:
    """The subset of the full grid that `Figure2_DiffusivityVsAddedNoise.ipynb` draws.

    'default' initialization at N_INF in {2000, 8000, 32000}, every sigma_add, every
    replicate.  The trajectory is integrated only as far as N_INF = 32000 needs; the
    simulator draws its noise step by step, so those samples are bit-identical to the
    first 32000 of the full-grid trajectory.
    """
    cfg = Config() if cfg is None else cfg
    return replace(cfg, N_grid=(2000, 8000, 32000), init_modes=("default",),
                   N_INF_max=32000)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grid", choices=("figure", "full"), default="figure",
                    help="'figure': the cells Figure 2 draws (~4 CPU-h); "
                         "'full': the whole study (~99 CPU-h)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--outdir", type=Path, default=_HERE / "data")
    ap.add_argument("--no-resume", action="store_true",
                    help="recompute every cell instead of reusing compatible cached fits")
    ap.add_argument("--save-fits", action="store_true",
                    help="also store per-sample arrays of replicate-0 fits in OUTDIR/fits")
    ap.add_argument("--check-sim", action="store_true",
                    help="only validate the thinned integrator against the reference one")
    args = ap.parse_args()

    if args.check_sim:
        rep = check_thinned_simulation()
        print("thinned vs reference integrator, max |difference|:")
        for k, v in rep.items():
            print(f"  {k:>14s} : {v:.3e}")
        print("estimator inputs bit-identical; fitted parameters identical.")
        return

    cfg = Config() if args.grid == "full" else figure_config()
    # run_sweep checkpoints after every fit, so the parquet is already up to date
    # (and is left untouched when every cell was cached).
    rows = run_sweep(cfg, args.outdir, workers=args.workers,
                     resume=not args.no_resume, save_fits=args.save_fits)
    print(f"{args.outdir / 'sweep_results.parquet'}: {len(rows)} rows")


if __name__ == "__main__":
    main()
