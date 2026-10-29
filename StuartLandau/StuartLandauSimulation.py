"""
Stochastic Stuart–Landau oscillator with isotropic Cartesian noise.

Implements:
  - Euler–Maruyama simulation in Cartesian coordinates
  - Geometric phase θ and asymptotic phase Φ = θ + (b/a) ln r
  - Ensemble phase MSD with theoretical predictions
  - Long-time diffusivity extraction via linear fit
  - Phase geometry visualization (isochrons + ∇Φ vectors)
  - Radial variance vs OU prediction

All key formulas follow:
  D_eff = Dx (1 + b²/a²)
  MSD_θ_OU = 2Dx t + (2b²Dx/a²)[t − (1−e^{−2at})/a + (1−e^{−4at})/(4a)]
  MSD_θ_short = 2Dx t + (8/3) b² Dx t³          [valid t << 1/(2a)]
  Var[ρ] = (Dx/2a)(1 − e^{−4at})                  [OU prediction]
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ──────────────────────────────────────────────────────────────────────────────
# 1.  Simulation
# ──────────────────────────────────────────────────────────────────────────────

def simulate_stuart_landau(
    a: float,
    b: float,
    Dx: float,
    dt: float,
    N: int,
    M: int,
    seed: int = 0,
    random_initial_phase: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Euler–Maruyama simulation of the stochastic Stuart–Landau oscillator.

    SDEs (Cartesian, Itô):
        dx1 = [a(1−r²)x1 − (1+b(r²−1))x2] dt + √(2Dx) dW1
        dx2 = [a(1−r²)x2 + (1+b(r²−1))x1] dt + √(2Dx) dW2

    Parameters
    ----------
    a : float
        Radial relaxation rate (a > 0 for a stable limit cycle).
    b : float
        Non-isochronicity / shear parameter.
    Dx : float
        Isotropic Cartesian noise strength [length² time⁻¹].
    dt : float
        Integration timestep.  Require dt ≪ 1/(2a) and √(2Dx dt) ≪ 1.
    N : int
        Number of integration steps.
    M : int
        Number of independent stochastic realizations.
    seed : int
        NumPy RNG seed for reproducibility.
    random_initial_phase : bool
        If True, draw initial angles uniformly on [0, 2π); otherwise θ₀ = 0.

    Returns
    -------
    t : np.ndarray, shape (N+1,)
    x1, x2 : np.ndarray, shape (M, N+1)
    r : np.ndarray, shape (M, N+1)
    theta : np.ndarray, shape (M, N+1)   – unwrapped polar angle
    Phi : np.ndarray, shape (M, N+1)     – asymptotic phase Φ = θ + (b/a) ln r
    """
    rng = np.random.default_rng(seed)
    t = np.arange(N + 1, dtype=float) * dt

    x1 = np.empty((M, N + 1))
    x2 = np.empty((M, N + 1))

    if random_initial_phase:
        theta0 = rng.uniform(0.0, 2.0 * np.pi, size=M)
    else:
        theta0 = np.zeros(M)

    x1[:, 0] = np.cos(theta0)
    x2[:, 0] = np.sin(theta0)

    noise_amp = np.sqrt(2.0 * Dx * dt)

    for n in range(N):
        r2 = x1[:, n] ** 2 + x2[:, n] ** 2
        factor = 1.0 + b * (r2 - 1.0)
        F1 = a * (1.0 - r2) * x1[:, n] - factor * x2[:, n]
        F2 = a * (1.0 - r2) * x2[:, n] + factor * x1[:, n]
        x1[:, n + 1] = x1[:, n] + F1 * dt + noise_amp * rng.standard_normal(M)
        x2[:, n + 1] = x2[:, n] + F2 * dt + noise_amp * rng.standard_normal(M)

    r = np.sqrt(x1 ** 2 + x2 ** 2)
    theta = np.unwrap(np.arctan2(x2, x1), axis=1)
    r_safe = np.maximum(r, 1e-12)
    Phi = theta + (b / a) * np.log(r_safe)

    return t, x1, x2, r, theta, Phi


# ──────────────────────────────────────────────────────────────────────────────
# 2.  Relative phases
# ──────────────────────────────────────────────────────────────────────────────

def compute_relative_phases(
    t: np.ndarray,
    theta: np.ndarray,
    Phi: np.ndarray,
    subtract: str = "theory",
    omega_custom: float | None = None,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """
    Compute relative phases by subtracting the mean rotation.

    φ_θ(t) = θ(t) − θ(0) − ω_θ t
    φ_Φ(t) = Φ(t) − Φ(0) − ω_Φ t

    Parameters
    ----------
    t : (N+1,)
    theta, Phi : (M, N+1)
    subtract : str
        "theory"   → ω = 1 (exact deterministic frequency)
        "empirical" → ω = ⟨θ(T)−θ(0)⟩ / T  (removes any noise-induced drift)
    omega_custom : float, optional
        Required when subtract="custom".

    Returns
    -------
    phi_theta, phi_Phi : (M, N+1)
    omega_theta, omega_Phi : float
    """
    T = t[-1]
    if subtract == "empirical":
        omega_theta = float(np.mean(theta[:, -1] - theta[:, 0]) / T)
        omega_Phi = float(np.mean(Phi[:, -1] - Phi[:, 0]) / T)
    elif subtract == "custom":
        if omega_custom is None:
            raise ValueError("Must provide omega_custom when subtract='custom'")
        omega_theta = float(omega_custom)
        omega_Phi = float(omega_custom)
    else:
        omega_theta = 1.0
        omega_Phi = 1.0

    phi_theta = theta - theta[:, :1] - omega_theta * t[np.newaxis, :]
    phi_Phi = Phi - Phi[:, :1] - omega_Phi * t[np.newaxis, :]

    return phi_theta, phi_Phi, omega_theta, omega_Phi


# ──────────────────────────────────────────────────────────────────────────────
# 3.  MSD
# ──────────────────────────────────────────────────────────────────────────────

def compute_msd(phi: np.ndarray) -> np.ndarray:
    """
    Compute phase MSD, choosing the estimator based on number of realizations.

    M > 1 — Ensemble MSD
    --------------------
    MSD(t_k) = (1/M) Σ_m φ_m(t_k)²
    (φ(t₀) = 0 by construction, so no additional reference subtraction needed.)

    M = 1 — Time-averaged MSD with overlapping windows
    ---------------------------------------------------
    For each integer lag ℓ, average over all N+1-ℓ overlapping increments:

        MSD_TA(τ_ℓ) = (1/(N+1-ℓ)) Σ_{i=0}^{N-ℓ} [φ(t_{i+ℓ}) − φ(t_i)]²

    Important caveats for M = 1:
    - Use ``subtract="theory"`` (not "empirical") when building φ.  Empirical
      frequency subtraction forces φ(t_N) = 0, pinning the endpoints and
      strongly biasing the TA-MSD toward zero at large lags.
    - Only fit diffusivity in the reliable lag range  5τ_r < τ < T/4.
      At τ → T only one increment contributes; variance diverges.
    - Expect ~10–20 % statistical error even for N ~ 40 000.  For precise
      D_eff estimates use M > 1.

    Parameters
    ----------
    phi : (M, N+1)  – relative phase array; φ(t₀) = 0

    Returns
    -------
    msd : (N+1,)    – MSD[k] at lag time t_k = k·dt
    """
    M, Np1 = phi.shape
    if M > 1:
        # Ensemble average: ⟨φ(t_k)²⟩_M
        return np.mean(phi ** 2, axis=0)

    # Single trajectory: time-averaged MSD over overlapping increments
    traj = phi[0]           # (N+1,)
    N = Np1 - 1
    msd = np.empty(Np1)
    msd[0] = 0.0
    for lag in range(1, Np1):
        increments = traj[lag:] - traj[:N + 1 - lag]   # N+1-lag samples
        msd[lag] = np.mean(increments ** 2)
    return msd


# ──────────────────────────────────────────────────────────────────────────────
# 4.  Theoretical curves
# ──────────────────────────────────────────────────────────────────────────────

def theoretical_curves(t: np.ndarray, a: float, b: float, Dx: float) -> dict:
    """
    Analytical predictions for phase MSD and radial variance.

    Parameters
    ----------
    t  : (N+1,)  – time array
    a, b, Dx : floats

    Returns
    -------
    dict with keys:
        D_eff               – long-time effective phase diffusion coefficient
        tau_r               – radial relaxation time 1/(2a)
        MSD_theta_OU        – finite-time OU geometric phase theory
        MSD_theta_short     – short-time expansion  (valid t ≪ 1/(2a))
        MSD_long            – 2 D_eff t  (long-time asymptote)
        MSD_Phi_PR          – phase-reduction prediction  2 D_eff t
        radial_variance_OU  – OU prediction for Var[ρ(t)]
    """
    D_eff = Dx * (1.0 + b ** 2 / a ** 2)
    tau_r = 1.0 / (2.0 * a)

    # Shear bracket from OU calculation (exact in linearized theory)
    shear_bracket = (
        t
        - (1.0 - np.exp(-2.0 * a * t)) / a
        + (1.0 - np.exp(-4.0 * a * t)) / (4.0 * a)
    )

    MSD_theta_OU = 2.0 * Dx * t + (2.0 * b ** 2 * Dx / a ** 2) * shear_bracket
    MSD_theta_short = 2.0 * Dx * t + (8.0 / 3.0) * b ** 2 * Dx * t ** 3
    MSD_long = 2.0 * D_eff * t
    MSD_Phi_PR = 2.0 * D_eff * t                   # same formula; valid for all t in PR theory
    radial_variance_OU = (Dx / (2 * a)) * (1.0 - np.exp(-4.0 * a * t))

    return {
        "D_eff": D_eff,
        "tau_r": tau_r,
        "MSD_theta_OU": MSD_theta_OU,
        "MSD_theta_short": MSD_theta_short,
        "MSD_long": MSD_long,
        "MSD_Phi_PR": MSD_Phi_PR,
        "radial_variance_OU": radial_variance_OU,
    }


# ──────────────────────────────────────────────────────────────────────────────
# 5.  Long-time diffusivity fit
# ──────────────────────────────────────────────────────────────────────────────

def fit_long_time_diffusivity(
    t: np.ndarray,
    msd: np.ndarray,
    tau_r: float,
    fit_start_factor: float = 5.0,
    fit_end_factor: float | None = None,
) -> tuple[float, float, np.ndarray]:
    """
    Linear fit MSD(t) ≈ 2 D_fit t + c.

    Fit window: fit_start_factor × τ_r  <  t  ≤  fit_end_factor × T
      - Lower bound removes transient shear-crossover.
      - Upper bound (default: no cap for ensemble; T/4 recommended for TA-MSD)
        avoids the high-variance tail of single-trajectory estimates.

    Parameters
    ----------
    t : (N+1,)
    msd : (N+1,)
    tau_r : float
    fit_start_factor : float
        Start fit at t > fit_start_factor × τ_r.  Default 5.0.
    fit_end_factor : float or None
        If given, end fit at t ≤ fit_end_factor × t[-1].
        Recommended value for TA-MSD (M=1): 0.25.

    Returns
    -------
    D_fit : float
    intercept : float
    fit_mask : bool ndarray of shape (N+1,)
    """
    fit_mask = t > fit_start_factor * tau_r
    if fit_end_factor is not None:
        fit_mask &= t <= fit_end_factor * t[-1]
    if fit_mask.sum() < 2:
        return float("nan"), float("nan"), fit_mask
    slope, intercept = np.polyfit(t[fit_mask], msd[fit_mask], 1)
    return slope / 2.0, float(intercept), fit_mask


# ──────────────────────────────────────────────────────────────────────────────
# 6.  Plots
# ──────────────────────────────────────────────────────────────────────────────

def plot_phase_msd(
    t: np.ndarray,
    msd_theta: np.ndarray,
    msd_Phi: np.ndarray,
    theory: dict,
    D_fit_theta: float | None = None,
    D_fit_Phi: float | None = None,
    title: str = "",
    M: int = 2,
) -> plt.Figure:
    """
    Log-log MSD plot combining simulation and all theoretical predictions.

    Curves plotted
    --------------
    - Simulated ⟨φ_θ²⟩  (steelblue solid)
    - Simulated ⟨φ_Φ²⟩  (tomato solid)
    - OU theory for geometric phase  (steelblue dashed)
    - 2 D_eff t  (black dashed) – long-time asymptote and PR prediction
    - Short-time: 2Dx t + (8/3) b² Dx t³  (purple dotted, only t < 5τ_r)
    - Vertical line at τ_r = 1/(2a)  (green dashed)

    Parameters
    ----------
    M : int
        Number of realizations used.  When M=1 (TA-MSD), simulated curves
        are clipped to t ≤ T/4 to hide the high-variance long-lag tail.
    """
    T = t[-1]
    plot_t_max = T / 4.0 if M == 1 else T
    mask = (t > 0) & (t <= plot_t_max)
    t_ = t[mask]
    tau_r = theory["tau_r"]
    D_eff = theory["D_eff"]

    fig, ax = plt.subplots(figsize=(9, 6))

    # ── Simulated ──
    ax.loglog(t_, msd_theta[mask], color="steelblue", lw=2,
              label=r"Sim. $\langle\phi_\theta^2\rangle$")
    ax.loglog(t_, msd_Phi[mask], color="tomato", lw=2,
              label=r"Sim. $\langle\phi_\Phi^2\rangle$")

    # ── OU theory for θ ──
    ax.loglog(t_, theory["MSD_theta_OU"][mask], "--", color="steelblue",
              lw=1.5, alpha=0.65, label=r"OU theory $\langle\phi_\theta^2\rangle$")

    # ── Phase-reduction / long-time: 2 D_eff t  (extend to full T for reference) ──
    full_mask = t > 0
    ax.loglog(t[full_mask], theory["MSD_long"][full_mask], "k--", lw=1.5, alpha=0.55,
              label=rf"$2D_{{\rm eff}}\,t$  ($D_{{\rm eff}}={D_eff:.3g}$)")

    # ── Short-time geometric theory (plot only for t < 5 τ_r) ──
    st_mask = (t > 0) & (t < 5.0 * tau_r)
    if st_mask.sum() > 1:
        ax.loglog(t[st_mask], theory["MSD_theta_short"][st_mask], ":",
                  color="purple", lw=2.0, alpha=0.80,
                  label=r"Short-time: $2D_x t+\frac{8}{3}b^2D_x t^3$")

    # ── Radial relaxation time marker ──
    ax.axvline(tau_r, color="forestgreen", ls="--", lw=1.3, alpha=0.85,
               label=rf"$\tau_r=1/(2a)={tau_r:.3g}$")
    # Annotate above the x-axis: defer to after autoscale
    ax.set_xlabel("Time $t$", fontsize=13)
    ax.set_ylabel(r"Phase MSD $\langle\phi^2\rangle$", fontsize=13)
    ax.set_title(title or "Phase MSD (log-log)", fontsize=12)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, which="both", alpha=0.3, ls=":")

    plt.tight_layout()
    return fig


def plot_phase_geometry(
    a: float,
    b: float,
    grid_lim: float = 2.0,
    grid_N: int = 200,
    quiver_step: int = 15,
    title: str = "",
) -> plt.Figure:
    """
    Phase geometry plot: limit cycle r = 1, Φ-isochron contours, ∇Φ arrows.

    The asymptotic phase is  Φ(x1,x2) = arctan2(x2,x1) + (b/a) ln r.
    Its gradient in Cartesian coordinates:
        ∂Φ/∂x1 = (b/a) x1/r² − x2/r²
        ∂Φ/∂x2 = (b/a) x2/r² + x1/r²

    Interpretation
    --------------
    - Φ-contours are isochrons (curves of equal phase).
    - For b ≠ 0 they spiral: radial noise contributes to phase noise.
    - ∇Φ arrows show sensitivity direction; on r=1: ∇Φ = (b/a) ê_r + ê_θ.
    """
    x = np.linspace(-grid_lim, grid_lim, grid_N)
    y = np.linspace(-grid_lim, grid_lim, grid_N)
    X, Y = np.meshgrid(x, y)

    R = np.sqrt(X ** 2 + Y ** 2)
    R_safe = np.maximum(R, 1e-10)
    R2 = R_safe ** 2

    Phi_grid = np.arctan2(Y, X) + (b / a) * np.log(R_safe)

    # ∇Φ in Cartesian
    dPhi_dx = (b / a) * X / R2 - Y / R2
    dPhi_dy = (b / a) * Y / R2 + X / R2

    # Mask near origin where Φ is singular
    near_origin = R < 0.15
    Phi_masked = np.ma.masked_where(near_origin, Phi_grid)

    # Contour level range covers the actual range of Φ on the grid
    lev_half = np.pi + abs(b / a) * np.log(max(grid_lim / 0.2, 1.0))
    n_lev = 18
    levels = np.linspace(-lev_half, lev_half, 2 * n_lev + 1)

    fig, ax = plt.subplots(figsize=(7, 7))

    cf = ax.contourf(X, Y, Phi_masked, levels=levels, cmap="twilight",
                     alpha=0.35, extend="both")
    ax.contour(X, Y, Phi_masked, levels=levels[1::2], colors="dimgray",
               linewidths=0.7, alpha=0.65)
    plt.colorbar(cf, ax=ax, label=r"Asymptotic phase $\Phi$", fraction=0.04, pad=0.04)

    # Limit cycle r = 1
    th = np.linspace(0.0, 2.0 * np.pi, 400)
    ax.plot(np.cos(th), np.sin(th), "k-", lw=2.5, zorder=5,
            label="Limit cycle $r=1$")

    # ∇Φ quiver (normalized arrows, skip region near origin)
    qs = quiver_step
    Xq, Yq = X[::qs, ::qs], Y[::qs, ::qs]
    Uq = dPhi_dx[::qs, ::qs]
    Vq = dPhi_dy[::qs, ::qs]
    Rq = np.sqrt(Xq ** 2 + Yq ** 2)
    show_q = Rq > 0.22
    mag = np.sqrt(Uq ** 2 + Vq ** 2)
    mag = np.where(mag > 0, mag, 1.0)
    ax.quiver(
        Xq[show_q], Yq[show_q],
        (Uq / mag)[show_q], (Vq / mag)[show_q],
        color="black", alpha=0.55, scale=22.0, width=0.003, zorder=6,
        label=r"$\nabla\Phi$ (unit vectors)",
    )

    ax.set_xlim(-grid_lim, grid_lim)
    ax.set_ylim(-grid_lim, grid_lim)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$x_1$", fontsize=13)
    ax.set_ylabel(r"$x_2$", fontsize=13)
    ax.set_title(title or rf"Phase geometry: $a={a}$, $b={b}$", fontsize=12)
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(True, alpha=0.25)

    plt.tight_layout()
    return fig


def plot_radial_variance(
    t: np.ndarray,
    r: np.ndarray,
    a: float,
    Dx: float,
    title: str = "",
) -> plt.Figure:
    """
    Radial fluctuation variance: simulation vs linearized OU prediction.

    Var[ρ(t)] = (Dx/2a)(1 − e^{−4at})
    """
    rho = r - 1.0
    var_sim = np.mean(rho ** 2, axis=0)
    var_ou = (Dx / (2 * a)) * (1.0 - np.exp(-4.0 * a * t))

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(t, var_sim, color="steelblue", lw=2,
            label=r"Simulated $\langle\rho^2\rangle$")
    ax.plot(t, var_ou, "k--", lw=1.5, alpha=0.75,
            label=r"OU: $\frac{D_x}{2a}(1-e^{-4at})$")
    ax.axhline(Dx / (2 * a), color="gray", ls=":", lw=1.2,
               label=rf"Asymptote $D_x/(2a) = {Dx/(2 * a):.3g}$")

    ax.set_xlabel("Time $t$", fontsize=12)
    ax.set_ylabel(r"$\mathrm{Var}[\rho] = \langle(r-1)^2\rangle$", fontsize=12)
    ax.set_title(title or rf"Radial variance: $a={a}$, $D_x={Dx:.1e}$", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig

def plot_radial_shift(
    t: np.ndarray,
    r: np.ndarray,
    a: float,
    Dx: float,
    title: str = "",
) -> plt.Figure:
    """
    Radial shift: simulation vs linearized OU prediction.

    <ρ(t)> = -(Dx/4a)
    """
    rho = r - 1.0
    mean_sim = np.mean(rho, axis=0)
    mean_ou = -(Dx / (4 * a)) * np.ones_like(t)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(t, mean_sim, color="steelblue", lw=2,
            label=r"Simulated $\langle\rho\rangle$")
    ax.plot(t, mean_ou, "k--", lw=1.5, alpha=0.75,
            label=r"OU: $-\frac{D_x}{4a}$")
    ax.axhline(-Dx / (4 * a), color="gray", ls=":", lw=1.2,
               label=rf"Asymptote $-D_x/4a = {-Dx/(4 * a):.3g}$")

    ax.set_xlabel("Time $t$", fontsize=12)
    ax.set_ylabel(r"$\mathrm{Var}[\rho] = \langle(r-1)^2\rangle$", fontsize=12)
    ax.set_title(title or rf"Radial shift: $a={a}$, $D_x={Dx:.1e}$", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# 7.  Summary table
# ──────────────────────────────────────────────────────────────────────────────

def print_summary_table(
    a: float,
    b: float,
    Dx: float,
    D_fit_theta: float,
    D_fit_Phi: float,
) -> None:
    """Print comparison of theoretical and fitted diffusivities."""
    tau_r = 1.0 / (2.0 * a)
    D_direct = Dx
    D_shear = Dx * b ** 2 / a ** 2
    D_eff_theory = Dx * (1.0 + b ** 2 / a ** 2)
    err_theta = (D_fit_theta - D_eff_theory) / D_eff_theory if D_eff_theory else float("nan")
    err_Phi = (D_fit_Phi - D_eff_theory) / D_eff_theory if D_eff_theory else float("nan")

    rows = [
        ("a",                           f"{a:.4g}"),
        ("b",                           f"{b:.4g}"),
        ("Dx",                          f"{Dx:.4g}"),
        ("tau_r = 1/(2a)",              f"{tau_r:.4g}"),
        ("D_direct = Dx",               f"{D_direct:.4g}"),
        ("D_shear = Dx*b**2/a**2",      f"{D_shear:.4g}"),
        ("D_eff_theory",                f"{D_eff_theory:.4g}"),
        ("D_theta_fit",                 f"{D_fit_theta:.4g}"),
        ("D_Phi_fit",                   f"{D_fit_Phi:.4g}"),
        ("relative error theta",        f"{err_theta:+.3f}"),
        ("relative error Phi",          f"{err_Phi:+.3f}"),
    ]
    w = max(len(r[0]) for r in rows) + 2
    print(f"{'Quantity':<{w}} | {'Value':>14}")
    print("-" * (w + 17))
    for name, val in rows:
        print(f"{name:<{w}} | {val:>14}")
    print()


# ──────────────────────────────────────────────────────────────────────────────
# 8.  Experiment runner
# ──────────────────────────────────────────────────────────────────────────────

def run_experiment(
    name: str,
    a: float,
    b: float,
    Dx: float,
    dt: float = 0.005,
    N: int = 10_000,
    M: int = 500,
    seed: int = 42,
    outdir: Path = Path("plots"),
    show: bool = False,
) -> None:
    """Full analysis pipeline for one parameter set."""
    tau_r = 1.0 / (2.0 * a)
    rho_rms = np.sqrt(Dx / a)
    noise_step = np.sqrt(2.0 * Dx * dt)

    print(f"\n{'='*68}")
    print(f"  {name}")
    print(f"  a={a}, b={b}, Dx={Dx:.1e}, dt={dt}, N={N}, M={M}")
    print(f"  T={N*dt:.1f},  tau_r={tau_r:.4f},  sqrt(Dx/a)={rho_rms:.5f},  "
          f"sqrt(2Dx dt)={noise_step:.5f}")
    weak_noise = "✓ (small radial fluctuations)" if rho_rms < 0.3 else "⚠ WARN: large excursions"
    print(f"  Weak-noise condition sqrt(Dx/a) ≪ 1: {rho_rms:.4f}  {weak_noise}")
    print(f"{'='*68}")

    # ── Simulate ──
    t, x1, x2, r, theta, Phi = simulate_stuart_landau(
        a=a, b=b, Dx=Dx, dt=dt, N=N, M=M, seed=seed
    )

    # ── Relative phases ──
    # For M=1 (TA-MSD) use theoretical ω to avoid endpoint-pinning bias.
    subtract_mode = "theory" if M == 1 else "empirical"
    phi_theta, phi_Phi, omega_theta, omega_Phi = compute_relative_phases(
        t, theta, Phi, subtract=subtract_mode
    )
    print(f"  subtract={subtract_mode!r},  ω_θ = {omega_theta:.6f},  ω_Φ = {omega_Phi:.6f}  "
          f"(theory: 1.000000)")

    # ── MSDs ──
    msd_theta = compute_msd(phi_theta)
    msd_Phi = compute_msd(phi_Phi)
    theory = theoretical_curves(t, a, b, Dx)

    # ── Fit long-time diffusivities ──
    # For TA-MSD cap fit at T/4 to stay in the low-variance regime.
    fit_end = 0.25 if M == 1 else None
    D_fit_theta, _, _ = fit_long_time_diffusivity(t, msd_theta, tau_r, fit_end_factor=fit_end)
    D_fit_Phi, _, _ = fit_long_time_diffusivity(t, msd_Phi, tau_r, fit_end_factor=fit_end)

    print_summary_table(a, b, Dx, D_fit_theta, D_fit_Phi)

    # ── Figures ──
    slug = (name.lower()
            .replace(" ", "_").replace(":", "")
            .replace("/", "").replace(".", ""))

    fig1 = plot_phase_msd(
        t, msd_theta, msd_Phi, theory,
        D_fit_theta=D_fit_theta, D_fit_Phi=D_fit_Phi,
        title=rf"{name}  ($a={a}$, $b={b}$, $D_x={Dx:.1e}$)",
        M=M,
    )
    path1 = outdir / f"{slug}_msd.png"
    fig1.savefig(path1, dpi=150, bbox_inches="tight")

    fig2 = plot_phase_geometry(a, b, title=rf"Phase geometry: $a={a}$, $b={b}$")
    path2 = outdir / f"{slug}_geometry.png"
    fig2.savefig(path2, dpi=150, bbox_inches="tight")

    fig3 = plot_radial_variance(
        t, r, a, Dx, title=rf"Radial variance: $a={a}$, $D_x={Dx:.1e}$"
    )
    path3 = outdir / f"{slug}_radvar.png"
    fig3.savefig(path3, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    plt.close("all")

    print(f"  Saved: {path1.name},  {path2.name},  {path3.name}")


# ──────────────────────────────────────────────────────────────────────────────
# 9.  Main: five canonical experiments
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    """Run all five canonical experiments from the theory notes."""
    outdir = Path(__file__).resolve().parent / "plots"
    outdir.mkdir(exist_ok=True)

    # (name, a, b, Dx, dt, N, M)
    experiments = [
        # b=0: no shear → D_eff = Dx
        ("Exp1: Isochronous",                1.0, 0.0, 1e-3, 0.005, 10_000, 100),
        # moderate shear → D_eff = 5 Dx
        ("Exp2: Moderate non-isochronicity", 1.0, 2.0, 1e-3, 0.005, 10_000, 100),
        # fast radial relaxation → D_eff ≈ 1.16 Dx
        ("Exp3: Fast radial relaxation",     5.0, 2.0, 1e-3, 0.005, 10_000, 100),
        # slow radial relaxation → D_eff ≈ 17 Dx  (double N for long-time fit)
        ("Exp4: Slow radial relaxation",     0.5, 2.0, 1e-3, 0.005, 20_000, 100),
        # strong shear → D_eff = 26 Dx  (small Dx for validity)
        ("Exp5: Strong shear",               1.0, 5.0, 1e-4, 0.005, 10_000, 100),
    ]

    for name, a, b, Dx, dt, N, M in experiments:
        run_experiment(name, a, b, Dx, dt=dt, N=N, M=M, outdir=outdir)

    print("\nAll experiments complete. Figures saved to ./plots/")


if __name__ == "__main__":
    main()
