import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize
import pandas as pd
jax.config.update("jax_enable_x64", True)

import importlib
import Hippocampal as HP
importlib.reload(HP)

def fft_integer_modes_jax(n_theta):
    """
    Integer FFT modes for a 2π-periodic grid in JAX.
    """
    return jnp.fft.fftfreq(n_theta, d=1.0 / n_theta)


def spectral_heat_kernel_hat_jax(n_theta, mean_step, Q):
    """
    Fourier multiplier for circular drift-diffusion.

        theta_k = theta_{k-1} + mean_step + noise
        noise variance Q = 2 Dtheta dt

    K_hat[m] = exp(-i m mean_step - 0.5 m^2 Q)
    """
    m = fft_integer_modes_jax(n_theta)

    return jnp.exp(
        -1j * m * mean_step
        -0.5 * (m ** 2) * Q
    )


def spectral_predict_jax(p, K_hat):
    """
    Spectral circular prediction.

        p_pred = ifft(K_hat * fft(p))
    """
    return jnp.real(jnp.fft.ifft(K_hat * jnp.fft.fft(p)))

def prepare_jax_inputs(fit_data, priors):
    """Convert NumPy fitting data and prior tuple to JAX arrays."""
    tk_jax = jnp.asarray(fit_data["tk"])
    yk_jax = jnp.asarray(fit_data["yk"])
    priors_jax = tuple(jnp.asarray(p) for p in priors)
    return tk_jax, yk_jax, priors_jax


def fit_map_parameters_jax(
    fit_data,
    priors,
    param0,
    M,
    *,
    n_theta=128,
    method="BFGS",
    options=None,
):
    """
    MAP fit using the JAX posterior and gradient.

    This is the gradient-based replacement for the old Powell fit.
    The JAX likelihood uses the spectral grid transition.
    """
    x0 = HP.pack_param(*param0)
    tk_jax, yk_jax, priors_jax = prepare_jax_inputs(fit_data, priors)

    def objective(x):
        return float(
            neg_log_posterior_flat_jit(
                jnp.asarray(x),
                tk_jax,
                yk_jax,
                priors_jax,
                M=M,
                n_theta=n_theta,
            )
        )

    def gradient(x):
        return np.asarray(
            grad_neg_log_posterior_flat_jit(
                jnp.asarray(x),
                tk_jax,
                yk_jax,
                priors_jax,
                M=M,
                n_theta=n_theta,
            )
        )

    res = minimize(
        objective,
        x0,
        jac=gradient,
        method=method,
        options=options,
    )

    return {
        "result": res,
        "param": HP.unpack_param(res.x, M),
        "x": np.asarray(res.x),
        "fun": float(res.fun),
        "success": bool(res.success),
        "message": res.message,
        "nfev": getattr(res, "nfev", np.nan),
        "njev": getattr(res, "njev", np.nan),
        "nit": getattr(res, "nit", np.nan),
        "tk_jax": tk_jax,
        "yk_jax": yk_jax,
        "priors_jax": priors_jax,
    }


def summarize_laplace_uncertainty_jax(
    x_map,
    tk_jax,
    yk_jax,
    priors_jax,
    *,
    M,
    n_theta=128,
    hessian_method="direct",
    step=None,
):
    """
    Local Laplace covariance for the JAX MAP fit.

    Parameters
    ----------
    hessian_method : {"direct", "fd_grad"}
        "direct" uses jax.hessian of the spectral-grid posterior. This is now
        the preferred path after the spectral transition refactor.

        "fd_grad" keeps the older finite-difference-of-gradient fallback.
    """
    if hessian_method == "direct":
        H = np.asarray(
            hessian_neg_log_posterior_flat_jit(
                jnp.asarray(x_map),
                tk_jax,
                yk_jax,
                priors_jax,
                M=M,
                n_theta=n_theta,
            )
        )
    elif hessian_method == "fd_grad":
        H = hessian_neg_log_posterior_fd(
            x_map,
            tk_jax,
            yk_jax,
            priors_jax,
            M=M,
            n_theta=n_theta,
            step=step,
        )
    else:
        raise ValueError("hessian_method must be 'direct' or 'fd_grad'.")

    H = 0.5 * (H + H.T)
    eig = np.linalg.eigvalsh(H)

    cov = np.linalg.inv(H)
    se = np.sqrt(np.maximum(np.diag(cov), 0.0))

    return {
        "H": H,
        "cov": cov,
        "hessian_method": hessian_method,
        "hessian_min_eig": float(eig[0]),
        "hessian_max_eig": float(eig[-1]),
        "hessian_condition": float(eig[-1] / eig[0]) if eig[0] != 0 else np.inf,
        "omega_se": float(se[0]),
        "lnDtheta_se": float(se[1]),
        "lnR_se": float(se[2]),
        "Dtheta_mult_se": float(np.exp(se[1])),
        "R_eff_mult_se": float(np.exp(se[2])),
    }


def run_added_noise_fit_jax(
    data,
    R_add,
    M,
    rng,
    *,
    n_theta=128,
    param0=None,
    initialize_kwargs=None,
    prior_kwargs=None,
    optimizer_kwargs=None,
    compute_uncertainty=True,
    hessian_method="direct",
):
    """
    One JAX MAP fit at a fixed R_add.

    Workflow
    --------
    1. Add noise to create the fitting data.
    2. Initialize omega, phase, and waveform from the raw data, not yk_reg.
       This matches the notebook and avoids unstable waveform initialization
       from noisy observations.
    3. Fit the regularized data yk_reg.
    4. Set R_eff prior mode to the current R_add.
    5. Use continuation only through param0, not through the priors.
    """
    if initialize_kwargs is None:
        initialize_kwargs = {}

    if prior_kwargs is None:
        prior_kwargs = {}

    if optimizer_kwargs is None:
        optimizer_kwargs = {}

    raw_tk, raw_yk = HP.as_inference_data(data)
    raw_data = {"tk": raw_tk, "yk": raw_yk}

    data_reg = HP.add_regularizing_noise(
        data,
        R_add,
        rng,
        copy=True,
        key="yk_reg",
    )

    fit_data = {
        "tk": np.asarray(data_reg["tk"], float),
        "yk": np.asarray(data_reg["yk_reg"], float),
    }

    if param0 is None:
        init_kwargs = dict(initialize_kwargs)

        if "Dtheta0" not in init_kwargs and "Dtheta_mode" in prior_kwargs:
            init_kwargs["Dtheta0"] = prior_kwargs["Dtheta_mode"]

        if "R0" not in init_kwargs:
            init_kwargs["R0"] = R_add

        init = HP.initialize_map_parameters(
            # raw_data,
            fit_data,
            M,
            **init_kwargs,
        )

        param0 = init["param0"]
    else:
        init = None

    prior_kwargs_i = dict(prior_kwargs)
    prior_kwargs_i["R_mode"] = R_add

    if init is not None:
        phase_init = init["regression"]
        omega0 = init["omega0"]

        prior_kwargs_i.setdefault(
            "mu_theta",
            phase_init["phi_mean"] + omega0 * raw_tk[0],
        )
        prior_kwargs_i.setdefault(
            "sigma_theta",
            max(2.0 * phase_init["phi_std"], 1e-6),
        )

    priors = HP.make_default_priors(
        fit_data,
        param0,
        **prior_kwargs_i,
    )

    fit = fit_map_parameters_jax(
        fit_data,
        priors,
        param0,
        M,
        n_theta=n_theta,
        **optimizer_kwargs,
    )

    row = HP.summarize_map(fit["param"])
    row.update({
        "R_add": float(R_add),
        "success": fit["success"],
        "fun": fit["fun"],
        "nfev": fit["nfev"],
        "njev": fit["njev"],
        "nit": fit["nit"],
        "message": str(fit["message"]),
    })

    cov_out = None
    if compute_uncertainty and fit["success"]:
        cov_out = summarize_laplace_uncertainty_jax(
            fit["x"],
            fit["tk_jax"],
            fit["yk_jax"],
            fit["priors_jax"],
            M=M,
            n_theta=n_theta,
            hessian_method=hessian_method,
        )

        row.update({
            "omega_se": cov_out["omega_se"],
            "lnDtheta_se": cov_out["lnDtheta_se"],
            "lnR_se": cov_out["lnR_se"],
            "Dtheta_mult_se": cov_out["Dtheta_mult_se"],
            "R_eff_mult_se": cov_out["R_eff_mult_se"],
            "hessian_method": cov_out["hessian_method"],
            "hessian_min_eig": cov_out["hessian_min_eig"],
            "hessian_max_eig": cov_out["hessian_max_eig"],
            "hessian_condition": cov_out["hessian_condition"],
        })
    else:
        row.update({
            "omega_se": np.nan,
            "lnDtheta_se": np.nan,
            "lnR_se": np.nan,
            "Dtheta_mult_se": np.nan,
            "R_eff_mult_se": np.nan,
            "hessian_method": hessian_method,
            "hessian_min_eig": np.nan,
            "hessian_max_eig": np.nan,
            "hessian_condition": np.nan,
        })

    return {
        "row": row,
        "fit": fit,
        "param": fit["param"],
        "param0": param0,
        "priors": priors,
        "init": init,
        "fit_data": fit_data,
        "data_reg": data_reg,
        "cov": cov_out,
    }


# def sweep_added_noise_jax(
#     data,
#     R_add_values,
#     M,
#     rng,
#     *,
#     n_reps=1,
#     n_theta=128,
#     warm_start=True,
#     initialize_kwargs=None,
#     prior_kwargs=None,
#     optimizer_kwargs=None,
#     compute_uncertainty=True,
#     hessian_method="direct",
# ):
#     """
#     JAX continuation sweep over added-noise variance.

#     Priors
#     ------
#     omega:
#         prior center estimated once from the raw-data FFT.

#     Dtheta:
#         prior mode fixed across the sweep through Dtheta_mode.

#     R_eff:
#         prior mode reset at each point to the current R_add.

#     Continuation
#     ------------
#     If warm_start=True, the previous MAP estimate initializes the next fit.
#     This changes only the optimizer seed, not the statistical prior.
#     """
#     if initialize_kwargs is None:
#         initialize_kwargs = {}

#     if prior_kwargs is None:
#         prior_kwargs = {}

#     if optimizer_kwargs is None:
#         optimizer_kwargs = {}

#     rows = []
#     results = []

#     R_add_values = list(R_add_values)

#     tk0, yk0 = HP.as_inference_data(data)
#     omega_fft, _, _ = HP.estimate_frequency_fft((tk0, yk0))

#     for rep in range(n_reps):
#         param0 = None

#         for i, R_add in enumerate(R_add_values):
#             init_kwargs_i = dict(initialize_kwargs)
#             init_kwargs_i["omega0"] = omega_fft

#             out = run_added_noise_fit_jax(
#                 data,
#                 R_add,
#                 M,
#                 rng,
#                 n_theta=n_theta,
#                 param0=param0 if warm_start else None,
#                 initialize_kwargs=init_kwargs_i,
#                 prior_kwargs=prior_kwargs,
#                 optimizer_kwargs=optimizer_kwargs,
#                 compute_uncertainty=compute_uncertainty,
#                 hessian_method=hessian_method,
#             )

#             row = dict(out["row"])
#             row["rep"] = rep
#             row["sweep_index"] = i
#             row["omega_prior_mode"] = omega_fft
#             row["R_mode"] = R_add

#             rows.append(row)
#             results.append(out)

#             if warm_start and out["fit"]["success"]:
#                 param0 = out["param"]

#     return {
#         "summary": pd.DataFrame(rows),
#         "results": results,
#     }


def sweep_added_noise_jax(
    data,
    R_add_values,
    M,
    rng,
    *,
    n_reps=1,
    n_theta=128,
    warm_start=True,
    initialize_kwargs=None,
    prior_kwargs=None,
    optimizer_kwargs=None,
    compute_uncertainty=True,
    hessian_method="direct",
    compute_ppc=False,
    ppc_kwargs=None,
):
    if initialize_kwargs is None:
        initialize_kwargs = {}

    if prior_kwargs is None:
        prior_kwargs = {}

    if optimizer_kwargs is None:
        optimizer_kwargs = {}

    if ppc_kwargs is None:
        ppc_kwargs = {}

    rows = []
    results = []

    R_add_values = list(R_add_values)

    tk0, yk0 = HP.as_inference_data(data)
    omega_fft, _, _ = HP.estimate_frequency_fft((tk0, yk0))

    for rep in range(n_reps):
        param0 = None

        for i, R_add in enumerate(R_add_values):
            init_kwargs_i = dict(initialize_kwargs)
            init_kwargs_i["omega0"] = omega_fft

            out = run_added_noise_fit_jax(
                data,
                R_add,
                M,
                rng,
                n_theta=n_theta,
                param0=param0 if warm_start else None,
                initialize_kwargs=init_kwargs_i,
                prior_kwargs=prior_kwargs,
                optimizer_kwargs=optimizer_kwargs,
                compute_uncertainty=compute_uncertainty,
                hessian_method=hessian_method,
            )

            row = dict(out["row"])
            row["rep"] = rep
            row["sweep_index"] = i
            row["omega_prior_mode"] = omega_fft
            row["R_mode"] = R_add

            if compute_ppc and out["fit"]["success"] and out["cov"] is not None:
                ppc = HP.run_interval_ppcs(
                    out,
                    M=M,
                    rng=rng,
                    n_theta=n_theta,
                    **ppc_kwargs,
                )

                out["ppc"] = ppc

                row.update({
                    "ppc_tau": ppc["tau"],
                    "ppc_lag": ppc["lag"],

                    "ppc_msd_p": ppc["msd_p_value"],
                    "ppc_msd_pass": ppc["msd_passes"],
                    "ppc_msd_delta_mean": ppc["msd_delta_mean"],
                    "ppc_msd_delta_median": ppc["msd_delta_median"],

                    "ppc_tail_p": ppc["tail_p_value"],
                    "ppc_tail_pass": ppc["tail_passes"],
                    "ppc_tail_delta_mean": ppc["tail_delta_mean"],
                    "ppc_tail_delta_median": ppc["tail_delta_median"],
                    "ppc_tail_z": ppc["tail_z"],

                    "ppc_ng_p": ppc["ng_p_value"],
                    "ppc_ng_pass": ppc["ng_passes"],
                    "ppc_ng_delta_mean": ppc["ng_delta_mean"],
                    "ppc_ng_delta_median": ppc["ng_delta_median"],
                })

            else:
                out["ppc"] = None

                row.update({
                    "ppc_tau": np.nan,
                    "ppc_lag": np.nan,

                    "ppc_msd_p": np.nan,
                    "ppc_msd_pass": np.nan,
                    "ppc_msd_delta_mean": np.nan,
                    "ppc_msd_delta_median": np.nan,

                    "ppc_tail_p": np.nan,
                    "ppc_tail_pass": np.nan,
                    "ppc_tail_delta_mean": np.nan,
                    "ppc_tail_delta_median": np.nan,
                    "ppc_tail_z": np.nan,
                })

            rows.append(row)
            results.append(out)

            if warm_start and out["fit"]["success"]:
                param0 = out["param"]

    return {
        "summary": pd.DataFrame(rows),
        "results": results,
    }

def unpack_param_jax(x, M):
    omega = x[0]
    lnDtheta = x[1]
    lnR = x[2]

    nA = M + 1
    A = x[3:3 + nA]

    B = jnp.zeros(M + 1)
    if M >= 2:
        B = B.at[2:].set(x[3 + nA:])

    return omega, lnDtheta, lnR, A, B


def wrap_to_pi_jax(x):
    return (x + jnp.pi) % (2.0 * jnp.pi) - jnp.pi


def q_fourier_jax(theta, A, B):
    M = A.shape[0] - 1
    m = jnp.arange(M + 1)

    q_cos = jnp.sum(A[None, :] * jnp.cos(theta[:, None] * m[None, :]), axis=1)
    q_sin = jnp.sum(B[None, :] * jnp.sin(theta[:, None] * m[None, :]), axis=1)

    return q_cos + q_sin


def wrapped_gaussian_kernel_jax(theta_grid, mean_step, Q, eps=1e-30):
    """
    Wrapped Gaussian transition kernel.

    The small eps floor prevents exact zeros in the transition kernel.
    Exact zeros are usually harmless for function evaluation but can poison
    autodiff through the filtering recursion at low process/noise levels.
    """
    delta = wrap_to_pi_jax(theta_grid - mean_step)

    log_kernel = -0.5 * delta**2 / Q
    log_kernel = log_kernel - jax.scipy.special.logsumexp(log_kernel)

    kernel = jnp.exp(log_kernel)

    # Keep all transition probabilities strictly positive.
    kernel = kernel + eps
    kernel = kernel / jnp.sum(kernel)

    return kernel


def circular_convolve_jax(kernel, p):
    return jnp.real(jnp.fft.ifft(jnp.fft.fft(kernel) * jnp.fft.fft(p)))


def neg_log_prior_jax(x, priors, M):
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

    omega, lnDtheta, lnR, A, B = unpack_param_jax(x, M)

    Dtheta = jnp.exp(lnDtheta)
    R = jnp.exp(lnR)

    nlp = 0.0

    # omega prior
    nlp = nlp + 0.5 * ((omega - mu_omega) / sigma_omega) ** 2

    # inverse-gamma priors in log coordinates
    nlp = nlp + (alpha_D + 1.0) * lnDtheta + beta_D / Dtheta
    nlp = nlp + (alpha_R + 1.0) * lnR + beta_R / R

    # Jacobian correction for log coordinates
    nlp = nlp - lnDtheta
    nlp = nlp - lnR

    # waveform priors
    nlp = nlp + 0.5 * ((A[0] - mu_A) / sigma_A) ** 2
    # nlp = nlp + 0.5 * jnp.sum((A[1:] / sigma_q) ** 2)

    # if M >= 2:
    #     nlp = nlp + 0.5 * jnp.sum((B[2:] / sigma_q) ** 2)
    p_wave = 1.0
    
    mA = jnp.arange(1, M + 1)
    sigmaA = sigma_q / (mA ** p_wave)
    
    nlp = nlp + 0.5 * jnp.sum((A[1:] / sigmaA) ** 2)
    
    if M >= 2:
        mB = jnp.arange(2, M + 1)
        sigmaB = sigma_q / (mB ** p_wave)
    
        nlp = nlp + 0.5 * jnp.sum((B[2:] / sigmaB) ** 2)
    
    return nlp


def neg_log_marginal_likelihood_jax(x, tk, yk, priors, M, n_theta):
    (
        mu_theta,
        sigma_theta,
        *_,
    ) = priors

    omega, lnDtheta, lnR, A, B = unpack_param_jax(x, M)

    Dtheta = jnp.exp(lnDtheta)
    R = jnp.exp(lnR)

    N = yk.shape[0]
    dt = tk[1] - tk[0]

    theta_grid = jnp.linspace(0.0, 2.0 * jnp.pi, n_theta, endpoint=False)

    Q = 2.0 * Dtheta * dt
    mean_step = omega * dt

    K_hat = spectral_heat_kernel_hat_jax(n_theta, mean_step, Q)
    q_grid = q_fourier_jax(theta_grid, A, B)

    def log_likelihood_row(y):
        resid = y - q_grid
        return -0.5 * jnp.log(2.0 * jnp.pi * R) - 0.5 * resid**2 / R

    logL = jax.vmap(log_likelihood_row)(yk)

    prior_delta = wrap_to_pi_jax(theta_grid - mu_theta)
    log_prior_theta = -0.5 * prior_delta**2 / sigma_theta**2
    log_prior_theta = log_prior_theta - jax.scipy.special.logsumexp(log_prior_theta)

    # eps_alpha = 1e-30

    # def step(log_alpha_prev, logL_k):
    #     alpha_prev = jnp.exp(log_alpha_prev)
    
    #     alpha_pred = spectral_predict_jax(alpha_prev, K_hat)
    
    #     alpha_pred = jnp.maximum(alpha_pred, 0.0)
    #     alpha_pred = alpha_pred + eps_alpha
    #     alpha_pred = alpha_pred / jnp.sum(alpha_pred)
    
    #     log_alpha_pred = jnp.log(alpha_pred)
    #     log_unnorm = log_alpha_pred + logL_k
    
    #     log_scale = jax.scipy.special.logsumexp(log_unnorm)
    #     log_alpha = log_unnorm - log_scale
    
    #     return log_alpha, log_scale

    eps_soft = 1e-12

    def project_probability_softplus_jax(p, eps=eps_soft):
        p_pos = eps * jax.nn.softplus(p / eps)
        return p_pos / jnp.sum(p_pos)
    
    def step(log_alpha_prev, logL_k):
        alpha_prev = jnp.exp(log_alpha_prev)
    
        alpha_pred_raw = spectral_predict_jax(alpha_prev, K_hat)
    
        alpha_pred = project_probability_softplus_jax(
            alpha_pred_raw,
            eps=eps_soft,
        )
    
        log_alpha_pred = jnp.log(alpha_pred)
        log_unnorm = log_alpha_pred + logL_k
    
        log_scale = jax.scipy.special.logsumexp(log_unnorm)
        log_alpha = log_unnorm - log_scale
    
        return log_alpha, log_scale
    
    # k = 0 update
    log_unnorm0 = log_prior_theta + logL[0]
    log_scale0 = jax.scipy.special.logsumexp(log_unnorm0)
    log_alpha0 = log_unnorm0 - log_scale0

    # k >= 1 updates
    _, log_scales_tail = jax.lax.scan(step, log_alpha0, logL[1:])

    loglik = log_scale0 + jnp.sum(log_scales_tail)

    return -loglik


def neg_log_posterior_flat_jax(x, tk, yk, priors, M, n_theta=256):
    nll = neg_log_marginal_likelihood_jax(
        x,
        tk,
        yk,
        priors,
        M,
        n_theta,
    )

    nlp = neg_log_prior_jax(
        x,
        priors,
        M,
    )

    return nll + nlp


neg_log_posterior_flat_jit = jax.jit(
    neg_log_posterior_flat_jax,
    static_argnames=("M", "n_theta"),
)

grad_neg_log_posterior_flat_jit = jax.jit(
    jax.grad(neg_log_posterior_flat_jax),
    static_argnames=("M", "n_theta"),
)

hessian_neg_log_posterior_flat_jit = jax.jit(
    jax.hessian(neg_log_posterior_flat_jax),
    static_argnames=("M", "n_theta"),
)


def fd_hessian_from_grad(
    grad_fun,
    x0,
    *,
    step=None,
):
    """
    Finite-difference Hessian from a gradient function.

    Parameters
    ----------
    grad_fun : callable
        Returns gradient vector.

    x0 : ndarray
        Evaluation point.

    step : float or ndarray, optional
        Finite-difference step size.

    Returns
    -------
    H : ndarray
        Symmetric Hessian approximation.
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

    for j in range(n):
        dx = np.zeros(n)
        dx[j] = step[j]

        gp = np.asarray(grad_fun(x0 + dx))
        gm = np.asarray(grad_fun(x0 - dx))

        H[:, j] = (gp - gm) / (2.0 * step[j])

    return 0.5 * (H + H.T)


def hessian_neg_log_posterior_fd(
    x0,
    tk,
    yk,
    priors,
    *,
    M,
    n_theta=256,
    step=None,
):
    """
    Finite-difference Hessian of the JAX posterior gradient.
    """

    def grad_fun(x):
        return np.asarray(
            grad_neg_log_posterior_flat_jit(
                jnp.asarray(x),
                tk,
                yk,
                priors,
                M=M,
                n_theta=n_theta,
            )
        )

    return fd_hessian_from_grad(
        grad_fun,
        x0,
        step=step,
    )