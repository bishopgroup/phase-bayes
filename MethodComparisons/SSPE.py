"""State-space phase estimation for sums of stochastic oscillators.

This module independently implements the linear Gaussian oscillator model
described by Matsuda and Komaki (2017) and adapted for state-space phase
estimation by Wodeyar et al. (2021).  It requires only NumPy and SciPy.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import block_diag


_TWO_PI = 2.0 * np.pi
_MIN_VARIANCE = np.finfo(float).eps


def oscillator_matrices(
    frequencies,
    sampling_frequency,
    damping,
    process_variances,
):
    """Construct the state-space matrices for stochastic oscillators.

    Parameters
    ----------
    frequencies : array_like, shape (n_components,)
        Oscillator frequencies in hertz.
    sampling_frequency : float
        Observation sampling frequency in hertz.
    damping : array_like, shape (n_components,)
        Per-sample damping factors in the open interval ``(0, 1)``.
    process_variances : array_like, shape (n_components,)
        Isotropic process-noise variances for the oscillator coordinates.

    Returns
    -------
    transition : ndarray, shape (2*n_components, 2*n_components)
        Block-diagonal damped rotation matrix.
    process_covariance : ndarray, shape (2*n_components, 2*n_components)
        Block-diagonal process covariance.
    observation : ndarray, shape (2*n_components,)
        Observation vector selecting and summing each real coordinate.

    Raises
    ------
    ValueError
        If the parameters have incompatible shapes or invalid values.
    """
    frequencies = np.atleast_1d(np.asarray(frequencies, dtype=float))
    damping = np.atleast_1d(np.asarray(damping, dtype=float))
    process_variances = np.atleast_1d(np.asarray(process_variances, dtype=float))

    if frequencies.ndim != 1 or frequencies.size == 0:
        raise ValueError("frequencies must be a nonempty one-dimensional array.")
    if damping.shape != frequencies.shape:
        raise ValueError("damping must have the same shape as frequencies.")
    if process_variances.shape != frequencies.shape:
        raise ValueError(
            "process_variances must have the same shape as frequencies."
        )
    if not np.isfinite(sampling_frequency) or sampling_frequency <= 0:
        raise ValueError("sampling_frequency must be finite and positive.")
    if not np.all(np.isfinite(frequencies)):
        raise ValueError("frequencies must be finite.")
    if np.any(frequencies <= 0) or np.any(frequencies >= sampling_frequency / 2):
        raise ValueError("frequencies must lie strictly between zero and Nyquist.")
    if not np.all(np.isfinite(damping)) or np.any(damping <= 0) or np.any(
        damping >= 1
    ):
        raise ValueError("damping factors must lie strictly between zero and one.")
    if not np.all(np.isfinite(process_variances)) or np.any(
        process_variances <= 0
    ):
        raise ValueError("process_variances must be finite and positive.")

    transition_blocks = []
    process_blocks = []
    for frequency, decay, variance in zip(
        frequencies, damping, process_variances
    ):
        angle = _TWO_PI * frequency / sampling_frequency
        rotation = np.array(
            [
                [np.cos(angle), -np.sin(angle)],
                [np.sin(angle), np.cos(angle)],
            ]
        )
        transition_blocks.append(decay * rotation)
        process_blocks.append(variance * np.eye(2))

    transition = block_diag(*transition_blocks)
    process_covariance = block_diag(*process_blocks)
    observation = np.zeros(2 * frequencies.size)
    observation[::2] = 1.0
    return transition, process_covariance, observation


def kalman_filter(
    signal,
    transition,
    process_covariance,
    observation,
    observation_variance,
    *,
    initial_mean=None,
    initial_covariance=None,
):
    """Run a Kalman filter for a scalar observation sequence.

    Parameters
    ----------
    signal : array_like, shape (n_time,)
        Scalar observations.
    transition, process_covariance : ndarray, shape (n_state, n_state)
        State transition and process covariance matrices.
    observation : ndarray, shape (n_state,)
        Linear observation vector.
    observation_variance : float
        Scalar observation-noise variance.
    initial_mean : array_like, shape (n_state,), optional
        Mean of the state prior before the first observation.
    initial_covariance : ndarray, shape (n_state, n_state), optional
        Covariance of the state prior before the first observation.

    Returns
    -------
    result : dict
        Predicted and filtered means and covariances, Kalman gains,
        innovations, innovation variances, and total log likelihood.

    Raises
    ------
    ValueError
        If shapes are inconsistent or variances are invalid.
    """
    signal = np.asarray(signal, dtype=float)
    transition = np.asarray(transition, dtype=float)
    process_covariance = np.asarray(process_covariance, dtype=float)
    observation = np.asarray(observation, dtype=float)

    if signal.ndim != 1 or signal.size < 2 or not np.all(np.isfinite(signal)):
        raise ValueError("signal must be a finite one-dimensional array of length >= 2.")
    if transition.ndim != 2 or transition.shape[0] != transition.shape[1]:
        raise ValueError("transition must be square.")
    n_state = transition.shape[0]
    if process_covariance.shape != (n_state, n_state):
        raise ValueError("process_covariance has an incompatible shape.")
    if observation.shape != (n_state,):
        raise ValueError("observation has an incompatible shape.")
    if not np.all(np.isfinite(transition)) or not np.all(
        np.isfinite(observation)
    ):
        raise ValueError("transition and observation must be finite.")
    _validate_covariance(process_covariance, "process_covariance")
    if not np.isfinite(observation_variance) or observation_variance <= 0:
        raise ValueError("observation_variance must be finite and positive.")

    if initial_mean is None:
        initial_mean = np.zeros(n_state)
    else:
        initial_mean = np.asarray(initial_mean, dtype=float)
    if initial_covariance is None:
        initial_covariance = np.eye(n_state)
    else:
        initial_covariance = np.asarray(initial_covariance, dtype=float)
    if initial_mean.shape != (n_state,):
        raise ValueError("initial_mean has an incompatible shape.")
    if initial_covariance.shape != (n_state, n_state):
        raise ValueError("initial_covariance has an incompatible shape.")
    if not np.all(np.isfinite(initial_mean)):
        raise ValueError("initial_mean must be finite.")
    _validate_covariance(initial_covariance, "initial_covariance")

    n_time = signal.size
    predicted_mean = np.empty((n_time, n_state))
    predicted_covariance = np.empty((n_time, n_state, n_state))
    filtered_mean = np.empty_like(predicted_mean)
    filtered_covariance = np.empty_like(predicted_covariance)
    kalman_gain = np.empty((n_time, n_state))
    innovation = np.empty(n_time)
    innovation_variance = np.empty(n_time)

    mean_prior = initial_mean.copy()
    covariance_prior = _symmetrize(initial_covariance)
    log_likelihood = 0.0

    for index, observation_value in enumerate(signal):
        predicted_mean[index] = mean_prior
        predicted_covariance[index] = covariance_prior

        residual = observation_value - observation @ mean_prior
        residual_variance = float(
            observation @ covariance_prior @ observation + observation_variance
        )
        if not np.isfinite(residual_variance) or residual_variance <= 0:
            raise RuntimeError("Encountered a nonpositive innovation variance.")

        gain = covariance_prior @ observation / residual_variance
        mean_filtered = mean_prior + gain * residual
        covariance_filtered = covariance_prior - np.outer(
            gain, observation @ covariance_prior
        )
        covariance_filtered = _symmetrize(covariance_filtered)

        filtered_mean[index] = mean_filtered
        filtered_covariance[index] = covariance_filtered
        kalman_gain[index] = gain
        innovation[index] = residual
        innovation_variance[index] = residual_variance
        log_likelihood -= 0.5 * (
            np.log(_TWO_PI * residual_variance)
            + residual * residual / residual_variance
        )

        mean_prior = transition @ mean_filtered
        covariance_prior = _symmetrize(
            transition @ covariance_filtered @ transition.T + process_covariance
        )

    return {
        "predicted_mean": predicted_mean,
        "predicted_covariance": predicted_covariance,
        "filtered_mean": filtered_mean,
        "filtered_covariance": filtered_covariance,
        "kalman_gain": kalman_gain,
        "innovation": innovation,
        "innovation_variance": innovation_variance,
        "log_likelihood": float(log_likelihood),
    }


def rts_smoother(filter_result, transition):
    """Apply Rauch--Tung--Striebel smoothing to Kalman-filter output.

    Parameters
    ----------
    filter_result : dict
        Output from :func:`kalman_filter`.
    transition : ndarray, shape (n_state, n_state)
        State transition matrix.

    Returns
    -------
    result : dict
        Smoothed means, covariances, smoother gains, and lag-one
        covariances ``Cov(x[k], x[k-1] | y)``.
    """
    transition = np.asarray(transition, dtype=float)
    predicted_mean = np.asarray(filter_result["predicted_mean"])
    predicted_covariance = np.asarray(filter_result["predicted_covariance"])
    filtered_mean = np.asarray(filter_result["filtered_mean"])
    filtered_covariance = np.asarray(filter_result["filtered_covariance"])

    n_time, n_state = filtered_mean.shape
    if transition.shape != (n_state, n_state):
        raise ValueError("transition has an incompatible shape.")

    smoothed_mean = filtered_mean.copy()
    smoothed_covariance = filtered_covariance.copy()
    smoother_gain = np.zeros((n_time - 1, n_state, n_state))

    for index in range(n_time - 2, -1, -1):
        cross = filtered_covariance[index] @ transition.T
        gain = np.linalg.solve(
            predicted_covariance[index + 1], cross.T
        ).T
        smoother_gain[index] = gain
        smoothed_mean[index] = filtered_mean[index] + gain @ (
            smoothed_mean[index + 1] - predicted_mean[index + 1]
        )
        smoothed_covariance[index] = _symmetrize(
            filtered_covariance[index]
            + gain
            @ (
                smoothed_covariance[index + 1]
                - predicted_covariance[index + 1]
            )
            @ gain.T
        )

    lag_covariance = np.zeros_like(smoothed_covariance)
    for index in range(1, n_time):
        lag_covariance[index] = (
            smoothed_covariance[index] @ smoother_gain[index - 1].T
        )

    return {
        "smoothed_mean": smoothed_mean,
        "smoothed_covariance": smoothed_covariance,
        "smoother_gain": smoother_gain,
        "lag_covariance": lag_covariance,
    }


def fit_sspe(
    signal,
    sampling_frequency,
    initial_frequencies,
    *,
    initial_damping=0.99,
    initial_process_variances=None,
    initial_observation_variance=None,
    initial_mean=None,
    initial_covariance=None,
    max_iterations=400,
    frequency_tolerance=1e-3,
    minimum_iterations=2,
    variance_floor=None,
):
    """Fit a multi-oscillator SSPE model by structured EM.

    Parameters
    ----------
    signal : array_like, shape (n_time,)
        Finite, mean-centered scalar observations.
    sampling_frequency : float
        Sampling frequency in hertz.
    initial_frequencies : array_like, shape (n_components,)
        Initial component frequencies in hertz. Component order is preserved.
    initial_damping : float or array_like, optional
        Initial per-sample damping factors.
    initial_process_variances : array_like, optional
        Initial isotropic component process variances.
    initial_observation_variance : float, optional
        Initial observation-noise variance.
    initial_mean : array_like, optional
        State prior mean.
    initial_covariance : ndarray, optional
        State prior covariance. By default, stationary component variances
        implied by the initial parameters are used.
    max_iterations : int, optional
        Maximum number of EM parameter updates.
    frequency_tolerance : float, optional
        Convergence threshold for the sum of absolute frequency changes,
        in hertz.
    minimum_iterations : int, optional
        Minimum number of EM updates before convergence is accepted.
    variance_floor : float, optional
        Positive numerical floor for fitted process and observation variances.

    Returns
    -------
    result : dict
        Fitted parameters, matrices, filter and smoother results,
        reconstructions, phases, likelihood and parameter histories,
        convergence flag, and iteration count.

    Raises
    ------
    ValueError
        If data or fitting controls are invalid.
    RuntimeError
        If filtering or smoothing becomes numerically invalid.
    """
    signal = np.asarray(signal, dtype=float)
    frequencies = np.atleast_1d(np.asarray(initial_frequencies, dtype=float))
    if signal.ndim != 1 or signal.size < 3 or not np.all(np.isfinite(signal)):
        raise ValueError("signal must be a finite one-dimensional array of length >= 3.")
    if abs(float(np.mean(signal))) > 1e-8 * max(1.0, float(np.std(signal))):
        raise ValueError("signal must be mean-centered before fitting.")
    if not isinstance(max_iterations, (int, np.integer)) or max_iterations < 1:
        raise ValueError("max_iterations must be a positive integer.")
    if (
        not isinstance(minimum_iterations, (int, np.integer))
        or minimum_iterations < 1
        or minimum_iterations > max_iterations
    ):
        raise ValueError(
            "minimum_iterations must be between one and max_iterations."
        )
    if not np.isfinite(frequency_tolerance) or frequency_tolerance <= 0:
        raise ValueError("frequency_tolerance must be finite and positive.")

    n_components = frequencies.size
    damping = np.broadcast_to(
        np.asarray(initial_damping, dtype=float), (n_components,)
    ).copy()
    signal_variance = float(np.var(signal))
    if signal_variance <= 0:
        raise ValueError("signal must have positive variance.")

    if initial_process_variances is None:
        stationary_share = signal_variance / n_components
        process_variances = stationary_share * (1.0 - damping**2)
    else:
        process_variances = np.broadcast_to(
            np.asarray(initial_process_variances, dtype=float),
            (n_components,),
        ).copy()
    if initial_observation_variance is None:
        observation_variance = 0.5 * signal_variance
    else:
        observation_variance = float(initial_observation_variance)

    if variance_floor is None:
        variance_floor = max(_MIN_VARIANCE, 1e-12 * signal_variance)
    if not np.isfinite(variance_floor) or variance_floor <= 0:
        raise ValueError("variance_floor must be finite and positive.")

    transition, process_covariance, observation = oscillator_matrices(
        frequencies,
        sampling_frequency,
        damping,
        process_variances,
    )
    n_state = 2 * n_components
    if initial_mean is None:
        initial_mean = np.zeros(n_state)
    else:
        initial_mean = np.asarray(initial_mean, dtype=float)
    if initial_covariance is None:
        stationary_variances = process_variances / (1.0 - damping**2)
        initial_covariance = block_diag(
            *(variance * np.eye(2) for variance in stationary_variances)
        )
    else:
        initial_covariance = np.asarray(initial_covariance, dtype=float)

    likelihood_history = []
    frequency_history = [frequencies.copy()]
    damping_history = [damping.copy()]
    process_variance_history = [process_variances.copy()]
    observation_variance_history = [observation_variance]
    converged = False

    for iteration in range(1, max_iterations + 1):
        filter_result = kalman_filter(
            signal,
            transition,
            process_covariance,
            observation,
            observation_variance,
            initial_mean=initial_mean,
            initial_covariance=initial_covariance,
        )
        smoother_result = rts_smoother(filter_result, transition)
        likelihood_history.append(filter_result["log_likelihood"])

        frequencies_new, damping_new, process_variances_new, observation_variance_new = (
            _structured_m_step(
                signal,
                smoother_result,
                observation,
                sampling_frequency,
                variance_floor,
            )
        )
        frequency_change = float(np.sum(np.abs(frequencies_new - frequencies)))

        frequencies = frequencies_new
        damping = damping_new
        process_variances = process_variances_new
        observation_variance = observation_variance_new
        transition, process_covariance, observation = oscillator_matrices(
            frequencies,
            sampling_frequency,
            damping,
            process_variances,
        )

        frequency_history.append(frequencies.copy())
        damping_history.append(damping.copy())
        process_variance_history.append(process_variances.copy())
        observation_variance_history.append(observation_variance)

        if iteration >= minimum_iterations and frequency_change < frequency_tolerance:
            converged = True
            break

    filter_result = kalman_filter(
        signal,
        transition,
        process_covariance,
        observation,
        observation_variance,
        initial_mean=initial_mean,
        initial_covariance=initial_covariance,
    )
    smoother_result = rts_smoother(filter_result, transition)
    likelihood_history.append(filter_result["log_likelihood"])

    filtered_phase = component_phase(filter_result["filtered_mean"])
    smoothed_phase = component_phase(smoother_result["smoothed_mean"])
    filtered_components = filter_result["filtered_mean"][:, ::2]
    smoothed_components = smoother_result["smoothed_mean"][:, ::2]

    return {
        "frequencies": frequencies,
        "damping": damping,
        "process_variances": process_variances,
        "observation_variance": observation_variance,
        "transition": transition,
        "process_covariance": process_covariance,
        "observation": observation,
        "filter": filter_result,
        "smoother": smoother_result,
        "filtered_components": filtered_components,
        "smoothed_components": smoothed_components,
        "filtered_reconstruction": np.sum(filtered_components, axis=1),
        "smoothed_reconstruction": np.sum(smoothed_components, axis=1),
        "filtered_phase": filtered_phase,
        "smoothed_phase": smoothed_phase,
        "likelihood_history": np.asarray(likelihood_history),
        "frequency_history": np.asarray(frequency_history),
        "damping_history": np.asarray(damping_history),
        "process_variance_history": np.asarray(process_variance_history),
        "observation_variance_history": np.asarray(
            observation_variance_history
        ),
        "converged": converged,
        "n_iterations": iteration,
        "sampling_frequency": float(sampling_frequency),
    }


def simulate_sspe(
    n_samples,
    sampling_frequency,
    frequencies,
    damping,
    process_variances,
    observation_variance,
    *,
    initial_state=None,
    rng=None,
):
    """Simulate observations and latent states from an SSPE model.

    Parameters
    ----------
    n_samples : int
        Number of samples to simulate.
    sampling_frequency : float
        Sampling frequency in hertz.
    frequencies : array_like, shape (n_components,)
        Oscillator frequencies in hertz.
    damping : array_like, shape (n_components,)
        Per-sample damping factors in the open interval ``(0, 1)``.
    process_variances : array_like, shape (n_components,)
        Isotropic process-noise variances.
    observation_variance : float
        Scalar observation-noise variance.
    initial_state : array_like, shape (2*n_components,), optional
        Initial latent state. By default, each component is drawn from its
        exact stationary Gaussian distribution.
    rng : numpy.random.Generator, optional
        Random-number generator. Supply a seeded generator for reproducible
        simulations.

    Returns
    -------
    result : dict
        Latent states, real-coordinate components, amplitudes, wrapped and
        unwrapped phases, noiseless reconstruction, observation noise, noisy
        observations, and the model matrices.

    Raises
    ------
    ValueError
        If the sample count, parameters, variance, or initial state is invalid.
    """
    if not isinstance(n_samples, (int, np.integer)) or n_samples < 1:
        raise ValueError("n_samples must be a positive integer.")
    if not np.isfinite(observation_variance) or observation_variance <= 0:
        raise ValueError("observation_variance must be finite and positive.")
    if rng is None:
        rng = np.random.default_rng()

    frequencies = np.atleast_1d(np.asarray(frequencies, dtype=float))
    damping = np.atleast_1d(np.asarray(damping, dtype=float))
    process_variances = np.atleast_1d(
        np.asarray(process_variances, dtype=float)
    )
    transition, process_covariance, observation = oscillator_matrices(
        frequencies,
        sampling_frequency,
        damping,
        process_variances,
    )
    n_components = frequencies.size
    n_state = 2 * n_components

    states = np.empty((n_samples, n_state))
    if initial_state is None:
        stationary_variances = process_variances / (1.0 - damping**2)
        initial_covariance = block_diag(
            *(variance * np.eye(2) for variance in stationary_variances)
        )
        states[0] = rng.multivariate_normal(
            np.zeros(n_state),
            initial_covariance,
            check_valid="raise",
        )
    else:
        initial_state = np.asarray(initial_state, dtype=float)
        if initial_state.shape != (n_state,) or not np.all(
            np.isfinite(initial_state)
        ):
            raise ValueError(
                "initial_state must be finite with shape (2*n_components,)."
            )
        states[0] = initial_state

    zero_mean = np.zeros(n_state)
    for index in range(1, n_samples):
        states[index] = transition @ states[index - 1] + rng.multivariate_normal(
            zero_mean,
            process_covariance,
            check_valid="raise",
        )

    components = states[:, ::2]
    amplitudes = np.hypot(states[:, ::2], states[:, 1::2])
    phases_wrapped = component_phase(states)
    phases_unwrapped = np.unwrap(phases_wrapped, axis=0)
    noiseless_observation = components @ np.ones(n_components)
    observation_noise = rng.normal(
        scale=np.sqrt(observation_variance),
        size=n_samples,
    )
    noisy_observation = noiseless_observation + observation_noise

    return {
        "states": states,
        "components": components,
        "amplitudes": amplitudes,
        "phases_wrapped": phases_wrapped,
        "phases_unwrapped": phases_unwrapped,
        "noiseless_observation": noiseless_observation,
        "observation_noise": observation_noise,
        "observation": noisy_observation,
        "transition": transition,
        "process_covariance": process_covariance,
        "observation_vector": observation,
        "sampling_frequency": float(sampling_frequency),
    }


def component_phase(state_mean):
    """Extract wrapped component phases from interleaved state coordinates.

    Parameters
    ----------
    state_mean : array_like, shape (n_time, 2*n_components)
        State means ordered as real and imaginary coordinates per component.

    Returns
    -------
    phase : ndarray, shape (n_time, n_components)
        Wrapped component phases in ``[-pi, pi)``.
    """
    state_mean = np.asarray(state_mean, dtype=float)
    if state_mean.ndim != 2 or state_mean.shape[1] % 2:
        raise ValueError("state_mean must have shape (n_time, 2*n_components).")
    return np.arctan2(state_mean[:, 1::2], state_mean[:, ::2])


def phase_uncertainty_delta(state_mean, state_covariance, component):
    """Approximate phase uncertainty by first-order covariance propagation.

    Parameters
    ----------
    state_mean : ndarray, shape (n_time, n_state)
        State means.
    state_covariance : ndarray, shape (n_time, n_state, n_state)
        State covariances.
    component : int
        Zero-based oscillator index.

    Returns
    -------
    result : dict
        Wrapped and unwrapped mean phase, phase standard deviation, and
        amplitude. Standard deviation is infinite when the mean amplitude
        is numerically zero.
    """
    means, covariances = _component_moments(
        state_mean, state_covariance, component
    )
    real = means[:, 0]
    imag = means[:, 1]
    amplitude_squared = real**2 + imag**2
    phase = np.arctan2(imag, real)
    phase_variance = np.full(real.shape, np.inf)
    valid = amplitude_squared > np.finfo(float).eps
    gradient = np.column_stack(
        (
            -imag[valid] / amplitude_squared[valid],
            real[valid] / amplitude_squared[valid],
        )
    )
    phase_variance[valid] = np.einsum(
        "ni,nij,nj->n",
        gradient,
        covariances[valid],
        gradient,
    )
    phase_variance[valid] = np.maximum(phase_variance[valid], 0.0)
    return {
        "phase_wrapped": phase,
        "phase_unwrapped": np.unwrap(phase),
        "phase_std": np.sqrt(phase_variance),
        "amplitude": np.sqrt(amplitude_squared),
    }


def phase_uncertainty_monte_carlo(
    state_mean,
    state_covariance,
    component,
    *,
    n_samples=2000,
    quantiles=(0.025, 0.25, 0.75, 0.975),
    rng=None,
):
    """Estimate circular phase uncertainty by Gaussian state sampling.

    Parameters
    ----------
    state_mean : ndarray, shape (n_time, n_state)
        State means.
    state_covariance : ndarray, shape (n_time, n_state, n_state)
        State covariances.
    component : int
        Zero-based oscillator index.
    n_samples : int, optional
        Gaussian state samples per time point.
    quantiles : sequence of float, optional
        Quantiles of phase deviations from the mean phase.
    rng : numpy.random.Generator, optional
        Random-number generator. A fixed generator should be supplied for
        reproducible intervals.

    Returns
    -------
    result : dict
        Wrapped and unwrapped mean phase, requested unwrapped phase
        quantiles, circular standard deviation, and amplitude.
    """
    if not isinstance(n_samples, (int, np.integer)) or n_samples < 2:
        raise ValueError("n_samples must be an integer >= 2.")
    quantiles = np.asarray(quantiles, dtype=float)
    if (
        quantiles.ndim != 1
        or quantiles.size == 0
        or np.any(quantiles <= 0)
        or np.any(quantiles >= 1)
        or np.any(np.diff(quantiles) <= 0)
    ):
        raise ValueError("quantiles must be strictly increasing within (0, 1).")
    if rng is None:
        rng = np.random.default_rng()

    means, covariances = _component_moments(
        state_mean, state_covariance, component
    )
    phase_wrapped = np.arctan2(means[:, 1], means[:, 0])
    phase_unwrapped = np.unwrap(phase_wrapped)
    phase_quantiles = np.empty((means.shape[0], quantiles.size))
    circular_std = np.empty(means.shape[0])

    for index, (mean, covariance) in enumerate(zip(means, covariances)):
        covariance = _positive_semidefinite(covariance)
        samples = rng.multivariate_normal(
            mean,
            covariance,
            size=n_samples,
            check_valid="ignore",
        )
        sample_phase = np.arctan2(samples[:, 1], samples[:, 0])
        deviations = np.angle(
            np.exp(1j * (sample_phase - phase_wrapped[index]))
        )
        phase_quantiles[index] = phase_unwrapped[index] + np.quantile(
            deviations, quantiles
        )
        resultant_length = np.abs(np.mean(np.exp(1j * deviations)))
        circular_std[index] = np.sqrt(
            -2.0 * np.log(np.clip(resultant_length, 1e-15, 1.0))
        )

    return {
        "phase_wrapped": phase_wrapped,
        "phase_unwrapped": phase_unwrapped,
        "quantiles": quantiles,
        "phase_quantiles": phase_quantiles,
        "circular_std": circular_std,
        "amplitude": np.linalg.norm(means, axis=1),
    }


def _structured_m_step(
    signal,
    smoother_result,
    observation,
    sampling_frequency,
    variance_floor,
):
    """Update structured oscillator parameters from smoothed moments."""
    mean = smoother_result["smoothed_mean"]
    covariance = smoother_result["smoothed_covariance"]
    lag_covariance = smoother_result["lag_covariance"]
    n_time, n_state = mean.shape
    n_components = n_state // 2

    second_moment = covariance + np.einsum("ni,nj->nij", mean, mean)
    previous_sum = np.sum(second_moment[:-1], axis=0)
    current_sum = np.sum(second_moment[1:], axis=0)
    lag_sum = np.sum(
        lag_covariance[1:]
        + np.einsum("ni,nj->nij", mean[1:], mean[:-1]),
        axis=0,
    )

    frequencies = np.empty(n_components)
    damping = np.empty(n_components)
    process_variances = np.empty(n_components)
    for component in range(n_components):
        component_slice = slice(2 * component, 2 * component + 2)
        previous = previous_sum[component_slice, component_slice]
        current = current_sum[component_slice, component_slice]
        lagged = lag_sum[component_slice, component_slice]

        cosine_statistic = float(np.trace(lagged))
        sine_statistic = float(lagged[1, 0] - lagged[0, 1])
        angle = abs(np.arctan2(sine_statistic, cosine_statistic))
        angle = np.clip(angle, np.finfo(float).eps, np.pi - np.finfo(float).eps)
        frequencies[component] = (
            angle * sampling_frequency / _TWO_PI
        )

        denominator = float(np.trace(previous))
        if not np.isfinite(denominator) or denominator <= 0:
            raise RuntimeError("Degenerate smoothed state energy in EM update.")
        damping_raw = np.hypot(cosine_statistic, sine_statistic) / denominator
        damping[component] = np.clip(
            damping_raw,
            np.sqrt(np.finfo(float).eps),
            1.0 - np.sqrt(np.finfo(float).eps),
        )

        rotation = np.array(
            [
                [np.cos(angle), -np.sin(angle)],
                [np.sin(angle), np.cos(angle)],
            ]
        )
        transition_block = damping[component] * rotation
        residual_trace = np.trace(
            current
            - transition_block @ lagged.T
            - lagged @ transition_block.T
            + transition_block @ previous @ transition_block.T
        )
        process_variances[component] = max(
            float(residual_trace) / (2.0 * (n_time - 1)),
            variance_floor,
        )

    predicted_observation = mean @ observation
    observation_uncertainty = np.einsum(
        "i,nij,j->n", observation, covariance, observation
    )
    observation_variance = max(
        float(
            np.mean(
                (signal - predicted_observation) ** 2
                + observation_uncertainty
            )
        ),
        variance_floor,
    )
    return frequencies, damping, process_variances, observation_variance


def _component_moments(state_mean, state_covariance, component):
    """Extract one component's two-dimensional Gaussian moments."""
    state_mean = np.asarray(state_mean, dtype=float)
    state_covariance = np.asarray(state_covariance, dtype=float)
    if state_mean.ndim != 2 or state_mean.shape[1] % 2:
        raise ValueError("state_mean must have shape (n_time, 2*n_components).")
    n_time, n_state = state_mean.shape
    if state_covariance.shape != (n_time, n_state, n_state):
        raise ValueError("state_covariance has an incompatible shape.")
    n_components = n_state // 2
    if (
        not isinstance(component, (int, np.integer))
        or component < 0
        or component >= n_components
    ):
        raise ValueError("component is out of range.")
    component_slice = slice(2 * component, 2 * component + 2)
    return (
        state_mean[:, component_slice],
        state_covariance[:, component_slice, component_slice],
    )


def _symmetrize(matrix):
    """Return the symmetric part of a matrix."""
    return 0.5 * (matrix + matrix.T)


def _positive_semidefinite(matrix):
    """Project a symmetric 2D covariance onto the PSD cone."""
    eigenvalues, eigenvectors = np.linalg.eigh(_symmetrize(matrix))
    eigenvalues = np.maximum(eigenvalues, 0.0)
    return (eigenvectors * eigenvalues) @ eigenvectors.T


def _validate_covariance(matrix, name):
    """Validate a finite, symmetric positive-semidefinite covariance."""
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must be finite.")
    scale = max(1.0, float(np.max(np.abs(matrix))))
    tolerance = 1e-10 * scale
    if not np.allclose(matrix, matrix.T, rtol=1e-10, atol=tolerance):
        raise ValueError(f"{name} must be symmetric.")
    if float(np.min(np.linalg.eigvalsh(_symmetrize(matrix)))) < -tolerance:
        raise ValueError(f"{name} must be positive semidefinite.")
