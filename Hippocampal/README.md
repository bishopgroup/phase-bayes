# Hippocampal Oscillations

This directory contains the hippocampal EEG example for Bayesian inference of
latent phase dynamics and a periodic observation waveform. The notebooks
generate components of the main-text and supporting-information figures.

## Contents

- `Hippocampal.ipynb` demonstrates data loading and thinning, regularizing
  noise, frequency and initial-phase estimation, MAP inference, uncertainty
  quantification, global phase filtering and smoothing, and posterior
  predictive checks.
- `Hippocampal_Performance.ipynb` evaluates the effects of added regularizing
  noise and sampling interval on inferred frequency and phase diffusivity.
- `Hippocampal.py` provides the NumPy/SciPy implementation, including a
  JAX-free fitting fallback, phase inference, trajectory sampling, and
  posterior-predictive diagnostics.
- `HippocampalJAX.py` provides the gradient-based JAX fitting, Hessian, and
  added-noise sweep workflows.
- `data/` contains the source EEG data and saved analysis results, while
  `plots/` contains exported figure panels.

## Running the notebooks

Run the notebooks from this directory using the `phase-bayes` Python
environment. The main dependencies are NumPy, SciPy, JAX, Matplotlib, pandas,
and joblib.

The notebooks use fixed random-number-generator seeds where reproducibility is
needed. Re-running analysis or plotting cells may overwrite files in `data/`
or `plots/`. The JAX workflows require JAX with 64-bit calculations enabled;
the CPU fitting functions in `Hippocampal.py` provide a JAX-free alternative.
