# Synthetic Phase with Nonlinear Measurements

This directory contains the synthetic-data example for Bayesian inference of
phase drift, phase diffusion, and additive noise from nonlinear
two-dimensional measurements. The notebooks generate components of the
main-text and supporting-information figures.

## Contents

- `SyntheticPhaseNonlinearMeasurement.ipynb` develops the model and
  demonstrates simulation, frequency and initial-phase estimation, extended
  Kalman filtering, RTS smoothing, MAP parameter estimation, uncertainty
  estimation, and EM convergence.
- `SPNM_Performance.ipynb` evaluates inference performance as the number of
  observations, sampling interval, measurement-noise variance, and phase
  diffusivity are varied.
- `SyntheticPhaseNonlinearMeasurement.py` provides the simulation and
  inference functions shared by both notebooks.
- `data/` contains saved synthetic data, and `plots/` contains exported figure
  panels.

## Running the notebooks

Run the notebooks from this directory using the `phase-bayes` Python
environment. The main dependencies are NumPy, SciPy, Matplotlib, pandas, and
numdifftools.

The notebooks use fixed random-number-generator seeds where reproducibility is
needed. Re-running plotting cells may overwrite files in `plots/`.
