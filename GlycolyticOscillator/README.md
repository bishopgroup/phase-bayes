# Glycolytic Oscillator

This directory contains the noisy glycolytic-oscillator example for Bayesian
inference of latent phase dynamics and a periodic observation waveform. The
notebook generates components of the main-text and supporting-information
figures.

## Contents

- `GlycolyticOscillatorGPF.ipynb` demonstrates stochastic oscillator
  simulation, regularizing-noise addition, frequency and initial-phase
  estimation, global phase filtering and smoothing, posterior predictive
  checks, phase mean-squared displacement, MAP parameter estimation, and EM.
- `GlycolyticOscillator.py` provides the oscillator, Fourier-waveform, global
  phase inference, trajectory-sampling, and parameter-estimation functions
  used by the notebook.
- `data/` contains saved simulated data, and `plots/` contains exported figure
  panels.

## Running the notebook

Run the notebook from this directory using the `phase-bayes` Python
environment. The main dependencies are NumPy, SciPy, Matplotlib, pandas, and
numdifftools.

The notebook uses fixed random-number-generator seeds where reproducibility is
needed. Re-running simulation or plotting cells may overwrite files in
`data/` or `plots/`.
