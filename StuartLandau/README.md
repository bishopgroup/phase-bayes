# Stuart–Landau Oscillator

This directory contains the Stuart–Landau (SL) example for Bayesian inference
of phase drift, phase diffusion, and additive noise from nonlinear
two-dimensional measurements. The notebooks generate components of the
main-text and supporting-information figures, including the inferred
diffusivity versus added measurement noise (Figure 2).

The estimator is the one used in `../SyntheticPhaseNonlinearMeasurement/`.
Here it is applied to data from a stochastic SL oscillator with isotropic
Cartesian noise, which the estimator's random-walk phase model does not
describe exactly:

- the radius fluctuates;
- radial errors are correlated over $\tau_r = 1/(2a)$;
- the phase diffusivity crosses over from $D_x$ at short lags to
  $D_{\rm eff} = D_x(1 + b^2/a^2)$ at long lags.

The physical parameters are $a = b = 1$ and $D_x = 10^{-3}$.

## Contents

- `StuartLandauInference.ipynb` runs the end-to-end analysis at
  $D_x = 10^{-3}$, $a = b = 1$, and $N_{\rm INF} = 32000$.
  - **§0** validates the simulator against the analytical limits: $D_x$,
    $D_{\rm eff}$, the OU crossover, the radial variance, and the
    Euler–Maruyama frequency $\omega_{\rm sim}$.
  - **Three measurement regimes:**
    1. added noise, $\sigma_{\rm add} = 0.125$, $\Delta t = 0.02$
       (pseudo-true $D^\ast = 0.80\,D_{\rm eff}$);
    2. no added noise, $\Delta t = 2.16 = 4.3\tau_r$ (amplitude decorrelated
       between samples; $D^\ast = 0.80\,D_{\rm eff}$);
    3. no added noise, $\Delta t = 0.02$ ($D^\ast = 0.31\,D_{\rm eff}$, the
       Figure 2 point at $\sigma_{\rm add} = 0$);

    2b. the limiting regime of case 2: $\Delta t = 2.16$ with a stiff
        amplitude, $a = b = 100$, $D_x = 10^{-4}$
        ($\Delta t/\tau_r = 432$, $D^\ast = 0.998\,D_{\rm eff}$);

    3b. the limiting regime of case 3: $\Delta t = 0.02$ with
        $a = b = 5000$ ($\Delta t/\tau_r = 200$,
        $D^\ast = 0.995\,D_{\rm eff}$).
  - **Limiting law.** Without added noise
    $D^\ast = D_{\rm eff}(1 - \tau_r/\Delta t)$: the coarse-grained model is
    exact once the amplitude relaxes within one sample. At $a = b = 1$ Nyquist
    caps $\Delta t/\tau_r$ at $2\pi$, so cases 2b and 3b use a stiffer
    amplitude and put $D_{\rm eff}$ inside the 95 % interval
    (`plots/SL_LimitingRegime.pdf`).
  - **Each regime ends with a parameter-posterior figure**
    (`plots/SL_Parameters_Case*.pdf`) showing the marginal and joint Laplace
    posteriors of $\omega$, $D_\theta$, and $R$.
  - **Case 3 checks itself** against the cached sweep row. Every case draws
    the pseudo-true $D^\ast$ of `InferenceExplainer.ipynb` §5.
  - **Consistency with Example 1b** (`../SyntheticPhaseNonlinearMeasurement/`):
    the inference module is checked to be code-identical, the pipeline is
    compared step by step, and the three parameter conditions of
    `SPNM_Performance.ipynb` are evaluated for every case. Condition 3,
    $(\omega\Delta t)^2/\sqrt R \ll 1$, is violated by the coarse-sampling
    cases without harm; see `InferenceExplainer.ipynb` §3d for why it is
    superseded.
- `InferenceExplainer.ipynb` explains what the estimator measures and where it
  fails: the two measurement channels, the Whittle information, the KL
  projection $D^\ast(\sigma_{\rm add},\Delta t)$ that reproduces the sweep to
  1 %, the error-propagation diagnostics, and a regime map.
- `Figure2_DiffusivityVsAddedNoise.ipynb` draws Figure 2 (main panel) and
  Figure 2b (three data budgets) from `data/sweep_results.parquet`.
- `StuartLandauSimulation.py` provides the Euler–Maruyama integrator, the
  geometric and asymptotic phases, the phase MSD, the analytical curves, and
  the long-time diffusivity fit.
- `StuartLandauPhaseInference.py` provides the inference functions: EKF, RTS
  smoother, marginal likelihood, generalized EM, and frozen-linearization
  information matrices. Apart from comments, it is the same code as
  `../SyntheticPhaseNonlinearMeasurement/SyntheticPhaseNonlinearMeasurement.py`.
- `FastPhaseEM.py` provides an O(N) implementation of the EM and Laplace
  steps. The reference module builds a dense N×N covariance on every
  iteration; this module uses the RTS lag-one identity and a scalar EKF update
  instead. `python FastPhaseEM.py` checks it against the reference module to
  machine precision.
- `NoiseInjectionSweep.py` runs the noise-injection study. It simulates 8
  independent SL realizations, subsamples them at $\Delta t = 0.02$, adds
  isotropic Cartesian noise $\sigma_{\rm add}$, and fits every window.
  Results are written to `data/sweep_results.parquet`.
- `figstyle.py` holds the shared figure style: fixed physical axes size,
  fonts, and colours.
- `data/` contains the cached sweep (1872 fits) and its configuration
  fingerprint (`grid_signature.json`), while `plots/` contains exported figure
  panels.

## Running the notebooks

Run the notebooks from this directory using the `phase-bayes` Python
environment (Python ≥ 3.10). The main dependencies are NumPy, SciPy, pandas,
pyarrow (for parquet), Matplotlib, and Jupyter.

1. `StuartLandauInference.ipynb` takes about 35–40 min. Each $N = 32000$
   Laplace covariance takes 1–3 min; case 2 integrates 35 M Euler–Maruyama
   steps (about 5 min), case 2b 69 M (about 11 min), and case 3b 32 M (about
   5 min).
2. `InferenceExplainer.ipynb` takes about 3 min, mostly from the cached sweep.
3. `Figure2_DiffusivityVsAddedNoise.ipynb` takes a few seconds, because it
   reads the shipped `data/sweep_results.parquet`.

To recompute the sweep:

```bash
python NoiseInjectionSweep.py --check-sim                       # thinned integrator == reference (≈10 s)
python NoiseInjectionSweep.py --no-resume --outdir data_regen   # Figure 2 cells from scratch
python NoiseInjectionSweep.py --grid full                       # the full study grid
```

- The default `--grid figure` computes the cells the figures draw: `default`
  initialization, $N_{\rm INF} \in \{2000, 8000, 32000\}$, 13 noise levels,
  and 8 realizations, 312 fits in total. This costs about 4 CPU-hours, or
  about 40 min on 7 workers.
- `--grid full` adds $N_{\rm INF}$ up to 512000 and the `informed`
  initialization, 1872 fits in total, at about 99 CPU-hours.
- Fits already present in `--outdir` are reused unless `--no-resume` is given.
  Running the default command against the shipped `data/` therefore does
  nothing.
- To draw Figure 2 from a regenerated sweep, point `SWEEP` in the notebook's
  setup cell at `data_regen/sweep_results.parquet`.

The notebooks and the sweep use fixed random-number-generator seeds: the
trajectory seed is 42, and the injected-noise seeds come from
`NoiseInjectionSweep.noise_seed_for`. A regenerated sweep reproduces the
shipped one. For the 312 Figure 2 fits, every fitted quantity ($\omega$, $D$,
$R$, their Laplace standard deviations, and EM iteration counts) is
bit-identical, and the figures drawn from it are pixel-identical. Only the
phase-tracking diagnostics (`rmse_*`, `sigma_psi_*`, `omega_*_emp`) differ, by
at most $2\times10^{-13}$ rad; they come from summing the unwrapped true phase
in a different order and never enter the fit.

Re-running the notebooks may overwrite files in `plots/`.
`Figure2_DiffusivityVsAddedNoise.ipynb` also writes PNG and SVG copies of its
figures, which are not tracked in this repository.
