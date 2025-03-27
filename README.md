# CityCOVID Calibration

This repository contains the code used for the papers:

- [Bayesian calibration of stochastic agent based model via random forest](https://onlinelibrary.wiley.com/doi/abs/10.1002/sim.70029)
- [Advancing calibration for stochastic agent-based models in epidemiology with stein variational inference and gaussian process surrogates](https://arxiv.org/abs/2502.19550v1)

It also contains hospitalization and death data produced by the [CityCOVID](https://www.anl.gov/dis/citycovid) agent based model.
As part of these papers, it provides code to train a Random Forest or Gaussian Process surrogate model for CityCOVID hospitalizations and deaths, calculate a Bayesian or Stein variational estimate of parameters from CityCOVID using the surrogate, and then produce plots and data of these estimates.

## Installation
Package installation for this code is mostly easily accomplished using [pixi](https://pixi.sh/latest/).
All dependencies can be installed by running `pixi install` in the repository root.
However, specifics can be found below.

### Python
The code in this repository makes use of the following `python` packages:
- `matplotlib`
- `numpy`
- `pandas`
- `properscoring`
- `pymcmcstat` (from updated version [here](https://github.com/cnrrobertson/pymcmcstat))
- `seaborn`
- `scikit-learn`

### R
The code in this repository makes use of the following `R` packages:
- `ggdist`
- `coda`
- `mcgibbsit`
- `scoringutils`

## Usage
Instructions to reproduce each paper can be found in the `papers/**/README.md` files.
However, generally, the various scripts from `scripts/**` will be used to call utility functions in `src/` using data from `data/` and will output files into `results/**` and plots into `plots/**`.
