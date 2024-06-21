# Bayesian calibration of stochastic agent based model via random forest

This repository contains the code used for the paper "Bayesian calibration of stochastic agent based model via random forest surrogate modeling".
It contains hospitalization and death data produced by the CityCOVID agent based model.
It also provides code to train a Random Forest surrogate model for CityCOVID hospitalizations and deaths, calculate a Bayesian estimate of parameters from CityCOVID using this surrogate, and then produce plots and data of this calibration.

## Installation
All dependencies can be installed by running `make install_deps`.
Specifics can be found below.

### Python
The code in this repository makes use of the following `python` packages:
- `matplotlib`
- `numpy`
- `pandas`
- `properscoring`
- `pymcmcstat` (from updated version [here](https://github.com/cnrrobertson/pymcmcstat))
- `seaborn`
- `scikit-learn`

These can easily be installed by running `make install_python` if you have `conda` or `pip` already on your system.
Alternatively, they can be installed with `pip` via `pip install -r requirements.txt` or with `conda/mamba` via `conda env create -f environment.yml`.

### R
The code in this repository makes use of the following `R` packages:
- `ggdist`
- `coda`
- `mcgibbsit`
- `scoringutils`

These can easily be installed by running `make install_r` as long as `R` is already installed on your system.

## Usage
Reproduction of the results and plots can most easily be done by running `make all_surrogate` followed by `make all_calibration` and `make compare`.

The above runs various scripts from `scripts/` which make use of some utility functions in `src/` and data from `data/` to output files into `results/` and plots into `plots/`.
See `make help` or `make` for more information on individual scripts.
