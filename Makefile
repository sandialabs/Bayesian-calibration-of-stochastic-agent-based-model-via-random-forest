.PHONY: help

export PROJLOC=$(pwd)
RPKG1 := scoringutils_1.2.2.tar.gz
RPKG2 := mcgibbsit_1.2.2.tar.gz
RURL := https://cran.r-project.org/src/contrib/
RDOWN := lib/

help:
	@echo =====================================================================
	@echo This makefile assists in running scripts to\:
	@echo     - Tune, train, and save a random forest surrogate of CityCOVID
	@echo     - Run Bayesian DRAM calibration for CityCOVID using the surrogate
	@echo     - Output plots of the surrogate accuracy and calibration results
	@echo =====================================================================
	@echo
	@echo --------------------------- Install ---------------------------------
	@echo install_packages  - install dependencies
	@echo install_python    - install python dependencies
	@echo install_r         - install r dependencies
	@echo
	@echo --------------------------- Surrogate -------------------------------
	@echo tune_surrogate    - find optimal parameters for rf surrogate
	@echo train_surrogate   - train and save model and feature importance
	@echo plot_surrogate    - make plots of model performance
	@echo all_surrogate     - tune, train, and plot
	@echo
	@echo --------------------------- Calibration -----------------------------
	@echo calibrate         - calibrate CityCOVID with surrogate and DRAM
	@echo convergence       - check convergence of calibration
	@echo score             - use proper scoring rules to evaluate calibration
	@echo plot_calibration  - make plots of calibration results
	@echo all_calibration   - tune, train, and plot
	@echo
	@echo --------------------------- Pushforward -----------------------------
	@echo compare           - compare ABC and MCMC calibration pushforwards
	@echo
	@echo --------------------------- Documentation ---------------------------
	@echo docs              - compare ABC and MCMC calibration pushforwards
	@echo
	@echo =====================================================================

install_deps: install_python install_r
install_python:
	@which conda >/dev/null 2>&1;\
	if [ $$? -eq 0 ]; then\
		echo "conda is installed. Installing python dependencies with conda...";\
		conda env create -y --file environment.yml;\
	else\
		echo "conda is not installed. Falling back to pip...";\
		pip install requirements.txt;\
	fi
install_r:
	wget -P ${RDOWN} ${RURL}${RPKG1}
	wget -P ${RDOWN} ${RURL}${RPKG2}
	R -e "install.packages('ggdist', repo='https://cran.rstudio.com/')"
	R -e "install.packages('coda', repo='https://cran.rstudio.com/')"
	R CMD INSTALL ${RDOWN}${RPKG1}
	R CMD INSTALL ${RDOWN}${RPKG2}

tune_surrogate:
	python scripts/tune_surrogate.py
train_surrogate:
	python scripts/train_surrogate.py
plot_surrogate:
	python scripts/plot_surrogate.py
all_surrogate: tune_surrogate train_surrogate plot_surrogate

calibrate:
	python scripts/calibrate.py
convergence:
	Rscript scripts/test_convergence.R
score:
	Rscript scripts/score_calibration.R
plot_calibration:
	python scripts/plot_calibration.py
all_calibration: calibrate convergence score plot_calibration

compare:
	python scripts/compare_mcmc_abc.py

docs:

