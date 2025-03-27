help:
	@echo =====================================================================
	@echo This makefile assists in running scripts to\:
	@echo     - Tune, train, and save a random forest surrogate of CityCOVID
	@echo     - Run Bayesian DRAM calibration for CityCOVID using the surrogate
	@echo     - Output plots of the surrogate accuracy and calibration results
	@echo =====================================================================
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
	@echo =====================================================================

tune_surrogate:
	python tune_surrogate.py
train_surrogate:
	python train_surrogate.py
plot_surrogate:
	python plot_surrogate.py
all_surrogate: tune_surrogate train_surrogate plot_surrogate

calibrate:
	python calibrate.py
convergence:
	Rscript test_convergence.R
plot_calibration:
	python plot_calibration.py
all_calibration: calibrate convergence plot_calibration

compare:
	python compare_mcmc_abc.py
