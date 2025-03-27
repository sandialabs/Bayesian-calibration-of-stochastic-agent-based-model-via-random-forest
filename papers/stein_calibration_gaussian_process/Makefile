help:
	@echo =====================================================================
	@echo This makefile assists in running scripts to\:
	@echo     - Tune, train, and save a gaussian process surrogate of CityCOVID
	@echo     - Run Stein variational inference and Bayesian DRAM calibration for CityCOVID using the surrogate
	@echo     - Output plots of the surrogate accuracy and calibration results
	@echo =====================================================================
	@echo
	@echo --------------------------- Surrogate -------------------------------
	@echo tune_surrogate    - find optimal parameters for gp surrogate
	@echo train_surrogate   - train and save model and feature importance
	@echo plot_surrogate    - make plots of model performance
	@echo all_surrogate     - tune, train, and plot
	@echo
	@echo --------------------------- Calibration -----------------------------
	@echo calibrate_mcmc    - calibrate CityCOVID with surrogate and DRAM
	@echo calibrate_stein   - calibrate CityCOVID with surrogate and SteinVI
	@echo mcmc_convergence  - check convergence of mcmc calibration
	@echo plot_calibration  - make plots of mcmc calibration results
	@echo all_calibration   - mcmc+stein calibration, check mcmc convergence, make plots
	@echo
	@echo --------------------------- Pushforward -----------------------------
	@echo compare           - Generate plots comparing MCMC and Stein outcomes
	@echo self_consistency  - create plots showing self consistency across numbers of particles in Stein calibrations
	@echo compare_seeds     - create plot comparing stein results for different seeds
	@echo
	@echo =====================================================================

tune_surrogate:
	python tune_surrogate.py
train_surrogate:
	python train_surrogate.py
plot_surrogate:
	python plot_surrogate.py
all_surrogate: tune_surrogate train_surrogate plot_surrogate

calibrate_mcmc:
	python calibrate_mcmc.py
calibrate_stein:
	python calibrate_stein.py
mcmc_convergence:
	Rscript test_convergence.R
plot_calibration:
	python plot_calibration.py
all_calibration: calibrate_mcmc calibrate_stein mcmc_convergence plot_calibration

compare:
	python compare_mcmc_stein.py
	python compare_posteriors.py
self_consistency:
	python self_consistency_plots.py
compare_seeds:
	python compare_seeds.py
