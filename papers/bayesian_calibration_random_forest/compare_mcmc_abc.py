import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import properscoring as scoring
import os
env = os.environ["PROJLOC"]
plt.style.use(env+"/src/utils/rf_mcmc.mplstyle")
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

from scipy.stats import gaussian_kde, entropy, wasserstein_distance, chisquare
import pickle

nice_param_map = {
    'infected.count': "Number of initial infected",
    'susceptible.to.exposed.probability': "Probability of infection from exposure",
    'seasonality.multiplier': "Seasonality multiplier",
    'shielding.scaling': "Shielding by susceptible",
    'isolate.infectivity.household': "Proportion of isolated in homes",
    'isolate.infectivity.nursinghome': "Proportion of isolated in nursing homes",
    'initial.exposure.tick': "Time of initial exposure",
    'stay.at.home.probability': "Probability of self-isolation",
    'stoe.behavioral.adjustment.probability': "Probability of protective behaviors",
}

# ------------------------------------------------------------------------------
# Load data
# ------------------------------------------------------------------------------
plotdir = env+"/plots/bayesian_calibration_random_forest/pushforward/"
resdir = env+"/results/bayesian_calibration_random_forest/"

# Load surrogate
with open(resdir+"surrogate/surrogate.pkl", "rb") as file:
    loaded_model = pickle.load(file)
    rf = loaded_model["rf"]
    transformer = loaded_model["transformer"]
    pca_components = loaded_model["pca_components"]

def PCA_RF_surrogate(ps):
    coefs = rf.predict(ps)
    raw_result = pca_components @ coefs.flatten()
    result = transformer.inverse_transform(raw_result.reshape((1,-1))).flatten()
    return result

data_real = pd.read_csv(env+"/data/observed_chicago.csv", index_col=0).dropna()

# Average posterior predictive
data_pred = pd.read_csv(env+"/data/bayesian_pushforward_results.csv", index_col=0, header=[0,1]).dropna()
full_time = pd.to_datetime(data_pred.columns.get_level_values(1))
data_time = pd.to_datetime(data_real.index)
time_mask = np.array([True if (ft in data_time) else False for ft in full_time])
data_pred = data_pred.loc[:,time_mask]
params_pred = pd.read_csv(env+"/data/bayesian_pushforward_parameters.csv", index_col=None)
params_pred = params_pred.rename(nice_param_map, axis=1)
key_vars = params_pred.columns[params_pred.var() > 1e-10]

hosp_pred = data_pred.loc[:,"hospitalizations"]
death_pred = data_pred.loc[:,"deaths"]

hosp_mean = hosp_pred.mean()
death_mean = death_pred.mean()
hosp_5 = hosp_pred.quantile(.05)
death_5 = death_pred.quantile(.05)
hosp_25 = hosp_pred.quantile(.25)
death_25 = death_pred.quantile(.25)
hosp_75 = hosp_pred.quantile(.75)
death_75 = death_pred.quantile(.75)
hosp_95 = hosp_pred.quantile(.95)
death_95 = death_pred.quantile(.95)

# Load ABC data
data_abc = pd.read_csv(env+"/data/abc_results.csv", index_col=0, header=[0,1]).dropna()
params_abc = pd.read_csv(env+"/data/abc_parameters.csv", index_col=0)
params_abc = params_abc.rename(nice_param_map, axis=1)
data_abc = data_abc.loc[params_abc.index, time_mask]

hosp_abc = data_abc.loc[:,"hospitalizations"]
death_abc = data_abc.loc[:,"deaths"]

habc_mean = hosp_abc.mean()
dabc_mean = death_abc.mean()
# habc_wmean = hosp_abc.apply(np.average, weights=params_abc.weight)
# dabc_wmean = death_abc.apply(np.average, weights=params_abc.weight)
habc_5 = hosp_abc.quantile(.05)
dabc_5 = death_abc.quantile(.05)
habc_25 = hosp_abc.quantile(.25)
dabc_25 = death_abc.quantile(.25)
habc_75 = hosp_abc.quantile(.75)
dabc_75 = death_abc.quantile(.75)
habc_95 = hosp_abc.quantile(.95)
dabc_95 = death_abc.quantile(.95)
habc_45 = hosp_abc.quantile(.45)
dabc_45 = death_abc.quantile(.45)
habc_55 = hosp_abc.quantile(.55)
dabc_55 = death_abc.quantile(.55)

# Surrogate predictions from posterior samples
hosp_surr = hosp_pred.copy()
death_surr = death_pred.copy()
for i in params_pred.index:
    ps = pd.DataFrame(params_pred.loc[i,key_vars]).T
    temp_data = PCA_RF_surrogate(ps)
    temp_data = temp_data[time_mask]
    hosp_surr.loc[i+1] = temp_data[:64]
    death_surr.loc[i+1] = temp_data[64:]

hsurr_mean = hosp_surr.mean()
dsurr_mean = death_surr.mean()
hsurr_5 = hosp_surr.quantile(.05)
dsurr_5 = death_surr.quantile(.05)
hsurr_25 = hosp_surr.quantile(.25)
dsurr_25 = death_surr.quantile(.25)
hsurr_75 = hosp_surr.quantile(.75)
dsurr_75 = death_surr.quantile(.75)
hsurr_95 = hosp_surr.quantile(.95)
dsurr_95 = death_surr.quantile(.95)

# %%
# ------------------------------------------------------------------------------
# Plot data
# ------------------------------------------------------------------------------

augmented_ticks = pd.to_datetime(data_real.index).strftime("%m/%d")
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

f = plt.figure(figsize=(8,6))
axes = f.subplots(2, 2, sharex=True, sharey='row')
axes[0,0].set_title("Surrogate Calibrated ABM Pushforward")
axes[0,0].set_ylabel("Hospitalizations")
axes[0,0].scatter(data_real.index, data_real.loc[:,"hospitalizations"], c="black", s=10, label="Data", zorder=15)
axes[0,0].plot(data_real.index, hosp_mean, lw=2, label="Mean", zorder=10)
axes[0,0].fill_between(data_real.index, hosp_5, hosp_95, label="5%-95%", alpha=.4, zorder=5)
axes[0,0].fill_between(data_real.index, hosp_25, hosp_75, label="25%-75%", alpha=.4, zorder=5)

axes[0,1].set_title("ABC Pushforward")
axes[0,1].scatter(data_real.index, data_real.loc[:,"hospitalizations"], c="black", s=10, label="Data", zorder=15)
axes[0,1].plot(data_real.index, habc_mean, lw=2, label="Mean", zorder=10)
# axes[0,1].plot(data_real.index, habc_wmean, lw=2, label="Weighted Mean", zorder=10)
axes[0,1].fill_between(data_real.index, habc_5, habc_95, label="5%-95%", alpha=.4, zorder=2)
axes[0,1].fill_between(data_real.index, habc_25, habc_75, label="25%-75%", alpha=.4, zorder=3)
axes[0,1].legend()

axes[1,0].set_ylabel("Deaths")
axes[1,0].scatter(data_real.index, data_real.loc[:,"deaths"], c="black", s=10, label="Data", zorder=15)
axes[1,0].plot(data_real.index, death_mean, lw=2, label="Mean", zorder=10)
axes[1,0].fill_between(data_real.index, death_5, death_95, label="5%-95%", alpha=.4, zorder=5)
axes[1,0].fill_between(data_real.index, death_25, death_75, label="25%-75%", alpha=.4, zorder=5)

axes[1,1].scatter(data_real.index, data_real.loc[:,"deaths"], c="black", s=10, label="Data", zorder=15)
axes[1,1].plot(data_real.index, dabc_mean, lw=2, label="Mean", zorder=10)
# axes[1,1].plot(data_real.index, dabc_wmean, lw=2, label="Weighted Mean", zorder=10)
axes[1,1].fill_between(data_real.index, dabc_5, dabc_95, label="5%-95%", alpha=.4, zorder=2)
axes[1,1].fill_between(data_real.index, dabc_25, dabc_75, label="25%-75%", alpha=.4, zorder=3)

axes[1,0].set_xticks(data_real.index[::15], augmented_ticks[::15])

plt.tight_layout()
plt.savefig(plotdir+"rf_posterior_runs.pdf")
plt.savefig(plotdir+"rf_posterior_runs.png")
plt.show()

# %%
# ------------------------------------------------------------------------------
# Plot predictions
# ------------------------------------------------------------------------------
hscores = scoring.crps_ensemble(data_real.hospitalizations.to_numpy(), hosp_pred.to_numpy().T)
dscores = scoring.crps_ensemble(data_real.deaths.to_numpy(), death_pred.to_numpy().T)

f = plt.figure(figsize=(8,5/2 + (.25/.75)*(5/2)))
thin = 0.05
thick = 2
axes = f.subplots(2, 2, sharex=True, height_ratios=[0.75,0.25])
hplot = data_real.hospitalizations.to_numpy()
dplot = data_real.deaths.to_numpy()
xplot = np.arange(len(hplot))
hosp_len = len(hplot)
for i,(ax,inter,tdata,name) in enumerate(zip(
    axes[0],
    [hosp_pred, death_pred],
    [hplot, dplot],
    ["Hospitalizations (MCMC)", "Deaths (MCMC)"],
)):
    sns.lineplot(x=xplot.flatten(), y=tdata, ax=ax, color="black", lw=thick, label="Observed")
    q5 = ax.fill_between(
        x=xplot.flatten(),
        y1=np.quantile(np.nan_to_num(inter), .05, axis=0),
        y2=np.quantile(np.nan_to_num(inter), .95, axis=0),
        color=colors[0],
        # label=r"5\% - 95\%"
    )
    q25 = ax.fill_between(
        x=xplot.flatten(),
        y1=np.quantile(np.nan_to_num(inter), .25, axis=0),
        y2=np.quantile(np.nan_to_num(inter), .75, axis=0),
        color=colors[1],
        # label=r"25\% - 75\%"
    )
    q45 = ax.fill_between(
        x=xplot.flatten(),
        y1=np.quantile(np.nan_to_num(inter), .45, axis=0),
        y2=np.quantile(np.nan_to_num(inter), .55, axis=0),
        color=colors[2],
        # label=r"45\% - 55\%"
    )
    ax.set_xticks(xplot[0:hosp_len:15].flatten(), pd.to_datetime(data_real.index).strftime("%m/%d").to_numpy()[::15])
    ax.set_title(name)

    if i == 0:
        first_legend = ax.legend(loc="lower left", framealpha=1)
    else:
        second_legend = ax.legend([q5, q25, q45], [r"5\% - 95\%", r"25\% - 75\%", r"45\% - 55\%"], title="Pushforward Intervals")
sns.lineplot(x=xplot, y=hscores, color=colors[3], ax=axes[1,0], label="CRPS")
sns.lineplot(x=xplot, y=dscores, color=colors[3], ax=axes[1,1])
axes[1,0].set_ylim(0, 150)
axes[1,1].set_ylim(0, 100)
f.tight_layout()
# plt.savefig(plotdir+filename+"_prediction.pdf")
plt.savefig(plotdir+"rf_posterior_runs.pdf")
plt.show()
# %%
# ------------------------------------------------------------------------------
# Plot predictions
# ------------------------------------------------------------------------------
hscores = scoring.crps_ensemble(data_real.hospitalizations.to_numpy(), hosp_abc.to_numpy().T)
dscores = scoring.crps_ensemble(data_real.deaths.to_numpy(), death_abc.to_numpy().T)

f = plt.figure(figsize=(8,5/2 + (.25/.75)*(5/2)))
thin = 0.05
thick = 2
axes = f.subplots(2, 2, sharex=True, height_ratios=[0.75,0.25])
hplot = data_real.hospitalizations.to_numpy()
dplot = data_real.deaths.to_numpy()
xplot = np.arange(len(hplot))
hosp_len = len(hplot)
for i,(ax,inter,tdata,name) in enumerate(zip(
    axes[0],
    [hosp_abc, death_abc],
    [hplot, dplot],
    ["Hospitalizations (ABC)", "Deaths (ABC)"],
)):
    sns.lineplot(x=xplot.flatten(), y=tdata, ax=ax, color="black", lw=thick, label="Observed")
    q5 = ax.fill_between(
        x=xplot.flatten(),
        y1=np.quantile(np.nan_to_num(inter), .05, axis=0),
        y2=np.quantile(np.nan_to_num(inter), .95, axis=0),
        color=colors[0],
        # label=r"5\% - 95\%"
    )
    q25 = ax.fill_between(
        x=xplot.flatten(),
        y1=np.quantile(np.nan_to_num(inter), .25, axis=0),
        y2=np.quantile(np.nan_to_num(inter), .75, axis=0),
        color=colors[1],
        # label=r"25\% - 75\%"
    )
    q45 = ax.fill_between(
        x=xplot.flatten(),
        y1=np.quantile(np.nan_to_num(inter), .45, axis=0),
        y2=np.quantile(np.nan_to_num(inter), .55, axis=0),
        color=colors[2],
        # label=r"45\% - 55\%"
    )
    ax.set_xticks(xplot[0:hosp_len:15].flatten(), pd.to_datetime(data_real.index).strftime("%m/%d").to_numpy()[::15])
    ax.set_title(name)
    if i == 0:
        first_legend = ax.legend(loc="lower left")
    else:
        second_legend = ax.legend([q5, q25, q45], [r"5\% - 95\%", r"25\% - 75\%", r"45\% - 55\%"], title="Pushforward Intervals")

sns.lineplot(x=xplot, y=hscores, color=colors[3], ax=axes[1,0], label="CRPS")
sns.lineplot(x=xplot, y=dscores, color=colors[3], ax=axes[1,1])
axes[1,0].set_ylim(0, 150)
axes[1,1].set_ylim(0, 100)

f.tight_layout()
# plt.savefig(plotdir+filename+"_prediction.pdf")
plt.savefig(plotdir+"rf_posterior_runs_abc.pdf")
plt.show()

# %%
# ------------------------------------------------------------------------------
# Plot posteriors
# ------------------------------------------------------------------------------

f = plt.figure(figsize=(7,6))
axes = f.subplots(2, 2).flatten()
for i,kv in enumerate(key_vars):
    pred_var = params_pred.loc[:,kv]
    abc_var = params_abc.loc[:,kv]
    pkernel = gaussian_kde(pred_var)
    akernel = gaussian_kde(abc_var)
    ubound = max(pred_var.max(), abc_var.max())
    lbound = min(pred_var.min(), abc_var.min())
    dom = np.linspace(lbound, ubound, 1000)
    prange = pkernel(dom)
    arange = akernel(dom)
    axes[i].plot(dom, prange, label="RF+PCA")
    axes[i].plot(dom, arange, label="ABC")
    axes[i].set_title(kv)
    if i == 0:
        axes[i].legend()
plt.tight_layout()
plt.savefig(plotdir+"rf_posterior_runs_pdf.pdf")
plt.show()
# %%

# ------------------------------------------------------------------------------
# Plot prediction rank histogram
# ------------------------------------------------------------------------------
predictions_h = data_pred["hospitalizations"].to_numpy()
predictions_d = data_pred["deaths"].to_numpy()
all_data_h = np.vstack((predictions_h, data_real["hospitalizations"].to_numpy().reshape((1,-1))))
all_data_d = np.vstack((predictions_d, data_real["deaths"].to_numpy().reshape((1,-1))))

obs_rank_h = np.zeros(predictions_h.shape[1])
obs_rank_d = np.zeros(predictions_d.shape[1])
for i in range(predictions_h.shape[1]):
    obs_rank_h[i] = np.where(np.argsort(all_data_h[:,i]) == predictions_h.shape[0])[0]
    obs_rank_d[i] = np.where(np.argsort(all_data_d[:,i]) == predictions_d.shape[0])[0]

obs_rank = pd.DataFrame(np.hstack([obs_rank_h, obs_rank_d]).T, columns=["Observations"])
obs_rank[""] = str(np.zeros(obs_rank.shape[0]))
obs_rank.loc[:predictions_h.shape[1], ""] = "Hospitalizations"
obs_rank.loc[predictions_h.shape[1]:, ""] = "Deaths"

fig = plt.figure(figsize=(4, 5/2))
ax = fig.add_subplot()
sh = sns.histplot(obs_rank, x="Observations", hue="", ax=ax, multiple="layer", bins=10)
sh.legend_.set_loc("lower right")
plt.xlabel("")
max_rank = max(max(obs_rank_d), max(obs_rank_h))
plt.xticks([0, max_rank/2, max_rank], ["Overpredicted", "Centered", "Underpredicted"])
ax.yaxis.label.set_visible(False)
plt.tight_layout()
plt.savefig(plotdir+"rf_posterior_runs_predictive_rank.pdf")
plt.show()

# %%
##############################################################################
# Put together scores
##############################################################################
crps_hmcmc = scoring.crps_ensemble(data_real.hospitalizations.to_numpy(), hosp_pred.to_numpy().T).mean()
crps_dmcmc = scoring.crps_ensemble(data_real.deaths.to_numpy(), death_pred.to_numpy().T).mean()

crps_habc = scoring.crps_ensemble(data_real.hospitalizations.to_numpy(), hosp_abc.to_numpy().T).mean()
crps_dabc = scoring.crps_ensemble(data_real.deaths.to_numpy(), death_abc.to_numpy().T).mean()

# RVH
nbins = 10
predictions_h = data_pred["hospitalizations"].to_numpy()
predictions_d = data_pred["deaths"].to_numpy()
all_data_h = np.vstack((predictions_h, data_real["hospitalizations"].to_numpy().reshape((1,-1))))
all_data_d = np.vstack((predictions_d, data_real["deaths"].to_numpy().reshape((1,-1))))
mcmc_rank_h = np.zeros(predictions_h.shape[1])
mcmc_rank_d = np.zeros(predictions_d.shape[1])
for i in range(predictions_h.shape[1]):
    mcmc_rank_h[i] = np.where(np.argsort(all_data_h[:,i]) == predictions_h.shape[0])[0]
    mcmc_rank_d[i] = np.where(np.argsort(all_data_d[:,i]) == predictions_d.shape[0])[0]
mcmc_h_hist_full = np.histogram(mcmc_rank_h, range=(0,predictions_h.shape[0]), bins=nbins)[0]
mcmc_d_hist_full = np.histogram(mcmc_rank_d, range=(0,predictions_d.shape[0]), bins=nbins)[0]
mcmc_h_hist = np.histogram(mcmc_rank_h, density=True, range=(0,predictions_h.shape[0]), bins=nbins)
mcmc_d_hist = np.histogram(mcmc_rank_d, density=True, range=(0,predictions_d.shape[0]), bins=nbins)
mcmc_h_hist = mcmc_h_hist[0] * (mcmc_h_hist[1][1] - mcmc_h_hist[1][0])
mcmc_d_hist = mcmc_d_hist[0] * (mcmc_d_hist[1][1] - mcmc_d_hist[1][0])

abc_h = data_abc["hospitalizations"].to_numpy()
abc_d = data_abc["deaths"].to_numpy()
all_abc_h = np.vstack((abc_h, data_real["hospitalizations"].to_numpy().reshape((1,-1))))
all_abc_d = np.vstack((abc_d, data_real["deaths"].to_numpy().reshape((1,-1))))
abc_rank_h = np.zeros(abc_h.shape[1])
abc_rank_d = np.zeros(abc_d.shape[1])
for i in range(predictions_h.shape[1]):
    abc_rank_h[i] = np.where(np.argsort(all_abc_h[:,i]) == abc_h.shape[0])[0]
    abc_rank_d[i] = np.where(np.argsort(all_abc_d[:,i]) == abc_d.shape[0])[0]
abc_h_hist_full = np.histogram(abc_rank_h, range=(0,abc_h.shape[0]), bins=nbins)[0]
abc_d_hist_full = np.histogram(abc_rank_d, range=(0,abc_d.shape[0]), bins=nbins)[0]
abc_h_hist = np.histogram(abc_rank_h, density=True, range=(0,abc_h.shape[0]), bins=nbins)
abc_d_hist = np.histogram(abc_rank_d, density=True, range=(0,abc_d.shape[0]), bins=nbins)
abc_h_hist = abc_h_hist[0] * (abc_h_hist[1][1] - abc_h_hist[1][0])
abc_d_hist = abc_d_hist[0] * (abc_d_hist[1][1] - abc_d_hist[1][0])

uniform_hist_full = np.ones(len(abc_h_hist)) * (mcmc_h_hist_full.sum() / len(abc_h_hist))
uniform_hist = np.ones(len(abc_h_hist)) / len(abc_h_hist)

# KL Divergence of RVH and Uniform
kl_hmcmc = entropy(mcmc_h_hist, uniform_hist)
kl_dmcmc = entropy(mcmc_d_hist, uniform_hist)

kl_habc = entropy(abc_h_hist, uniform_hist)
kl_dabc = entropy(abc_d_hist, uniform_hist)

# Earth movers distance
wass_hmcmc = wasserstein_distance(mcmc_h_hist, uniform_hist)
wass_dmcmc = wasserstein_distance(mcmc_d_hist, uniform_hist)

wass_habc = wasserstein_distance(abc_h_hist, uniform_hist)
wass_dabc = wasserstein_distance(abc_d_hist, uniform_hist)

# Chi-square distance (frequency comparison)
chi_hmcmc = chisquare(mcmc_h_hist_full, uniform_hist_full)[0]
chi_dmcmc = chisquare(mcmc_d_hist_full, uniform_hist_full)[0]

chi_habc = chisquare(abc_h_hist_full, uniform_hist_full)[0]
chi_dabc = chisquare(abc_d_hist_full, uniform_hist_full)[0]

# DIC
smoothing = 7
scut = smoothing // 2
real_daily_h = np.convolve(data_real.hospitalizations.to_numpy(), np.ones(smoothing)/smoothing, 'same')
real_daily_d = np.convolve(data_real.deaths.to_numpy(), np.ones(smoothing)/smoothing, 'same')
real_daily_h = np.diff(real_daily_h)[scut:-scut]
real_daily_d = np.diff(real_daily_d)[scut:-scut]

mcmc_daily_h = np.diff(predictions_h, axis=1)[:,scut:-scut]
mcmc_daily_d = np.diff(predictions_d, axis=1)[:,scut:-scut]
abc_daily_h = np.diff(abc_h, axis=1)[:,scut:-scut]
abc_daily_d = np.diff(abc_d, axis=1)[:,scut:-scut]

## Deviance
loglikelihood_h = lambda y, sig: np.sum((-0.5*((y - real_daily_h) / sig)**2) + np.log(1 / (sig * np.sqrt(2*np.pi))), axis=1)
loglikelihood_d = lambda y, sig: np.sum((-0.5*((y - real_daily_d) / sig)**2) + np.log(1 / (sig * np.sqrt(2*np.pi))), axis=1)

s_h_mcmc = np.std(mcmc_daily_h)
s_d_mcmc = np.std(mcmc_daily_d)
s_h_abc = np.std(abc_daily_h)
s_d_abc = np.std(abc_daily_d)

ll_mean_hmcmc = np.mean(loglikelihood_h(mcmc_daily_h, s_h_mcmc))
ll_mean_dmcmc = np.mean(loglikelihood_d(mcmc_daily_d, s_d_mcmc))
ll_mean_habc = np.mean(loglikelihood_h(abc_daily_h, s_h_abc))
ll_mean_dabc = np.mean(loglikelihood_d(abc_daily_d, s_d_abc))

## theta Bayes
mcmc_theta_bayes = params_pred.mean()
rel_mcmc_diff = (params_pred - mcmc_theta_bayes) / mcmc_theta_bayes
mcmc_bayes_h = mcmc_daily_h[rel_mcmc_diff.apply(np.linalg.norm, axis=1).argmin()]
mcmc_bayes_d = mcmc_daily_d[rel_mcmc_diff.apply(np.linalg.norm, axis=1).argmin()]
abc_theta_bayes = params_abc.mean()
rel_abc_diff = (params_abc - abc_theta_bayes) / abc_theta_bayes
abc_bayes_h = abc_daily_h[rel_abc_diff.apply(np.linalg.norm, axis=1).argmin()]
abc_bayes_d = abc_daily_d[rel_abc_diff.apply(np.linalg.norm, axis=1).argmin()]
ll_at_mean_hmcmc = np.mean(loglikelihood_h(mcmc_bayes_h.reshape((1,-1)), s_h_mcmc))
ll_at_mean_dmcmc = np.mean(loglikelihood_d(mcmc_bayes_d.reshape((1,-1)), s_d_mcmc))
ll_at_mean_habc = np.mean(loglikelihood_h(abc_bayes_h.reshape((1,-1)), s_h_abc))
ll_at_mean_dabc = np.mean(loglikelihood_d(abc_bayes_d.reshape((1,-1)), s_d_abc))

## Compute DIC
dic_hmcmc = -2*ll_at_mean_hmcmc + 4*(ll_at_mean_hmcmc - ll_mean_hmcmc)
dic_dmcmc = -2*ll_at_mean_dmcmc + 4*(ll_at_mean_dmcmc - ll_mean_dmcmc)
dic_habc = -2*ll_at_mean_habc + 4*(ll_at_mean_habc - ll_mean_habc)
dic_dabc = -2*ll_at_mean_dabc + 4*(ll_at_mean_dabc - ll_mean_dabc)

# Collect results
all_scores = pd.DataFrame(
    np.array([
        [crps_habc, crps_hmcmc, crps_dabc, crps_dmcmc],
        [kl_habc, kl_hmcmc, kl_dabc, kl_dmcmc],
        [wass_habc, wass_hmcmc, wass_dabc, wass_dmcmc],
        [chi_habc, chi_hmcmc, chi_dabc, chi_dmcmc],
        [dic_habc, dic_hmcmc, dic_dabc, dic_dmcmc],
    ]),
    columns = pd.MultiIndex.from_tuples([
        ("Hospitalizations", "ABC"),
        ("Hospitalizations", "MCMC"),
        ("Deaths", "ABC"),
        ("Deaths", "MCMC"),
    ]),
    index = [
        "CRPS",
        "KL Divergence (VRH)",
        "Wasserstein Distance (VRH)",
        "Chi-Squared Distance (VRH)",
        "DIC"
    ]
)

all_scores.to_csv(resdir+"pushforward/scores.csv")

# %%
plt.figure(figsize=(10,6))
plt.subplot(221)
plt.plot(mcmc_daily_h.T, alpha=0.5, lw=0.5)
plt.plot(real_daily_h, color="black", lw=2)
plt.title("MCMC (h)")
plt.subplot(222)
plt.plot(mcmc_daily_d.T, alpha=0.5, lw=0.5)
plt.plot(real_daily_d, color="black", lw=2)
plt.title("MCMC (d)")
plt.subplot(223)
plt.plot(abc_daily_h.T, alpha=0.5, lw=0.5)
plt.plot(real_daily_h, color="black", lw=2)
plt.title("ABC (h)")
plt.subplot(224)
plt.plot(abc_daily_d.T, alpha=0.5, lw=0.5)
plt.plot(real_daily_d, color="black", lw=2)
plt.title("ABC (d)")
plt.tight_layout()
plt.savefig(plotdir+"rf_posterior_daily_comparison.pdf")
plt.show()
