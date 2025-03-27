import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import properscoring as scoring
import seaborn as sns
import pickle
import os
env = os.environ["PROJLOC"]
sys.path.append(env+"/src/utils")
from load_data import *
plt.style.use(env+"/src/utils/rf_mcmc.mplstyle")
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

plotdir = env+"/plots/stein_calibration_gaussian_process/pushforward/"
resdir = env+"/results/stein_calibration_gaussian_process/"

# ------------------------------------------------------------------------------
# Load data
# ------------------------------------------------------------------------------
# Load surrogate
surr_loc = resdir + "surrogate/surrogate.pkl"
with open(surr_loc, "rb") as file:
    loaded_model = pickle.load(file)
gp = loaded_model["gp"]
transformer = loaded_model["transformer"]

def gp_surrogate(ps):
    transformed_ps = pd.DataFrame(
        transformer.transform(ps), columns=ps.columns)
    return gp.predict(transformed_ps).flatten()

data_real = get_real_data()
hosp_data = data_real.reindex(columns=["hospitalizations","deaths"]).T.stack().hospitalizations

# Average posterior predictive
stein_data_pred = pd.read_csv(env+"/data/stein_pushforward_results.csv", index_col=0, header=[0,1]).dropna()
full_time = pd.to_datetime(stein_data_pred.columns.get_level_values(1))
data_time = pd.to_datetime(data_real.index)
time_mask = np.array(
    [True if (ft in data_time) else False for ft in full_time])
stein_data_pred = stein_data_pred.loc[:, time_mask]
stein_params_pred = pd.read_csv(env+"/data/stein_pushforward_parameters.csv", index_col=0)
key_vars = stein_params_pred.columns[stein_params_pred.var() > 1e-10]

# Comparing the last peak in the marinal posterior
stein_sub_data = stein_data_pred.iloc[(
    stein_params_pred["Rate of exposure to infected"] > 0.063).to_numpy(), :]
stein_sub_hosp = stein_sub_data.loc[:, "hospitalizations"].diff(
    axis=1).dropna(axis=1)
stein_sub_death = stein_sub_data.loc[:, "deaths"].diff(axis=1).dropna(axis=1)

stein_hosp_pred = stein_data_pred.loc[:, "hospitalizations"].diff(
    axis=1).dropna(axis=1)
stein_death_pred = stein_data_pred.loc[:, "deaths"].diff(axis=1).dropna(axis=1)
data_real = data_real.diff().dropna()

stein_hosp_mean = stein_hosp_pred.mean()
stein_death_mean = stein_death_pred.mean()
stein_hosp_5 = stein_hosp_pred.quantile(.05)
stein_death_5 = stein_death_pred.quantile(.05)
stein_hosp_25 = stein_hosp_pred.quantile(.25)
stein_death_25 = stein_death_pred.quantile(.25)
stein_hosp_75 = stein_hosp_pred.quantile(.75)
stein_death_75 = stein_death_pred.quantile(.75)
stein_hosp_95 = stein_hosp_pred.quantile(.95)
stein_death_95 = stein_death_pred.quantile(.95)

mcmc_data_pred = pd.read_csv(env+"/data/bayesian_gp_pushforward_results.csv", index_col=0, header=[0,1]).dropna()
mcmc_data_pred = mcmc_data_pred.loc[:, time_mask]
mcmc_params_pred = pd.read_csv(env+"/data/bayesian_gp_pushforward_parameters.csv", index_col=0)

mcmc_hosp_pred = mcmc_data_pred.loc[:, "hospitalizations"].diff(
    axis=1).dropna(axis=1)
mcmc_death_pred = mcmc_data_pred.loc[:, "deaths"].diff(axis=1).dropna(axis=1)

mcmc_hosp_mean = mcmc_hosp_pred.mean()
mcmc_death_mean = mcmc_death_pred.mean()
mcmc_hosp_5 = mcmc_hosp_pred.quantile(.05)
mcmc_death_5 = mcmc_death_pred.quantile(.05)
mcmc_hosp_25 = mcmc_hosp_pred.quantile(.25)
mcmc_death_25 = mcmc_death_pred.quantile(.25)
mcmc_hosp_75 = mcmc_hosp_pred.quantile(.75)
mcmc_death_75 = mcmc_death_pred.quantile(.75)
mcmc_hosp_95 = mcmc_hosp_pred.quantile(.95)
mcmc_death_95 = mcmc_death_pred.quantile(.95)

# Load ABC data
data_abc = get_data(abc=True)
params_abc = get_params(abc=True, drop_vars=True)
data_abc = data_abc.loc[params_abc.index, time_mask]

hosp_abc = data_abc.loc[:, "hospitalizations"].diff(axis=1).dropna(axis=1)
death_abc = data_abc.loc[:, "deaths"].diff(axis=1).dropna(axis=1)

habc_mean = hosp_abc.mean()
dabc_mean = death_abc.mean()
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
hosp_surr_stein = stein_hosp_pred.copy()
death_surr_stein = stein_death_pred.copy()
for i in stein_params_pred.index:
    ps = pd.DataFrame(stein_params_pred.loc[i, key_vars]).T
    temp_data = gp_surrogate(ps)
    temp_data = temp_data[time_mask]
    hosp_surr_stein.loc[i+1] = np.diff(temp_data[:64])
    death_surr_stein.loc[i+1] = np.diff(temp_data[64:])

hsurr_stein_mean = hosp_surr_stein.mean()
dsurr_stein_mean = death_surr_stein.mean()
hsurr_stein_5 = hosp_surr_stein.quantile(.05)
dsurr_stein_5 = death_surr_stein.quantile(.05)
hsurr_stein_25 = hosp_surr_stein.quantile(.25)
dsurr_stein_25 = death_surr_stein.quantile(.25)
hsurr_stein_75 = hosp_surr_stein.quantile(.75)
dsurr_stein_75 = death_surr_stein.quantile(.75)
hsurr_stein_95 = hosp_surr_stein.quantile(.95)
dsurr_stein_95 = death_surr_stein.quantile(.95)

hosp_surr_mcmc = mcmc_hosp_pred.copy()
death_surr_mcmc = mcmc_death_pred.copy()
for i in mcmc_params_pred.index:
    ps = pd.DataFrame(mcmc_params_pred.loc[i, key_vars]).T
    temp_data = gp_surrogate(ps)
    temp_data = temp_data[time_mask]
    hosp_surr_mcmc.loc[i+1] = np.diff(temp_data[:64])
    death_surr_mcmc.loc[i+1] = np.diff(temp_data[64:])

hsurr_mcmc_mean = hosp_surr_mcmc.mean()
dsurr_mcmc_mean = death_surr_mcmc.mean()
hsurr_mcmc_5 = hosp_surr_mcmc.quantile(.05)
dsurr_mcmc_5 = death_surr_mcmc.quantile(.05)
hsurr_mcmc_25 = hosp_surr_mcmc.quantile(.25)
dsurr_mcmc_25 = death_surr_mcmc.quantile(.25)
hsurr_mcmc_75 = hosp_surr_mcmc.quantile(.75)
dsurr_mcmc_75 = death_surr_mcmc.quantile(.75)
hsurr_mcmc_95 = hosp_surr_mcmc.quantile(.95)
dsurr_mcmc_95 = death_surr_mcmc.quantile(.95)

# %%
# ------------------------------------------------------------------------------
# Plot data
# ------------------------------------------------------------------------------

augmented_ticks = pd.to_datetime(data_real.index).strftime("%m/%d")
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

f = plt.figure(figsize=(16, 6))
axes = f.subplots(2, 5, sharex=True, sharey='row')
axes[0, 0].set_title("CityCOVID+Stein Pushforward")
axes[0, 0].set_ylabel("Hospitalizations")
axes[0, 0].scatter(data_real.index, data_real.loc[:,
                   "hospitalizations"], c="black", s=10, label="Data", zorder=15)
axes[0, 0].plot(data_real.index, stein_hosp_mean,
                lw=2, label="Mean", zorder=10)
axes[0, 0].fill_between(data_real.index, stein_hosp_5,
                        stein_hosp_95, label="5\\%-95\\%", alpha=.4, zorder=5)
axes[0, 0].fill_between(data_real.index, stein_hosp_25,
                        stein_hosp_75, label="25\\%-75\\%", alpha=.4, zorder=5)
axes[0, 0].legend()

axes[0, 1].set_title("CityCOVID+MCMC Pushforward")
axes[0, 1].set_ylabel("Hospitalizations")
axes[0, 1].scatter(data_real.index, data_real.loc[:,
                   "hospitalizations"], c="black", s=10, label="Data", zorder=15)
axes[0, 1].plot(data_real.index, mcmc_hosp_mean, lw=2, label="Mean", zorder=10)
axes[0, 1].fill_between(data_real.index, mcmc_hosp_5,
                        mcmc_hosp_95, label="5\\%-95\\%", alpha=.4, zorder=5)
axes[0, 1].fill_between(data_real.index, mcmc_hosp_25,
                        mcmc_hosp_75, label="25\\%-75\\%", alpha=.4, zorder=5)

axes[0, 2].set_title("IMABC Pushforward")
axes[0, 2].scatter(data_real.index, data_real.loc[:,
                   "hospitalizations"], c="black", s=10, label="Data", zorder=15)
axes[0, 2].plot(data_real.index, habc_mean, lw=2, label="Mean", zorder=10)
axes[0, 2].fill_between(data_real.index, habc_5, habc_95,
                        label="5\\%-95\\%", alpha=.4, zorder=2)
axes[0, 2].fill_between(data_real.index, habc_25, habc_75,
                        label="25\\%-75\\%", alpha=.4, zorder=3)

axes[0, 3].set_title("GP+Stein Pushforward")
axes[0, 3].scatter(data_real.index, data_real.loc[:,
                   "hospitalizations"], c="black", s=10, label="Data", zorder=15)
axes[0, 3].plot(data_real.index, hsurr_stein_mean,
                lw=2, label="Mean", zorder=10)
axes[0, 3].fill_between(data_real.index, hsurr_stein_5,
                        hsurr_stein_95, label="5\\%-95\\%", alpha=.4, zorder=5)
axes[0, 3].fill_between(data_real.index, hsurr_stein_25,
                        hsurr_stein_75, label="25\\%-75\\%", alpha=.4, zorder=5)

axes[0, 4].set_title("GP+MCMC Pushforward")
axes[0, 4].scatter(data_real.index, data_real.loc[:,
                   "hospitalizations"], c="black", s=10, label="Data", zorder=15)
axes[0, 4].plot(data_real.index, hsurr_mcmc_mean,
                lw=2, label="Mean", zorder=10)
axes[0, 4].fill_between(data_real.index, hsurr_mcmc_5,
                        hsurr_mcmc_95, label="5\\%-95\\%", alpha=.4, zorder=5)
axes[0, 4].fill_between(data_real.index, hsurr_mcmc_25,
                        hsurr_mcmc_75, label="25\\%-75\\%", alpha=.4, zorder=5)

axes[1, 0].set_ylabel("Deaths")
axes[1, 0].scatter(data_real.index, data_real.loc[:, "deaths"],
                   c="black", s=10, label="Data", zorder=15)
axes[1, 0].plot(data_real.index, stein_death_mean,
                lw=2, label="Mean", zorder=10)
axes[1, 0].fill_between(data_real.index, stein_death_5,
                        stein_death_95, label="5\\%-95\\%", alpha=.4, zorder=5)
axes[1, 0].fill_between(data_real.index, stein_death_25,
                        stein_death_75, label="25\\%-75\\%", alpha=.4, zorder=5)

axes[1, 1].scatter(data_real.index, data_real.loc[:, "deaths"],
                   c="black", s=10, label="Data", zorder=15)
axes[1, 1].plot(data_real.index, mcmc_death_mean,
                lw=2, label="Mean", zorder=10)
axes[1, 1].fill_between(data_real.index, mcmc_death_5,
                        mcmc_death_95, label="5\\%-95\\%", alpha=.4, zorder=5)
axes[1, 1].fill_between(data_real.index, mcmc_death_25,
                        mcmc_death_75, label="25\\%-75\\%", alpha=.4, zorder=5)

axes[1, 2].scatter(data_real.index, data_real.loc[:, "deaths"],
                   c="black", s=10, label="Data", zorder=15)
axes[1, 2].plot(data_real.index, dabc_mean, lw=2, label="Mean", zorder=10)
axes[1, 2].fill_between(data_real.index, dabc_5, dabc_95,
                        label="5\\%-95\\%", alpha=.4, zorder=2)
axes[1, 2].fill_between(data_real.index, dabc_25, dabc_75,
                        label="25\\%-75\\%", alpha=.4, zorder=3)

axes[1, 3].scatter(data_real.index, data_real.loc[:, "deaths"],
                   c="black", s=10, label="Data", zorder=15)
axes[1, 3].plot(data_real.index, dsurr_stein_mean,
                lw=2, label="Mean", zorder=10)
axes[1, 3].fill_between(data_real.index, dsurr_stein_5,
                        dsurr_stein_95, label="5\\%-95\\%", alpha=.4, zorder=5)
axes[1, 3].fill_between(data_real.index, dsurr_stein_25,
                        dsurr_stein_75, label="25\\%-75\\%", alpha=.4, zorder=5)

axes[1, 4].scatter(data_real.index, data_real.loc[:, "deaths"],
                   c="black", s=10, label="Data", zorder=15)
axes[1, 4].plot(data_real.index, dsurr_mcmc_mean,
                lw=2, label="Mean", zorder=10)
axes[1, 4].fill_between(data_real.index, dsurr_mcmc_5,
                        dsurr_mcmc_95, label="5\\%-95\\%", alpha=.4, zorder=5)
axes[1, 4].fill_between(data_real.index, dsurr_mcmc_25,
                        dsurr_mcmc_75, label="25\\%-75\\%", alpha=.4, zorder=5)

axes[1, 0].set_xticks(data_real.index[::15], augmented_ticks[::15])

plt.tight_layout()
plt.savefig(plotdir+"citycovid_stein_mcmc_pushforward_comparison.pdf")
plt.show()

# %%
# ------------------------------------------------------------------------------
# Plot posteriors
# ------------------------------------------------------------------------------

f = plt.figure(figsize=(7, 6))
axes = f.subplots(2, 2).flatten()
for i, kv in enumerate(key_vars):
    stein_var = stein_params_pred.loc[:, kv]
    mcmc_var = mcmc_params_pred.loc[:, kv]
    abc_var = params_abc.loc[:, kv]
    skernel = gaussian_kde(stein_var)
    mkernel = gaussian_kde(mcmc_var)
    akernel = gaussian_kde(abc_var)
    ubound = max(stein_var.max(), abc_var.max(), mcmc_var.max())
    lbound = min(stein_var.min(), abc_var.min(), mcmc_var.min())
    dom = np.linspace(lbound, ubound, 1000)
    srange = skernel(dom)
    mrange = mkernel(dom)
    arange = akernel(dom)

    axes[i].plot(dom, srange, label="GP+Stein")
    axes[i].plot(dom, mrange, label="GP+MCMC")
    axes[i].plot(dom, arange, label="IMABC")

    if i == 0:
        # Mark final mode
        sdom = np.linspace(0.063, ubound, 1000)
    axes[i].set_title(kv)
    if i == 0:
        axes[i].legend()
plt.tight_layout()
plt.savefig(plotdir+"citycovid_pushforward_pdf.pdf")
plt.show()

# %%
# Compare citycovid pushforwards
hosp_len = data_real.shape[0]
smoothing = 7
comparison_data = data_real.reindex(columns=["hospitalizations","deaths"]).T.to_numpy()
comparison_data[0] = np.convolve(comparison_data[0], np.ones(smoothing)/smoothing, 'same')
comparison_data[1] = np.convolve(comparison_data[1], np.ones(smoothing)/smoothing, 'same')
y = comparison_data
x = np.arange(comparison_data.shape[1]).flatten()
xplot = x

fig = plt.figure(figsize=(12,(2+1)*(5/2)))
axes = fig.subplots(2+1, 2, sharex=True, sharey=False)
for i in range(2):
    if i == 0:
        p_hcurves = hosp_surr_mcmc
        p_dcurves = death_surr_mcmc
    else:
        p_hcurves = hosp_surr_stein
        p_dcurves = death_surr_stein
    hscores = scoring.crps_ensemble(data_real.hospitalizations.dropna().to_numpy(), p_hcurves.T)
    dscores = scoring.crps_ensemble(data_real.deaths.dropna().to_numpy(), p_dcurves.T)

    thin = 0.05
    thick = 2
    if i == 0:
        name = "DRAM"
    else:
        name = "SVI"
    for j,(ax,inter,tdata,pname) in enumerate(zip(
        axes[i],
        [p_hcurves, p_dcurves],
        [y[0], y[1]],
        [f"Hospitalizations ({name})", f"Deaths ({name})"],
    )):
        sns.lineplot(x=xplot.flatten(), y=tdata, ax=ax, color="black", lw=thick, label="Observed")
        if "Hosp" in pname:
            ax.set_ylim(-50, 80)
        else:
            ax.set_ylim(5, 55)
        q5 = ax.fill_between(
            x=xplot.flatten(),
            y1=np.quantile(np.nan_to_num(inter), .05, axis=0),
            y2=np.quantile(np.nan_to_num(inter), .95, axis=0),
            color=colors[4],
        )
        q25 = ax.fill_between(
            x=xplot.flatten(),
            y1=np.quantile(np.nan_to_num(inter), .25, axis=0),
            y2=np.quantile(np.nan_to_num(inter), .75, axis=0),
            color=colors[5],
        )
        q45 = ax.fill_between(
            x=xplot.flatten(),
            y1=np.quantile(np.nan_to_num(inter), .45, axis=0),
            y2=np.quantile(np.nan_to_num(inter), .55, axis=0),
            color=colors[6],
        )
        ax.set_xticks(xplot.flatten()[0:hosp_len:15], pd.to_datetime(hosp_data.index).strftime("%m/%d").to_numpy()[::15])
        ax.set_title(pname)
        if j == 0:
            first_legend = ax.legend(loc="lower left")
            second_legend = ax.legend([q5, q25, q45], ["5\\% - 95\\%", "25\\% - 75\\%", "45\\% - 55\\%"], title="Pushforward Intervals")
            ax.add_artist(first_legend)
        else:
            ax.legend_.set_visible(False)

    # Add CRPS
    xplot = np.arange(len(hscores))
    hscore_mean = np.mean(hscores)
    dscore_mean = np.mean(dscores)
    sns.lineplot(x=xplot, y=hscores, ax=axes[-1,0], lw=thick, label=name+f" (Mean: {hscore_mean:.2f})")
    sns.lineplot(x=xplot, y=dscores, ax=axes[-1,1], lw=thick, label=name+f" (Mean: {dscore_mean:.2f})")

    axes[-1,0].yaxis.label.set_visible(False)
    axes[-1,0].set_ylim(0, 200)
    axes[-1,0].set_title("CRPS (Hospitalizations)")
    axes[-1,1].yaxis.label.set_visible(False)
    axes[-1,1].set_ylim(0, 20)
    axes[-1,1].set_title("CRPS (Deaths)")

plt.tight_layout()
plt.savefig(plotdir+"compare_citycovid_pushforward.pdf")
plt.show()
# %%
