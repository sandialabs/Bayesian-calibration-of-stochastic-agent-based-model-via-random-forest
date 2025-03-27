import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import properscoring as scoring
from pymcmcstat.MCMC import MCMC
from pymcmcstat.propagation import calculate_intervals

from scipy.stats.qmc import Halton
from pymcmcstat.chain.ChainProcessing import load_json_object

import warnings
warnings.filterwarnings(action="ignore",category=UserWarning)
warnings.filterwarnings(action="ignore",category=RuntimeWarning)

import os
env = os.environ["PROJLOC"]
plt.style.use(env+"/src/utils/rf_mcmc.mplstyle")
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

import sys
sys.path.append(env + "/src/utils")
from load_data import get_data, get_params, get_real_data

sys.path.append(env+"/src/calibration")
from mcgibbsit import mcgibbsit

smoothing = 7
seeds = 6
nparticles = 200

nsimu = 10000
verb = 100
lr = 1e-3
nr = 1

res_loc = env+"/results/stein_calibration_gaussian_process/"
loaddir = res_loc+"calibration/"
plotdir = env+"/plots/stein_calibration_gaussian_process/"

params_stein = []
for seed in range(seeds):
    filename = f"stein_calibration_seed{seed}"
    params_stein.append(pd.read_csv(loaddir+filename+".csv"))
params_stein = pd.concat(params_stein)
params_stein["Calibration"] = "Stein Posterior"

results = load_json_object(loaddir+"calibrator.json")
for k in results.keys():
    r = results[k]
    if isinstance(r, list):
        results[k] = np.array(r)

names = results['names']
params_mcmc = np.array(results['chain'])
mc_res = mcgibbsit(params_mcmc[:,None,:])
mc_start = mc_res["M"].max().astype(int)
mc_step = mc_res["I"].max().astype(int)
params_mcmc = params_mcmc[mc_start::mc_step,:]
params_mcmc = pd.DataFrame(params_mcmc, columns=names)
params_mcmc["Calibration"] = "DRAM Posterior"

params_abc = get_params(abc=True, drop_vars=True)
params_abc = params_abc.filter(names)
params_abc.reset_index(drop=True, inplace=True)
params_abc["Calibration"] = "IMABC"
params_abc = params_abc.loc[:,params_mcmc.columns]

# Prior
hsampler = Halton(d=len(names), scramble=True)
samples = hsampler.random(n=1000)
parameters = get_params(drop_vars=True)
param_upper = parameters.loc[:,names].max().to_numpy()
param_lower = parameters.loc[:,names].min().to_numpy()
scaled_samples = samples * (param_upper - param_lower) + param_lower
labels = np.ones(scaled_samples.shape[0])

approved_samples = scaled_samples[labels == 1]
prior_df = pd.DataFrame(approved_samples, columns=names)
prior_df["Calibration"] = "DRAM Prior"

# %%
# ------------------------------------------------------------------------------
# Plot posteriors
# ------------------------------------------------------------------------------
# pair_df = pd.concat([params_abc, prior_df, params_mcmc, params_stein])
pair_df = pd.concat([params_abc, params_mcmc, params_stein])
fig = plt.figure(figsize=(8, 5))
axes = fig.subplots(2, 2)
for i,ax in enumerate(axes.flatten(), 1):
    column = pair_df.columns[i-1]
    sns.kdeplot(
        data=pair_df.loc[:,[column, "Calibration"]],
        x=column,
        hue="Calibration",
        hue_order=["Stein Posterior", "DRAM Posterior", "IMABC"],
        multiple = "layer",
        common_norm = False,
        common_grid = True,
        legend = False,
        ax = ax
    )
    if i % 2 == 0:
        ax.set_ylabel("")

    for line,ls in zip(ax.lines, [":", "--", "-"]):
        line.set_linestyle(ls)

p1 = axes[0,0].plot([.06], [0], ls="-")[0]
p2 = axes[0,0].plot([.06], [0], ls="--")[0]
p4 = axes[0,0].plot([.06], [0], ls=":")[0]
axes[0,0].legend(
    [p1,p2,p4],
    ["Stein Posterior", "DRAM Posterior", "IMABC"],
)
plt.tight_layout()
plt.savefig(plotdir+filename+"_posterior_compare.pdf")
plt.show()

# %%
# ------------------------------------------------------------------------------
# Plot sampled posteriors
# ------------------------------------------------------------------------------
n_samples = 150
mcmc_samples = np.random.choice(np.arange(params_mcmc.shape[0]), size=n_samples, replace=False)
sub_params_mcmc = params_mcmc.iloc[mcmc_samples, :]
sub_params_mcmc["Calibration"] = f"DRAM ({n_samples} samples)"
stein_samples = np.random.choice(np.arange(params_stein.shape[0]), size=n_samples, replace=False)
sub_params_stein = params_stein.iloc[stein_samples,:]
sub_params_stein["Calibration"] = f"Stein ({n_samples} samples)"

pair_df = pd.concat([params_mcmc, params_stein, sub_params_mcmc, sub_params_stein])
fig = plt.figure(figsize=(8, 5))
axes = fig.subplots(2, 2)
for i,ax in enumerate(axes.flatten(), 1):
    column = pair_df.columns[i-1]
    sns.kdeplot(
        data=pair_df.loc[:,[column, "Calibration"]],
        x=column,
        hue="Calibration",
        hue_order=[f"Stein ({n_samples} samples)", f"DRAM ({n_samples} samples)", "Stein Posterior", "DRAM Posterior"],
        multiple = "layer",
        common_norm = False,
        common_grid = True,
        legend = True,
        ax = ax
    )

plt.legend()
plt.tight_layout()
plt.show()

sub_params_mcmc.drop("Calibration", axis=1)\
    .to_csv(loaddir+filename+"_posterior_samples_mcmc.csv", index=False)
sub_params_stein.drop("Calibration", axis=1)\
    .to_csv(loaddir+filename+"_posterior_samples_stein.csv", index=False)

# %%
# ------------------------------------------------------------------------------
# Plot pairwise posteriors
# ------------------------------------------------------------------------------
thin = 0.05
thick = 2
pair_df = pd.concat([params_stein, params_mcmc])
pplot = sns.pairplot(
    # pair_df,
    pair_df[pair_df.Calibration != "IMABC"],
    corner=True,
    hue = "Calibration",
    diag_kind="kde",
    plot_kws=dict(alpha=1, edgecolor=None, s=4),
    diag_kws=dict(
        multiple="layer",
        common_norm=False,
        alpha=0.3,
        linewidth=thick
    ),
    height = 5/4,
    # aspect = 1.25,
    aspect = 1.7,
)
ax_i = 0
for ax in pplot.axes.flatten():
    if ax:
        if ax.get_ylabel() != "":
            ylab = ax.get_ylabel().split()
            ylab.insert(2,"\n")
            ylab = " ".join(ylab)
            ax.set_ylabel("", rotation = 0, loc="center")
        xlab = ax.get_xlabel().split()
        xlab.insert(2,"\n")
        xlab = " ".join(xlab)
        ax.set_xlabel(xlab, rotation = 0, loc="center")
        if "Spine" in str(type(ax.get_children()[0])):
            # ax.set_xlim(prior_df.min().iloc[ax_i], prior_df.max().iloc[ax_i])
            ax.yaxis.set_visible(True)
            ax.yaxis.label.set_visible(True)
            ax.yaxis.set_label_position("right")
            ax.set_ylabel(ylab, rotation = 0, horizontalalignment="left", rotation_mode="anchor")
            ax_i += 1
        if "seeding" in ax.get_xlabel():
            ax.set_xlabel("Time of initial \nseeding of infection")

sns.move_legend(pplot, "center right", fontsize=12, markerscale=2, title_fontsize=12, fancybox=True, frameon=True)
pplot.figure.subplots_adjust(bottom=(0.12), right=(0.8))
plt.savefig(plotdir+filename+"_pairwise_compare.pdf")
plt.show()

# %%
# Comparison of predictive pushforward
data_real = get_real_data().diff()

hosp_data = data_real.reindex(columns=["hospitalizations","deaths"]).T.stack().hospitalizations
death_data = data_real.reindex(columns=["hospitalizations","deaths"]).T.stack().deaths
train_data = get_data()
hosp_len = data_real.shape[0]
smoothing = 7
comparison_data = data_real.reindex(columns=["hospitalizations","deaths"]).T.to_numpy()
comparison_data = comparison_data[:,1:]
comparison_data[0] = np.convolve(comparison_data[0], np.ones(smoothing)/smoothing, 'same')
comparison_data[1] = np.convolve(comparison_data[1], np.ones(smoothing)/smoothing, 'same')

fig = plt.figure(figsize=(12,(2+1)*(5/2)))
axes = fig.subplots(2+1, 2, sharex=True, sharey=False)
for i in range(2):
    surr_loc = res_loc + "surrogate/surrogate.pkl"
    with open(surr_loc, "rb") as file:
        loaded_model = pickle.load(file)
    smodel = loaded_model["gp"]
    transformer = loaded_model["transformer"]
    data_time = pd.to_datetime(data_real.index)
    full_time = pd.to_datetime(train_data.columns.get_level_values(1))
    time_mask = np.array([True if (ft in data_time) else False for ft in full_time])
    def surrogate_func(ps):
        tps = transformer.transform(ps.reshape((1,-1)))
        result = smodel.predict(tps).T.flatten()[time_mask]
        result = np.vstack((result[:hosp_len], result[hosp_len:]))
        result = np.diff(result)
        return result

    sampler = MCMC()
    x = np.arange(comparison_data.shape[1])
    y = comparison_data
    sampler.data.add_data_set(x,y)
    predmodelfun = lambda ps, _: surrogate_func(ps).T

    if i == 0:
        names = results['names']
        chain = np.array(results['chain'])
        s2chain = np.array(results['s2chain'])
        sschain = np.array(results['sschain'])

        # Filter with mcgibbsit
        mc_res = mcgibbsit(chain[:,np.newaxis,:])
        mc_start = mc_res["M"].max().astype(int)
        mc_step = mc_res["I"].max().astype(int)
        chain = chain[mc_start::mc_step,:]
        sschain = sschain[mc_start::mc_step,:]
        s2chain = s2chain[mc_start::mc_step,:]

        intervals = calculate_intervals(chain, results, sampler.data, predmodelfun,
                                       s2chain=s2chain, nsample=500, waitbar=True, sstype=0
        )
        hosp_intervals = {'credible': intervals[0]['credible'], 'prediction': intervals[0]['prediction']}
        death_intervals = {'credible': intervals[1]['credible'], 'prediction': intervals[1]['prediction']}
        xplot = sampler.data.xdata[0]

        p_hcurves = hosp_intervals["prediction"]
        p_dcurves = death_intervals["prediction"]
    else:
        scaler = lambda x: x
        def ssfun(ps,data):
            res = scaler(np.array(data["ydata"]).squeeze()) - scaler(surrogate_func(ps))
            return (res**2).sum(axis=1)
        errs = np.linalg.norm(train_data.diff(axis=1).loc[:,time_mask].iloc[:,2:] - comparison_data.flatten(), axis=1)
        ps0_loc = np.argsort(errs)[0]
        cut_parameters = get_params(drop_vars=True)
        ps0 = cut_parameters.iloc[ps0_loc].to_numpy()
        x = np.arange(comparison_data.shape[1]).flatten()
        y = comparison_data
        cdata = {"xdata": x, "ydata": y}
        S2_ols = ssfun(ps0.reshape((1,-1)),cdata) / (comparison_data.shape[1] - parameters.shape[1])
        pcurves = []
        for seed in range(seeds):
            pcurves.append(pd.read_csv(loaddir+filename+"_curves.csv", index_col=0))
        pcurves = pd.concat(pcurves, axis=1, ignore_index=True)
        pcurves_type = pcurves.iloc[:,-1]
        pcurves = pcurves.drop(pcurves.columns[pcurves.dtypes == object], axis=1)
        pcurves["type"] = pcurves_type
        hcurves = pcurves.loc[pcurves.type == "hospitalizations", :].drop("type", axis=1)
        dcurves = pcurves.loc[pcurves.type == "deaths", :].drop("type", axis=1)
        p_hcurves = (hcurves + np.random.normal(loc=0, scale=np.sqrt(S2_ols[0]), size=hcurves.shape)).T
        p_dcurves = (dcurves + np.random.normal(loc=0, scale=np.sqrt(S2_ols[1]), size=dcurves.shape)).T
        xplot = x

    # Compare predictive posteriors
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
            color=colors[0],
        )
        q25 = ax.fill_between(
            x=xplot.flatten(),
            y1=np.quantile(np.nan_to_num(inter), .25, axis=0),
            y2=np.quantile(np.nan_to_num(inter), .75, axis=0),
            color=colors[1],
        )
        q45 = ax.fill_between(
            x=xplot.flatten(),
            y1=np.quantile(np.nan_to_num(inter), .45, axis=0),
            y2=np.quantile(np.nan_to_num(inter), .55, axis=0),
            color=colors[2],
        )
        ax.set_xticks(sampler.data.xdata[0][0:hosp_len:15].flatten(), pd.to_datetime(hosp_data.index).strftime("%m/%d").to_numpy()[::15])
        ax.set_title(pname)
        if j == 0:
            first_legend = ax.legend(loc="lower left")
            second_legend = ax.legend([q5, q25, q45], ["5\\% - 95\\%", "25\\% - 75\\%", "45\\% - 55\\%"], title="Predictive Intervals")
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
plt.savefig(plotdir+filename+"_compare_pred_post.pdf")
plt.show()

# %%
# Comparison of error intervals
fig = plt.figure(figsize=(6,(5/2)))
plt.subplot(121)
# plt.hist(s2chain[:,0], density=True, label="MCMC", edgecolor="black")
sns.histplot(s2chain[:,0], label="MCMC", bins=10)
plt.vlines(S2_ols[0], 0, plt.ylim()[1], color="black", label="SVI")
plt.xlabel("$\\sigma_h^2$")
plt.ylabel("Count")
plt.subplot(122)
# plt.hist(s2chain[:,1], density=True, label="MCMC", edgecolor="black")
sns.histplot(s2chain[:,1], label="MCMC", bins=10)
plt.vlines(S2_ols[1], 0, plt.ylim()[1], color="black", label="SVI")
plt.xlabel("$\\sigma_d^2$")
plt.ylabel("Count")
plt.legend()
plt.tight_layout()
plt.savefig(plotdir+filename+"_compare_sigma2.pdf")
plt.show()

# %%
# Comparison of predictive pushforward
data_real = get_real_data().diff()
hosp_data = data_real.reindex(columns=["hospitalizations","deaths"]).T.stack().hospitalizations
death_data = data_real.reindex(columns=["hospitalizations","deaths"]).T.stack().deaths
train_data = get_data()
hosp_len = data_real.shape[0]
smoothing = 7
comparison_data = data_real.reindex(columns=["hospitalizations","deaths"]).T.to_numpy()
comparison_data = comparison_data[:,1:]
comparison_data[0] = np.convolve(comparison_data[0], np.ones(smoothing)/smoothing, 'same')
comparison_data[1] = np.convolve(comparison_data[1], np.ones(smoothing)/smoothing, 'same')

fig = plt.figure(figsize=(12,(2+1)*(5/2)))
axes = fig.subplots(2+1, 2, sharex=True, sharey=False)
for i in range(2):
    surr_loc = res_loc + "surrogate/surrogate.pkl"
    with open(surr_loc, "rb") as file:
        loaded_model = pickle.load(file)
    smodel = loaded_model["gp"]
    transformer = loaded_model["transformer"]
    data_time = pd.to_datetime(data_real.index)
    full_time = pd.to_datetime(train_data.columns.get_level_values(1))
    time_mask = np.array([True if (ft in data_time) else False for ft in full_time])
    def surrogate_func(ps):
        tps = transformer.transform(ps.reshape((1,-1)))
        result = smodel.predict(tps).T.flatten()[time_mask]
        result = np.vstack((result[:hosp_len], result[hosp_len:]))
        result = np.diff(result)
        return result

    sampler = MCMC()
    x = np.arange(comparison_data.shape[1])
    y = comparison_data
    sampler.data.add_data_set(x,y)
    predmodelfun = lambda ps, _: surrogate_func(ps).T

    if i == 0:
        names = results['names']
        chain = np.array(results['chain'])
        s2chain = np.array(results['s2chain'])
        sschain = np.array(results['sschain'])

        # Filter with mcgibbsit
        mc_res = mcgibbsit(chain[:,np.newaxis,:])
        mc_start = mc_res["M"].max().astype(int)
        mc_step = mc_res["I"].max().astype(int)
        chain = chain[mc_start::mc_step,:]
        sschain = sschain[mc_start::mc_step,:]
        s2chain = s2chain[mc_start::mc_step,:]

        intervals = calculate_intervals(chain, results, sampler.data, predmodelfun,
                                       s2chain=s2chain, nsample=500, waitbar=True, sstype=0
        )
        hosp_intervals = {'credible': intervals[0]['credible'], 'prediction': intervals[0]['prediction']}
        death_intervals = {'credible': intervals[1]['credible'], 'prediction': intervals[1]['prediction']}
        xplot = sampler.data.xdata[0]

        p_hcurves = hosp_intervals["credible"]
        p_dcurves = death_intervals["credible"]
    else:
        scaler = lambda x: x
        def ssfun(ps,data):
            res = scaler(np.array(data["ydata"]).squeeze()) - scaler(surrogate_func(ps))
            return (res**2).sum(axis=1)
        errs = np.linalg.norm(train_data.diff(axis=1).loc[:,time_mask].iloc[:,2:] - comparison_data.flatten(), axis=1)
        ps0_loc = np.argsort(errs)[0]
        cut_parameters = get_params(drop_vars=True)
        ps0 = cut_parameters.iloc[ps0_loc].to_numpy()
        x = np.arange(comparison_data.shape[1]).flatten()
        y = comparison_data
        cdata = {"xdata": x, "ydata": y}
        S2_ols = ssfun(ps0.reshape((1,-1)),cdata) / (comparison_data.shape[1] - parameters.shape[1])
        pcurves = []
        for seed in range(seeds):
            filename = f"stein_calibration_seed{seed}"
            pcurves.append(pd.read_csv(loaddir+filename+"_curves.csv", index_col=0))
        pcurves = pd.concat(pcurves, axis=1, ignore_index=True)
        pcurves_type = pcurves.iloc[:,-1]
        pcurves = pcurves.drop(pcurves.columns[pcurves.dtypes == object], axis=1)
        pcurves["type"] = pcurves_type
        hcurves = pcurves.loc[pcurves.type == "hospitalizations", :].drop("type", axis=1)
        dcurves = pcurves.loc[pcurves.type == "deaths", :].drop("type", axis=1)
        p_hcurves = hcurves.T
        p_dcurves = dcurves.T
        xplot = x

    # Compare predictive posteriors
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
            color=colors[0],
        )
        q25 = ax.fill_between(
            x=xplot.flatten(),
            y1=np.quantile(np.nan_to_num(inter), .25, axis=0),
            y2=np.quantile(np.nan_to_num(inter), .75, axis=0),
            color=colors[1],
        )
        q45 = ax.fill_between(
            x=xplot.flatten(),
            y1=np.quantile(np.nan_to_num(inter), .45, axis=0),
            y2=np.quantile(np.nan_to_num(inter), .55, axis=0),
            color=colors[2],
        )
        ax.set_xticks(sampler.data.xdata[0][0:hosp_len:15].flatten(), pd.to_datetime(hosp_data.index).strftime("%m/%d").to_numpy()[::15])
        ax.set_title(pname)
        if j == 0:
            first_legend = ax.legend(loc="lower left")
            second_legend = ax.legend([q5, q25, q45], [r"5\\% - 95\\%", r"25\\% - 75\\%", r"45\\% - 55\\%"], title="Pushforward Intervals")
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
plt.savefig(plotdir+filename+"_compare_pushforward.pdf")
plt.show()

# %%
# Compare verification rank histograms
fig,axes = plt.subplots(1, 2, figsize=(8, 5/2), sharey=True)
for ax_i in range(2):
    ax = axes[ax_i]
    if ax_i == 0:
        names = results['names']
        chain = np.array(results['chain'])
        s2chain = np.array(results['s2chain'])
        sschain = np.array(results['sschain'])

        # Filter with mcgibbsit
        mc_res = mcgibbsit(chain[:,np.newaxis,:])
        mc_start = mc_res["M"].max().astype(int)
        mc_step = mc_res["I"].max().astype(int)
        chain = chain[mc_start::mc_step,:]
        sschain = sschain[mc_start::mc_step,:]
        s2chain = s2chain[mc_start::mc_step,:]

        intervals = calculate_intervals(chain, results, sampler.data, predmodelfun,
                                       s2chain=s2chain, nsample=500, waitbar=True, sstype=0
        )
        hosp_intervals = {'credible': intervals[0]['credible'], 'prediction': intervals[0]['prediction']}
        death_intervals = {'credible': intervals[1]['credible'], 'prediction': intervals[1]['prediction']}
        xplot = sampler.data.xdata[0]

        p_hcurves = hosp_intervals["prediction"]
        p_dcurves = death_intervals["prediction"]
        all_data_h = np.vstack((p_hcurves, comparison_data[0].reshape((1,-1))))
        all_data_d = np.vstack((p_dcurves, comparison_data[1].reshape((1,-1))))
    else:
        scaler = lambda x: x
        def ssfun(ps,data):
            res = scaler(np.array(data["ydata"]).squeeze()) - scaler(surrogate_func(ps))
            return (res**2).sum(axis=1)
        errs = np.linalg.norm(train_data.diff(axis=1).loc[:,time_mask].iloc[:,2:] - comparison_data.flatten(), axis=1)
        ps0_loc = np.argsort(errs)[0]
        cut_parameters = get_params(drop_vars=True)
        ps0 = cut_parameters.iloc[ps0_loc].to_numpy()
        x = np.arange(comparison_data.shape[1]).flatten()
        y = comparison_data
        cdata = {"xdata": x, "ydata": y}
        S2_ols = ssfun(ps0.reshape((1,-1)),cdata) / (comparison_data.shape[1] - parameters.shape[1])
        pcurves = []
        for seed in range(seeds):
            pcurves.append(pd.read_csv(loaddir+filename+"_curves.csv", index_col=0))
        pcurves = pd.concat(pcurves, axis=1, ignore_index=True)
        pcurves_type = pcurves.iloc[:,-1]
        pcurves = pcurves.drop(pcurves.columns[pcurves.dtypes == object], axis=1)
        pcurves["type"] = pcurves_type
        hcurves = pcurves.loc[pcurves.type == "hospitalizations", :].drop("type", axis=1)
        dcurves = pcurves.loc[pcurves.type == "deaths", :].drop("type", axis=1)
        p_hcurves = (hcurves + np.random.normal(loc=0, scale=np.sqrt(S2_ols[0]), size=hcurves.shape)).T
        p_dcurves = (dcurves + np.random.normal(loc=0, scale=np.sqrt(S2_ols[1]), size=dcurves.shape)).T
        xplot = x

        all_data_h = np.hstack((p_hcurves.T, comparison_data[0].reshape((-1,1)))).T
        all_data_d = np.hstack((p_dcurves.T, comparison_data[1].reshape((-1,1)))).T

    obs_rank_h = np.zeros(p_hcurves.shape[1])
    obs_rank_d = np.zeros(p_dcurves.shape[1])
    for i in range(hcurves.shape[0]):
        obs_rank_h[i] = np.where(np.argsort(all_data_h[:,i]) == p_hcurves.shape[0])[0]
        obs_rank_d[i] = np.where(np.argsort(all_data_d[:,i]) == p_dcurves.shape[0])[0]

    obs_rank = pd.DataFrame(np.hstack([obs_rank_h, obs_rank_d]).T, columns=["Observations"])
    obs_rank[""] = str(np.zeros(obs_rank.shape[0]))
    obs_rank.loc[:hcurves.shape[0], ""] = "Hospitalizations"
    obs_rank.loc[hcurves.shape[0]:, ""] = "Deaths"

    sh = sns.histplot(obs_rank, x="Observations", hue="", ax=ax, multiple="layer", bins=10)
    ax.set_xlabel("")
    ax.set_xticks([0, max(obs_rank_h)/2, max(obs_rank_h)], ["Overpredicted", "Centered", "Underpredicted"])
    if ax_i == 0:
        ax.set_title("DRAM")
        ax.legend().remove()
    else:
        ax.set_title("SVI")
        sh.legend_.set_loc("lower right")

plt.tight_layout()
plt.savefig(plotdir+filename+"_compare_vrh.pdf")
plt.show()
