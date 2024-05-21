import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import pickle
import warnings
warnings.filterwarnings(action="ignore",category=UserWarning)
warnings.filterwarnings(action="ignore",category=RuntimeWarning)
import os
env = os.environ["PROJLOC"]
plt.style.use(env+"/src/utils/rf_mcmc.mplstyle")
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

import sys
sys.path.append(env + "/src/utils")
from load_data import get_data, get_params
sys.path.append(env + "/src/calibration")
from distance_correlation import distcorr

from scipy.stats.qmc import Halton
from scipy.stats import gaussian_kde
from pymcmcstat.propagation import calculate_intervals, plot_intervals
from pymcmcstat import mcmcplot as mcp
from pymcmcstat.chain.ChainProcessing import load_json_object
from pymcmcstat.MCMC import MCMC

# %%
# ------------------------------------------------------------------------------
# Options
# ------------------------------------------------------------------------------
smoothing = 7
show_plots = True

loaddir = env + "/results/calibration"
plotdir = env + "/plots/calibration"

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

# %%
# ------------------------------------------------------------------------------
# Load data
# ------------------------------------------------------------------------------
data_real = pd.read_csv(env+"/data/observed_chicago.csv", index_col=0).dropna()
data_real = data_real.diff()
hosp_len = data_real.shape[0]
train_data = get_data(abc=False)
parameters = get_params(abc=False, drop_vars=False)
comparison_data = data_real.reindex(columns=["hospitalizations","deaths"]).T.to_numpy()
comparison_data = comparison_data[:,1:]
comparison_data[0] = np.convolve(comparison_data[0], np.ones(smoothing)/smoothing, 'same')
comparison_data[1] = np.convolve(comparison_data[1], np.ones(smoothing)/smoothing, 'same')
hosp_data = data_real.reindex(columns=["hospitalizations","deaths"]).T.stack().hospitalizations
death_data = data_real.reindex(columns=["hospitalizations","deaths"]).T.stack().deaths

surr_loc = env + "/results/surrogate/surrogate.pkl"
with open(surr_loc, "rb") as file:
    loaded_model = pickle.load(file)
smodel = loaded_model["rf"]
pca_components = loaded_model["pca_components"]
transformer = loaded_model["transformer"]

# Restrict pca components to when we have data
full_time = pd.to_datetime(train_data.columns.get_level_values(1))
times = np.arange(full_time.unique().shape[0])
data_time = pd.to_datetime(data_real.index)
time_mask = np.array([True if (ft in data_time) else False for ft in full_time])

# Define surrogate model function
pca_components_all_time = np.zeros(len(pca_components))
pca_components = pca_components[time_mask]
def surrogate_func(ps):
    coefs = smodel.predict(ps.reshape((1,-1)))
    pca_components_all_time[time_mask] = pca_components @ coefs.flatten()
    raw_result = transformer.inverse_transform(pca_components_all_time.reshape((1,-1)))
    result = raw_result.flatten()[time_mask]
    result = np.vstack((result[:hosp_len], result[hosp_len:]))
    result = np.diff(result)
    return result

sampler = MCMC()
x = np.arange(comparison_data.shape[1])
y = comparison_data
sampler.data.add_data_set(x,y)

# ------------------------------------------------------------------------------
# Display results
# ------------------------------------------------------------------------------
results = load_json_object(loaddir+"/calibrator.json")
for k in results.keys():
    r = results[k]
    if isinstance(r, list):
        results[k] = np.array(r)

# names = [nice_param_map[n] for n in results['names']]
names = results['names']
chain = np.array(results['chain'])
s2chain = np.array(results['s2chain'])
sschain = np.array(results['sschain'])
opt_params = chain[np.argmin(sschain.sum(axis=1))]

# %%
# ------------------------------------------------------------------------------
# Plot predictions
# ------------------------------------------------------------------------------
predmodelfun = lambda ps, _: surrogate_func(ps).T
intervals = calculate_intervals(chain, results, sampler.data, predmodelfun,
                               s2chain=s2chain, nsample=500, waitbar=True, sstype=0
)
hosp_intervals = {'credible': intervals[0]['credible'], 'prediction': intervals[0]['prediction']}
death_intervals = {'credible': intervals[1]['credible'], 'prediction': intervals[1]['prediction']}
xplot = sampler.data.xdata[0]
hplot = sampler.data.ydata[0][0]
dplot = sampler.data.ydata[0][1]

f = plt.figure(figsize=(8,5/2))
thin = 0.05
thick = 2
axes = f.subplots(1, 2, sharex=True)
for i,(ax,inter,tdata,name) in enumerate(zip(
    axes,
    [hosp_intervals, death_intervals],
    [hplot, dplot],
    ["Hospitalizations (Pushfoward)", "Deaths (Pushfoward)"],
)):
    sns.lineplot(x=xplot.flatten(), y=tdata, ax=ax, color="black", lw=thick, label="Observed")
    q5 = ax.fill_between(
        x=xplot.flatten(),
        y1=np.quantile(np.nan_to_num(inter["credible"]), .05, axis=0),
        y2=np.quantile(np.nan_to_num(inter["credible"]), .95, axis=0),
        color=colors[4],
        # label=r"5\% - 95\%"
    )
    q25 = ax.fill_between(
        x=xplot.flatten(),
        y1=np.quantile(np.nan_to_num(inter["credible"]), .25, axis=0),
        y2=np.quantile(np.nan_to_num(inter["credible"]), .75, axis=0),
        color=colors[5],
        # label=r"25\% - 75\%"
    )
    q45 = ax.fill_between(
        x=xplot.flatten(),
        y1=np.quantile(np.nan_to_num(inter["credible"]), .45, axis=0),
        y2=np.quantile(np.nan_to_num(inter["credible"]), .55, axis=0),
        color=colors[6],
        # label=r"45\% - 55\%"
    )
    ax.set_xticks(sampler.data.xdata[0][0:hosp_len:15].flatten(), pd.to_datetime(hosp_data.index).strftime("%m/%d").to_numpy()[::15])
    ax.set_title(name)
    if i == 0:
        first_legend = ax.legend(loc="lower left")
        second_legend = ax.legend([q5, q25, q45], [r"5\% - 95\%", r"25\% - 75\%", r"45\% - 55\%"], title="Pushforward Intervals")
        ax.add_artist(first_legend)
    else:
        ax.legend_.set_visible(False)
f.tight_layout()
plt.savefig(plotdir+"/mcmc_prediction.pdf")
if show_plots:
    plt.show()
else:
    plt.close()

# %%
# ------------------------------------------------------------------------------
# Plot prediction intervals
# ------------------------------------------------------------------------------
f = plt.figure(figsize=(8,5/2))
thin = 0.05
thick = 2
axes = f.subplots(1, 2, sharex=True)
for i,(ax,inter,tdata,name) in enumerate(zip(
    axes,
    [hosp_intervals, death_intervals],
    [hplot, dplot],
    ["Hospitalizations (Predictive)", "Deaths (Predictive)"],
)):
    sns.lineplot(x=xplot.flatten(), y=tdata, ax=ax, color="black", lw=thick, label="Observed")
    q5 = ax.fill_between(
        x=xplot.flatten(),
        y1=np.quantile(np.nan_to_num(inter["prediction"]), .05, axis=0),
        y2=np.quantile(np.nan_to_num(inter["prediction"]), .95, axis=0),
        color=colors[0],
        # label=r"5\% - 95\%"
    )
    q25 = ax.fill_between(
        x=xplot.flatten(),
        y1=np.quantile(np.nan_to_num(inter["prediction"]), .25, axis=0),
        y2=np.quantile(np.nan_to_num(inter["prediction"]), .75, axis=0),
        color=colors[1],
        # label=r"25\% - 75\%"
    )
    q45 = ax.fill_between(
        x=xplot.flatten(),
        y1=np.quantile(np.nan_to_num(inter["prediction"]), .45, axis=0),
        y2=np.quantile(np.nan_to_num(inter["prediction"]), .55, axis=0),
        color=colors[2],
        # label=r"45\% - 55\%"
    )
    ax.set_xticks(sampler.data.xdata[0][0:hosp_len:15].flatten(), pd.to_datetime(hosp_data.index).strftime("%m/%d").to_numpy()[::15])
    ax.set_title(name)
    if i == 0:
        first_legend = ax.legend(loc="lower left")
        second_legend = ax.legend([q5, q25, q45], [r"5\% - 95\%", r"25\% - 75\%", r"45\% - 55\%"], title="Predictive Intervals")
        ax.add_artist(first_legend)
    else:
        ax.legend_.set_visible(False)
f.tight_layout()
plt.savefig(plotdir+"/mcmc_prediction_intervals.pdf")
if show_plots:
    plt.show()
else:
    plt.close()

# %%
f = plt.figure(figsize=(8,5/4))
thin = 0.05
thick = 2
ax = f.subplots(1, 1)
for i,name in enumerate(["$\\sigma_h$", "$\\sigma_d$"]):
    sns.histplot(np.sqrt(s2chain[:,i]), ax=ax, label=name, color=colors[i], stat="density", bins=20)
ax.set_ylabel("")
ax.legend()
f.tight_layout()
plt.savefig(plotdir+"/mcmc_prediction_sigmas.pdf")
if show_plots:
    plt.show()
else:
    plt.close()

# %%
temp_data_real = pd.read_csv(env+"/data/observed_chicago.csv", index_col=0).dropna()
d0,h0 = temp_data_real.iloc[0]

temp_xplot = np.vstack((-1, xplot))
temp_hplot = np.cumsum(np.hstack((h0, hplot)))
temp_dplot = np.cumsum(np.hstack((d0, dplot)))
npred = hosp_intervals['credible'].shape[0]
npred,ntime = hosp_intervals['credible'].shape
temp_hosp_intervals = {
    'credible': np.cumsum(np.hstack((h0*np.ones((npred,1)), hosp_intervals['credible'])), axis=1),
    'prediction': np.cumsum(np.hstack((h0*np.ones((npred,1)), hosp_intervals['prediction'])), axis=1)
}
temp_death_intervals = {
    'credible': np.cumsum(np.hstack((d0*np.ones((npred,1)), death_intervals['credible'])), axis=1),
    'prediction': np.cumsum(np.hstack((d0*np.ones((npred,1)), death_intervals['prediction'])), axis=1)
}

f = plt.figure(figsize=(8,5/2))
thin = 0.05
thick = 2
axes = f.subplots(1, 2, sharex=True)
for i,(ax,inter,tdata,name) in enumerate(zip(
    axes,
    [temp_hosp_intervals, temp_death_intervals],
    [temp_hplot, temp_dplot],
    ["Hospitalizations (daily)", "Deaths (daily)"],
)):
    sns.lineplot(x=temp_xplot.flatten(), y=tdata, ax=ax, color="black", lw=thick, label="Observed")
    q5 = ax.fill_between(
        x=temp_xplot.flatten(),
        y1=np.quantile(np.nan_to_num(inter["prediction"]), .05, axis=0),
        y2=np.quantile(np.nan_to_num(inter["prediction"]), .95, axis=0),
        color=colors[0],
        # label=r"5\% - 95\%"
    )
    q25 = ax.fill_between(
        x=temp_xplot.flatten(),
        y1=np.quantile(np.nan_to_num(inter["prediction"]), .25, axis=0),
        y2=np.quantile(np.nan_to_num(inter["prediction"]), .75, axis=0),
        color=colors[1],
        # label=r"25\% - 75\%"
    )
    q45 = ax.fill_between(
        x=temp_xplot.flatten(),
        y1=np.quantile(np.nan_to_num(inter["prediction"]), .45, axis=0),
        y2=np.quantile(np.nan_to_num(inter["prediction"]), .55, axis=0),
        color=colors[2],
        # label=r"45\% - 55\%"
    )
    ax.set_xticks(sampler.data.xdata[0][0:hosp_len:15].flatten(), pd.to_datetime(hosp_data.index).strftime("%m/%d").to_numpy()[::15])
    ax.set_title(name)
    if i == 0:
        first_legend = ax.legend(loc="lower left")
    else:
        second_legend = ax.legend([q5, q25, q45], [r"5\% - 95\%", r"25\% - 75\%", r"45\% - 55\%"], title=r"\begin{center}Predictive Intervals \\ (Census)\end{center}", loc="upper left")
f.tight_layout()
plt.savefig(plotdir+"/mcmc_prediction_census_intervals_predictive.pdf")
if show_plots:
    plt.show()
else:
    plt.close()

f = plt.figure(figsize=(8,5/2))
thin = 0.05
thick = 2
axes = f.subplots(1, 2, sharex=True)
for i,(ax,inter,tdata,name) in enumerate(zip(
    axes,
    [temp_hosp_intervals, temp_death_intervals],
    [temp_hplot, temp_dplot],
    ["Hospitalizations (daily)", "Deaths (daily)"],
)):
    sns.lineplot(x=temp_xplot.flatten(), y=tdata, ax=ax, color="black", lw=thick, label="Observed")
    q5 = ax.fill_between(
        x=temp_xplot.flatten(),
        y1=np.quantile(np.nan_to_num(inter["credible"]), .05, axis=0),
        y2=np.quantile(np.nan_to_num(inter["credible"]), .95, axis=0),
        color=colors[0],
        # label=r"5\% - 95\%"
    )
    q25 = ax.fill_between(
        x=temp_xplot.flatten(),
        y1=np.quantile(np.nan_to_num(inter["credible"]), .25, axis=0),
        y2=np.quantile(np.nan_to_num(inter["credible"]), .75, axis=0),
        color=colors[1],
        # label=r"25\% - 75\%"
    )
    q45 = ax.fill_between(
        x=temp_xplot.flatten(),
        y1=np.quantile(np.nan_to_num(inter["credible"]), .45, axis=0),
        y2=np.quantile(np.nan_to_num(inter["credible"]), .55, axis=0),
        color=colors[2],
        # label=r"45\% - 55\%"
    )
    ax.set_xticks(sampler.data.xdata[0][0:hosp_len:15].flatten(), pd.to_datetime(hosp_data.index).strftime("%m/%d").to_numpy()[::15])
    ax.set_title(name)
    if i == 0:
        first_legend = ax.legend(loc="lower left")
    else:
        second_legend = ax.legend([q5, q25, q45], [r"5\% - 95\%", r"25\% - 75\%", r"45\% - 55\%"], title=r"\begin{center}Pushforward Intervals \\ (Census)\end{center}", loc="upper left")
f.tight_layout()
plt.savefig(plotdir+"/mcmc_prediction_census_intervals_pushforward.pdf")
if show_plots:
    plt.show()
else:
    plt.close()

# %%
# ------------------------------------------------------------------------------
# Plot prediction rank histogram
# ------------------------------------------------------------------------------
predictions_h = intervals[0]['prediction']
predictions_d = intervals[1]['prediction']
all_data_h = np.vstack((predictions_h, comparison_data[0].reshape((1,-1))))
all_data_d = np.vstack((predictions_d, comparison_data[1].reshape((1,-1))))

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
plt.xticks([0, max(obs_rank_h)/2, max(obs_rank_h)], ["Overpredicted", "Centered", "Underpredicted"])
ax.yaxis.label.set_visible(False)
plt.tight_layout()
plt.savefig(plotdir+"/mcmc_predictive_rank.pdf")
if show_plots:
    plt.show()
else:
    plt.close()

# %%
# ------------------------------------------------------------------------------
# Plot CRPS and DSS scores
# ------------------------------------------------------------------------------
scores = pd.read_csv(loaddir+"/scores.csv",index_col=0)
hscores = scores[scores.target_type == "hosps"]
dscores = scores[scores.target_type == "deaths"]

f = plt.figure(figsize=(8,5/2 + 5/4 + 5/4))
# fig, axes = plt.subplots(3, 2, figsize=(8,5), sharex=True, height_ratios=[.6,.2,.2])
thin = 0.05
thick = 2
axes = f.subplots(3, 2, sharex=True, height_ratios=[.5,.25,.25])
for i,(ax,inter,tdata,name) in enumerate(zip(
    axes[0],
    [hosp_intervals, death_intervals],
    [hplot, dplot],
    ["Hospitalizations (Surrogate)", "Deaths (Surrogate)"],
)):
    sns.lineplot(x=xplot.flatten(), y=tdata, ax=ax, color="black", lw=thick, label="Observed")
    q5 = ax.fill_between(
        x=xplot.flatten(),
        y1=np.quantile(np.nan_to_num(inter["prediction"]), .05, axis=0),
        y2=np.quantile(np.nan_to_num(inter["prediction"]), .95, axis=0),
        color=colors[0],
        # label=r"5\% - 95\%"
    )
    q25 = ax.fill_between(
        x=xplot.flatten(),
        y1=np.quantile(np.nan_to_num(inter["prediction"]), .25, axis=0),
        y2=np.quantile(np.nan_to_num(inter["prediction"]), .75, axis=0),
        color=colors[1],
        # label=r"25\% - 75\%"
    )
    q45 = ax.fill_between(
        x=xplot.flatten(),
        y1=np.quantile(np.nan_to_num(inter["prediction"]), .45, axis=0),
        y2=np.quantile(np.nan_to_num(inter["prediction"]), .55, axis=0),
        color=colors[2],
        # label=r"45\% - 55\%"
    )
    ax.set_xticks(sampler.data.xdata[0][0:hosp_len:15].flatten(), pd.to_datetime(hosp_data.index).strftime("%m/%d").to_numpy()[::15])
    ax.set_title(name)
    if i == 0:
        first_legend = ax.legend(loc="lower left")
        second_legend = ax.legend([q5, q25, q45], [r"5\% - 95\%", r"25\% - 75\%", r"45\% - 55\%"], title="Predictive Intervals")
        ax.add_artist(first_legend)
    else:
        ax.legend_.set_visible(False)

# Add CRPS and DSS
sns.lineplot(x=hscores.target_end_date, y=hscores.crps, ax=axes[1,0], color=colors[3], lw=thick, label="CRPS")
axes[1,0].yaxis.label.set_visible(False)
sns.lineplot(x=hscores.target_end_date, y=hscores.dss, ax=axes[2,0], color=colors[7], lw=thick, label="DSS")
axes[2,0].yaxis.label.set_visible(False)
axes[2,0].xaxis.label.set_visible(False)

sns.lineplot(x=dscores.target_end_date, y=dscores.crps, ax=axes[1,1], color=colors[3], lw=thick)
axes[1,1].yaxis.label.set_visible(False)
sns.lineplot(x=dscores.target_end_date, y=dscores.dss, ax=axes[2,1], color=colors[7], lw=thick)
axes[2,1].yaxis.label.set_visible(False)
axes[2,1].xaxis.label.set_visible(False)

f.tight_layout()
plt.savefig(plotdir+"/mcmc_scores.pdf")
if show_plots:
    plt.show()
else:
    plt.close()

# %%
# ------------------------------------------------------------------------------
# Plot chain outcomes
# ------------------------------------------------------------------------------
f = mcp.plot_chain_panel(chain, names, settings={
    'fig':{'figsize':(10,8)},
    'plot':{'linestyle':'-', 'marker':None},
})
for i in range(len(names)):
    f.axes[i].hlines([parameters.loc[:,names[i]].min(), parameters.loc[:,names[i]].max()], 0, chain.shape[0], color="black")
plt.savefig(plotdir+"/mcmc_chain.pdf")
if show_plots:
    plt.show()
else:
    plt.close()

# %%
# Find prior samples
hsampler = Halton(d=len(names), scramble=True)
samples = hsampler.random(n=1000)

param_upper = parameters.loc[:,names].max().to_numpy()
param_lower = parameters.loc[:,names].min().to_numpy()
scaled_samples = samples * (param_upper - param_lower) + param_lower
labels = np.ones(scaled_samples.shape[0])
approved_samples = scaled_samples[labels == 1]
prior_df = pd.DataFrame(approved_samples, columns=names)
prior_df["Calibration"] = "DRAM Prior"

# Load ABC results
params_abc = get_params(abc=True)
params_abc.columns = [nice_param_map[n] for n in params_abc.columns]
params_abc = params_abc.filter(names)
params_abc.reset_index(drop=True, inplace=True)
params_abc["Calibration"] = "ABC"

chain_df = pd.DataFrame(chain, columns=names)
chain_df["Calibration"] = "DRAM Posterior"
pair_df = pd.concat([chain_df])
pplot = sns.pairplot(
    pair_df,
    corner=True,
    diag_kind="kde",
    plot_kws=dict(alpha=1, edgecolor=None, s=2),
    diag_kws=dict(
        multiple="layer",
        common_norm=False,
        alpha=0.3,
        linewidth=thick
    ),
    height = 5/4,
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
            ax.yaxis.set_visible(True)
            ax.yaxis.label.set_visible(True)
            ax.yaxis.set_label_position("right")
            ax.set_ylabel(ylab, rotation = 0, horizontalalignment="left", rotation_mode="anchor")
            ax_i += 1

pplot.figure.subplots_adjust(bottom=(0.12), right=(0.8))
plt.savefig(plotdir+"/mcmc_pairwise.pdf")
if show_plots:
    plt.show()
else:
    plt.close()

# %%
pair_df = pd.concat([params_abc, prior_df, chain_df])
fig = plt.figure(figsize=(8, 5))
axes = fig.subplots(2, 2)
for i,ax in enumerate(axes.flatten(), 1):
    column = pair_df.columns[i-1]
    sns.kdeplot(
        data=pair_df.loc[:,[column, "Calibration"]],
        x=column,
        hue="Calibration",
        hue_order=["DRAM Posterior", "DRAM Prior", "ABC"],
        multiple = "layer",
        common_norm = False,
        common_grid = True,
        legend = False,
        ax = ax
    )
    if i % 2 == 0:
        ax.set_ylabel("")

    for line,ls in zip(ax.lines, ["--", ":", "-"]):
        line.set_linestyle(ls)

p1 = axes[0,0].plot([.06], [0], ls="-")[0]
p2 = axes[0,0].plot([.06], [0], ls=":")[0]
p3 = axes[0,0].plot([.06], [0], ls="--")[0]
axes[0,0].legend(
    [p1,p2,p3],
    ["DRAM Posterior", "DRAM Prior", "ABC"],
)
plt.tight_layout()
plt.savefig(plotdir+"/mcmc_posterior_compare.pdf")
if show_plots:
    plt.show()
else:
    plt.close()
# %%
# ------------------------------------------------------------------------------
# Compute distance correlation between samples
# ------------------------------------------------------------------------------
dc = distcorr(approved_samples)
plt.figure(figsize=(8,8))
plt.imshow(dc, clim=(0,1))
plt.colorbar()
plt.xticks(range(dc.shape[0]), names, rotation=70)
plt.yticks(range(dc.shape[0]), names)
plt.grid(False)
plt.tight_layout()
plt.savefig(plotdir+"/mcmc_dist_corr.pdf")
if show_plots:
    plt.show()
else:
    plt.close()
# %%

