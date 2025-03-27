import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
env = os.environ["PROJLOC"]
plt.style.use(env+"/src/utils/rf_mcmc.mplstyle")
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

import sys
sys.path.append(env+"src/forConnor/utils")
from load_data import *

from scipy.stats import gaussian_kde, ecdf
from scipy.spatial.distance import jensenshannon

# %%
# Load data
seeds = range(6)
nparticles_list = range(10, 400, 20)
nhparticles = nparticles_list[-1]
model = "double"
replicate = -1
data_type = "daily"
dataset = 9
smoothing = 7
surrogate = "gp"
show_plots = True

nsimu = 10000
verb = 100
nr = 1
lr = 1e-3

plot_str = "stein"
plotdir = env+"/plots/stein_calibration_gaussian_process"

hparticles = []
for seed in seeds:
    hero_file = add_result_str_kwargs(
        f"results/{plot_str}", ".csv",
        dataset=dataset, model=model, dtype=data_type, replicate=replicate,
        smoothing=smoothing, surrogate=surrogate, seed=seed,
        nparticles=nhparticles, nr=nr, verb=verb, lr=lr
    )
    hero_particles = pd.read_csv(hero_file)
    hero_particles["Num. Particles"] = nhparticles
    hparticles.append(hero_particles)
hparticles = pd.concat(hparticles)

particles = []
for nparticles in nparticles_list:
    seed_particles = []
    for seed in seeds:
        filename = add_result_str_kwargs(
            f"results/{plot_str}", ".csv",
            dataset=dataset, model=model, dtype=data_type, replicate=replicate,
            smoothing=smoothing, surrogate=surrogate, seed=seed,
            nparticles=nparticles, nr=nr, verb=verb, lr=lr
        )

        seed_particles.append(pd.read_csv(filename))
    seed_df = pd.concat(seed_particles)
    seed_df["Num. Particles"] = nparticles
    particles.append(seed_df)
particles = pd.concat(particles)
particles.rename(columns={"Time of initial exposure": "Time of initial seeding of infection"}, inplace=True)

grid = sns.FacetGrid(
    pd.melt(particles, id_vars="Num. Particles"),
    hue="Num. Particles",
    col="variable",
    sharex=False,
    sharey=False,
    palette="flare",
)

grid.map(sns.kdeplot, "value")
grid.set_xlabels("")
grid.set_titles("{col_name}")
grid.add_legend(ncol=2)
plt.savefig(f"{plotdir}/{plot_str}_distribution_convergence.pdf")
plt.show()

# %%
# Make kdes of all combined particles, find distance between each
minx = particles.drop("Num. Particles", axis=1).min().iloc[:-1]
maxx = particles.drop("Num. Particles", axis=1).max().iloc[:-1]
domains = [np.linspace(mi,ma,100) for mi,ma in zip(minx,maxx)]

run_probs = []
for nparticles in nparticles_list:
    sub_particles = particles.loc[particles.loc[:,"Num. Particles"] == nparticles, particles.columns != nparticles]

    probs = np.zeros((4,len(domains[0])))
    for col,dom in enumerate(domains):
        kde = gaussian_kde(sub_particles.iloc[:,col])
        probs[col] = kde(dom)
    run_probs.append(probs)
run_probs = np.array(run_probs)

js_dist = np.zeros((len(nparticles_list)-1, 4))
for i in range(js_dist.shape[0]):
    for j in range(js_dist.shape[1]):
        js_dist[i,j] = jensenshannon(run_probs[i,j], run_probs[-1,j])

js_col_names = [f"$\\theta_{i}:$ {c}" for i,c in enumerate(particles.columns[:-1])]
js_dist = pd.DataFrame(js_dist, columns=js_col_names, index=nparticles_list[:-1])

f = plt.figure(figsize=(8/1.5,5/1.5))
sns.lineplot(js_dist)
plt.ylabel(f"Jensen Shannon distance\n(from results at {nparticles_list[-1]} particles)")
plt.xlabel("Number of particles")
# plt.yscale("log")
plt.tight_layout()
plt.savefig(f"{plotdir}/{plot_str}_js_convergence.pdf")
plt.show()

# %%
# Find distances between quantiles
quantiles = [.05, .25, .5, .75, .95]

q_probs = np.zeros((len(nparticles_list), len(quantiles), 4))
for i,nparticles in enumerate(nparticles_list):
    sub_particles = particles.drop("Num. Particles", axis=1).loc[particles.loc[:, "Num. Particles"] == nparticles]

    for j,q in enumerate(quantiles):
        q_probs[i,j,:] = np.quantile(sub_particles, q, axis=0)

qs_dist = np.zeros((len(nparticles_list)-1, len(quantiles), 4))
for i in range(qs_dist.shape[0]):
    for j in range(qs_dist.shape[1]):
        qs_dist[i,j,:] = (q_probs[i,j] - q_probs[-1,j])**2 / (q_probs[-1,j])**2

qs_col_names = [f"$\\theta_{i}:$ {c}" for i,c in enumerate(particles.columns[:-1])]
for i,q in enumerate(quantiles):
    tmp_qs_dist = pd.DataFrame(qs_dist[:,i,:], columns=qs_col_names, index=nparticles_list[:-1])
    f = plt.figure(figsize=(8/1.5,5/1.5))
    sns.lineplot(tmp_qs_dist)
    plt.ylabel(f"Quantile ({int(100*q)}\\%) distance\n(from results at {nparticles_list[-1]} particles)")
    plt.xlabel("Number of particles")
    # plt.yscale("log")
    plt.tight_layout()
    plt.savefig(f"{plotdir}/{plot_str}_quantile-{q}_convergence.pdf")
    plt.show()


# %%
# Find KS, Cramer Mises distances between all particles
minx = particles.drop("Num. Particles", axis=1).min()
maxx = particles.drop("Num. Particles", axis=1).max()
domains = [np.linspace(mi,ma,100) for mi,ma in zip(minx,maxx)]

particle_cdfs = []
for i, nparticles in enumerate(nparticles_list):
    sub_particles = particles.drop("Num. Particles", axis=1).loc[particles.loc[:, "Num. Particles"] == nparticles]

    cdfs = []
    for j in range(sub_particles.shape[1]):
        cdfs.append(ecdf(sub_particles.iloc[:,j]))
    particle_cdfs.append(cdfs)

ks_dist = np.zeros((len(nparticles_list)-1, 4))
cvm_dist = np.zeros((len(nparticles_list)-1, 4))
wass_dist = np.zeros((len(nparticles_list)-1, 4))
for i in range(ks_dist.shape[0]):
    for j in range(ks_dist.shape[1]):
        ks_dist[i,j] = np.max(np.abs(particle_cdfs[i][j].cdf.evaluate(domains[j]) - particle_cdfs[-1][j].cdf.evaluate(domains[j])))
        cvm_dist[i,j] = np.mean((particle_cdfs[i][j].cdf.evaluate(domains[j]) - particle_cdfs[-1][j].cdf.evaluate(domains[j]))**2)
        wass_dist[i,j] = np.mean(np.abs(particle_cdfs[i][j].cdf.evaluate(domains[j]) - particle_cdfs[-1][j].cdf.evaluate(domains[j])))

col_names = [f"$\\theta_{i}:$ {c}" for i,c in enumerate(particles.columns[:-1])]

tmp_ks_dist = pd.DataFrame(ks_dist, columns=col_names, index=nparticles_list[:-1])
f = plt.figure(figsize=(8/1.5,5/1.5))
sns.lineplot(tmp_ks_dist)
plt.ylabel(f"Kolmogorov Smirnov CDF distances\n(from results at {nparticles_list[-1]} particles)")
plt.xlabel("Number of particles")
plt.tight_layout()
plt.savefig(f"{plotdir}/{plot_str}_ks_convergence.pdf")
plt.show()

tmp_cvm_dist = pd.DataFrame(cvm_dist, columns=col_names, index=nparticles_list[:-1])
f = plt.figure(figsize=(8/1.5,5/1.5))
sns.lineplot(tmp_cvm_dist)
plt.ylabel(f"Cramer-von Mises CDF distances\n(from results at {nparticles_list[-1]} particles)")
plt.xlabel("Number of particles")
plt.tight_layout()
plt.savefig(f"{plotdir}/{plot_str}_cvm_convergence.pdf")
plt.show()

tmp_wass_dist = pd.DataFrame(wass_dist, columns=col_names, index=nparticles_list[:-1])
f = plt.figure(figsize=(8/1.5,5/1.5))
sns.lineplot(tmp_wass_dist)
plt.ylabel(f"Wasserstein CDF distances\n(from results at {nparticles_list[-1]} particles)")
plt.xlabel("Number of particles")
plt.tight_layout()
plt.savefig(f"{plotdir}/{plot_str}_wass_convergence.pdf")
plt.show()
# %%
