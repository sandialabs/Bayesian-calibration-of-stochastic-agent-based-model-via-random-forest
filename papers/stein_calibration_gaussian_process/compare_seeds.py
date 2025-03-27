import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
env = os.environ["PROJLOC"]
plt.style.use(env+"/src/utils/rf_mcmc.mplstyle")
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

import sys
sys.path.append(env+"/src/utils")
from load_data import *
# %%
# Load data
seeds = range(6)
nparticles = 200

seed_particles = []
for seed in seeds:
    filename = f"stein_calibration_seed{seed}"

    sdf = pd.read_csv(env+f"/results/stein_calibration_gaussian_process/calibration/{filename}.csv")
    sdf["Random Seed"] = seed
    seed_particles.append(sdf)
particles = pd.concat(seed_particles)

grid = sns.FacetGrid(
    pd.melt(particles, id_vars="Random Seed"),
    hue="Random Seed",
    col="variable",
    sharex=False,
    sharey=False,
    palette="tab10",
    height=5/2,
    aspect=8/5,
    col_wrap=2
)

grid.map(sns.kdeplot, "value")
grid.set_titles("{col_name}")
grid.set_xlabels("")
grid.set_ylabels("Probability Density")
grid.add_legend(ncol=2)

plt.savefig(env+"/plots/stein_calibration_gaussian_process/calibration/seed_convergence.pdf")
plt.show()
