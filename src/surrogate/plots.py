import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_data(data, savefile=None, show_plots=True):
    """
    Plot hospitalization and deaths trajectories in dataset

    Parameters
    ----------
    data: dataset of ABM runs
        - rows: parameters
        - columns: multiindex (variable, timestamps)
    """
    plt.figure(figsize=(10,3))
    ticks = pd.to_datetime(data['hospitalizations'].columns)
    for i in range(data.shape[0]):
      plt.subplot(121)
      plt.plot(ticks, data['hospitalizations'].iloc[i])
      plt.subplot(122)
      plt.plot(ticks, data['deaths'].iloc[i])
    plt.subplot(121); plt.title("Hospitalizations")
    plt.xticks(ticks[::15], ticks.strftime("%m/%d")[::15])
    plt.subplot(122); plt.title("Deaths")
    plt.xticks(ticks[::15], ticks.strftime("%m/%d")[::15])
    plt.tight_layout()
    if savefile != None:
        plt.savefig(savefile)
    if show_plots:
        plt.show()
    else:
        plt.close()

def plot_pca_components(components, savefile=None, show_plots=True):
    """
    Plot PCA components (split into hospitalizations / deaths)

    Parameters
    ----------
    components: PCA components (columnwise)
    """
    plt.figure(figsize=(12,3))
    for i in range(components.shape[1]):
      comp = components[:,i]
      n = len(comp) // 2
      comp_hosp = comp[0*n:1*n]
      comp_dead = comp[1*n:2*n]
      plt.subplot(121)
      plt.plot(comp_hosp,label="$\\vec{{c}}_{{{}}}$".format(i+1))
      plt.subplot(122)
      plt.plot(comp_dead,label="$\\vec{{c}}_{{{}}}$".format(i+1))
    plt.subplot(121); plt.title("Hospitalizations"); plt.legend()
    plt.subplot(122); plt.title("Deaths"); plt.legend()
    if savefile != None:
        plt.savefig(savefile)
    if show_plots:
        plt.show()
    else:
        plt.close()

def plot_errs_w_variables(n_components, errs, variables, suptitle=None, savefile=None, show_plots=True, **kwargs):
    """
    Plot errors for each variable across ticks

    Parameters
    ----------
    ticks: range of errors
    errs: the errors for each number of components
    variables: the output variables in consideration
    suptitle: the title for all subplots
    kwargs: keywords to pass on to `plot_errs`
    """
    fig = plt.figure(figsize=(7,5))
    for i,output in enumerate(variables):
      plt.subplot(2,1,i+1)
      plot_errs(n_components,errs[i],fig=fig,title=output,**kwargs)
    if suptitle is not None:
        plt.suptitle(suptitle)
    plt.tight_layout()
    if savefile != None:
        plt.savefig(savefile)
    if show_plots:
        plt.show()
    else:
        plt.close()

def plot_errs(ticks,errs,exp_variance=None,ax_jump=9,
              fig=None,xticks=None,yticks=[1e-1,1e-2,1e-3],
              xlabel="Number of components", var_name="y",
              xlabel2="Variance explained by components",
              title=None,labels=["5%","Median","95%"],
              figsize=(7,5)):
    """
    Plot errors across ticks

    * Most common usage are used as defaults

    Parameters
    ----------
    ticks: range of errors
    errs: errors
    exp_variance: explained variances across ticks (if applicable)
    ax_jump: how much to jump in xticks
    fig: Matplotlib figure (if None will be created)
    xticks: Xticks for the plot (if None, will use ticks with ax_jump)
    yticks: yticks for the plot
    xlabel: label for x axis
    var_name: name to be used in ylabel relative error
    xlabel2: label for explained variance second axis (if applicable)
    title: title of plot
    labels: labels for curves in errs
    figsize: size of figure
    """
    if fig is None:
        plt.figure(figsize=figsize)
    plt.plot(ticks,errs,label=labels)
    plt.yscale('log')
    if xticks is None:
        plt.xticks(ticks[::ax_jump])
    plt.yticks(yticks)
    plt.xlabel(xlabel)
    plt.ylabel("$(\\frac{{|{{{}}} - \hat{{{}}}|}}{{|{}|}})$".format(var_name,var_name,var_name))
    plt.grid(); plt.legend()
    if title is not None:
        plt.title(title)
    if exp_variance is not None:
        ax = plt.gca()
        secax = ax.secondary_xaxis('top')
        secax.set_ticks(ticks[::ax_jump], ["{:.3f}".format(i) for i in exp_variance[::ax_jump]])
        secax.set_xlabel(xlabel2)

def plot_statistics_over_axes(param_five,param_med,param_nfive,time_five,time_med,time_nfive,n_comp,savefile=None,xlabel="Number of components",show_plots=True):
    """
    Plot pca 5%,median,95% statistics applied over each axes

    Parameters
    ----------
    param_five: 5% quantile over axis 0
    param_med: median over axis 0
    param_nfive: 95% quantile over axis 0
    time_five: 5% quantile over axis 1
    time_med: median over axis 1
    time_nfive: 95% quantile over axis 1
    n_comp: number of components used
    """
    fig = plt.figure(figsize=(7,5))

    plt.subplot(211)
    ticks = pd.to_datetime(param_med.index).strftime("%m/%d")
    plot_errs(ticks,pd.concat([param_five,param_med,param_nfive],axis=1),title="Absolute relative error (median over parameters)",fig=fig,xlabel=xlabel)

    plt.subplot(212)
    ticks = time_med.index
    plt.step(ticks, time_five,label="5%")
    plt.step(ticks, time_med,label="Median")
    plt.step(ticks, time_nfive,label="95%")
    plt.yscale('log')
    plt.xlabel("Parameter set")
    plt.ylabel("$(\\frac{|y - \hat{y}}{|y|})$")
    plt.title("Absolute relative error (median over time)")
    plt.grid();plt.legend()

    plt.suptitle("Deaths ({} components)".format(n_comp))
    plt.tight_layout()
    if savefile != None:
        plt.savefig(savefile)
    if show_plots:
        plt.show()
    else:
        plt.close()

def plot_data_comparisons(data1,data2,err,variables,pset=None,savefile=None,show_plots=True,pred_name="Prediction"):
    """
    Plot overlayed data1,data2 and error between them for each variable

    Parameters
    ----------
    data1: true data
    data2: predicted data
    err: error between data1 and data2 for plotting
    variables: output variables in the dataset
    pset: parameter set number for title if desired
    """
    fig = plt.figure(figsize=(9,2.5))
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    ticks = pd.to_datetime(data1[variables[0]].index)
    nvar = len(variables)
    for i,v in enumerate(variables):
        ax = plt.subplot(1,nvar,i+1)
        p1 = plt.plot(ticks, data1[v],label="CityCOVID", color=colors[0])
        p2 = plt.plot(ticks, data2[v],label="Prediction", color=colors[1])
        plt.title(v.capitalize())
        plt.xticks(ticks[::15], ticks.strftime("%m/%d")[::15])

        sax = ax.twinx()
        # if i == 1:
        #     sax.set_ylabel("Median rel error", color=colors[2])
        p3 = sax.plot(ticks,err[v], color=colors[3], ls="--")
        sax.set_ylim(0, .15)
        sax.set_yticks([0, .05, .1, .15], ["0\%", "5\%", "10\%", "15\%"])

        yticks1 = ax.get_yticks()
        if i == 0:
            sax.yaxis.set_ticklabels([])
        ax.set_yticks(yticks1)
        sax.grid(False)

        if i == 1:
            sax.legend(
                [p1[0], p2[0], p3[0]],
                ["CityCOVID", pred_name, "Median Rel Err"],
                framealpha=1,
                loc="upper left"
            )

    if pset != None:
        fig.suptitle("Parameter set: {}".format(pset))

    plt.tight_layout()
    if savefile != None:
        plt.savefig(savefile)
    if show_plots:
        plt.show()
    else:
        plt.close()
