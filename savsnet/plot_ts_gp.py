#!/usr/bin/python -o

import argparse
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import pandas as pd
import numpy as np
import pymc3 as pm
import pickle
from prev_ts_GP import extractData


def plotPrediction(ax,X,y,N,pred,mindate,lag=None,prev_mult=1,plot_gp=False):
    """Predictive time series plot with (by default) the prediction
    summarised as [0.01,0.05,0.5,0.95,0.99] quantiles, and observations colour-coded
    by tail-probability.

    Parameters
    ==========
    ax -- a set of axes on which to plot
    X  -- 1D array-like of times of length n
    y  -- 1D array-like of observed number of cases at each time of length n
    N  -- 1D array-like of total number at each time of length n
    pred -- 2D m x n array with numerical draws from posterior
    mindate -- a pandas.Timestamp representing the time origin wrt X
    lag     -- how many days prior to max(X) to plot
    prev_mult -- prevalence multiplier (to get in, eg. prev per 1000 population)
    plot_gp -- plots a GP smudge-o-gram rather than 95% and 99% quantiles.

    Returns
    =======
    Nothing.   Just modifies ax
    """

    from pymc3.gp.util import plot_gp_dist

    # Time slice
    ts = slice(0,X.shape[0])
    if lag is not None:
        ts = slice(X.shape[0]-lag, X.shape[0])
    
    # Data
    x = np.array([mindate + pd.Timedelta(d,unit='D') for d in X[ts]])
    pbar = np.array(y/N)[ts] * prev_mult

    # Prediction quantiles
    phat = pm.invlogit(pred[:,ts]).eval() * prev_mult
    pctiles = np.percentile(phat, [1,5,50,95,99], axis=0)

    # Tail probabilities for observed p
    prp = np.sum(pbar > phat, axis=0)/phat.shape[0]
    prp[prp > .5] = 1. - prp[prp > .5]
    
    # Risk masks
    red = prp <= 0.01
    orange = (0.01 < prp) & (prp <= 0.05)
    green = 0.05 < prp

    # Construct plot
    if plot_gp is True:
        plot_gp_dist(ax,phat,x,plot_samples=False, palette="Blues")
    else:
        ax.fill_between(x, pctiles[4,:], pctiles[0,:], color='lightgrey',alpha=.5,label="99% credible interval")
        ax.fill_between(x, pctiles[3,:], pctiles[1,:], color='lightgrey',alpha=1,label='95% credible interval')
    ax.plot(x, pctiles[2,:], c='grey', ls='-', label="Predicted prevalence")
    ax.scatter(x[green],pbar[green],c='green',s=8,alpha=0.5,label='0.05<p')
    ax.scatter(x[orange],pbar[orange],c='orange',s=8,alpha=0.5,label='0.01<p<=0.05')
    ax.scatter(x[red],pbar[red],c='red',s=8,alpha=0.5,label='p<=0.01')


def plotAllPred(data,predictions,species,condition,lag=None):
    """Convenience function to plot a grid of species and conditions.

    Parameters
    ==========
    data - a 2D sequence of datasets for each species/condition
    predictions - a 2D sequence of predictions for each species/condition
    species - a sequence of species names
    condition - a sequence of condition names
    axlabels - y axis labels
    """
    
    nSpecies = len(species)
    nConditions = len(condition)

    fig,ax = plt.subplots(nConditions,nSpecies,sharex=True,figsize=(10,5))
    ax = np.atleast_2d(ax)

    for i in range(nConditions):
        for j in range(nSpecies):
            plotPrediction(ax[i,j], data[i][j].day, 
                           data[i][j].cases,data[i][j].N,
                           predictions[i][j]['s_star'],
                           np.min(data[i][j].date),lag=lag,prev_mult=1000)
            if i==0:
                ax[i,j].set_title(species[j].capitalize())
            box = ax[i,j].get_position()
            ax[i,j].set_position([box.x0, box.y0, box.width, box.height*.7])
            ax[i,j].yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
        ax[i,0].set(ylabel=condition[i].capitalize())
    
    fig.autofmt_xdate()
    #ax[nConditions-1,0].legend(ncol=3,loc='lower left',
    #                           bbox_to_anchor=(.3,-1))
    fig.tight_layout(rect=(0.02,0.1,1,1))
    fig.subplots_adjust(hspace=0)
    h,l = ax[0,0].get_legend_handles_labels()
    fig.legend(handles=h,labels=l,loc='lower center',ncol=3)
    fig.text(0.01,0.5,'Prevalence / 1000 consults', va='center', rotation='vertical',size='large')
    return fig,ax
    


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="Plot timeseries predictions")
    parser.add_argument("-d", "--data", nargs=1, dest="data", type=str, required=True,
                        help="Input data file with (at least) columns Date, Species, Consult_reason")
    parser.add_argument("-s", "--species", dest="species", nargs='+', type=str, required=True,
                        help="Species as appearing in data (columns)")
    parser.add_argument("-c", "--condition", dest="condition", nargs='+', type=str, required=True,
                        help="Condition as appearing in data (rows)")
    parser.add_argument("-l", "--lag", dest="lag", type=int, default=None,
                        help="Number of timesteps prior to maximum time to plot")
    parser.add_argument("-p", "--posterior", dest="posterior", nargs='+', type=str, required=True,
                        help="Space-separated list of posterior files in row major order.  Number of files should equal species X conditions")
    parser.add_argument("-o", "--output", dest="output", type=str, default=None)
    args = parser.parse_args()

    # Check number of posterior files equals nSpecies*nConditions
    if len(args.posterior) != (len(args.species)*len(args.condition)):
        raise IndexError("len(posterior) does not equal len(species)*len(condition)")
    
    # Read data in
    print ("Reading data")
    data = pd.read_csv(args.data[0])
    print("Packaging data")
    # Construct data and prediction matrices
    predictionBlock = []
    dataBlock = []
    nSpecies = len(args.species)
    for c,condition in enumerate(args.condition):
        p = []
        d = []
        for s,species in enumerate(args.species):
            d.append(extractData(data,species,condition))
            p.append(pickle.load(open(args.posterior[nSpecies*c+s],'rb'))[2])
        
        dataBlock.append(d)
        predictionBlock.append(p)
    print("Plotting data")
    fig,ax = plotAllPred(dataBlock,predictionBlock,args.species,args.condition,lag=args.lag)
    print("Done")
    if args.output is not None:
        fig.savefig(args.output)
    else:
        fig.show()
