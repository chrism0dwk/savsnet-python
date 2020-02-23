#!/usr/bin/python -o
# SAVSNet model

import sys
import pandas as pd
import numpy as np
import pymc3 as pm
import theano.tensor as tt
import theano as t
import theano.tensor.slinalg as tla

__refdate = pd.to_datetime('2000-01-01')

def match(x, y):
    """Returns the positions of the first occurrence of values of x in y."""
    def indexof(a, b):
        i = np.where(b == a)[0]
        return i if i.shape[0] > 0 else [np.nan]
    return np.concatenate([indexof(xx, y) for xx in x])


def BinomGP(y, N, time, time_pred, mcmc_iter, start={}):
    """Fits a logistic Gaussian process regression timeseries model.

    Details
    =======
    Let $y_t$ be the number of cases observed out of $N_t$ individuals tested.  Then
    $$y_t \sim Binomial(N_t, p_t)$$
    with 
    $$logit(p_t) \sim s_t$$

    $s_t$ is modelled as a Gaussian process such that
    $$s_t \sim \mbox{GP}(\mu_t, \Sigma)$$
    with a mean function capturing a linear time trend
    $$\mu_t = \alpha + \beta X_t$$
    and a periodic covariance plus white noise
    $$\Sigma_{i,j} = \sigma^2 \exp{ \frac{2 \sin^2(|x_i - x_j|/365)}{\phi^2}} + \tau^2$$

    Parameters
    ==========
    y -- a vector of cases
    N -- a vector of number tested
    X -- a vector of times at which y is observed
    T -- a vector of time since recruitment
    X_star -- a vector of times at which predictons are to be made
    start -- a dictionary of starting values for the MCMC.

    Returns
    =======
    A tuple of (model,trace,pred) where model is a PyMC3 Model object, trace is a PyMC3 Multitrace object, and pred is a 5000 x X_star.shape[0] matrix of draws from the posterior predictive distribution of $\pi_t$.
    """
    time = np.array(time)[:, None]  # Inputs must be arranged as column vector
    offset = np.mean(time)
    time = time - offset  # Center time
    model = pm.Model()

    with model:
        alpha = pm.Normal('alpha', 0, 1000, testval=0.)
        beta = pm.Normal('beta', 0, 100, testval=0.)
        sigmasq_s = pm.HalfNormal('sigmasq_s', 5., testval=0.1)
        phi_s = 0.16 #pm.HalfNormal('phi_s', 5., testval=0.5)
        tau2 = pm.Gamma('tau2', .1, .1, testval=0.1)

        # Construct GPs
        cov_t = sigmasq_s * pm.gp.cov.Periodic(1, 365., phi_s)
        mean_t = pm.gp.mean.Linear(coeffs=beta, intercept=alpha)
        gp_period = pm.gp.Latent(mean_func=mean_t, cov_func=cov_t)

        cov_nugget = pm.gp.cov.WhiteNoise(tau2)
        gp_nugget = pm.gp.Latent(cov_func=cov_nugget)

        gp_t = gp_period + gp_nugget
        s = gp_t.prior('gp_t', X=time[:, None])

        Y_obs = pm.Binomial('y_obs', N, pm.invlogit(s), observed=y)

        # Sample
        trace = pm.sample(mcmc_iter,
                          chains=1,
                          start=start,
                          tune=1000,
                          adapt_step_size=True)

        # Prediction
        time_pred -= offset
        s_star = gp_t.conditional('s_star', time_pred[:, None])
        pi_star = pm.Deterministic('pi_star', pm.invlogit(s_star))
        pred = pm.sample_posterior_predictive(trace, var_names=['y_obs', 'pi_star'])

        return {'model': model, 'trace': trace, 'pred': pred}


def extractData(dat, species, condition, date_range=None):
    """Extracts weekly records for 'condition' in 'species'.

    Parameters
    ==========
    dat -- pandas data_frame with at least columns Date,Species,Consult_reason.
    species -- name of a species in column Species.
    condition -- name of a condition in column Consult_reason.

    Returns
    =======
    A pandas data_frame containing weekly aggregated data:
      date  -- date of week commencing
      day   -- day since start of dataset
      cases -- number of weekly cases of condition in species
      N     -- total number of cases seen weekly for species

    """
    if date_range is None:
        date_range = [np.min(dat['Date']), np.max(dat['Date'])]

    dat = dat[dat['Species'] == species]
    dat['Date'] = pd.to_datetime(dat['Date'])
    dat = dat[dat['Date'] >= date_range[0]]
    dat = dat[dat['Date'] < date_range[1]]

    # Turn date into days, and quantize into weeks
    dat['day'] = (dat['Date'] - __refdate) // np.timedelta64(1,'W') * 7.0 # Ensure this is float

    # Now group by date combination
    datgrp = dat.groupby(by='day')
    aggTable = datgrp['Consult_reason'].agg([['cases', lambda x: np.sum(x == condition)],
                                          ['N', len]])
    aggTable['day'] = np.array(aggTable.index)
    aggTable.index = __refdate + pd.to_timedelta(aggTable.index, unit='D')
    return aggTable


# def predProb(y, ystar):
#     """Calculates predictive tail probability of y wrt ystar.
#
#     Parameters
#     ==========
#     y     -- a vector of length n of observed values of y
#     ystar -- a m x n matrix containing draws from the joint distribution of ystar.
#
#     Returns
#     =======
#     A vector of tail probabilities for y wrt ystar.
#     """
#     q = np.sum(y > ystar, axis=0) / ystar.shape[0]
#     q[q > .5] = 1. - q[q > .5]
#     return q


if __name__ == '__main__':

    import argparse
    import pickle

    parser = argparse.ArgumentParser(description='Fit Binomial GP timeseries model')
    parser.add_argument("data", nargs=1, type=str,
                        help="Input data file with (at least) columns Date, Species, Consult_reason")
    parser.add_argument("-o", "--prefix", dest="prefix", type=str, default=None,
                        help="Output file prefix [optional].")
    parser.add_argument("-c", "--condition", dest="condition", nargs='+', type=str,
                        required=True,
                        help="One or more space-separated conditions to analyse")
    parser.add_argument("-s", "--species", dest="species", nargs='+', type=str,
                        required=True,
                        help="One or more space-separated species to analyse")
    parser.add_argument("-i", "--iterations", dest="iterations", type=int,
                        default=5000, nargs=1,
                        help="Number of MCMC iterations (default 5000)")
    args = parser.parse_args()

    data = pd.read_csv(args.data[0])

    for species in args.species:
        for condition in args.condition:
            print("Running GP smoother for condition '%s' in species '%s'" % (condition, species))
            sys.stdout.write("Extracting data...")
            sys.stdout.flush()
            d = extractData(data, species, condition, date_range=['2016-02-02','2019-12-01'])
            sys.stdout.write("done\nCalculating...")
            sys.stdout.flush()

            pred_range = pd.date_range(start='2018-02-14', end='2020-02-14', freq='W')
            day_pred = (pred_range - __refdate) / np.timedelta64(1, 'D')

            res = BinomGP(np.array(d.cases, dtype=float), np.array(d.N, dtype=float),
                          np.array(d.day, dtype=float), np.array(day_pred, dtype=float),
                          mcmc_iter=args.iterations[0])
            res['data'] = extractData(data, species, condition, date_range=['2018-02-14','2020-02-14'])
            res['pred'] = pd.DataFrame(res['pred']['pi_star'])
            res['pred'].columns = pd.Index(pred_range)
            sys.stdout.write("done\n")
            sys.stdout.flush()
            filename = "%s_%s.pkl" % (species, condition)
            if args.prefix is not None:
                filename = "%s%s" % (args.prefix, filename)
            print("Saving '%s'" % filename)
            with open(filename, 'wb') as f:
                pickle.dump(res, f, pickle.HIGHEST_PROTOCOL)
