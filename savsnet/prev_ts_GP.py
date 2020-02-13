#!/usr/bin/python -o
# SAVSNet model

import sys
import pandas as pd
import numpy as np
import pymc3 as pm
import theano.tensor as tt
import theano as t
import theano.tensor.slinalg as tla


def match(x, y):
    """Returns the positions of the first occurrence of values of x in y."""
    def indexof(a, b):
        i = np.where(b == a)[0]
        return i if i.shape[0] > 0 else [np.nan]
    return np.concatenate([indexof(xx, y) for xx in x])


def BinomGP(y, N, X, prac_id, X_star, mcmc_iter, start={}):
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

    n_prac = np.unique(prac_idx).shape[0]
    print(f"Received {n_prac} practices")

    X = np.array(X)[:, None]  # Inputs must be arranged as column vector
    X = X - np.mean(X) # Center covariates
    T = np.unique(X)
    prac_unique = np.unique(prac_id)

    model = pm.Model()

    with model:
        alpha = pm.Normal('alpha', 0, 1000, testval=0.)
        beta = pm.Normal('beta', 0, 100, testval=0.)
        sigmasq_s = pm.HalfNormal('sigmasq_s', 5., testval=0.1)
        phi_s = 0.16 #pm.HalfNormal('phi_s', 5., testval=0.5)
        tau2 = pm.HalfNormal('tau2', 5., testval=0.1)

        # Construct GPs
        cov_s = sigmasq_s * pm.gp.cov.Periodic(1, 365., phi_s)
        mean_f = pm.gp.mean.Linear(coeffs=beta, intercept=alpha)
        gp_t = pm.gp.Latent(mean_func=mean_f, cov_func=cov_s)

        cov_nugget = pm.gp.cov.WhiteNoise(tau2)
        nugget = pm.gp.Latent(cov_func=cov_nugget)

        sigma_u = pm.HalfNormal('sigma_u', 5.)
        u = pm.Normal('u', alpha, sigma_u, shape=n_prac)

        gp = gp_s + nugget
        model.gp = gp
        s = gp.prior('s', X=X) + u[prac_idx]

        Y_obs = pm.Binomial('y_obs', N, pm.invlogit(s), observed=y)

        # Sample
        trace = pm.sample(mcmc_iter,
                          chains=1,
                          start=start,
                          tune=500)
        # Predictions
        s_star = gp_t.conditional('s_star', T[:, None])
        pred = pm.sample_ppc(trace, vars=[s_star, Y_obs])

        return model, trace, pred


def extractData(dat, species, condition):
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
    dat = dat.copy()[dat['Species'] == species]
    dat.Date = pd.DatetimeIndex(dat.Date)
    dat = dat[dat.Date > '2016-08-01']
    # # Throw out recently added practices
    # byPractice = dat.groupby(['Practice_ID'])
    # firstAppear = byPractice['Date'].agg([['mindate','min']])
    # dat = pd.merge(dat, firstAppear, how='left', on='Practice_ID')
    # dat = dat[dat.mindate<'2018-06-01']

    # Turn date into days, and quantize into weeks
    minDate = np.min(dat.Date)
    day = ((pd.DatetimeIndex(dat.Date) - minDate).days // 7) * 7
    day.name = 'Day'
    dat['day'] = day

    # Calculate the earliest day for each practice
    day0 = dat.groupby(dat['Practice_ID'])['day'].agg([['day0', np.min]])

    # Now group by Practice/day combination
    dat = dat.groupby(by=[dat['Practice_ID'], day])
    aggTable = dat['Consult_reason'].agg([['cases', lambda x: np.sum(x == condition)],
                                          ['N', len]])
    aggTable = aggTable.reset_index()
    aggTable['date'] = [minDate + pd.Timedelta(dt, unit='D') for dt in aggTable['Day']]
    aggTable = pd.merge(aggTable, day0, how='left', on='Practice_ID')
    aggTable['time_since_recruitment'] = aggTable['Day'] - aggTable['day0']
    aggTable['Practice_ID'] = aggTable['Practice_ID'].astype('category')
    return aggTable


def predProb(y, ystar):
    """Calculates predictive tail probability of y wrt ystar.

    Parameters
    ==========
    y     -- a vector of length n of observed values of y
    ystar -- a m x n matrix containing draws from the joint distribution of ystar.

    Returns
    =======
    A vector of tail probabilities for y wrt ystar.
    """
    q = np.sum(y > ystar, axis=0) / ystar.shape[0]
    q[q > .5] = 1. - q[q > .5]
    return q


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
            d = extractData(data, species, condition)
            sys.stdout.write("done\nCalculating...")
            sys.stdout.flush()
            res = BinomGP(np.array(d.cases), np.array(d.N),
                          np.array(d.Day), np.array(d.Practice_ID.cat.codes), np.array(d.Day),
                          mcmc_iter=args.iterations[0])
            sys.stdout.write("done\n")
            sys.stdout.flush()
            filename = "%s_%s.pkl" % (species, condition)
            if args.prefix is not None:
                filename = "%s%s" % (args.prefix, filename)
            print("Saving '%s'" % filename)
            with open(filename, 'wb') as f:
                pickle.dump(res, f, pickle.HIGHEST_PROTOCOL)
