#!/usr/bin/python -o
# SAVSNet model

import pandas as pd
import numpy as np
import pymc3 as pm
import theano.tensor as tt
import theano as t
import theano.tensor.slinalg as tla


def BinomGP(y, N, X, X_star,mcmc_iter,start={}):
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
    X_star -- a vector of times at which predictons are to be made
    start -- a dictionary of starting values for the MCMC.

    Returns
    =======
    A tuple of (model,trace,pred) where model is a PyMC3 Model object, trace is a PyMC3 Multitrace object, and pred is a 5000 x X_star.shape[0] matrix of draws from the posterior predictive distribution of $\pi_t$.
    """

    X = np.array(X)[:,None] # Inputs must be arranged as column vector
    X_star = X_star[:,None]

    model = pm.Model()

    with model:
        
        alpha      = pm.Normal('alpha',0, 1000, testval=0.)
        beta       = pm.Normal('beta',0, 100, testval=0.)
        sigmasq_s  = pm.Gamma('sigmasq_s',.1,.1,testval=0.1)
        phi_s      = pm.Gamma('phi_s', 1., 1., testval=0.5)
        tau2        = pm.Gamma('tau2',.1,.1,testval=0.1)

        # Construct GPs
        cov_s = sigmasq_s * pm.gp.cov.Periodic(1,365.,phi_s)
        mean_f = pm.gp.mean.Linear(coeffs=beta, intercept=alpha)
        gp_s = pm.gp.Latent(mean_func=mean_f,cov_func=cov_s)
        
        cov_nugget = pm.gp.cov.WhiteNoise(tau2)
        nugget = pm.gp.Latent(cov_func=cov_nugget)

        gp = gp_s + nugget
        model.gp = gp
        s = gp.prior('s',X=X)
    
        Y_obs = pm.Binomial('y_obs', N, pm.invlogit(s), observed=y)

        # Sample
        trace = pm.sample(mcmc_iter,
                          chains=1,
                          start=start)
        # Predictions
        s_star = gp.conditional('s_star', X_star)
        pred = pm.sample_ppc(trace, vars=[s_star,Y_obs])

        return (model,trace,pred)


def extractData(dat,species,condition):
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
    dat = dat[dat.Species==species]
    dates = pd.DatetimeIndex(dat.Date)
    minDate = np.min(dates)
    week = (dates-minDate).days // 7

    dat = dat.groupby(week)
    aggTable = dat['Consult_reason'].agg([['cases',lambda x: np.sum(x==condition)],
                                          ['N',len]])

    aggTable['day']  = aggTable.index * 7
    aggTable['date'] = [minDate + pd.Timedelta(dt*7, unit='D') for dt in aggTable.index]
    
    return (aggTable)



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







if __name__=='__main__':

    import argparse
    import pickle

    parser = argparse.ArgumentParser(description='Fit Binomial GP timeseries model')
    parser.add_argument("data", nargs=1, type=str, 
                        help="Input data file with (at least) columns Date, Species, Consult_reason") 
    parser.add_argument("-o", "--prefix", dest="prefix",nargs=1,type=str,
                        help="Output file prefix.")
    parser.add_argument("-c", "--condition", dest="condition",nargs='+',type=str,
                        help="One or more space-separated conditions to analyse")
    parser.add_argument("-s", "--species", dest="species",nargs='+',type=str,
                        help="One or more space-separated species to analyse")
    parser.add_argument("-i", "--iterations", dest="iterations", type=int, 
                        default=5000, nargs=1,
                        help="Number of MCMC iterations (default 5000)")
    args = parser.parse_args()

    data = pd.read_csv(args.data[0])

    for species in args.species:
        for condition in args.condition:
            print ("Running GP smoother for condition '%s' in species '%s'" % (species, condition))
            d = extractData(data, species, condition)
            res = BinomGP(np.array(d.cases),np.array(d.N),
                          np.array(d.day), np.array(d.day),
                          mcmc_iter=args.iterations[0])
            filename = "%s_%s_%s.pkl" % (args.prefix[0], species, condition)
            print ("Saving '%s'" % filename)
            with open(filename, 'wb') as f:
                pickle.dump(res,f,pickle.HIGHEST_PROTOCOL)
