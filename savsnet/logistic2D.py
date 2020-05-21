"""Functions to perform 2D logistic kriging"""

import sys

import geopandas as gp
import numpy as np
import pymc3 as pm
import theano as t
import theano.tensor as tt
from pymc3.gp.util import stabilize
from theano.tensor.nlinalg import matrix_inverse

from savsnet.gis_util import make_masked_raster, raster2coords
from savsnet.raster import make_raster


class Data:
    def __init__(self, y, coords_obs, coords_knots):
        self.y = np.squeeze(np.array(y))
        self.coords = np.array(coords_obs)
        self.knots = np.array(coords_knots)
        assert y.shape[0] == coords_obs.shape[0]
        assert self.coords.shape[1] == 2
        assert self.knots.shape[1] == 2

    def __len__(self):
        return self.y.shape[0]


def project(x_val, x_coords, x_star_coords, cov_fn):
    """Projects a Gaussian process defined by `x_val` at `x_coords`
       onto a set of coordinates `x_star_coords`.
    :param x_val: values of the GP at `x_coords`
    :param x_coords: a set of coordinates for each `x_val`
    :param x_star_coords: a set of coordinates onto which to project the GP
    :param cov_fn: a covariance function returning a covariance matrix given a set of coordinates
    :returns: a vector of projected values at `x_star_coords`
    """
    kxx = cov_fn(x_coords)
    kxxtx = matrix_inverse(stabilize(kxx))
    kxxs = tt.dot(kxxtx, x_val)
    knew = cov_fn(x_star_coords, x_coords)
    return tt.dot(knew, kxxs)


def logistic2D(y, coords, knots, pred_coords):
    """Returns an instance of a logistic geostatistical model with
       Matern32 covariance.
       
       Let $y_i, i=1,\dots,n$ be a set of binary observations at locations $x_i, i=1,\dots,n$.
       We model $y_i$ as
       $$
       y_i \sim Bernoulli(p_i)
       $$
       with
       $$
       \mbox{logit}(p_i) = \alpha + S(x_i).
       $$
       
       $S(x_i)$ is a latent Gaussian process defined as
       $$
       S(\bm{x}) \sim \mbox{MultivariateNormal}(\bm{0}, \Sigma^2)
       $$
       where
       $$
       \Sigma_{ij}^2 = \sigma^2 \left(1 + \frac{\sqrt{3(x - x')^2}}{\ell}\right)
                  \mathrm{exp}\left[ - \frac{\sqrt{3(x - x')^2}}{\ell} \right]
       $$

       The model evaluates $S(x_i)$ approximately using a set of inducing points $x^\star_i, i=1,\dots,m$ for $m$ auxilliary locations.  See [Banerjee \emph{et al.} (2009)](https://dx.doi.org/10.1111%2Fj.1467-9868.2008.00663.x) for further details.

       :param y: a vector of binary outcomes {0, 1}
       :param coords: a matrix of coordinates of `y` of shape `[n, d]` for `d`-dimensions and `n` observations.
       :param knots: a matrix of inducing point coordinates of shape `[m, d]`.
       :param pred_coords: a matrix of coordinates at which predictions are required.
       :returns: a dictionary containing the PyMC3 `model`, and the `posterior` PyC3 `Multitrace` object. 
       """
    model = pm.Model()
    with model:
        alpha = pm.Normal('alpha', mu=0., sd=1.)
        sigma_sq = pm.Gamma('sigma_sq', 1., 1.)
        phi = pm.Gamma('phi', 2., 0.1)

        spatial_cov = sigma_sq * pm.gp.cov.Matern32(2, phi)
        spatial_gp = pm.gp.Latent(cov_func=spatial_cov)
        s = spatial_gp.prior('s', X=knots)
        s_star_ = pm.Deterministic('s_star', project(s, knots, pred_coords, spatial_cov))

        eta = alpha + project(s, knots, coords, spatial_cov)
        y_rv = pm.Bernoulli('y', p=pm.invlogit(eta), observed=y)


    def sample_fn(*args, **kwargs):
        with model:
            trace = pm.sample(*args, **kwargs)
        return {'model': model, 'posterior': trace}

    return sample_fn


if __name__ == '__main__':
    import argparse
    import pickle as pkl
    import pandas as pd

    parser = argparse.ArgumentParser(description='Fit Binomial GP timeseries model')
    parser.add_argument("data", nargs=1, type=str,
                        help="Input data file with (at least) columns consult_date, gi_mpc_selected, Easting, Northing")
    parser.add_argument("--startdate", "-s", type=np.datetime64, nargs=1, help="Start date")
    parser.add_argument("--period", "-p", type=int, default=[7], nargs=1, help="Period in days")
    parser.add_argument("--iterations", "-i", type=int, default=[5000], nargs=1,
                        help="Number of MCMC iterations")
    args = parser.parse_args()

    data = pd.read_csv(args.data[0])
    data['consult_date'] = pd.to_datetime(data['consult_date'])
    week = pd.to_timedelta(args.period[0], unit='day')

    start = args.startdate[0]

    end = start + week
    d = data[(start <= data['consult_date']) & (data['consult_date'] < end)]

    d = d.groupby([d['person_easting'], d['person_northing']])
    aggr = d['gi_mpc_selected'].agg([['case', lambda x: np.sum(x) > 0]])
    aggr.reset_index(inplace=True)

    y = np.array(aggr['case'])
    coords = np.array(aggr[['person_easting', 'person_northing']], dtype='float32')
    knots = pm.gp.util.kmeans_inducing_points(300, coords)

    # GIS
    poly = gp.read_file('/home/jewellcp/Documents/GIS/gadm36_GBR.gpkg').to_crs(epsg=27700)
    r = make_masked_raster(poly, 5000., crs='+init=epsg:27700')
    pred_points = raster2coords(r)

    sampler = logistic2D(y, coords=coords/1000., knots=knots/1000., pred_coords=pred_points/1000.)
    result = sampler(args.iterations[0], chains=1)
    with open(f"vom_dog_spatial_week{start}.pkl", 'wb') as f:
        pkl.dump(result, f)

    start = end

    sys.exit(0)
