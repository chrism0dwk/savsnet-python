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
    kxx = cov_fn(x_coords)
    kxxtx = matrix_inverse(stabilize(kxx))
    kxxs = tt.dot(kxxtx, x_val)
    knew = cov_fn(x_star_coords, x_coords)
    return tt.dot(knew, kxxs)


def logistic2D(y, coords, knots, pred_coords):
    model = pm.Model()
    with model:
        alpha = pm.Normal('alpha', mu=0., sd=1.)
        sigma_sq = pm.Gamma('sigma_sq', 1., 1.)
        phi = pm.Gamma('phi', 2., 0.1)

        spatial_cov = sigma_sq * pm.gp.cov.Matern32(2, phi)
        spatial_gp = pm.gp.Latent(cov_func=spatial_cov)
        s = spatial_gp.prior('s', X=knots)

        eta = alpha + project(s, knots, coords, spatial_cov)
        y_rv = pm.Bernoulli('y', p=pm.invlogit(eta), observed=y)

        s_star_ = pm.Deterministic('s_star', project(s, knots, pred_coords, spatial_cov))

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
