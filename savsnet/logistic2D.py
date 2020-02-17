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


def logistic2D(y, N, coords, pred_coords):
    model = pm.Model()
    with model:
        alpha = pm.Normal('alpha', mu=0., sd=1.)
        sigma_sq = pm.Gamma('sigma_sq', 1., 1.)
        phi = pm.Gamma('phi', 5., 1.)

        spatial_cov = sigma_sq * pm.gp.cov.Matern32(2, phi)
        spatial_gp = pm.gp.Latent(cov_func=spatial_cov)
        s = spatial_gp.prior('s', X=coords)

        eta = alpha + s
        y_rv = pm.Binomial('y', n=N, p=pm.invlogit(eta), observed=y)

        kxx = spatial_cov(coords)
        kxxtx = matrix_inverse(stabilize(kxx))
        kxxs = tt.dot(kxxtx, s)
        knew = spatial_cov(pred_coords, coords)
        s_star = tt.dot(knew, kxxs)
        s_star_ = pm.Deterministic('s_star', s_star)

    def sample_fn(*args, **kwargs):
        with model:
            trace = pm.sample(*args, **kwargs)
        return {'model': model, 'posterior': trace}

    return sample_fn


# Prediction



def build_predictor(orig_coords, pred_coords):
    t.config.compute_test_value = 'off'
    sigma_sq = tt.dscalar('sigma_sq', )
    phi = tt.dscalar('phi')
    s = tt.dvector('s')
    spatial_cov = sigma_sq * pm.gp.cov.Matern32(2, phi)
    cov_x = spatial_cov(orig_coords)
    kxx = matrix_inverse(stabilize(cov_x))
    kxxs = tt.dot(kxx, s)
    kss = spatial_cov(pred_coords, orig_coords)
    s_star = tt.dot(kss, kxxs)

    return t.function([sigma_sq, phi, s], [s_star])



if __name__ == '__main__':
    import argparse
    import pickle as pkl
    import pandas as pd

    parser = argparse.ArgumentParser(description='Fit Binomial GP timeseries model')
    parser.add_argument("data", nargs=1, type=str,
                        help="Input data file with (at least) columns consult_date, gi_mpc_selected, Easting, Northing")
    parser.add_argument("--iterations", "-i", type=int, default=[5000], nargs=1,
                        help="Number of MCMC iterations")
    args = parser.parse_args()

    data = pd.read_csv(args.data[0])
    data['consult_date'] = pd.to_datetime(data['consult_date'])
    week = pd.to_timedelta(7, unit='day')

    start = pd.to_datetime(['2019-12-02'])

    for i in range(9):
        end = start + week
        d = data[(start[0] <= data['consult_date']) & (data['consult_date'] < end[0])]

        d = d.groupby([d['Easting'], d['Northing']])
        aggr = d['gi_mpc_selected'].agg([['cases', np.sum], ['N', len]])
        aggr.reset_index(inplace=True)

        y = np.array(aggr['cases'])
        N = np.array(aggr['N'])
        coords = np.array(aggr[['Easting', 'Northing']], dtype='float32')

        # GIS
        poly = gp.read_file('/home/jewellcp/Documents/GIS/gadm36_GBR.gpkg').to_crs(epsg=27700)
        r = make_masked_raster(poly, 5000., crs='+init=epsg:27700')
        pred_points = raster2coords(r)

        sampler = logistic2D(y, N, coords / 1000., pred_points / 1000.)
        result = sampler(args.iterations[0], chains=1)
        with open(f"vom_dog_week{i}.pkl", 'wb') as f:
            pkl.dump(result, f)

        start = end

    sys.exit(0)
