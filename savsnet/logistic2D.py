"""Functions to perform 2D logistic kriging"""

import numpy as np
import theano as t
import theano.tensor as tt
from theano.tensor.nlinalg import matrix_inverse
import pymc3 as pm
from pymc3.gp.util import stabilize
from pymc3.gp.util import kmeans_inducing_points

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



class Logistic2D(pm.Model):
    def __init__(self, y, coords, n_knots, name='', model=None):
        super().__init__(name, model)

        alpha = pm.Normal('alpha', mu=0., sd=1.)
        sigma_sq = pm.Gamma('sigma_sq', 1., 1.)
        phi = pm.Gamma('phi', 1., 1.)

        # Inducing points
        knots = kmeans_inducing_points(n_knots, coords)
        spatial_cov = sigma_sq * pm.gp.cov.Matern32(2, phi)
        spatial_gp = pm.gp.Latent(cov_func=spatial_cov)
        inducing = spatial_gp.prior('k', X=knots)

        # Project inducing points onto real points
        cov_kk = spatial_cov(knots)
        kk_inducing = tt.dot(matrix_inverse(stabilize(cov_kk)), inducing)
        cov_sk = spatial_cov(coords, knots)
        s = tt.dot(cov_sk, kk_inducing)

        eta = alpha + s
        y_rv = pm.Bernoulli('y', pm.invlogit(eta), observed=y)


    def sample(self, *args, **kwargs):
        with self:
            return pm.sample(*args, **kwargs)


if __name__ == '__main__':

    import argparse
    import pickle as pkl
    import pandas as pd

    parser = argparse.ArgumentParser(description='Fit Binomial GP timeseries model')
    parser.add_argument("data", nargs=1, type=str,
                        help="Input data file with (at least) columns Date, Species, Consult_reason")
    args = parser.parse_args()

    data = pd.read_csv(args.data[0])

    data = data[('2020-01-01' <= data['consult_date']) & (data['consult_date'] < '2020-01-07')]
    print(data.columns)

    y = np.array(data['gi_mpc_selected']).astype(np.int)
    coords = np.array(data[['Easting','Northing']], dtype='float32')

    model = Logistic2D(y, coords, n_knots=300)
    trace = model.sample(5000, chains=1)
    with model:
        pm.traceplot(trace)
    with open('vom_dog_post.pkl','wb') as f:
        pkl.dump(trace, f)



