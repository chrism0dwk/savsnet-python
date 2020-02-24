""" Performs logistic Kriging """

import sys
import argparse
import pickle as pkl
import numpy as np
import pandas as pd
import geopandas as gp
import pymc3 as pm

from savsnet.logistic2D import logistic2D
from savsnet.gis_util import gen_raster, make_masked_raster, raster2coords


def get_mean(x):
    return np.mean(x, axis=0)


def get_exceed(x):
    p = np.sum(x > 0., axis=0)/x.shape[0]
    return (p < 0.05) | (p > 0.95)


if __name__ == '__main__':
    import argparse
    import pickle as pkl
    import pandas as pd

    parser = argparse.ArgumentParser(description='Fit Binomial GP timeseries model')
    parser.add_argument("data", nargs=1, type=str,
                        help="Input data file with (at least) columns consult_date, gi_mpc_selected, Easting, Northing")
    parser.add_argument("--iterations", "-i", type=int, default=[5000], nargs=1,
                        help="Number of MCMC iterations")
    parser.add_argument("--startdate", "-s", type=np.datetime64, nargs=1, help="Start date")
    parser.add_argument("--period", "-d", type=int, default=[7], nargs=1, help="Period in days")
    parser.add_argument("--polygon", "-p", type=str, nargs=1, help="Polygon file")
    args = parser.parse_args()

    data = pd.read_csv(args.data[0])
    data['consult_date'] = pd.to_datetime(data['consult_date'])
    period = pd.to_timedelta(args.period[0], unit='day')

    start = args.startdate[0]
    end = start + period

    d = data[(start <= data['consult_date']) & (data['consult_date'] < end)]
    
    d = d.groupby([d['person_easting'], d['person_northing']])
    aggr = d['gi_mpc_selected'].agg([['case', lambda x: np.sum(x) > 0]])
    aggr.reset_index(inplace=True)
    
    y = np.array(aggr['case'])
    coords = np.array(aggr[['person_easting', 'person_northing']], dtype='float32')
    knots = pm.gp.util.kmeans_inducing_points(300, coords)
    
    # GIS
    poly = gp.read_file(args.polygon[0]).to_crs(epsg=27700)
    r = make_masked_raster(poly, 5000., crs='+init=epsg:27700')
    pred_points = raster2coords(r)
    
    sampler = logistic2D(y, coords=coords/1000., knots=knots/1000., pred_coords=pred_points/1000.)
    result = sampler(args.iterations[0], chains=1)
    with open(f"posterior_week{start}.pkl", 'wb') as f:
        pkl.dump(result, f)
        
    gen_raster(result['posterior']['s_star'], poly, filename=f"raster_{start}.tiff",
               summary=[['mean', get_mean], ['exceed', get_exceed]])
    sys.exit(0)
