"""Generate rasters for logistic 2D"""

import pickle as pkl
import geopandas as gp
from savsnet.gis_util import gen_raster


def make_raster(s_star, filename):
    poly = gp.read_file('/home/jewellcp/Documents/GIS/gadm36_GBR.gpkg').to_crs(epsg=27700)
    gen_raster(s_star, poly, filename=filename)


if __name__ == '__main__':

    with open('../../vom_dog_post.pkl', 'rb') as f:
        trace = pkl.load(f)

    make_raster(trace['posterior']['s_star'], 'vom_dog_2020-01-01.tiff')
