# Raster generator
import numpy as np
import rasterio
import rasterio.mask
from rasterio.io import MemoryFile
from rasterio.transform import from_origin
from rasterio.plot import show
import matplotlib.pyplot as plt

def make_masked_raster(polygon, resolution, bands=1, all_touched=False, 
                       nodata=-9999., crs='+init=epsg:27700', filename=None):
    """Generates a raster with points outside poly set to nodata. Pixels
    are set to be of dimension res x res"""
    x = np.arange(polygon.bounds['minx'].values, polygon.bounds['maxx'].values, resolution)
    y = np.arange(polygon.bounds['miny'].values, polygon.bounds['maxy'].values, resolution)
    X,Y = np.meshgrid(x,y)
    Z = np.ones(shape=X.shape)
    transform = from_origin(x[0] - resolution/2, y[-1]+resolution/2, resolution, resolution)

    raster_args = {'driver': 'GTiff',
                   'height': Z.shape[0],
                   'width': Z.shape[1],
                   'count': bands,
                   'dtype': Z.dtype,
                   'crs': crs,
                   'transform': transform,
                   'nodata': nodata}

    if filename is None:
        memfile = MemoryFile()
        raster = memfile.open(**raster_args)
    else:
        raster = rasterio.open(filename, 'w+', **raster_args)
    
    for i in range(bands):
        raster.write(Z, i+1)

    mask = rasterio.mask.mask(raster, polygon.geometry, crop=False, all_touched=all_touched)
    raster.write_mask(mask[0][0])
    return raster


def raster2coords(raster):
    x, y = np.meshgrid(np.arange(raster.shape[0]), np.arange(raster.shape[1]))
    coords = np.array([raster.xy(x, y) for x, y in zip(x.ravel(), y.ravel())])
    return coords[raster.read_masks(1).T.ravel().astype(np.bool), :]


def fill_raster(data, raster, band=1):
    r = raster.read(1, masked=True).T.flatten()
    r[~r.mask.flatten()] = data
    r = r.reshape([raster.shape[1], raster.shape[0]]).T
    raster.write(r, band)


def gen_raster(posterior, poly, filename=None,
               summary=lambda x: np.mean(x, axis=0)):
    raster = make_masked_raster(polygon=poly, resolution=5000.,
                                bands=1, filename=filename)

    fill_raster(summary(posterior),
                raster, 1)

    raster.update_tags(1, surface='s_star')
    raster.close()

    return raster


def plot_raster(ax, raster, transform, title=None, z_range=[None, None], log=False, alpha=1.0):

    show(raster, transform=transform, vmin=z_range[0], vmax=z_range[1], ax=ax, cmap='coolwarm', alpha=alpha)
    ax.set_title(title)
    ax.axis('off')
    cb = plt.colorbar(ax.images[1], ax=ax, shrink=0.7)
    if log is True:
        ticks = cb.get_ticks()
        cb.set_ticks(ticks)
        cb.set_ticklabels(np.round(np.exp(ticks), 1))
    return ax, cb
