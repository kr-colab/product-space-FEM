import numpy as np

import matplotlib
import matplotlib.pyplot as plt

from PIL import Image
from scipy import interpolate

def floats_to_rgb(x, min=-1, max=1):
    """
    Translates floats in [min, max) to valid RBG integers, in [0, 255].
    Values are clamped to min and max.
    """
    x = np.array(x)
    out = 256 * (x - min) / (max - min)
    out[out < 0] = 0
    out[out > 255] = 255
    assert np.min(out) >= 0 and np.max(out) <= 255
    return out.astype("uint8")


def rgb_to_floats(x, min=-1, max=1):
    """
    The "inverse" of floats_to_rgb.
    Note the denominator is 255, mirroring SLiM.
    """
    x = np.array(x, dtype='float')
    out = min + (max - min) * x / 255
    return out


def xyz_to_array(x, y, z):
    """
    Given arrays of regularly-spaced x and y values, with z[i] corresponding to the value at
    (x[i], y[i]), return the triple
      xx, yy, zz
    where zz is just z, reshaped, and xx and yy are such that zz[i, j] corresponds to (xx[i], yy[j]).
    """
    xx = np.unique(x)
    yy = np.unique(y)
    nr, nc = len(yy), len(xx)
    zz = np.zeros((nr, nc))
    ii = np.searchsorted(yy, y)
    jj = np.searchsorted(xx, x)
    for i, j, zval in zip(ii, jj, z):
        zz[i, j] = zval
    return xx, yy, zz


def xyz_to_function(x, y, z, **kwargs):
    """
    Given arrays of regularly-spaced x and y values, with z[i] corresponding to the value at
    (x[i], y[i]), return the function that linearly interpolates the values of z to other
    values of x and y. Will extrapolate outside of the given domain.
    """
    xx, yy, zz = xyz_to_array(x, y, z)
    return interpolate.RegularGridInterpolator((xx, yy), zz.T, **kwargs, fill_value=None, bounds_error=False)


def slope_layers(height, f=None):
    """
    Given an (n + 1, m + 1)-layer ``height``, return the (n, m, 2) layer that has the
    x- and y-components of the slope of ``height``, as follows: if the heights surrounding
    a square are
    > c d
    > a b
    then we compute the slope there as
    > ( (b - a)/2 + (d - c)/2, (c - a)/2 + (d - b)/2 )
    """
    if f is None:
        f = (1, 1)
    dx = f[0] * np.diff(height, axis=1)
    dy = f[1] * (-1) * np.diff(height, axis=0) # -1 because images have (0,0) in lower-left
    return np.stack([
            (dx[1:,:] + dx[:-1,:]) / 2,
            (dy[:,1:] + dy[:,:-1]) / 2
        ], axis=-1)


def function_height(f, nrow, ncol, xrange, yrange, **kwargs):
    """
    Return a (nrow x ncol) numpy array with values given by
    >  f(x[i], y[j])
    where x ranges from xrange[0] to xrange[1].
    and likewise for y, defaulting to both being in [0, 1).
    """
    xvals = np.linspace(xrange[0], xrange[1], nrow)
    yvals = np.linspace(yrange[0], yrange[1], ncol)
    x = np.repeat([xvals], ncol, axis=1)
    y = np.repeat([yvals], nrow, axis=0).flatten()
    out = f(x, y, **kwargs)
    out.shape = (nrow, ncol)
    return(out)


def bump_height(nrow, ncol, width=None, center=None):
    """
    Return a (nrow x ncol) numpy array with values given by the bump function
    >  exp(- 1 / (1 - r^2) ),
    where
    >  r = sqrt( (x/width[0])^2 + (y/width[1])^2 )
    for -width[0] < x < width[0] and -width[1] , y < width[1].
    """
    if center is None:
        center = np.array([(nrow - 1) / 2, (ncol - 1) / 2])
    if width is None:
        width = center
    x = np.repeat([np.arange(nrow) - center[0]], ncol, axis=1)
    y = np.repeat([np.arange(ncol) - center[1]], nrow, axis=0).flatten()
    z = np.maximum(0.05, 1 - ((x/width[0]) ** 2 + (y/width[1]) ** 2))
    out = np.exp(- 1 / z)
    out[out < 0] = 0.0
    out.shape = (nrow, ncol)
    return(out)


def gaussian_height(nrow, ncol, width=None, center=None):
    """
    Return a (nrow x ncol) numpy array with values given by the gaussian density
    >  exp(- r^2 / 2 ),
    where
    >  r = sqrt( (x/width[0])^2 + (y/width[1])^2 )
    for -width[0] < x < width[0] and -width[1] , y < width[1].
    """
    if center is None:
        center = np.array([(nrow - 1) / 2, (ncol - 1) / 2])
    if width is None:
        width = center
    x = np.repeat([np.arange(nrow) - center[0]], ncol, axis=1)
    y = np.repeat([np.arange(ncol) - center[1]], nrow, axis=0).flatten()
    z = (x/width[0]) ** 2 + (y/width[1]) ** 2
    out = np.exp(- z/2)
    out[out < 0] = 0.0
    out.shape = (nrow, ncol)
    return(out)


def saddle_height(nrow, ncol, width=None, center=None):
    """
    Return a (nrow x ncol) numpy array with values given by the gaussian density
    >  exp( - ((x/width[0])^2 - (y/width[1])^2) / 2 ),
    for -width[0] < x < width[0] and -width[1] , y < width[1].
    """
    if center is None:
        center = np.array([(nrow - 1) / 2, (ncol - 1) / 2])
    if width is None:
        width = center
    x = np.repeat([np.arange(nrow) - center[0]], ncol, axis=1)
    y = np.repeat([np.arange(ncol) - center[1]], nrow, axis=0).flatten()
    z = (x/width[0]) ** 2 - (y/width[1]) ** 2
    out = np.exp(- z/2)
    out[out < 0] = 0.0
    out.shape = (nrow, ncol)
    return(out)


def mountain_height(nrow, ncol, slope=None, center=None):
    """
    Return a (nrow x ncol) numpy array that has value 1.0 at ``center``
    and declines linearly with ``slope`` to zero.
    """
    if center is None:
        center = np.array([(nrow - 1) / 2, (ncol - 1) / 2])
    if slope is None:
        # put 0.0 at the further edge of the smaller dimension
        slope = 1.0 / min(max(ncol - center[0], center[0]),
                          max(nrow - center[1], center[1]))
    x = np.repeat([np.arange(nrow) - center[0]], ncol, axis=1)
    y = np.repeat([np.arange(ncol) - center[1]], nrow, axis=0).flatten()
    dist = np.sqrt(x ** 2 + y ** 2)
    out = 1.0 - dist * slope
    out[out < 0] = 0.0
    out.shape = (nrow, ncol)
    return(out)


def make_slope_rgb(nrow, ncol, height_fn, f=None, **kwargs):
    if 'center' in kwargs:
        center = kwargs['center']
        kwargs['center'] = [center[0] * (1 + 1/nrow), center[1] * (1 + 1/ncol)]
    height = height_fn(
            nrow + 1,
            ncol + 1,
            **kwargs)
    slope = slope_layers(height, f=f)
    out = np.concatenate([
            floats_to_rgb(slope / np.max(np.abs(slope)), min=-1, max=1),
            np.full((nrow, ncol, 1), 0, dtype='uint8'),
            np.full((nrow, ncol, 1), 255, dtype='uint8')
        ], axis=-1)
    return out.astype("uint8")


def make_sigma_rgb(nrow, ncol, height_fn, **kwargs):
    # uses same sigma in x and y direction; no correlation
    # do it by averaging the +1 grid to agree with slope
    if 'center' in kwargs:
        center = kwargs['center']
        kwargs['center'] = [center[0] * (1 + 1/nrow), center[1] * (1 + 1/ncol)]
    height = height_fn(
            nrow + 1,
            ncol + 1,
            **kwargs)
    sigma = floats_to_rgb((height[:-1,:-1]
                           + height[1:,:-1]
                           + height[:-1,1:]
                           + height[1:,1:])[:,:,np.newaxis] / 4,
                          min=-1, max=1)
    zero = floats_to_rgb(np.full((nrow, ncol, 1), 0), min=-1, max=1)
    out = np.concatenate([
            sigma,
            sigma,
            zero,
            np.full((nrow, ncol, 1), 255, dtype='uint8')
        ], axis=-1)
    return out.astype("uint8")


def mountain_slope(nrow, ncol, slope=None, center=None):
    """
    Make a (nrow, ncol, 4) RGBA array with layers corresponding to
    (downslope bias x, downslope bias y, 0, 255)
    on a "stratovolcano" (linear cone).
    """
    if center is None:
        center = np.array([nrow / 2, ncol / 2])
    return make_slope_rgb(
            nrow, ncol, mountain_height,
            slope=slope, center=center)

def mountain_sigma(nrow, ncol, slope=None, center=None):
    """
    Make a (nrow, ncol, 4) RGBA array with layers corresponding to
    (sigma x, sigma y, 0, 255)
    on a "stratovolcano" (linear cone).
    """
    if center is None:
        center = np.array([nrow / 2, ncol / 2])
    return make_sigma_rgb(
            nrow, ncol, mountain_height,
            slope=slope, center=center)

def saddle_slope(nrow, ncol, width=None, center=None):
    """
    Make a (nrow, ncol, 4) RGBA array with layers corresponding to
    (downslope bias x, downslope bias y, 0, 255)
    on the saddle exp(-(x^2-y^2)/2)
    """
    if center is None:
        center = np.array([nrow / 2, ncol / 2])
    return make_slope_rgb(
            nrow, ncol, saddle_height,
            width=width, center=center)

def gaussian_slope(nrow, ncol, width=None, center=None):
    """
    Make a (nrow, ncol, 4) RGBA array with layers corresponding to
    (downslope bias x, downslope bias y, 0, 255)
    on a "butte" (a bump function).
    """
    if center is None:
        center = np.array([nrow / 2, ncol / 2])
    return make_slope_rgb(
            nrow, ncol, gaussian_height,
            width=width, center=center)

def butte_slope(nrow, ncol, width=None, center=None):
    """
    Make a (nrow, ncol, 4) RGBA array with layers corresponding to
    (downslope bias x, downslope bias y, 0, 255)
    on a "butte" (a bump function).
    """
    if center is None:
        center = np.array([nrow / 2, ncol / 2])
    return make_slope_rgb(
            nrow, ncol, bump_height,
            width=width, center=center)

def butte_sigma(nrow, ncol, width=None, center=None):
    """
    Make a (nrow, ncol, 4) RGBA array with layers corresponding to
    (sigma x, sigma y, 0, 255)
    on a "butte" (a bump function).
    """
    if center is None:
        center = np.array([nrow / 2, ncol / 2])
    return make_sigma_rgb(
            nrow, ncol, bump_height,
            width=width, center=center)
