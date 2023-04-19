#!/usr/bin/python3
import numpy as np
import PIL.Image as Image

from map_utils import *

# format for bias layers is:
#  RGB = (bias_x, bias_y, unused)
# format for covariance layers is:
#  RGB = (sigma_x, sigma_y, correlation)
# format for habitat (monochrome) is:
#  K = carrying capacity

for (h, w, name) in [
        (20 * 20, 20 * 30, ""),
        (20 * 2, 20 * 30, "skinny_"),
    ]:

    # DEFAULT (constant) MAPS
    # all ones
    one = np.full((h, w, 4), 1.0)
    rgb = floats_to_rgb(one, min=0, max=1)
    im = Image.fromarray(rgb)
    im.save(f"{name}one.png", mode="L")
    im = Image.fromarray(rgb[:,:,:2])
    im.save(f"{name}one_bw.png", mode="L")

    # all ones, b&w

    # the identity map for sigma
    ident = np.full((h, w, 4), 1.0)
    ident[:,:,2] = 0.0
    rgb = floats_to_rgb(ident, min=-1, max=1)
    im = Image.fromarray(rgb)
    im.save(f"{name}identity.png", mode="L")

    # MOUNTAIN: downhill on a stratovolcano
    rgb = mountain_slope(h, w)
    im = Image.fromarray(rgb)
    im.save(f"{name}mountain_bias.png")

    rgb = mountain_sigma(h, w)
    im = Image.fromarray(rgb)
    im.save(f"{name}mountain_sigma.png")

    height = floats_to_rgb(mountain_height(h, w), min=0, max=1)
    im = Image.fromarray(height)
    im.save(f"{name}mountain_height.png", mode="L")

    # DOWNHILL: downhill on the gaussian density
    rgb = gaussian_slope(h, w)
    im = Image.fromarray(rgb)
    im.save(f"{name}gaussian_bias.png")

    # SADDLE: saddlewise on the gaussian density
    rgb = saddle_slope(h, w)
    im = Image.fromarray(rgb)
    im.save(f"{name}saddle_bias.png")

    x = saddle_height(h, w)
    x -= np.min(x)
    x /= np.max(x)
    height = floats_to_rgb(x, min=0, max=1)
    im = Image.fromarray(height)
    im.save(f"{name}saddle_height.png", mode="L")

    # BUTTE: downhill on a bump function
    rgb = butte_slope(h, w)
    im = Image.fromarray(rgb)
    im.save(f"{name}butte_bias.png")

    rgb = butte_sigma(h, w)
    im = Image.fromarray(rgb)
    im.save(f"{name}butte_sigma.png")

    x = bump_height(h, w)
    x /= np.max(x)
    height = floats_to_rgb(x, min=0, max=1)
    im = Image.fromarray(height)
    im.save(f"{name}butte_height.png", mode="L")
