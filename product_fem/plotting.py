import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from matplotlib.animation import FuncAnimation
from mpl_toolkits.axes_grid1 import make_axes_locatable

import numpy as np
import fenics as fx
from .functions import Control
from .transforms import sig_to_cov


# these all return Axes objects
def fenics_plot(f, **kwargs):
    assert isinstance(f, fx.Function)
    ax = plt.gca()
    ax = fx.plot(f).axes
    return ax

def covariance_plot(sig, **kwargs):
    assert isinstance(sig, fx.Function) and len(sig)==3
    return plot_ellipse_field(sig, **kwargs)
    
def control_plot(control, **kwargs):
    assert isinstance(control, Control)
    ax = []
    for m in control:
        if len(m) < 3:
            ax.append(fenics_plot(m))
        elif len(m) == 3:
            ax.append(covariance_plot(m))
    return ax

def plot(func, **kwargs):
    is_control = isinstance(func, Control)
    is_function = isinstance(func, fx.Function)
    ncols = len(func) if is_control else 1
    fig, ax = plt.subplots(1, ncols)
    
    if is_control:
        axs = control_plot(func, **kwargs)
        ax = tuple(axs)
    elif is_function and len(func) < 3:
        ax = fenics_plot(func, **kwargs)
        ax.figure = fig
    elif is_function and len(func) == 3:
        ax = covariance_plot(func, **kwargs)
        ax.figure = fig

    return fig, ax

def ellipse_patch(cov, x, y, ax, **kwargs):
    r = cov[0,1] / np.sqrt(cov[0,0] * cov[1,1])
    radius_x = np.sqrt(1 + r)
    radius_y = np.sqrt(1 - r)
    ellipse = Ellipse((0, 0), width=radius_x * 2, height=radius_y * 2,
                      facecolor='lightblue', alpha=0.5, edgecolor='blue', zorder=0)
    
    s = kwargs.get('s', 0.03)
    scale_x = np.sqrt(cov[0,0]) * s
    scale_y = np.sqrt(cov[1,1]) * s
    
    transf = transforms.Affine2D().rotate_deg(45).scale(scale_x, scale_y).translate(x, y)
    ellipse.set_transform(transf + ax.transData)
    return ellipse

def plot_ellipse_field(sig, ax, **kwargs):
    coords = sig.function_space().tabulate_dof_coordinates()[::3]
    cov = sig_to_cov(sig)
    ax.set_aspect('equal')
    for i, c in enumerate(cov):
        x, y = coords[i]
        ax.scatter(x, y, color='black', s=1)
        ax.add_patch(ellipse_patch(c, x, y, ax, **kwargs))
    return ax