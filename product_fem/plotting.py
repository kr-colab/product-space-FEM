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

def plot(func, ax, **kwargs):
    is_control = isinstance(func, Control)
    is_function = isinstance(func, fx.Function)
    ncols = len(func) if is_control else 1
    if ax is None:
        _, ax = plt.subplots(1, ncols)
    fig = ax.figure
    
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

def plot_peclets(mesh, peclets):
    xs = np.array([x for x, _ in peclets])
    ps = np.array([p for _, p in peclets])
    
    fig, ax = plt.subplots()

    lines = fx.plot(mesh)
    for line in lines:
        ax.add_line(line)
    s = ax.scatter(xs[:,0], xs[:,1], s=ps*100)

    plt.legend(*s.legend_elements("sizes", color="C0", num=6, func=lambda s: s/100), 
               bbox_to_anchor=(1, 1), title="Peclet numbers")
    return fig, ax

def animate_control(m, m_hats, save_as, duration=5, df=None, **kwargs):
    """
    Animation of a control object:
    `m`: a control
    `m_hats`: a list of objects that can be used to update m
    `save_as`: the output mp4 filename
    `duration`: duration of the animation, in seconds
    `df`: an additional data frame to be plotted below
    """
    m.update(m_hats[0])
    # fenics.plot seems to ONLY be able to plot to last axes
    layout = [[]]
    if df is not None:
        layout[0] += ["df"]
    layout[0] += ["ellipse", "vector"]
    fig = plt.figure(layout="constrained", dpi=150, figsize=(9, 3))
    axes = fig.subplot_mosaic(layout)

    s = axes["ellipse"].scatter([], [])
    q = axes["vector"].quiver(np.empty(49), np.empty(49), np.empty(49), np.empty(49))
    d = None
    if df is not None:
        d = df.plot(ax=axes["df"])

    # plot frame i
    def animate(i):
        if i % 10 == 0: print(f'animating frame {i} / {len(m_hats)}')
        # update to control at ith iteration
        m.update(m_hats[i])
        q.axes.clear()
        s.axes.clear()
        qu = fx.plot(m[0])
        ax = plot_ellipse_field(m[1], axes["ellipse"])
        out = [qu] + ax.get_children()
        if d is not None:
            d.clear()
            dp = df.plot(ax=d)
            sc = d.scatter([i for _ in df.columns], df.iloc[i].to_numpy())
            out += dp.get_children()
        return out

    interval = duration/len(m_hats) * 1000  # displayed in python
    anim = FuncAnimation(
            fig,
            animate,
            frames=len(m_hats),
            interval=interval,
            blit=True
    )
    
    writer = kwargs.get('writer', 'ffmpeg')
    fps = len(m_hats) / duration  # for saved file
    anim.save(save_as, writer='ffmpeg', fps=fps)
