import sys, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if not len(sys.argv) == 2:
    print(f"""
    Usage:
        python {sys.argv[0]} <input>.spstats.csv
    """)
    sys.exit()

sp_file = sys.argv[1]
if not os.path.isfile(sp_file):
    raise ValueError(f"File {sp_file} does not exist.")

basename = sp_file.replace(".spstats.csv", "")
outfile = f"{basename}.spstats.png"

sp = pd.read_csv(sp_file)
xvals = np.unique(sp['x'])
yvals = np.unique(sp['y'])
nr, nc = len(xvals), len(yvals)

def plot_heatmap(ax, n):
    X = sp['x'].to_numpy().reshape((nr, nc))
    Y = sp['y'].to_numpy().reshape((nr, nc))
    Z = sp[n].to_numpy().reshape((nr, nc))
    # return ax.pcolormesh(X, Y, Z)
    return ax.contourf(X, Y, Z)

fig, axes = plt.subplots(2, 2, figsize=(8, 8))
for (ax, n) in zip(axes.flatten(), sp.columns[2:]):
    im = plot_heatmap(ax, n)
    fig.colorbar(im, ax=ax)
    ax.set_title(n)
    ax.set_aspect(1.0)

fig.savefig(outfile)
print(f"Figure saved to {outfile}")
