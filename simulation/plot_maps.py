import sys, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import maps

if not len(sys.argv) >= 2:
    print(f"""
    Usage:
        python {sys.argv[0]} <input>.spstats.csv [more files]
    """)
    sys.exit()

for sp_file in sys.argv[1:]:
    if not os.path.isfile(sp_file):
        raise ValueError(f"File {sp_file} does not exist.")

    basename = sp_file.replace(".spstats.csv", "")
    outfile = f"{basename}.spstats.png"

    sp = pd.read_csv(sp_file)

    def plot_heatmap(ax, n):
        xvals, yvals, Z = maps.xyz_to_array(sp['x'], sp['y'], sp[n])
        X = sp['x'].to_numpy().reshape(Z.shape)
        Y = sp['y'].to_numpy().reshape(Z.shape)
        # return ax.pcolormesh(X, Y, Z)
        return ax.contourf(X, Y, Z)

    plot_cols = ["density", "fecundity", "mortality", "establishment"]
    fig, axes = plt.subplots(2, 2, figsize=(15, 8))
    for (ax, n) in zip(axes.flatten(), plot_cols):
        im = plot_heatmap(ax, n)
        fig.colorbar(im, ax=ax)
        ax.set_title(n)
        ax.set_aspect(1.0)

    fig.savefig(outfile)
    print(f"Figure saved to {outfile}")
