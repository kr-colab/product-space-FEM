import os, sys, pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['figure.dpi'] = 150

import product_fem as pf
import fenics
import inference

usage = f"""
Usage:
    {sys.argv[0]} (pickle file with results in)
where the pickle file is as produced by crossvalidation.py
"""

if len(sys.argv) != 2:
    print(usage)
    sys.exit()

results_file = sys.argv[1]
outdir = os.path.dirname(results_file)
with open(results_file, 'rb') as f:
    results = pickle.load(f)

params = results['params']

losses = [pd.DataFrame(results[fold]['losses']) for fold in range(params['folds'])]
test_errors = np.array([results[fold]['test_error'] for fold in range(params['folds'])])

# loss history
lossfile = os.path.join(outdir, "losses_history.png")
fig, axes = plt.subplots(params['folds'], 2, figsize=(5, params['folds'] * 2))
for k, (ax, df) in enumerate(zip(axes[:,0], losses)):
    df.plot(ax=ax, legend=(k == 0))
    ax.set_ylabel("loss")
    ax.set_title(f"fold {k}")

ax.set_xlabel("optimize() iteration")

for k, (ax, df) in enumerate(zip(axes[:,1], losses)):
    df.plot(ax=ax, legend=(k == 0), ylim=(0, 1.2 * max(df.iloc[int(df.shape[0]/2):,:].max())))
    ax.set_ylabel("loss")
    ax.set_title(f"fold {k}")

ax.set_xlabel("optimize() iteration")
plt.tight_layout()
plt.savefig(lossfile)

### BEGIN STUFF COPIED FROM crossvalidation.py (it doesn't pickle)
# need this to plot the solution
spatial_data = pd.read_csv(os.path.join(outdir, params['spatial_data']), index_col=0).rename(
        columns={"site_name": "name", "long": "x", "lat": "y"}
)
genetic_data = pd.read_csv(os.path.join(outdir, params['genetic_data']), index_col=0).rename(
        columns={"loc1": "name1", "loc2": "name2", "dxy": "divergence"}
)
data = inference.SpatialDivergenceData(spatial_data, genetic_data)
data.normalise(min_xy=0.2, max_xy=0.8)

boundary = data.boundary_fn(eps0=params['boundary']['eps0'], eps1=params['boundary']['eps1'])

if len(params['mesh']) == 1 and isinstance(params['mesh'], str):
    mesh = fenics.Mesh(params['mesh'])
elif len(params['mesh']) == 2:
    mesh = fenics.UnitSquareMesh(*params['mesh'])
else:
    print(f"mesh must be an xml file name or (width, height), got {params['mesh']}")
    sys.exit()
V = fenics.FunctionSpace(mesh, 'CG', 1)
W = pf.ProductFunctionSpace(V)

eqn = pf.HittingTimes(W, boundary, epsilon=params['boundary']['epsilon'])
control = eqn.control
### END STUFF COPIED FROM crossvalidation.py

# solutions
solnfile = os.path.join(outdir, "solutions.png")
fig, axes = plt.subplots(params['folds'], 3, figsize=(8, 2 * params['folds']))
for fold, axs in enumerate(axes):
    m_hats = results[fold]['m_hats']
    control.update(m_hats[-1])
    u_hat = eqn.solve()
    eqn.plot_control(axs=axs[:2])
    u_hat.plot((0.5, 0.5), ax=axs[2])

plt.tight_layout()
plt.savefig(solnfile)

for fold in range(params['folds']):
    animfile = os.path.join(outdir, f"history_{fold}.mp4")
    pf.animate_control(
            control,
            results[fold]['m_hats'],
            save_as=animfile,
            df=losses[fold],
    )
