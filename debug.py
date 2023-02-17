import os, sys, pickle
import numpy as np
import pandas as pd
import json

import product_fem as pf
from fenics import UnitSquareMesh, Mesh, FunctionSpace
import inference

cv_dir = "debug"

# load spatial and genetic data
spatial_data = pd.read_csv("data/nebria/stats.csv", index_col=0).rename(
        columns={"site_name": "name", "long": "x", "lat": "y"}
)
genetic_data = pd.read_csv("data/nebria/pairstats.csv", index_col=0).rename(
        columns={"loc1": "name1", "loc2": "name2", "dxy": "divergence"}
)
data = inference.SpatialDivergenceData(spatial_data, genetic_data)
data.normalise(min_xy=0.2, max_xy=0.8)

# boundary = data.boundary_fn(eps0=0.05, eps1=0.1)
divvec = data.genetic_data['divergence'].copy().to_numpy()
slocs = data.spatial_data.loc[:,('x', 'y')].copy().to_numpy()
mindiv = np.min(divvec)
def boundary(*args):
    return mindiv

mesh = UnitSquareMesh(4, 4)
V = FunctionSpace(mesh, 'CG', 1)
W = pf.ProductFunctionSpace(V)

print("hitting times")
eqn = pf.HittingTimes(W, boundary, epsilon=0.03)
control = eqn.control
print("loss functionals")
train_sd = pf.SpatialData(
        divvec,
        slocs,
        W,
)
train_loss = pf.LossFunctional(train_sd, control, {"l2": [100, 100.], "smoothing": [1, 1.]})
print("inv problem")
invp = pf.InverseProblem(eqn, train_loss)
options = {'gtol': 1e-8,
           'xrtol': 1e-8,
           'maxiter': 100}
print("optimize")
m_hats, losses, results = invp.optimize(control, 
                                        method='BFGS', 
                                        options=options)
print("solve")
control.update(m_hats[-1])
u_hat = eqn.solve()
print("DONE")

