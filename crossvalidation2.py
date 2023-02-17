# imports 
import os, sys, pickle
import numpy as np
import pandas as pd
import json

import product_fem as pf
from fenics import UnitSquareMesh, Mesh, FunctionSpace
import inference

usage = f"""
Usage:
    {sys.argv[0]} (json file with parameters in)
and the json file should include:
    "spatial_data", "genetic_data" : file paths
    "mesh" : either an XML file or a [width, height] of integers
    "folds": number of folds
    "regularization": {{ "l2": [a, b], "smoothing": [c, d] }}
    "boundary": {{ "epsilon": x, "eps0": y, "eps1": z }}
    "method": method for optimizing (default: BFGS)
    "options": additional options to the optimizer
"""

if len(sys.argv) != 2:
    print(usage)
    sys.exit()

# defaults
params = {
    "method": "BFGS",
    "options": {
        'gtol': 1e-8,
        'xrtol': 1e-8,
        'maxiter': 100
    },
    "boundary": None,
}

paramsfile = sys.argv[1]
outdir = os.path.dirname(paramsfile)
with open(paramsfile, 'r') as f:
    file_params = json.load(f)
params.update(file_params)

def pickle_dump(filename, obj):
    print(f'saving file {filename}')
    with open(filename, 'wb') as file:
        pickle.dump(obj, file, protocol=pickle.HIGHEST_PROTOCOL)

# load spatial and genetic data
spatial_data = pd.read_csv(os.path.join(outdir, params['spatial_data']), index_col=0).rename(
        columns={"site_name": "name", "long": "x", "lat": "y"}
)
genetic_data = pd.read_csv(os.path.join(outdir, params['genetic_data']), index_col=0).rename(
        columns={"loc1": "name1", "loc2": "name2", "dxy": "divergence"}
)
data = inference.SpatialDivergenceData(spatial_data, genetic_data)
data.normalise(min_xy=0.2, max_xy=0.8)

# optionally determine the boundary parameters
if params['boundary'] is None:
    bdry_params = data.choose_epsilons()
    params['boundary'] = {
            "epsilon": bdry_params['eps0'],
            "eps0": bdry_params['eps0'],
            "eps1": bdry_params['eps1']
    }

boundary = data.boundary_fn(eps0=params['boundary']['eps0'], eps1=params['boundary']['eps1'])

if len(params['mesh']) == 1 and isinstance(params['mesh'], str):
    mesh = Mesh(params['mesh'])
elif len(params['mesh']) == 2:
    mesh = UnitSquareMesh(*params['mesh'])
else:
    print(f"mesh must be an xml file name or (width, height), got {params['mesh']}")
    sys.exit()
V = FunctionSpace(mesh, 'CG', 1)
W = pf.ProductFunctionSpace(V)

test_errors = []
for fold, (train, test) in enumerate(data.split(k=params["folds"])):
    print(f"Doing fold {fold} with {params['method']}...")
    print("\t".join(["", "total_loss", "error", "regularization", "smoothing"]))
    eqn = pf.HittingTimes(W, boundary, epsilon=params['boundary']['epsilon'])
    control = eqn.control
    # loss functionals
    train_sd = pf.SpatialData(
            train.genetic_data['divergence'].to_numpy(),
            train.spatial_data.loc[:,('x', 'y')].to_numpy(),
            W,
    )
    train_loss = pf.LossFunctional(train_sd, control, params['regularization'])
    
    test_sd = pf.SpatialData(
            test.genetic_data['divergence'].to_numpy(),
            test.spatial_data.loc[:,('x', 'y')].to_numpy(),
            W,
    )
    test_loss = pf.LossFunctional(test_sd, control, params['regularization'])
    
    invp = pf.InverseProblem(eqn, train_loss)
    m_hats, losses, results = invp.optimize(
            control, 
            method=params['method'],
            options=params['options'],
    )
    
    # test set error
    control.update(m_hats[-1])
    u_hat = eqn.solve()
    test_errors.append(test_loss.l2_error(u_hat))
    print(f"Done: test error {test_errors[-1]}")
    
    # pickle results
    pickle_dump(os.path.join(outdir, f'fold_{fold}_m_hats.pkl'), m_hats)
    pickle_dump(os.path.join(outdir, f'fold_{fold}_losses.pkl'), losses)
    pickle_dump(os.path.join(outdir, f'fold_{fold}_results.pkl'), results)

pickle_dump(os.path.join(outdir, f'test_errors.pkl'), np.array(test_errors))
