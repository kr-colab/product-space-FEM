# imports 
import os, sys, pickle
import numpy as np
import pandas as pd
import product_fem as pf
from sklearn.model_selection import KFold
from fenics import UnitSquareMesh, Mesh, FunctionSpace

import inference


k = int(sys.argv[1]) # number of xval folds
penalty = float(sys.argv[2]) # regularization penalty

cv_dir = f'data/nebria/crossval_penalty_{penalty}/'

try:
    os.mkdir(cv_dir)
    os.mkdir(cv_dir + f'{k}_fold/')
except OSError as error:
    print(error)

# load spatial and genetic data
spatial_data = pd.read_csv("data/nebria/stats.csv", index_col=0).rename(
        columns={"site_name": "name", "long": "x", "lat": "y"}
)
spatial_data['name'] = [
        sites[n] for n in spatial_data['site_name']
]
genetic_data = pd.read_csv("data/nebria/pairstats.csv", index_col=0).rename(
        columns={"loc1": "name1", "loc2": "name2", "dxy": "divergence"}
)
data = inference.SpatialDivergenceData(spatial_data, genetic_data)
data.normalise(min_xy=0.2, max_xy=0.8)
bdry_params = data.choose_epsilons()

mesh = UnitSquareMesh(4,4)
# mesh = Mesh('data/nebria/mesh.xml')
V = FunctionSpace(mesh, 'CG', 1)
W = pf.ProductFunctionSpace(V)

for fold, (train, test) in enumerate(data.split(k=k)):
    eqn = pf.HittingTimes(W, boundary, epsilon=0.03)
    control = eqn.control
    # loss functionals
    reg = {'l2': [100*penalty, 100*penalty], 'smoothing': [penalty, penalty]}
    train_sd = pf.SpatialData(
            train.genetic_data['divergence'].to_numpy(),
            train.spatial_data.loc[:,('x', 'y')].to_numpy(),
            W,
    )
    train_loss = pf.LossFunctional(train_sd, control, reg)
    
    test_sd = pf.SpatialData(
            test.genetic_data['divergence'].to_numpy(),
            test.spatial_data.loc[:,('x', 'y')].to_numpy(),
            W,
    )
    test_loss = pf.LossFunctional(test_sd, control, reg)
    
    invp = pf.InverseProblem(eqn, train_loss)
    options = {'ftol': 1e-8, 
               'gtol': 1e-8, 
               'maxcor': 15,
               'maxiter': 100}
    m_hats, losses, results = invp.optimize(control, 
                                            method='L-BFGS-B', 
                                            options=options)
    
    # test set error
    control.update(m_hats[-1])
    u_hat = eqn.solve()
    test_errors[fold] = test_loss.l2_error(u_hat)
    
    # pickle results
    pickle_dump(cv_dir + f'{k}_fold/fold_{fold}_m_hats.pkl', m_hats)
    pickle_dump(cv_dir + f'{k}_fold/fold_{fold}_losses.pkl', losses)
    pickle_dump(cv_dir + f'{k}_fold/fold_{fold}_results.pkl', results)

pickle_dump(cv_dir + f'{k}_fold/test_errors.pkl', test_errors)
