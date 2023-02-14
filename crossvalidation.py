import os, sys, pickle
import numpy as np
import pandas as pd
import product_fem as pf
from sklearn.model_selection import KFold
from fenics import UnitSquareMesh, Mesh, FunctionSpace


k = int(sys.argv[1]) # number of xval folds
penalty = float(sys.argv[2]) # regularization penalty

cv_dir = f'data/nebria/crossval_penalty_{penalty}/'

try:
    os.mkdir(cv_dir)
    os.mkdir(cv_dir + f'{k}_fold/')
except OSError as error:
    print(error)

# load spatial and genetic data
spatial_data = pd.read_csv("data/nebria/stats.csv", index_col=0)
genetic_data = pd.read_csv("data/nebria/pairstats.csv", index_col=0)

pair_names = genetic_data[['loc1', 'loc2']].to_numpy()
site_names = np.unique(genetic_data['loc1'])
def slice_pairwise_data(site_idxs):
    sites = site_names[site_idxs]
    return genetic_data[[l1 in sites and l2 in sites for l1, l2 in pair_names]]

def pickle_dump(filename, obj):
    print(f'saving file {filename}')
    with open(filename, 'wb') as file:
        pickle.dump(obj, file, protocol=pickle.HIGHEST_PROTOCOL)

def normalize_coords(longlats, lower, upper, ref_longlats):
        ncoords = np.zeros_like(longlats)
        longs, lats = longlats.T
        ref_long = ref_longlats[:,0]
        ref_lat = ref_longlats[:,1]
        longmin, longmax = ref_long.min(), ref_long.max()
        latmin, latmax = ref_lat.min(), ref_lat.max()
        
        # first do longs
        nlongs = (longs - longmin) / (longmax - longmin)
        nlongs *= upper - lower
        nlongs += lower
        # next do lats
        nlats = (lats - latmin) / (latmax - latmin)
        nlats *= upper - lower
        nlats += lower
        
        ncoords[:,0] = nlongs
        ncoords[:,1] = nlats
        return ncoords

# pairwise coordinates needed for kernel density estimate
def coords_to_pairs(points):
    n = len(points)
    N = int(n * (n + 1) / 2)
    xs = np.zeros((N, 2))
    ys = np.zeros((N, 2))
    x1, y1, x2, y2 = [], [], [], []
    for i in range(n):
        for j in range(i, n):
            x1.append(points[:,0][i])
            y1.append(points[:,1][i])
            x2.append(points[:,0][j])
            y2.append(points[:,1][j])

    xs[:,0] = x1
    xs[:,1] = y1
    ys[:,0] = x2
    ys[:,1] = y2
    return xs, ys

mesh = UnitSquareMesh(4,4)
# mesh = Mesh('data/nebria/mesh.xml')
V = FunctionSpace(mesh, 'CG', 1)
W = pf.ProductFunctionSpace(V)

# k-fold crossvalidation
kf = KFold(k, shuffle=True, random_state=2357)
fold = 1 # current xval fold
test_errors = np.zeros(k)
for train, test in kf.split(site_names):
    # spatial data
    train_spatial_data = spatial_data.iloc[train]
    test_spatial_data = spatial_data.iloc[test]
    
    # genetic data
    train_data = slice_pairwise_data(train)
    test_data = slice_pairwise_data(test)
    
    # normalize dxy
    train_dxy = train_data['dxy'].to_numpy()
    train_max_dxy = train_dxy.max()
    train_dxy /= train_max_dxy
    
    test_dxy = test_data['dxy'].to_numpy()
    test_dxy /= train_max_dxy
    
    # normalize lats and longs
    train_longlats = train_spatial_data[['long', 'lat']].to_numpy()
    test_longlats = test_spatial_data[['long', 'lat']].to_numpy()
    lower, upper = 0.2, 0.8
    
    train_points = normalize_coords(train_longlats, lower, upper, train_longlats)
    test_points = normalize_coords(test_longlats, lower, upper, train_longlats)
    
    # gaussian density estimate on boundary
    xs, ys = coords_to_pairs(train_points)
    def kernl(x, y):
        x = np.add(x, y) / 2
        def _dists(x_i, x_j, x):
            diffs = x_i - x_j, x_i - x, x_j - x
            dists = np.sum([np.hypot(*xx) for xx in diffs])
            return dists
        dists = np.array([_dists(x_i, ys[i], x) for i, x_i in enumerate(xs)])
        e = 0.1
        scale = 2 * e**2
        return np.exp(-dists / scale)

    def boundary(x, y):
        k = kernl(x, y)
        return (train_dxy * k).sum() / k.sum()
    
    # PDE constraint
    eqn = pf.HittingTimes(W, boundary, epsilon=0.03)
    control = eqn.control
    
    # loss functionals
    reg = {'l2': [100*penalty, 100*penalty], 'smoothing': [penalty, penalty]}
    train_sd = pf.SpatialData(train_dxy, train_points, W)
    train_loss = pf.LossFunctional(train_sd, control, reg)
    
    test_sd = pf.SpatialData(test_dxy, test_points, W)
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
    test_errors[fold-1] = test_loss.l2_error(u_hat)
    
    # pickle results
    pickle_dump(cv_dir + f'{k}_fold/fold_{fold}_m_hats.pkl', m_hats)
    pickle_dump(cv_dir + f'{k}_fold/fold_{fold}_losses.pkl', losses)
    pickle_dump(cv_dir + f'{k}_fold/fold_{fold}_results.pkl', results)
    
    fold += 1

pickle_dump(cv_dir + f'{k}_fold/test_errors.pkl', test_errors)
