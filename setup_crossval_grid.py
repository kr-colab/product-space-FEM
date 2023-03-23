#!python3
import os, sys
import copy
import json
import numpy as np

usage = f"""
Usage:
    {sys.argv[0]} (json file with parameters in) (n_l2) (min_l2) (max_l2) (n_smoothness) (min_smoothness) (max_smoothness)
where :
    - n_X is the number of subdivisions in the grid for parameter X
    - the json file should include the parameters required for
crossvalidation.py; the result will be n^2 subdirectories containing simple
json files that vary the l2 and smoothness regularization parameters over the
2-dimensional grid from min to max.
"""

if len(sys.argv) != 8:
    print(usage)
    sys.exit()

paramsfile = sys.argv[1]
n_l2, min_l2, max_l2 = int(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
n_sm, min_sm, max_sm = int(sys.argv[5]), float(sys.argv[6]), float(sys.argv[7])

outdir = os.path.dirname(paramsfile)
with open(paramsfile, 'r') as f:
    params = json.load(f)

range_l2 = np.linspace(min_l2, max_l2, n_l2)
range_sm = np.linspace(min_sm, max_sm, n_sm)

reg = params['regularization']
params["spatial_data"] = os.path.join("..", params["spatial_data"])
params["genetic_data"] = os.path.join("..", params["genetic_data"])
if isinstance(params["mesh"], str):
    params["mesh"] = os.path.join("..", params["mesh"])

j = 0
for l2 in range_l2:
    for sm in range_sm:
        reg['l2'][0] = reg['l2'][1] = l2
        reg['smoothing'][0] = reg['smoothing'][1] = sm
        xdir = os.path.join(outdir, f"xval_{j}")
        os.mkdir(xdir)
        outfile = os.path.join(xdir, "xval_params.json")
        with open(outfile, "w") as f:
            json.dump(params, f, indent=3)
        j += 1

