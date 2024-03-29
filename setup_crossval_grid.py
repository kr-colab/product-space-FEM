#!python3
import os, sys
import copy
import json
import numpy as np

usage = f"""
{' '.join(sys.argv)}

Usage:
    {sys.argv[0]} (json file with parameters in) (n_l2) (min_l2) (max_l2) (n_smoothness) (min_smoothness) (max_smoothness) (basename for spatial and genetic data files) [start]
where :
    - n_X is the number of subdivisions in the grid for parameter X, from min_X to max_X
    - the json file should include the parameters required for
crossvalidation.py; the result will be n^2 subdirectories containing simple
json files that vary the l2 and smoothness regularization parameters over the
2-dimensional grid from min to max.
    - spatial, genetic data files should be at (basename).stats.csv and (basename).pairstats.csv respectively;
        paths given to the files are *relative to the json file*
    - `start` should be an integer from which directory numbering will start (defaults to 0)
"""

if len(sys.argv) != 8 and len(sys.argv) != 9 and len(sys.argv) != 10:
    print(usage)
    sys.exit()

try:
    paramsfile = sys.argv[1]
    n_l2, min_l2, max_l2 = int(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
    n_sm, min_sm, max_sm = int(sys.argv[5]), float(sys.argv[6]), float(sys.argv[7])

    outdir = os.path.dirname(paramsfile)
    with open(paramsfile, 'r') as f:
        params = json.load(f)

    outbase = ""

    if len(sys.argv) > 8:
        sgbase = sys.argv[8]
        fname = os.path.basename(sgbase)
        sfile = os.path.join("..", f"{fname}.stats.csv")
        gfile = os.path.join("..", f"{fname}.pairstats.csv")
        params['spatial_data'] = sfile
        params['genetic_data'] = gfile
        outbase = f"{sgbase}_"

    start = 0
    if len(sys.argv) > 9:
        start = int(sys.argv[9])
except:
    print(usage)
    sys.exit()

range_l2 = np.linspace(min_l2, max_l2, n_l2)
range_sm = np.linspace(min_sm, max_sm, n_sm)

reg = params['regularization']
if isinstance(params["mesh"], str):
    params["mesh"] = os.path.join("..", params["mesh"])

j = start
for l2 in range_l2:
    for sm in range_sm:
        reg['l2'][0] = reg['l2'][1] = l2
        reg['smoothing'][0] = reg['smoothing'][1] = sm
        xdir = f"{outbase}xval_{j}"
        os.mkdir(xdir)
        outfile = os.path.join(xdir, "xval_params.json")
        with open(outfile, "w") as f:
            json.dump(params, f, indent=3)
        j += 1

print(f"Done with setup; next would be at {j}")
