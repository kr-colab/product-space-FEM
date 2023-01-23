import sys, os
import warnings
import pyslim, tskit, msprime
import numpy as np
import pandas as pd
import json
from string import ascii_lowercase as letters

import matplotlib.pyplot as plt


if not len(sys.argv) in (2, 3):
    print(f"""
    Usage:
        python {sys.argv[0]} <input>.trees [seed]
    """)
    sys.exit()

ts_file = sys.argv[1]
if not os.path.isfile(ts_file):
    raise ValueError(f"File {ts_file} does not exist.")

if len(sys.argv) < 3:
    rng = np.random.default_rng()
    seed = rng.integers(1000000)

rng = np.random.default_rng(seed=seed)

basename = ts_file.replace(".trees", "")
basedir = f"{basename}_stats"
if not os.path.exists(basedir):
    os.makedirs(basedir)
repname = os.path.join(basedir, f"rep{seed}")

# parameters
mut_rate = 2e-8  # mutation rate
num_samples = 20  # number of sampled diploids
ancestral_Ne = 1e3  # ancestral pop size

print(
        f"Reading in from {ts_file}, outputting to {basedir}"
)

warnings.simplefilter('ignore', msprime.TimeUnitsMismatchWarning)

# Load tree sequence, reduce to samples today 
ts = tskit.load(ts_file)
_alive_nodes = np.array(
        [n for i in pyslim.individuals_alive_at(ts, 0)
            for n in ts.individual(i).nodes
])
ts = ts.simplify(_alive_nodes, keep_input_roots=True)

# write out full parameters
params = ts.metadata['SLiM']['user_metadata']
with open(os.path.join(basedir, "params.json"), 'w') as f:
    f.write(json.dumps(params))

ts = pyslim.recapitate(
        ts,
        ancestral_Ne=ancestral_Ne,
        recombination_rate=1e-8,
)

ts = msprime.sim_mutations(
        ts,
        rate=mut_rate,
        model=msprime.SLiMMutationModel(type=0),
        keep=False,
)

####
# compute statistics
samples = rng.choice(pyslim.individuals_alive_at(ts, 0), size=num_samples)
sample_nodes = np.array([
    ts.individual(i).nodes
    for i in samples
])

lll = [letters[k] for k in range(len(letters))]
names = ["".join([a, b, c]) for a in rng.permuted(lll) for b in rng.permuted(lll) for c in rng.permuted(lll)][:len(samples)]

locs = pd.DataFrame({
    "loc" : names,
    "x" : ts.individual_locations[samples, 0],
    "y" : ts.individual_locations[samples, 1],
}).set_index("loc")

# heterozygosity
locs["het"] = ts.diversity(sample_nodes, mode='site')

# write out text file
locs.to_csv(f"{repname}.stats.csv")

###
# pairwise stats

pairs = pd.DataFrame(
            np.array(
                [[a, b] for a in locs.index for b in locs.index if a <= b]
            ),
            columns=["loc1", "loc2"],
)
pairs["dxy"] = np.nan

plist = []
nlist = list(locs.index)
for a, b in zip(pairs["loc1"], pairs["loc2"]):
    plist.append((nlist.index(a), nlist.index(b)))

pairs["dxy"] = ts.divergence(
        sample_sets = sample_nodes,
        indexes = plist,
        mode = 'site',
)

# write out text file
pairs.to_csv(f"{repname}.pairstats.csv")

# make plot of IBD
i1 = np.array([nlist.index(i) for i in pairs['loc1']])
i2 = np.array([nlist.index(i) for i in pairs['loc2']])
pairs['dist'] = np.sqrt(
    (ts.individual_locations[i1, 0] - ts.individual_locations[i2, 0])**2
    + 
    (ts.individual_locations[i1, 1] - ts.individual_locations[i2, 1])**2
)

fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(pairs['dist'], pairs['dxy'])
ax.set_xlabel("geographic distance")
ax.set_ylabel("genetic distance")
ax.set_title(repname)
plt.tight_layout()

fig.savefig(f"{repname}.ibd.png")


print(f"All done with {basedir} =) =)")
