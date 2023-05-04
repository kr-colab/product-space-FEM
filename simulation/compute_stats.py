import sys, os
import warnings
import pyslim, tskit, msprime
import numpy as np
import pandas as pd
import json
from string import ascii_lowercase as letters

import matplotlib.pyplot as plt


if not len(sys.argv) in (3, 4):
    print(f"""
    Usage:
        python {sys.argv[0]} <input>.trees num_samples [seed]
    where
    - num_samples is the number of sampled diploids
    """)
    sys.exit()

ts_file = sys.argv[1]
if not os.path.isfile(ts_file):
    raise ValueError(f"File {ts_file} does not exist.")

num_samples = int(sys.argv[2])

if len(sys.argv) < 4:
    rng = np.random.default_rng()
    seed = rng.integers(1000000)
else:
    seed = int(sys.argv[3])

rng = np.random.default_rng(seed=seed)

basename = ts_file.replace(".trees", "")
basedir = f"{basename}_stats"
if not os.path.exists(basedir):
    os.makedirs(basedir)
repname = os.path.join(basedir, f"rep{seed}")

# other parameters
mut_rate = 2e-8  # mutation rate
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
with open(os.path.join(basedir, "slim_params.json"), 'w') as f:
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

sd = pd.DataFrame({
    "name" : names,
    "x" : ts.individual_locations[samples, 0],
    "y" : ts.individual_locations[samples, 1],
}).set_index("name")

# heterozygosity
sd["het"] = ts.diversity(sample_nodes, mode='site')

# write out text file
sd.to_csv(f"{repname}.stats.csv")

###
# pairwise stats

gd = pd.DataFrame(
            np.array(
                [[a, b] for a in sd.index for b in sd.index if a <= b]
            ),
            columns=["name1", "name2"],
)
gd["divergence"] = np.nan

plist = []
nlist = list(sd.index)
for a, b in zip(gd["name1"], gd["name2"]):
    plist.append((nlist.index(a), nlist.index(b)))

gd["divergence"] = ts.divergence(
        sample_sets = sample_nodes,
        indexes = plist,
        mode = 'site',
)

# write out text file
gd.to_csv(f"{repname}.pairstats.csv")

# make plot of IBD
i1 = np.array([nlist.index(i) for i in gd['name1']])
i2 = np.array([nlist.index(i) for i in gd['name2']])
gd['dist'] = np.sqrt(
    (ts.individual_locations[i1, 0] - ts.individual_locations[i2, 0])**2
    + 
    (ts.individual_locations[i1, 1] - ts.individual_locations[i2, 1])**2
)

fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(gd['dist'], gd['divergence'])
ax.set_xlabel("geographic distance")
ax.set_ylabel("genetic distance")
ax.set_title(repname)
plt.tight_layout()

fig.savefig(f"{repname}.ibd.png")
plt.close(fig)

# make images of per-individual divergences
subdir = f"{repname}_ibd"
if not os.path.exists(subdir):
    os.makedirs(subdir)

for focal in sd.index:
    subfile = os.path.join(subdir, f"{focal}.ibd.png")
    fig, ax = plt.subplots(figsize=(6, 6))
    sub_gd = gd.loc[np.logical_or(gd['name1'] == focal, gd['name2'] == focal),:]
    other = [a if a != focal else b for a, b in zip(sub_gd['name1'], sub_gd['name2'])]
    s = ax.scatter(sd.loc[other,"x"],
               sd.loc[other,"y"],
               c=sub_gd['divergence'],
               s=100 * (sub_gd['divergence'] - 0.9 * min(sub_gd['divergence'])) / (max(sub_gd['divergence']) - min(sub_gd['divergence'])),
               vmin=min(gd['divergence']),
               vmax=max(gd['divergence'])
    )
    fig.colorbar(s)
    ax.set_aspect('equal')
    plt.savefig(subfile)
    plt.close(fig)

print(f"All done with {basedir} =) =)")
