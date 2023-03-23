Structure of this directory:

- `bias_saddle/` - one particular scenario
    * `slim_params.json` - parameters to run this simulation with
    * `out_2874405695390.trees` - the result of one such simulation
    * `out_2874405695390_stats/` - things computed on that result
        - `rep371579.{stats,pairstats}.csv` - the result of running `compute_stats.py` once on the result
        - `rep371579_0/` - one crossvalidation setup using those results

Outline:

Figure "uphill": two rows, one with 'truth', one with inferred bias;
    columns as:

     * downhill bias on a bump (`bias_bump`)
     * bump in fecundity-regulated density (`density_bump`)
     * downhill bias on a saddle (`bias_saddle`)
     * saddle in fecundity-regulated density (`density_saddle`)


In this directory:

- `fecundity_regulation.slim`: takes maps of bias, covariance, and 'habitat' (which regulates fecundity)
- `example_params.json`: example parameter file for `fecundity_regulation.slim`
- `compute_stats.py`: run to compute divergences necessary for inference
- `plot_maps.py`: plots descriptive maps using the output of `compute_stats.py`
- `maps/`: various pngs of simple landscapes, produced by `cd maps; python3 make_maps.py`

Workflow:

For **debugging/visualization**, open `fecundity_regulation.slim` in SLiMgui
with working directory set to a subdirectory with `slim_params.json`,
e.g., `mkdir test; cp example_params.json test/slim_params.json; cd test; SLiMgui ../fecundity_regulation.slim`
(and edit `test/params.json` as desired).

To **run a test** (e.g., `bias_bump`):
```
slim -d 'OUTDIR="bias_bump"' fecundity_regulation.slim
python3 compute_stats.py bias_bump/<base>.trees 20
python3 plot_maps.py bias_bump/<base>.spatstats.csv
<TODO write inference script>
```

Other things here:

- `fast_slow.slim`: attempt at writing a script with constant population density but mortality-fecundity variation (DOES NOT CURRENTLY WORK)
