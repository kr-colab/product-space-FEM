#!/bin/bash

set -euxo pipefail

SEED1=123
SEED2=456
BASE1="test/out_${SEED1}"
BASE2="simulation/${BASE1}_stats/rep${SEED2}"
BASEX="${BASE2}_xval_"

PYTHONPATH="$PWD:$PWD/simulation:$PYTHONPATH"
pushd simulation
rm -rf test/out_*
slim -d 'OUTDIR="test"' -s $SEED1 fecundity_regulation.slim
python compute_stats.py ${BASE1}.trees 20 $SEED2
python plot_maps.py ${BASE1}.spstats.csv
popd
python setup_crossval_grid.py simulation/test/crossvalidation.json 2 0.1 1 1 0.1 0.1 ${BASE2}
for j in $BASEX*/xval_params.json;
do
    python crossvalidation.py --json $j --max_iter 5
done
for p in $BASEX*/*.pkl
do
    python plot_crossvalidation.py $p
    q=${p%pkl}truth.qmd
    cp simulation/compute_truth.qmd $q
    quarto render $q -P results_file:$(basename $p)
done

echo "All done! Produced:"
ls -lR simulation/${BASE1}_stats
