# 4/15/23:

Have three simulations ready to go:
```bash
$ tail -n 1 simulation/*/*.log
==> simulation/bias_bump/out_123456.trees.log <==
19981,"end",42219,70.365,71.3181,8.90243

==> simulation/bias_saddle/out_123456.trees.log <==
19981,"end",42170,70.2833,71.4013,7.49978

==> simulation/density_bump/out_12345.trees.log <==
19981,"end",38788,64.6467,132.632,49.2455
```

Set up a crossvalidation grid on each,
with mesh sizes of n=5 and n=10
(note: next time don't be so clever with the setup for loop; just do it by hand):
WHOOPS: I started 2-fold crossvalidation but edited un-run things to be 5-fold...
```bash
SIMS="simulation/bias_bump/out_123456 simulation/bias_saddle/out_123456 simulation/density_bump/out_12345"
N=5
M=11
for x in $SIMS;
do
    STATS=$(ls ${x}_stats/*.stats.csv)
    BASE=${STATS%.stats.csv}
    for j in 5 10;
    do
        JSON="$(dirname $x)/crossval_n${j}.json"
        JBASE="${BASE}_n${j}"
        echo "---- $x $JSON $STATS $JBASE"
        for u in stats pairstats;
        do
            echo "ln -s $(basename $BASE.${u}.csv) ${JBASE}.${u}.csv"
            ln -s $(basename $BASE.${u}.csv) ${JBASE}.${u}.csv
        done
        ARGS="$JSON $N .01 1 $M 0.01 1000 ${JBASE} 0"
        echo "$ARGS"
        python3 setup_crossval_grid.py $ARGS
        ARGS="$JSON $N .00001 0.001 $M 0.01 1000 ${JBASE} $(( $N * $M ))"
        echo "$ARGS"
        python3 setup_crossval_grid.py $ARGS
    done
done
```

Started crossvalidation on these:
```bash
DIRS="simulation/density_bump/out_12345_stats/rep899640 simulation/bias_bump/out_123456_stats/rep233858 simulation/bias_saddle/out_123456_stats/rep921586"
for DIR in $DIRS;
do
    snakemake -C base_name=$DIR --jobs 10 --profile ~/.config/snakemake/talapas/ &>>plr_snakemake.log &
done
```

Made plots:
```bash
for x in $(find simulation/ -name "*.pkl" -ctime -1); do if [ ! -e $(dirname $x)/solutions.png ]; then python plot_crossvalidation.py $x; fi; done
```

Runs with n=10 ran out of memory with 10G; upping to 20G to re-run.

Summarise results (n=10 ran out of memory):
```bash
export XDG_RUNTIME_DIR=$(mktemp -d)
n=5
DIRS="simulation/density_bump/out_12345_stats/rep899640_n${n} simulation/bias_bump/out_123456_stats/rep233858_n${n} simulation/bias_saddle/out_123456_stats/rep921586_n${n}"
for DIR in $DIRS
do
    QMD=${DIR}.summary.qmd
    if [ ! -e $QMD ]; then cp simulation/summarise_crossvalidation.qmd $QMD; fi
    quarto render $QMD -P basename:$(basename $DIR)
done
rm -rf $XDG_RUNTIME_DIR
unset XDG_RUNTIME_DIR
```

Extended to smaller values of regularization parameters:
```bash
SIMS="simulation/bias_bump/out_123456 simulation/bias_saddle/out_123456 simulation/density_bump/out_12345"
N=5
M=11
for x in $SIMS;
do
    j=5
    JSTATS=$(ls ${x}_stats/*_n${j}.stats.csv)
    JBASE=${JSTATS%.stats.csv}
    JSON="$(dirname $x)/crossval_n${j}.json"
    ARGS="$JSON $N 0.0001 .01 $M 0.01 10 ${JBASE} $(( 2 * $N * $M ))"
    echo "$ARGS"
    python3 setup_crossval_grid.py $ARGS
done

j=5
DIRS="simulation/density_bump/out_12345_stats/rep899640_n${j} simulation/bias_bump/out_123456_stats/rep233858_n${j} simulation/bias_saddle/out_123456_stats/rep921586_n${j}"
for DIR in $DIRS;
do
    snakemake -C base_name=$DIR --jobs 10 --profile ~/.config/snakemake/talapas/ --resources mem_mb=15000 &>>plr_snakemake.log &
done
```

# 4/17/23:

Found solutions with no regularization:

```bash
DIRS="simulation/bias_bump/out_123456_stats/rep233858 simulation/bias_saddle/out_123456_stats/rep921586 simulation/density_bump/out_12345_stats/rep899640"
j=5
for DIR in $DIRS
do
    snakemake -C base_name=${DIR}_noreg_n${j} --profile ~/.config/snakemake/talapas/ --resources mem_mb=10000 &&>>plr_snakemake.log &
done

j=10
for DIR in $DIRS
do
    snakemake -C base_name=${DIR}_noreg_n${j} --profile ~/.config/snakemake/talapas/ --resources mem_mb=30000 &&>>plr_snakemake.log &
done
```

Made plots:
```bash
for x in simulation/*/out_*_stats/rep**noreg_n*_0/results.pkl; do if [ ! -e $(dirname $x)/solutions.png ]; then python plot_crossvalidation.py $x; fi; done
```

Found solutions with very little regularization:

```bash
DIRS="simulation/bias_bump/out_123456_stats/rep233858 simulation/bias_saddle/out_123456_stats/rep921586 simulation/density_bump/out_12345_stats/rep899640"
j=5
for DIR in $DIRS
do
    ODIR=${DIR}_smreg_n${j}_0
    mkdir $ODIR
    cp ${DIR}_noreg_n${j}_0/xval_params.json $ODIR
    sed -i -e 's/smoothing": .*/smoothing": [0.01, 1]/' $ODIR/xval_params.json
    snakemake -C base_name=${ODIR%_0} --profile ~/.config/snakemake/talapas/
    python plot_crossvalidation.py $ODIR/results.pkl
done

# 4/18/23

Did more sims: 
- original sims had sigma=1, interaction=1 and bias=1
- (v2) smaller sigma (=0.1) and smaller bias (=0.1)
- (v3) as v2 and smaller interaction distance (=0.1)
- (v4, bias only) as v3 with smaller bias (0.01) (FAILED TO RUN?!?)
- (v5) long, skinny habitats as in v3

# 4/21/23

Found solutions with very little regularization:

Set-up:
```bash
STATS=$(ls simulation/*_v*/out_*_stats/rep*.stats.csv)
j=5
for STAT in $STATS
do
    DIR=${STAT%.stats.csv}
    REP=$(basename $DIR)
    ODIR=${DIR}_smreg_n${j}_0
    mkdir $ODIR
    cp simulation/density_bump/out_12345_stats/rep899640_smreg_n5_0/xval_params.json $ODIR
    sed -i -e "s/rep899640/$REP/" $ODIR/xval_params.json
done
```

Run stuff:
```bash
STATS=$(ls simulation/*_v*/out_*_stats/rep*.stats.csv)
j=5
for STAT in $STATS
do
    DIR=${STAT%.stats.csv}
    ODIR=${DIR}_smreg_n${j}_0
    snakemake -C base_name=${ODIR%_0} --profile ~/.config/snakemake/talapas/ &
    sleep 5
done
```


# 4/25:

Now try after regularizing covariance much more strongly:
```bash
STATS=$(ls simulation/*_v*/out_*_stats/rep*.stats.csv)
j=5
for STAT in $STATS
do
    DIR=${STAT%.stats.csv}
    PDIR=${DIR}_smreg_n${j}_0
    ODIR=${DIR}_smreg2_n${j}_0
    mkdir $ODIR
    cp $PDIR/xval_params.json $ODIR
    sed -i -e 's/smoothing": .*/smoothing": [0.01, 10000]/' $ODIR/xval_params.json
done
```
Run
```bash
JSONS=$(ls simulation/*_v*/out_*_stats/*_smreg2_*/xval_params.json)
JSONS=$(ls simulation/*_v5/out_*_stats/*_smreg*/xval_params.json)
j=5
for JSON in $JSONS
do
    DIR=$(dirname $JSON)
    snakemake -C base_name=${DIR%_0} --profile ~/.config/snakemake/talapas/ &
    sleep 5
done
```

