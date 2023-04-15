# 4/15/23:

Have three simulations ready to go:
```
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
```
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
```
DIRS="simulation/density_bump/out_12345_stats/rep899640 simulation/bias_bump/out_123456_stats/rep233858 simulation/bias_saddle/out_123456_stats/rep921586"
for DIR in $DIRS;
do
    snakemake -C base_name=$DIR --jobs 10 --profile ~/.config/snakemake/talapas/ &>>plr_snakemake.log &
done
```
