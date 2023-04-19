#!/bin/bash

USAGE="Usage:
    $0 path/to/results.pkl
Will copy this file to path/to/results.crossvalidation_results.qmd
and render it into path/to/results.crossvalidation_results.html.
"

if [ $# -ne 1 ]
then
    echo "$USAGE"
    exit 0
fi

RESULTS="$1"
OUTPATH=${RESULTS%pkl}crossvalidation_results.qmd
if [ -e $OUTPATH ]
then
    echo "$OUTPATH already exists; aborting."
    exit 0
fi

cp crossvalidation_results.qmd $OUTPATH
quarto render $OUTPATH -P results_file:$(basename $RESULTS) && echo "Done; report at ${OUTPATH%qmd}html"
