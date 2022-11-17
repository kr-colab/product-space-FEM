#! /bin/bash

penalties=( 1e-1 1e-2 )
for p in "${penalties[@]}"
do
	python crossvalidation.py 3 $p
done
