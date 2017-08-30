#!/bin/bash -l
for oc in 1 2 4 8
do
    export oc=oc ##is this the right syntax?
    for p in 0.001 0.00316 0.01 0.0316 0.1 0.316
    do
        export p=$p
        for gain in 0.1 0.167 0.278 0.464, 0.774 1.29 2.15 3.59 5.99 10.0
        export gain=$gain
        sbatch runscaled.sh
    done
done