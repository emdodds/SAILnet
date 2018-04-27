#!/bin/bash -l
export oc=4
for p in 0.001 0.00316 0.01 0.0316
do
    export p=$p
    for gain in 0.0129 0.0215 0.0359 0.0599
    do
        export gain=$gain
        sbatch runscaled.sh
    done
done