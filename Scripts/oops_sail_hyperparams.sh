#!/bin/bash -l
for oc in 1
do
    export oc=$oc
    for p in 0.0464 0.0774 0.129 0.215 0.359 0.599 1.0
    do
        export p=$p
        for gain in 0.464
        do
            export gain=$gain
            sbatch runscaled.sh
        done
    done
done
export keeponly=--keep_only_error
for oc in 2 4 8
do
    export oc=$oc
    for p in 0.001 0.00316 0.01 0.0316 0.1 0.316
    do
        export p=$p
        for gain in 0.1 0.167 0.278 0.464 0.774 1.29 2.15 3.59 5.99 10.0
        do
            export gain=$gain
            sbatch runscaled.sh
        done
    done
done
