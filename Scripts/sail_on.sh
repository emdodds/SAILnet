#!/bin/bash -l
for data in spectro images pcaimages
do
    export data=$data
    for scaled in --scaled --not-scaled
    do
        export scaled=$scaled
        sbatch runscaled.sh
    done
done