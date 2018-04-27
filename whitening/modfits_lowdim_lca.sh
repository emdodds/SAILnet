#!/bin/bash
#
# Partition:
#SBATCH --partition=cortex
#
# Wall clock limit:
#SBATCH --time=48:00:00
#
# Memory:
#SBATCH --mem-per-cpu=4G

cd /global/home/users/edodds/SAILnet/Scripts
export MODULEPATH=/global/software/sl-6.x64_64/modfiles/apps:$MODULEPATH
module load ml/tensorflow/1.2.0
python model_recovery.py -g $gain -p $p --oc $oc --desphere $ds --lca