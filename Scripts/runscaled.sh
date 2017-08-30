#!/bin/bash
#
# Partition:
#SBATCH --partition=cortex
#
# Wall clock limit:
#SBATCH --time=48:00:00
#
# Memory:
#SBATCH --mem-per-cpu=8G

cd /global/home/users/edodds/SAILnet/Scripts
python trainscaled.py -g $gain -d smallpcaimages -p $p --oc $oc $keeponly