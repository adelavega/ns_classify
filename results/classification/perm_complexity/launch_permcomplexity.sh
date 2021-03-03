#!/bin/bash -l
#SBATCH --job-name=permcomp
#SBATCH --output=logs/bsub.out
#SBATCH --nodes 1
#SBATCH --error=logs/bsub.err

## Number of samples and id number
python perm_complexity.py $1 $2