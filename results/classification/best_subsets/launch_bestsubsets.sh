#!/bin/bash -l
#SBATCH --job-name=bsub
#SBATCH --output=logs/bsub.out
#SBATCH --nodes 1
#SBATCH --error=logs/bsub.err

## Number of clusters, number of iterations, id
python best_subsets.py