#!/bin/bash
#SBATCH -J GPU_Job           # job name
#SBATCH -o test_out       # output and error file name (%j expands to jobID)
#SBATCH -p devel-gpu     # queue (partition) -- normal, development, etc.
#SBATCH -N 1
#SBATCH -A TG-ASC130034
#SBATCH -n 1
#SBATCH -t 00:15:00        # run time (hh:mm:ss) - 1.5 hours
module load cuda   # run the MPI executable named a.out
./poisson
