#!/bin/bash
#SBATCH -J myMPI           # job name
#SBATCH -o myMPI.o%j       # output and error file name (%j expands to jobID)
#SBATCH -p gpu     # queue (partition) -- normal, development, etc.
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:15:00        # run time (hh:mm:ss) - 1.5 hours
module load cuda   # run the MPI executable named a.out
./a.out
