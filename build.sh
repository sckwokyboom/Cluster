#!/bin/bash

#PBS -l select=1:ncpus=4:mpiprocs=4:mem=4000m,place=scatter
#PBS -l walltime=00:01:00
#PBS -m n
#PBS -o out.txt
#PBS -e err.txt

MPI_NP=$(wc -l $PBS_NODEFILE | awk '{ print $1 }')

cd $PBS_O_WORKDIR

echo "Node file path: $PBS_NODEFILE"
echo "Node file contents:"
cat $PBS_NODEFILE

echo "Using mpicc at `which mpicc`"
echo "Running $MPI_NP MPI processes"

mpicxx -std=gnu++1y -O3 -Wall -Wextra -o cluster.out cluster.cpp -lm -lpthread