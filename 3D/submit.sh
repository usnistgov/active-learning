#!/bin/bash
#SBATCH --partition=rack4               # -p, first available from the list of partitions
#SBATCH --time=04:00:00                 # -t, time (hh:mm:ss or dd-hh:mm:ss)
#SBATCH --nodes=1                       # -N, total number of machines
#SBATCH --ntasks=1                      # -n, 64 MPI ranks per Opteron machine
#SBATCH --cpus-per-task=10               # threads per MPI rank
#SBATCH --job-name=test_active_learning                 # -J, for your records
#SBATCH --chdir=/working/wd15/active-learning/3D   # -D, full path to an existing directory
#SBATCH --qos=test
#SBATCH --mem=0G

~/bin/nix-root nix develop ../flake.nix --command bash -c "snakemake --nolock --cores 10 --configfile config-slurm.yaml"


	
