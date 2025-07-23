#!/bin/bash
#SBATCH -J My-Test-Job # Job Name
#SBATCH -o my_test_job.out # SLURM standard output to file
#SBATCH -e my_test_job.err # SLURM standard error to file
#SBATCH --nodes=1 # Utilize 2 nodes for this job
#SBATCH --ntasks-per-node=1 # Each node runs 4 tasks, so two nodes will totally run 8 tasks
#SBATCH --cpus-per-task=4 # Each task reserves 56 vcores on a node, default value is 1
#SBATCH --gres=gpu:1 # Reserve 8 GPUs on each node for this job
#SBATCH --mem=16G # Reserve memory size on each node
#SBATCH --time=00-00:30:00 # Set execution time limit to 3 mins, kill the job if it reaches
#SBATCH -p trialq # Partition/Queue name

#==========================
# Load modules
#==========================

module purge
module load slurm/slurm/23.02.4
module load nvidia-hpc/2024_241
module load nvhpc/24.1
module list

#==========================
# Execute My Program
#==========================
python main.py