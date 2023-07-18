#!/bin/bash

# Define the list of sigma and beta values
sigma_values=(0.001 0.003 0.01 0.03 0.1 0.3 1 3)
beta_values=(0.001 0.003 0.01 0.03 0.1 0.3 1 3)

# Loop through each combination of sigma and beta values
for sigma in "${sigma_values[@]}"; do
  for beta in "${beta_values[@]}"; do
    # Submit a job to SLURM for each combination
    job_name="activePSGD_${sigma}_${beta}"
    output_file="activePSGD_${sigma}_${beta}.out"
    sbatch --job-name="$job_name" --output="$output_file" job.sh "$sigma" "$beta"
  done
done

