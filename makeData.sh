#!/bin/bash

# Define the list of sigma values and beta values
sigma_values=(0.001 0.003 0.01 0.03 0.1 0.3 1 3)
beta_values=(0.001 0.003 0.01 0.03 0.1 0.3 1 3)

# Loop through each combination of sigma and beta values
for sigma in "${sigma_values[@]}"; do
  for beta in "${beta_values[@]}"; do
    # Submit a job to SLURM for each combination
    sbatch --job-name=activePSGD_"$sigma"_"$beta" --output=activePSGD_"$sigma"_"$beta".out --wrap="python activePSGD.py --sigma $sigma --beta $beta"
  done
done
