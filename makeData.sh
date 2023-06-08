#!/bin/bash
#SBATCH --job-name=MakeDataJob
#SBATCH --output=MakeDataJob_%j.out
#SBATCH --error=MakeDataJob_%j.err
#SBATCH --time=1:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2G

# Load any necessary modules and activate your environment
# module load python/3.7
# source activate my_env

# Define your specific values for alpha and B
alpha_values=(10 20 30 40 50)
B_values=(5 15 25 35 45)

# Loop over your two parameters
for alpha in "${alpha_values[@]}"
do
    for B in "${B_values[@]}"
    do
        # Call your python script with the values as parameters
        python MakeData.py $alpha $B
    done
done

