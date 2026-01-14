#!/bin/bash
#$ -N test_fork               # Job name
#$ -l a100=1,s_vmem=4G                # Resources: 4GB memory
#$ -cwd                         # Use current working directory
#$ -o logs/fork_$JOB_ID.out  # Standard output
#$ -e logs/fork_$JOB_ID.err  # Standard error
#$ -V                          # Export environment variables
#$ -S /bin/bash                # Explicitly use bash shell

# Create logs directory
mkdir -p logs

# Environment setup
conda activate project

# Run pickle test
echo "Testing dataset pickling..."
python test_fork.py
echo "Test completed."