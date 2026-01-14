#!/bin/bash
#$ -N pickle_test               # Job name
#$ -l a100=1,s_vmem=64G           # Resources: 4GB memory
#$ -cwd                         # Use current working directory
#$ -o logs/pickle_$JOB_ID.out  # Standard output
#$ -e logs/pickle_$JOB_ID.err  # Standard error
#$ -V                          # Export environment variables
#$ -S /bin/bash                # Explicitly use bash shell

echo "========== DATASET PICKLE TEST =========="
echo "Job ID      : $JOB_ID"
echo "Start time  : $(date)"
echo "Host        : $(hostname)"
echo "=========================================="

# Activate environment
conda activate project

python scripts/pickle_test.py
EXIT_STATUS=$?

echo "=========================================="
echo "Finished with exit status: $EXIT_STATUS"
echo "End time : $(date)"
echo "=========================================="

exit $EXIT_STATUS
