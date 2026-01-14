#!/bin/bash
#$ -N dataset_test             # Job name
#$ -l a100=1,s_vmem=512G        # Resources: 1 A100 GPU, 512GB memory
#$ -cwd                         # Use current working directory
#$ -o logs/dataset_$JOB_ID.out # Standard output
#$ -e logs/dataset_$JOB_ID.err # Standard error
#$ -V                          # Export environment variables
#$ -S /bin/bash                # Explicitly use bash shell

echo "=============================================="
echo "Dataset Test Job Started"
echo "=============================================="
echo "Job ID: $JOB_ID"
echo "Job Name: $JOB_NAME"
echo "Start Time: $(date)"
echo "Host: $(hostname)"
echo "Working Directory: $(pwd)"
echo ""

# Check if logs directory exists, create if not
mkdir -p logs

# Environment setup
echo "Activating conda environment..."
conda activate project

# Run test
python -m tests.models.test_dataset

# Capture exit status
EXIT_STATUS=$?

echo "=============================================="
echo "Dataset Test Job Completed"
echo "Exit Status: $EXIT_STATUS"
echo "End Time: $(date)"
echo "=============================================="

exit $EXIT_STATUS