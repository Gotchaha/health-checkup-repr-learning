#!/bin/bash
#$ -N DL_profiling             # Job name
#$ -l a100=1,s_vmem=512G        # Resources: 1 A100 GPU, mem usage soft limit: 512G 
#$ -cwd                         # Use current working directory
#$ -o DL_profiling/logs/dl_profile_$JOB_ID.out   # Standard output
#$ -e DL_profiling/logs/dl_profile_$JOB_ID.err   # Standard error
#$ -V                          # Export environment variables
#$ -S /bin/bash                # Explicitly use bash shell

echo "=============================================="
echo "DL Profiling Job Started"
echo "=============================================="
echo "Job ID: $JOB_ID"
echo "Job Name: $JOB_NAME"
echo "Start Time: $(date)"
echo "Host: $(hostname)"
echo "Working Directory: $(pwd)"
echo "Available Memory: $(free -h | grep '^Mem:' | awk '{print $7}')"
echo "CPU Cores: $(nproc)"
echo "Allocated Slots: $NSLOTS"
echo "Virtual Memory Limit: $(ulimit -v) KB"
echo "GPU Count: $(nvidia-smi --query-gpu=count --format=csv,noheader,nounits)"
echo ""

# Check if logs directory exists, create if not
mkdir -p DL_profiling/logs

# Environment setup
echo "Activating conda environment..."
conda activate project

# Set Python hash seed for full reproducibility
export PYTHONHASHSEED=42
echo "PYTHONHASHSEED set to: $PYTHONHASHSEED"

# GPU and system info
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo ""

echo "Starting DL profiling..."
echo "Config: config/experiments/dl_profiling.yaml"
echo "Profiling 200 iterations"
echo "Notes: optimized code, with optimization"
echo "=============================================="

# Run profiling
python DL_profiling/profile_dl.py config/experiments/dl_profiling.yaml

# Capture exit status
EXIT_STATUS=$?

echo "=============================================="
echo "SSL Pre Profiling Job Completed"
echo "Exit Status: $EXIT_STATUS"
echo "End Time: $(date)"
echo "=============================================="

exit $EXIT_STATUS