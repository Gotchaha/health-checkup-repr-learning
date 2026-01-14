#!/bin/bash
#$ -N ssl_pre_profiling             # Job name
#$ -l a100=1,s_vmem=32G        # Resources: 1 A100 GPU, mem usage soft limit: 32G per slot
#$ -pe def_slot 9               # 9 slots
#$ -cwd                         # Use current working directory
#$ -o logs/pre_profile_$JOB_ID.out # Standard output
#$ -e logs/pre_profile_$JOB_ID.err # Standard error
#$ -V                          # Export environment variables
#$ -S /bin/bash                # Explicitly use bash shell

echo "=============================================="
echo "SSL Pre Profiling Job Started"
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
mkdir -p logs

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

echo "Starting SSL pre profiling..."
echo "Config: config/experiments/ssl_pretraining_test.yaml"
echo "Profiling 10 iterations to identify bottlenecks"
echo "=============================================="

# Run profiling
python scripts/pre_profile_ssl.py config/experiments/ssl_pretraining_test.yaml

# Capture exit status
EXIT_STATUS=$?

echo "=============================================="
echo "SSL Pre Profiling Job Completed"
echo "Exit Status: $EXIT_STATUS"
echo "End Time: $(date)"
echo "=============================================="

exit $EXIT_STATUS