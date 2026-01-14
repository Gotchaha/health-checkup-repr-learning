#!/bin/bash
#$ -N ssl_profiling                    # Job name
#$ -l a100=1,s_vmem=256G              # Resources: 1 A100 GPU, mem usage soft limit: 256G
#$ -cwd                               # Use current working directory
#$ -o profile_results/profile_$JOB_ID.out   # Standard output
#$ -e profile_results/profile_$JOB_ID.err   # Standard error
#$ -V                                 # Export environment variables
#$ -S /bin/bash                       # Explicitly use bash shell

echo "=============================================="
echo "SSL Profiling Job Started"
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
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo ""

# Check if profile_results directory exists, create if not
mkdir -p profile_results

# Environment setup
echo "Activating conda environment..."
conda activate project

# Set Python hash seed for reproducibility
export PYTHONHASHSEED=42
echo "PYTHONHASHSEED set to: $PYTHONHASHSEED"

# Create output directory with timestamp
OUTPUT_DIR="profile_results/$(date +%Y%m%d_%H%M%S)"
mkdir -p $OUTPUT_DIR

echo "Starting SSL profiling..."
echo "Output directory: $OUTPUT_DIR"
echo "=============================================="

# Run profiling script
python scripts/profile_ssl.py \
    --config config/experiments/ssl_profiling.yaml \
    --output-dir $OUTPUT_DIR \
    --steps 100 \
    --warmup-steps 1000

# Capture exit status
EXIT_STATUS=$?

echo "=============================================="
echo "SSL Profiling Job Completed"
echo "Exit Status: $EXIT_STATUS"
echo "End Time: $(date)"
echo "=============================================="

exit $EXIT_STATUS