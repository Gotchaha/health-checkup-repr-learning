#!/bin/bash
#$ -N grade5_fttransformer          # Job name
#$ -l a100=1,s_vmem=512G            # Resources: 1 A100 GPU, mem usage soft limit: 512G total
#$ -cwd                             # Use current working directory
#$ -o logs/job_$JOB_ID.out          # Standard output
#$ -e logs/job_$JOB_ID.err          # Standard error
#$ -V                               # Export environment variables
#$ -S /bin/bash                     # Explicitly use bash shell

echo "=============================================="
echo "Grade5 FT-Transformer Benchmark Job Started"
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

# Check if logs directory exists, create if not
mkdir -p logs

# Environment setup
echo "Activating conda environment..."
conda activate project

RUN_DIR="/path/to/run_dir"

echo "Starting FT-Transformer benchmark..."
echo "RUN_DIR: outputs/downstream/grade/benchmark/20260107_145433_seed0"
echo "=============================================="

python scripts/downstream/grade/benchmark_fttransformer.py --run_dir outputs/downstream/grade/benchmark/20260107_145433_seed0

# Capture exit status
EXIT_STATUS=$?

echo "=============================================="
echo "Grade5 FT-Transformer Benchmark Job Completed"
echo "Exit Status: $EXIT_STATUS"
echo "End Time: $(date)"
echo "=============================================="

exit $EXIT_STATUS
