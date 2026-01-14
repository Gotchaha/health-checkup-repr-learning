#!/bin/bash
#$ -N data_profiling                      # Job name
#$ -l a100=1,s_vmem=128G
#$ -cwd                                   # Run from project_root
#$ -o data_profiling/logs/job_$JOB_ID.out # Stdout path
#$ -e data_profiling/logs/job_$JOB_ID.err # Stderr path
#$ -V                                     # Inherit environment
#$ -S /bin/bash                           # Use bash

echo "=============================================="
echo "Data Loading Profiling Job Started"
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

# Ensure log directory exists
mkdir -p data_profiling/logs

# Activate the same Conda environment as training
echo "Activating conda environment..."
conda activate project

# Reproducibility and reduce BLAS/NumPy implicit threading noise
export PYTHONHASHSEED=42
echo "PYTHONHASHSEED set to: $PYTHONHASHSEED"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

# Run the profiling entry script (it reads config from its own directory)
python -u data_profiling/profile_data_loading.py

# Capture and report exit status
EXIT_STATUS=$?
echo "=============================================="
echo "Data Loading Profiling Job Completed"
echo "Exit Status: $EXIT_STATUS"
echo "End Time: $(date)"
echo "=============================================="

exit $EXIT_STATUS
