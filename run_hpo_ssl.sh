#!/bin/bash
#$ -N ssl_hpo                   # Job name
#$ -l a100=1,s_vmem=512G        # Resources: 1 A100 GPU, 512G soft limit
#$ -cwd                         # Use current working directory
#$ -o logs/hpo/job_$JOB_ID.out  # Standard output
#$ -e logs/hpo/job_$JOB_ID.err  # Standard error
#$ -V                           # Export environment variables
#$ -S /bin/bash                 # Explicitly use bash shell

set -euo pipefail

DEFAULT_HPO_CONFIG="config/hpo/ssl_pretraining_v1_lr_wd.yaml"
HPO_CONFIG_PATH="${1:-$DEFAULT_HPO_CONFIG}"
N_TRIALS="${HPO_N_TRIALS:-}"

echo "=============================================="
echo "SSL HPO Job Started"
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

# Ensure log/output directories exist
mkdir -p logs/hpo
mkdir -p outputs/hpo/studies

echo "Activating conda environment..."
conda activate project

# OpenMP/MKL threading layer fix for HPC environment
unset MKL_THREADING_LAYER
export MKL_THREADING_LAYER=GNU

export PYTHONHASHSEED=42
echo "PYTHONHASHSEED set to: $PYTHONHASHSEED"

echo "Starting SSL HPO..."
echo "HPO Config: ${HPO_CONFIG_PATH}"
if [[ -n "$N_TRIALS" ]]; then
  echo "Overriding number of trials: $N_TRIALS"
fi
echo "=============================================="

CMD=(python scripts/hpo_ssl.py --hpo-config "$HPO_CONFIG_PATH")
if [[ -n "$N_TRIALS" ]]; then
  CMD+=(--n-trials "$N_TRIALS")
fi

"${CMD[@]}"
EXIT_STATUS=$?

echo "=============================================="
echo "SSL HPO Job Completed"
echo "Exit Status: $EXIT_STATUS"
echo "End Time: $(date)"
echo "=============================================="

exit $EXIT_STATUS
