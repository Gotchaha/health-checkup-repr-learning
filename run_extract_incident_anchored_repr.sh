#!/bin/bash
#$ -N incident_repr_extract      # Job name
#$ -l a100=1,s_vmem=256G         # Resources: 1 GPU, mem usage soft limit
#$ -cwd                          # Use current working directory
#$ -o logs/job_$JOB_ID.out       # Standard output
#$ -e logs/job_$JOB_ID.err       # Standard error
#$ -V                            # Export environment variables
#$ -S /bin/bash                  # Explicitly use bash shell

echo "=============================================="
echo "Incident-Anchored Representation Extraction"
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

# Set Python hash seed for reproducibility
export PYTHONHASHSEED=42
echo "PYTHONHASHSEED set to: $PYTHONHASHSEED"

echo "Starting incident-anchored representation extraction..."
echo "Config: config/downstream/incident_anchored_repr_config.yaml"
echo "=============================================="

# Run extraction
python scripts/downstream/phenotyping/incident_anchored/extract_representations.py \
  --config config/downstream/incident_anchored_repr_config.yaml

# Capture exit status
EXIT_STATUS=$?

echo "=============================================="
echo "Incident-Anchored Representation Extraction Completed"
echo "Exit Status: $EXIT_STATUS"
echo "End Time: $(date)"
echo "=============================================="

exit $EXIT_STATUS
