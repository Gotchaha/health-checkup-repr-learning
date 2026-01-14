#!/bin/bash
#$ -N system_info
#$ -l a100=1,s_vmem=64G
#$ -cwd
#$ -o logs/system_info_job_$JOB_ID.out
#$ -e logs/system_info_job_$JOB_ID.err
#$ -V
#$ -S /bin/bash

set -euo pipefail

mkdir -p logs
mkdir -p outputs/system_info

echo "=============================================="
echo "System Info Job Started"
echo "=============================================="
echo "Job ID: ${JOB_ID:-}"
echo "Job Name: ${JOB_NAME:-}"
echo "Start Time: $(date)"
echo "Host: $(hostname)"
echo "Working Directory: $(pwd)"
echo "Allocated Slots: ${NSLOTS:-}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-}"

if command -v free >/dev/null 2>&1; then
  echo "Available Memory: $(free -h | grep '^Mem:' | awk '{print $7}')"
fi

if command -v nproc >/dev/null 2>&1; then
  echo "CPU Cores (nproc): $(nproc)"
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "GPU Count: $(nvidia-smi --query-gpu=count --format=csv,noheader,nounits 2>/dev/null | head -n 1)"
  echo "GPU Information:"
  nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv 2>/dev/null || true
fi

echo ""

echo "Activating conda environment (best effort)..."
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)" 2>/dev/null || true
  conda activate project 2>/dev/null || true
fi

echo "CONDA_DEFAULT_ENV: ${CONDA_DEFAULT_ENV:-}"
echo "CONDA_PREFIX: ${CONDA_PREFIX:-}"

PYTHON_BIN="${PYTHON_BIN:-python}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  PYTHON_BIN=python3
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
HOST="$(hostname)"
JOB="${JOB_ID:-nojob}"
OUT_JSON="outputs/system_info/system_info_${JOB}_${HOST}_${STAMP}.json"

echo "Writing JSON artifact: $OUT_JSON"
set +e
"$PYTHON_BIN" scripts/system/collect_system_info.py --out-json "$OUT_JSON" --also-print
EXIT_STATUS=$?
set -e

echo "=============================================="
echo "System Info Job Completed"
echo "Exit Status: $EXIT_STATUS"
echo "End Time: $(date)"
echo "JSON: $OUT_JSON"
echo "=============================================="

exit $EXIT_STATUS
