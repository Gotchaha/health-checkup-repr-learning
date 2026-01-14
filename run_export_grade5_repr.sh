#!/bin/bash
#$ -N grade5_repr_export         # Job name
#$ -l a100=1,s_vmem=512G        # Resources: 1 A100 GPU
#$ -cwd                         # Use current working directory
#$ -o logs/job_$JOB_ID.out     # Standard output
#$ -e logs/job_$JOB_ID.err     # Standard error
#$ -V                          # Export environment variables
#$ -S /bin/bash                # Explicitly use bash shell

echo "=============================================="
echo "Grade5 Representation Export Job Started"
echo "=============================================="
echo "Job ID: $JOB_ID"
echo "Job Name: $JOB_NAME"
echo "Start Time: $(date)"
echo "Host: $(hostname)"
echo "Working Directory: $(pwd)"
echo "=============================================="

mkdir -p logs

echo "Activating conda environment..."
conda activate project

export PYTHONHASHSEED=42
echo "PYTHONHASHSEED set to: $PYTHONHASHSEED"

echo "Starting Grade5 representation export..."
echo "Config: config/downstream/grade5_task_config_v1.yaml"
echo "Output Dir: data/downstream/grade/repr"
echo "=============================================="

python scripts/downstream/export_ssl_representations.py \
  --config config/downstream/grade5_task_config_v1.yaml \
  --output_dir data/downstream/grade/repr \
  --split all \
  --dtype float16 \
  --device cuda \
  --log_dir outputs/downstream/repr_export \
  --include_columns exam_id,person_id,ExamDate,split,grade5,is_grade5_valid

EXIT_STATUS=$?

echo "=============================================="
echo "Grade5 Representation Export Job Completed"
echo "Exit Status: $EXIT_STATUS"
echo "End Time: $(date)"
echo "=============================================="

exit $EXIT_STATUS
