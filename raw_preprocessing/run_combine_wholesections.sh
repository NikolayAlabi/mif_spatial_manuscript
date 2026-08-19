#!/usr/bin/env bash
set -euo pipefail

ROOT_WHOLE="/projects/ovcare/users/nikolay_alabi/immuno/data/raw_phenoptr/Whole Sections"
PY_SCRIPT="/projects/ovcare/users/nikolay_alabi/immuno/data/combine_wholesection_panel_batch.py"

OUTROOT="/projects/ovcare/users/nikolay_alabi/immuno/data/raw_phenoptr/combined_wholesections"
LOGDIR="${OUTROOT}/logs"
WORKERS="${OUTROOT}/workers"

CPUS=2
MEM="32G"
TIME="24:00:00"

CONDA_SH="/home/nalabi/miniconda3/etc/profile.d/conda.sh"
CONDA_ENV="cuda6"

PART_SIZE_REGIONS=25
ALL_NEG_LABEL="ALL_NEG"

PANELS=("AR" "B&T" "Myeloid")
BATCHES=("1" "2" "3")

mkdir -p "$OUTROOT" "$LOGDIR" "$WORKERS"

sanitize() {
  local s="$1"
  echo "$s" | sed -E 's/[[:space:]]+/_/g; s/[^A-Za-z0-9_+\&-]/_/g'
}

submit_one() {
  local panel="$1"
  local batch="$2"
  local safe_panel
  safe_panel="$(sanitize "$panel")"

  local job_name="WHOLE_${safe_panel}_B${batch}"
  local worker="${WORKERS}/${job_name}.sh"

  cat > "$worker" <<EOF
#!/bin/bash
#SBATCH --job-name=${job_name}
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --mem=${MEM}
#SBATCH --time=${TIME}
#SBATCH --output=${LOGDIR}/${job_name}.out
#SBATCH --error=${LOGDIR}/${job_name}.err

set -euo pipefail
export OMP_NUM_THREADS=${CPUS}
export MKL_NUM_THREADS=${CPUS}

: "\${PS1:=}"
set +u
source "${CONDA_SH}"
set -u
conda activate "${CONDA_ENV}"

echo "----------------------------------------"
echo "Job:    \$SLURM_JOB_ID"
echo "Panel:  ${panel}"
echo "Batch:  ${batch}"
echo "Root:   ${ROOT_WHOLE}"
echo "Out:    ${OUTROOT}"
echo "Host:   \$(hostname)"
echo "----------------------------------------"

python "${PY_SCRIPT}" \\
  --root_whole "${ROOT_WHOLE}" \\
  --panel "${panel}" \\
  --batch "${batch}" \\
  --out_dir "${OUTROOT}" \\
  --part_size_regions ${PART_SIZE_REGIONS} \\
  --drop_blank_cells \\
  --all_negative_label "${ALL_NEG_LABEL}" \\
  --skip_if_done
  --final_parquet "${OUTROOT}/WholeSections_${safe_panel}_Batch${batch}.parquet"

echo "✅ Done ${job_name}"
EOF

  chmod +x "$worker"
  echo "[submit] $job_name"
  sbatch "$worker"
}

for panel in "${PANELS[@]}"; do
  for batch in "${BATCHES[@]}"; do
    submit_one "$panel" "$batch"
  done
done

echo "✅ Submitted 9 Whole Sections jobs."
echo "Logs:    $LOGDIR"
echo "Workers: $WORKERS"
echo "Out:     $OUTROOT"