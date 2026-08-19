#!/usr/bin/env bash
set -euo pipefail

# -------------------------
# USER EDITS
# -------------------------
ROOT_DIR="/projects/ovcare/users/nikolay_alabi/immuno/data/raw_phenoptr"
PY_SCRIPT="/projects/ovcare/users/nikolay_alabi/immuno/data/combine_cohort_from_raw.py"

OUTROOT="/projects/ovcare/users/nikolay_alabi/immuno/data/raw_phenoptr/combined_cohorts"
LOGDIR="/projects/ovcare/users/nikolay_alabi/immuno/data/logs"
WORKERS="${OUTROOT}/workers"

# Slurm resources (tune)
CPUS=2
MEM="32G"
TIME="24:00:00"

# Conda
CONDA_SH="/home/nalabi/miniconda3/etc/profile.d/conda.sh"
CONDA_ENV="cuda6"   # must have pandas + pyarrow for parquet

# Combine settings
PART_SIZE_CORES=25
ALL_NEG_LABEL="ALL_NEG"

# Cohorts
# COHORTS=(
#   "BCA 2020 RC ARP"
#   "BCA2020 TURBT NAC ARP"
#   "No-NAC TURBT TMA1 ARP"
#   "No-NAC TURBT TMA2 ARP"
#   "PURE01 TMA1 Pre NAC ARP"
#   "PURE01 TMA2 Pre NAC ARP"
#   "PURE01 TMA3 Pre + Post NAC ARP"

#   "BCA 2020 RC B&T"
#   "BCA2020 TURBT NAC B&T"
#   "No-NAC TURBT TMA1  B&T"
#   "No-NAC TURBT TMA2  B&T"
#   "PURE01 TMA1 Pre NAC  B&T"
#   "PURE01 TMA2 Pre NAC  B&T"
#   "PURE01 TMA3 Pre + Post NAC  B&T"

#   "BCA 2020 RC Myeloid"
#   "BCA2020 TURBT NAC  Myeloid"
#   "No-NAC TURBT TMA1  Myeloid"
#   "No-NAC TURBT TMA2  Myeloid"
#   "PURE01 TMA1 Pre NAC  Myeloid"
#   "PURE01 TMA2 Pre NAC  Myeloid"
#   "PURE01 TMA3 Pre + Post NAC  Myeloid"

#   "Bladder 19_AR"
#   "Bladder 26_AR"
#   "Bladder 19_BT"
#   "Bladder 26_BT"
#   "Bladder 19_M"
#   "Bladder 26_M"
# )
COHORTS=(
  "No-NAC TURBT TMA1  B+T"
  "No-NAC TURBT TMA2  B+T"
  "PURE01 TMA1 Pre NAC  B+T"
  "PURE01 TMA2 Pre NAC  B+T"
  "PURE01 TMA3 Pre + Post NAC  B+T"
)
# -------------------------
# DO NOT EDIT BELOW
# -------------------------
mkdir -p "$OUTROOT" "$LOGDIR" "$WORKERS"

sanitize() {
  local s="$1"
  echo "$s" | sed -E 's/[[:space:]]+/_/g; s/[^A-Za-z0-9_+\&-]/_/g'
}

submit_one() {
  local cohort="$1"
  local safe
  safe="$(sanitize "$cohort")"

  local job_name="PHENOCOH_${safe}"
  local worker="${WORKERS}/${job_name}.sh"

  local out_dir="${OUTROOT}"
  local final_parquet="${OUTROOT}/${safe}.parquet"  # optional merged output per cohort

  cat > "$worker" <<EOF
#!/bin/bash
#SBATCH --job-name=${job_name}
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --mem=${MEM}
#SBATCH --time=${TIME}
#SBATCH --output=${LOGDIR}/${job_name}.out
#SBATCH --error=${LOGDIR}/${job_name}.err
#SBATCH --partition=upgrade,general,kresearch,merge,debug2

set -euo pipefail

export OMP_NUM_THREADS=${CPUS}
export MKL_NUM_THREADS=${CPUS}

: "\${PS1:=}"
set +u
source "${CONDA_SH}"
set -u
conda activate "${CONDA_ENV}"

echo "----------------------------------------"
echo "Job:      \$SLURM_JOB_ID"
echo "Cohort:   ${cohort}"
echo "Root:     ${ROOT_DIR}"
echo "Outroot:  ${OUTROOT}"
echo "Host:     \$(hostname)"
echo "----------------------------------------"

python "${PY_SCRIPT}" \\
  --root_dir "${ROOT_DIR}" \\
  --cohort "${cohort}" \\
  --out_dir "${out_dir}" \\
  --part_size_cores ${PART_SIZE_CORES} \\
  --drop_blank_cells \\
  --all_negative_label "${ALL_NEG_LABEL}" \\
  --final_parquet "${final_parquet}" \\
  --skip_if_done

echo "✅ Done: ${job_name}"
EOF

  chmod +x "$worker"
  echo "[submit] $job_name"
  sbatch "$worker"
}

for cohort in "${COHORTS[@]}"; do
  submit_one "$cohort"
done

echo "✅ Submitted ${#COHORTS[@]} cohort jobs."
echo "Logs:    $LOGDIR"
echo "Workers: $WORKERS"
echo "Out:     $OUTROOT"