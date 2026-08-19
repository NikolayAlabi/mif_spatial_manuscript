#!/bin/bash
set -euo pipefail
MODULE_DIR="${MODULE_DIR:-/projects/ovcare/users/nikolay_alabi/immuno/manuscript/modules2}"
CONFIG="${CONFIG:-$MODULE_DIR/configs/stage2d_discover_meta_modules_v1.json}"
SCRIPT="$MODULE_DIR/stage2d_discover_meta_modules_v1.py"
ROOT="/projects/ovcare/users/nikolay_alabi/immuno/stage2_root_meta_modules_v1/stage2d_meta_modules_v1"
LOGDIR="$ROOT/logs"
mkdir -p "$LOGDIR"

sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=s2d_meta
#SBATCH --cpus-per-task=1
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=$LOGDIR/run_%j.out
#SBATCH --error=$LOGDIR/run_%j.err
set -euo pipefail
export PS1="${PS1-}"
source /home/nalabi/miniconda3/etc/profile.d/conda.sh
conda activate cuda6
python "$SCRIPT" --config "$CONFIG" run
EOF
