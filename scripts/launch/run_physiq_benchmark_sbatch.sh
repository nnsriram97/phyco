#!/usr/bin/env bash
# SLURM-friendly launcher for run_physiq_benchmark.py.
# Submit as:
#   sbatch --array=0-7 scripts/launch/run_physiq_benchmark_sbatch.sh \
#       --benchmark_root /path/to/physics-IQ-benchmark \
#       --output_dir /path/to/.model_name \
#       --num_splits 8 \
#       -- --dit_path ... --use_lora ...

set -euo pipefail

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    cat <<'EOF'
Usage: run_physiq_benchmark_sbatch.sh [OPTIONS] -- [EXTRA_PY_ARGS]

Required options:
  --benchmark_root PATH   Path passed to run_physiq_benchmark.py --benchmark_root
  --output_dir PATH       Path passed to --output_dir

Optional launcher options:
  --num_splits N          Total partitions (default: 1)
  --split_index I         Partition index for this job (default: SLURM_ARRAY_TASK_ID or 0)

Everything after "--" is forwarded verbatim to run_physiq_benchmark.py
(e.g., --dit_path, --switch_frames_dir, etc.).
EOF
    exit 0
fi

BENCHMARK_ROOT=""
OUTPUT_DIR=""
NUM_SPLITS=1
SPLIT_INDEX=""
PY_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --benchmark_root)
            BENCHMARK_ROOT="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --num_splits)
            NUM_SPLITS="$2"
            shift 2
            ;;
        --split_index)
            SPLIT_INDEX="$2"
            shift 2
            ;;
        --)
            shift
            PY_ARGS+=("$@")
            break
            ;;
        *)
            PY_ARGS+=("$1")
            shift
            ;;
    esac
done

if [[ -z "$BENCHMARK_ROOT" || -z "$OUTPUT_DIR" ]]; then
    echo "Error: --benchmark_root and --output_dir are required." >&2
    exit 1
fi

if [[ -z "$SPLIT_INDEX" ]]; then
    if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
        SPLIT_INDEX="${SLURM_ARRAY_TASK_ID}"
    else
        SPLIT_INDEX=0
    fi
fi

CMD=(
    python examples/run_physiq_benchmark.py
    --benchmark_root "$BENCHMARK_ROOT"
    --output_dir "$OUTPUT_DIR"
    --num_splits "$NUM_SPLITS"
    --split_index "$SPLIT_INDEX"
)

if [[ ${#PY_ARGS[@]} -gt 0 ]]; then
    CMD+=("${PY_ARGS[@]}")
fi

echo "Running split $SPLIT_INDEX/$NUM_SPLITS on host ${HOSTNAME}"
echo "Command: ${CMD[*]}"

"${CMD[@]}"
