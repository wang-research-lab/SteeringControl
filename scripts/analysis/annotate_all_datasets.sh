#!/bin/bash
# Batch script to create annotation sessions for all LLM-judge datasets
#
# Usage:
#   ./annotate_all_datasets.sh [num_samples]
#
# Example:
#   ./annotate_all_datasets.sh 200    # Create sessions with 200 samples each
#   ./annotate_all_datasets.sh 100    # Create sessions with 100 samples each

NUM_SAMPLES=${1:-200}  # Default to 200 if not specified

echo "========================================"
echo "Creating annotation sessions"
echo "Sample size per dataset: $NUM_SAMPLES"
echo "========================================"
echo ""

DATASETS=(
    "DarkBenchAnthro"
    "DarkBenchBrandBias"
    "DarkBenchRetention"
    "DarkBenchSneaking"
    "DarkBenchSynchopancy"
    "PreciseWiki"
    "SaladBench"
)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ANNOTATION_SCRIPT="$SCRIPT_DIR/annotate_llm_judgments.py"

for dataset in "${DATASETS[@]}"; do
    echo "========================================"
    echo "Creating session for: $dataset"
    echo "========================================"

    python "$ANNOTATION_SCRIPT" \
        --dataset "$dataset" \
        --num-samples "$NUM_SAMPLES" \
        --sample-across-experiments \
        --experiments-base-dir experiments

    if [ $? -eq 0 ]; then
        echo "✓ Created session for $dataset"
    else
        echo "✗ Failed to create session for $dataset"
    fi
    echo ""
done

echo "========================================"
echo "Summary"
echo "========================================"
echo "Total datasets: ${#DATASETS[@]}"
echo "Samples per dataset: $NUM_SAMPLES"
echo "Total annotations needed: $((${#DATASETS[@]} * NUM_SAMPLES))"
echo ""
echo "Session files created in: annotations/"
echo ""
echo "To start annotating, run:"
echo "  python $ANNOTATION_SCRIPT --resume annotations/session_YYYYMMDD_HHMMSS_DATASET.json"
echo ""
echo "Or to see all sessions:"
echo "  ls -lh annotations/"
