#!/usr/bin/env bash
# run_eval.sh — One-shot evaluation pipeline launcher
#
# Usage:
#   bash run_eval.sh                          # Run all metrics on all papers
#   bash run_eval.sh --skip-data-prep         # Skip PDF/PPTX → text conversion
#   bash run_eval.sh --metrics rouge_eval     # Run a single metric
#   bash run_eval.sh --papers attn camera     # Evaluate specific papers
#   bash run_eval.sh --backend openai         # Use OpenAI instead of vLLM
#   bash run_eval.sh --num-papers 5           # Random sample of 5 papers
#
# All arguments are forwarded to pipeline.py.
# Edit constants.py to permanently change paths, models, or thresholds.

set -euo pipefail

# ── Resolve script directory ──────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Python executable ─────────────────────────────────────────────────────────
# Change PYTHON to your venv / conda env python if needed.
PYTHON="${PYTHON:-python3}"

# ── Optional: activate a virtualenv ──────────────────────────────────────────
# Uncomment and edit the path below if you use a venv.
# source "$SCRIPT_DIR/.venv/bin/activate"

# ── Environment ───────────────────────────────────────────────────────────────
# Pass your OpenAI key via the environment if using --backend openai.
# export OPENAI_API_KEY="sk-..."

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/eval_${TIMESTAMP}.log"

echo "========================================================================"
echo "  Evaluation Pipeline"
echo "  Started : $(date)"
echo "  Log     : $LOG_FILE"
echo "  Args    : $*"
echo "========================================================================"

# ── Run pipeline ─────────────────────────────────────────────────────────────
"$PYTHON" "$SCRIPT_DIR/pipeline.py" "$@" 2>&1 | tee "$LOG_FILE"

EXIT_CODE="${PIPESTATUS[0]}"

echo "========================================================================"
if [ "$EXIT_CODE" -eq 0 ]; then
    echo "  Finished successfully at $(date)"
else
    echo "  FAILED with exit code $EXIT_CODE at $(date)"
fi
echo "  Log saved to: $LOG_FILE"
echo "========================================================================"

exit "$EXIT_CODE"
