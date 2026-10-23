#!/usr/bin/env bash
# Settings shared by the TextSteer experiment scripts.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/outputs}"
# Model responses are cached under $XDG_CACHE_HOME/textgrad.
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$OUTPUT_DIR/cache}"

if [[ -f "$ROOT/.env" ]]; then
  set -a
  source "$ROOT/.env"
  set +a
fi
: "${OPENAI_API_KEY:?Set OPENAI_API_KEY (gpt-4.1-mini backward model and prompt editor)}"
: "${OPENROUTER_API_KEY:?Set OPENROUTER_API_KEY (target models)}"

# Data, evaluation and optimization budget shared by every method.
TRAIN_ARGS=(
  --eval-engine gpt-4.1-mini
  --test-max-tokens 2000
  --eval-max-workers 4
  --openrouter-request-timeout 300
  --openrouter-max-retries 20
  --openrouter-transient-error-policy score-zero
  --train-size 50 --val-size 50 --test-size 100
  --max-val-samples 50 --max-test-samples 100
  --batch-size 8 --epochs 1 --max-steps 3
  --pass-at-k 1 --eval-temperature 0
  --optimizer-verbose 1
  --use-validation-revert --validation-revert-tolerance 0
)

# Target models.
GPT_OSS_ARGS=(--test-engine openrouter-openai/gpt-oss-20b --openrouter-provider groq --openrouter-provider-sort throughput)
DEEPSEEK_ARGS=(--test-engine openrouter-deepseek/deepseek-v4-flash-0731 --openrouter-reasoning off --openrouter-provider coreweave/fp8)

# TextSteer hyperparameters. All four components are on by default; each
# ablation switches one off.
TEXTSTEER_ARGS=(--group-temperature 0 --num-candidates 4 --candidate-temperature 0.7 --candidate-responses 4 --edit-budget 2)

VARIANTS="textsteer wo_group wo_diag wo_age wo_lr"

# Sets VARIANT_ARGS for TextSteer or one of its ablations.
variant_args() {
  case "$1" in
    textsteer) VARIANT_ARGS=() ;;
    wo_group) VARIANT_ARGS=(--no-group-inference) ;;
    wo_diag) VARIANT_ARGS=(--no-diagnostic-preconditioning) ;;
    wo_age) VARIANT_ARGS=(--no-advantage-guided-editing) ;;
    wo_lr) VARIANT_ARGS=(--no-textual-learning-rate) ;;
    *)
      echo "Unknown variant '$1' (choose from: $VARIANTS)" >&2
      exit 2 ;;
  esac
}

# run_seed OUT_DIR SEED ARGS...: one optimization run; skipped if results exist.
run_seed() {
  local out="$1" seed="$2"
  shift 2
  if [[ -f "$out/results.json" ]]; then
    echo "Skipping $out (results.json exists)"
    return 0
  fi
  mkdir -p "$out"
  "$PYTHON" "$ROOT/textsteer/run.py" "$@" --seed "$seed" --save-results "$out/results.json" 2>&1 | tee "$out/run.log"
}
