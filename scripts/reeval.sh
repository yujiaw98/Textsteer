#!/usr/bin/env bash
# Test-time re-evaluation of Reasoning Gym prompts on the same 100 test questions.
#
#   bash scripts/reeval.sh TASK RUN [SEED ...]
#
#   TASK: count_primes  final prompts at 8,192 output tokens (Table 2)
#         ab            prompts trained at low effort, tested at medium effort
#                       with 4,096 output tokens (Table 2)
#         graph_color   final prompts at 8,192 output tokens (appendix)
#   RUN:  a run directory under outputs/reasoning_gym/TASK (e.g. textsteer,
#         wo_age), or "initial" for the initial prompt.
#   SEED: defaults to 47 48 49
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

task="${1:?usage: reeval.sh TASK RUN [SEED ...]}"
run_name="${2:?usage: reeval.sh TASK RUN [SEED ...]}"
shift 2
seeds=("$@")
[[ ${#seeds[@]} -gt 0 ]] || seeds=(47 48 49)

GRAPH_COLOR_CONFIG='{"graph_color": {"num_colors": 3, "min_num_vertices": 16, "max_num_vertices": 22, "edge_probability": 0.2}}'
case "$task" in
  count_primes)
    TASK_ARGS=(--task RG_count_primes --rg-difficulty medium --openrouter-reasoning medium)
    OVERRIDES=(--test-max-tokens 8192); tag=test8192 ;;
  ab)
    TASK_ARGS=(--task RG_ab --rg-difficulty medium)
    OVERRIDES=(--openrouter-reasoning medium --test-max-tokens 4096); tag=test4096_medium ;;
  graph_color)
    TASK_ARGS=(--task RG_graph_color --rg-config-json "$GRAPH_COLOR_CONFIG" --openrouter-reasoning medium)
    OVERRIDES=(--test-max-tokens 8192); tag=test8192 ;;
  *) echo "Unknown task '$task' (ab | count_primes | graph_color)" >&2; exit 2 ;;
esac

for seed in "${seeds[@]}"; do
  folder="$OUTPUT_DIR/reasoning_gym/$task/$run_name/seed-$seed"
  if [[ "$run_name" == initial ]]; then
    SOURCE_ARGS=(--prompt initial)
  else
    SOURCE_ARGS=(--prompt final --results "$folder/results.json")
  fi
  if [[ -f "$folder/$tag.json" ]]; then
    echo "Skipping $folder/$tag.json (exists)"
    continue
  fi
  mkdir -p "$folder"
  "$PYTHON" "$ROOT/textsteer/reeval.py" "${SOURCE_ARGS[@]}" --output "$folder/$tag.json" \
    "${TASK_ARGS[@]}" "${TRAIN_ARGS[@]}" "${GPT_OSS_ARGS[@]}" --no-diagnostic-preconditioning --seed "$seed" \
    "${OVERRIDES[@]}"
done
