#!/usr/bin/env bash
# Reasoning Gym experiments with GPT-OSS-20B (Table 2).
#
#   bash scripts/reasoning_gym.sh TASK VARIANT [SEED ...]
#
#   TASK:    ab | count_primes | graph_color
#   VARIANT: textsteer | wo_group | wo_diag | wo_age | wo_lr
#   SEED:    defaults to 47 48 49
#
# GraphColor's TextSteer configuration has no diagnostic preconditioning;
# DIAGNOSTICS=on adds it (and DIAGNOSTICS=off removes it on the other tasks).
# Test accuracy for Count Primes and A::B is reported by scripts/reeval.sh.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

task="${1:?usage: reasoning_gym.sh TASK VARIANT [SEED ...]}"
variant="${2:?usage: reasoning_gym.sh TASK VARIANT [SEED ...]}"
shift 2
seeds=("$@")
[[ ${#seeds[@]} -gt 0 ]] || seeds=(47 48 49)

GRAPH_COLOR_CONFIG='{"graph_color": {"num_colors": 3, "min_num_vertices": 16, "max_num_vertices": 22, "edge_probability": 0.2}}'
case "$task" in
  ab)
    TASK_ARGS=(--task RG_ab --rg-difficulty medium --openrouter-reasoning low)
    default_diagnostics=on ;;
  count_primes)
    TASK_ARGS=(--task RG_count_primes --rg-difficulty medium --openrouter-reasoning medium)
    default_diagnostics=on ;;
  graph_color)
    TASK_ARGS=(--task RG_graph_color --rg-config-json "$GRAPH_COLOR_CONFIG" --openrouter-reasoning medium)
    default_diagnostics=off ;;
  *) echo "Unknown task '$task' (ab | count_primes | graph_color)" >&2; exit 2 ;;
esac
DIAGNOSTICS="${DIAGNOSTICS:-$default_diagnostics}"

variant_args "$variant"
name="$variant"
DIAG_ARGS=()
[[ "$DIAGNOSTICS" == on ]] || DIAG_ARGS=(--no-diagnostic-preconditioning)
if [[ "$variant" != wo_diag && "$DIAGNOSTICS" != "$default_diagnostics" ]]; then
  name="${variant}_diagnostics_${DIAGNOSTICS}"
fi

for seed in "${seeds[@]}"; do
  run_seed "$OUTPUT_DIR/reasoning_gym/$task/$name/seed-$seed" "$seed" \
    "${TASK_ARGS[@]}" "${TRAIN_ARGS[@]}" "${GPT_OSS_ARGS[@]}" "${TEXTSTEER_ARGS[@]}" \
    ${VARIANT_ARGS[@]+"${VARIANT_ARGS[@]}"} ${DIAG_ARGS[@]+"${DIAG_ARGS[@]}"}
done
