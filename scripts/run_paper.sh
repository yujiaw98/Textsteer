#!/usr/bin/env bash
# TextSteer and its ablations in Tables 1 and 2, then the
# Reasoning Gym test re-evaluations. Finished runs are skipped.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
SCRIPTS="$ROOT/scripts"

for model in gpt-oss-20b deepseek-v4-flash; do
  for variant in $VARIANTS; do
    bash "$SCRIPTS/finqa.sh" "$model" "$variant"
  done
done

for task in ab count_primes graph_color; do
  for variant in $VARIANTS; do
    # GraphColor's TextSteer configuration already runs without diagnostics.
    [[ "$task" == graph_color && "$variant" == wo_diag ]] && continue
    bash "$SCRIPTS/reasoning_gym.sh" "$task" "$variant"
  done
done
# GraphColor with diagnostic preconditioning added.
for variant in textsteer wo_group wo_age wo_lr; do
  DIAGNOSTICS=on bash "$SCRIPTS/reasoning_gym.sh" graph_color "$variant"
done

for task in count_primes ab; do
  for run in initial $VARIANTS; do
    bash "$SCRIPTS/reeval.sh" "$task" "$run"
  done
done
for run in initial textsteer textsteer_diagnostics_on; do
  bash "$SCRIPTS/reeval.sh" graph_color "$run"
done
