#!/usr/bin/env bash
# FinQA experiments (Table 1).
#
#   bash scripts/finqa.sh MODEL VARIANT [SEED ...]
#
#   MODEL:   gpt-oss-20b | deepseek-v4-flash
#   VARIANT: textsteer | wo_group | wo_diag | wo_age | wo_lr
#   SEED:    defaults to 47 48 49 50 51
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

model="${1:?usage: finqa.sh MODEL VARIANT [SEED ...]}"
variant="${2:?usage: finqa.sh MODEL VARIANT [SEED ...]}"
shift 2
seeds=("$@")
[[ ${#seeds[@]} -gt 0 ]] || seeds=(47 48 49 50 51)

case "$model" in
  gpt-oss-20b) MODEL_ARGS=("${GPT_OSS_ARGS[@]}" --openrouter-reasoning medium) ;;
  deepseek-v4-flash) MODEL_ARGS=("${DEEPSEEK_ARGS[@]}") ;;
  *) echo "Unknown model '$model' (gpt-oss-20b | deepseek-v4-flash)" >&2; exit 2 ;;
esac

variant_args "$variant"

for seed in "${seeds[@]}"; do
  run_seed "$OUTPUT_DIR/finqa/$model/$variant/seed-$seed" "$seed" \
    --task FinQA "${TRAIN_ARGS[@]}" "${MODEL_ARGS[@]}" \
    "${TEXTSTEER_ARGS[@]}" ${VARIANT_ARGS[@]+"${VARIANT_ARGS[@]}"}
done
