#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Usage: scripts/reproduce_results/robustness_experiment.sh

Runs the robustness evaluation of Table 1 of the paper on the 32-bit canonical
configurations produced by scripts/reproduce_results/main_experiment.sh, with:
  - word deletion (10%-50% of the words)
  - synonym substitution (10%-50% of the words)
  - paraphrasing with GPT-5-mini (only if OPENAI_API_KEY is set)

Edited completions and their detection results are written to
output/llm_completions/<OUTPUT_TAG>_robustness_eval/.

Environment:
  OPENAI_API_KEY    API key used for the paraphrasing attack. If unset, the
                    paraphrasing column is skipped.
  OPENAI_BASE_URL   Base URL of the OpenAI-compatible API
                    (default: https://api.openai.com/v1)
  PARAPHRASE_MODEL  Paraphraser (default: gpt-5-mini-2025-08-07)
  INPUT_PATH        Input path (default: output/llm_completions/main-v2)
  BATCH_SIZE        Batch size for robustness editing (default: 64)
EOF
  exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

# Use the repository virtualenv if present; otherwise rely on the active env.
if [[ -f .venv/bin/activate ]]; then source .venv/bin/activate; fi

INPUT_PATH="${INPUT_PATH:-output/llm_completions/main-v2}"
BATCH_SIZE="${BATCH_SIZE:-64}"
OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://api.openai.com/v1}"
PARAPHRASE_MODEL="${PARAPHRASE_MODEL:-gpt-5-mini-2025-08-07}"

# Watermarks reported in Table 1 of the paper.
WATERMARKS=(
  BiMark
  M2Mark
  KGW-mpac
  MirrorMark
  RSBHWatermark
  StealthInk
  PPLMark-dynobino
)

args=(
  python scripts/run_robustness_eval.py
  --input_path "${INPUT_PATH}"
  --batch_size "${BATCH_SIZE}"
  --payload-size 32
  --word_deletion
  --synonym_substitution
)
for watermark in "${WATERMARKS[@]}"; do
  args+=(-w "${watermark}")
done

if [[ -n "${OPENAI_API_KEY:-}" ]]; then
  args+=(
    --paraphrase
    --model "${PARAPHRASE_MODEL}"
    --openai_url "${OPENAI_BASE_URL}"
    --api_key "${OPENAI_API_KEY}"
  )
else
  echo "OPENAI_API_KEY is not set: skipping the paraphrasing attack." >&2
fi

"${args[@]}"
