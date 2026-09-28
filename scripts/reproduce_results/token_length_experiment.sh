#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Usage: scripts/reproduce_results/token_length_experiment.sh [TOKEN_LENGTH...]

Runs the token-length sweep behind Fig. 3 (message accuracy) and Fig. 5
(BA@1%FPR) of the paper using one canonical configuration per watermark
(no epsilon/distribution sweep), then evaluates perplexity.

Lengths above 500 tokens are generated in balanced chunks of at most 500
tokens (with distinct prompts) that are concatenated and decoded jointly.

Examples:
  bash scripts/reproduce_results/token_length_experiment.sh
  bash scripts/reproduce_results/token_length_experiment.sh 64 128 256

Environment:
  BASE_OUTPUT_PATH  Output subdirectory under output/llm_completions
                    (default: main-token-length)
  N_SAMPLES         Number of samples per configuration (default: 1000)
  RUN_PERPLEXITY    If 1, run evaluate_perplexity.py after generation (default: 1)
  PPL_BATCH_SIZE    Batch size for evaluate_perplexity.py (default: 8)
EOF
  exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

# Use the repository virtualenv if present; otherwise rely on the active env.
if [[ -f .venv/bin/activate ]]; then source .venv/bin/activate; fi

BASE_OUTPUT_PATH="${BASE_OUTPUT_PATH:-main-token-length}"
N_SAMPLES="${N_SAMPLES:-1000}"
RUN_PERPLEXITY="${RUN_PERPLEXITY:-1}"
PPL_BATCH_SIZE="${PPL_BATCH_SIZE:-8}"

if (( $# == 0 )); then
  TOKEN_LENGTHS=(50 100 200 300 500 1000 2000 3000 5000)
else
  TOKEN_LENGTHS=("$@")
fi

for token_length in "${TOKEN_LENGTHS[@]}"; do
  if ! [[ "${token_length}" =~ ^[0-9]+$ ]] || (( token_length <= 0 )); then
    echo "Invalid token length: ${token_length}" >&2
    exit 1
  fi
done

for token_length in "${TOKEN_LENGTHS[@]}"; do
  output_path="${BASE_OUTPUT_PATH}/tokens-${token_length}"
  echo "Running token length ${token_length} -> ${output_path}"
  cmd=(
    python scripts/run_canonical_config.py
    -w MirrorMark
    -w KGW-mpac
    -w PPLMark-bino
    -w PPLMark-dynobino
    -w RSBHWatermark
    -w StealthInk
    -w BiMark
    -w M2Mark
    --disable-metrics
    --payload-size 32
    --n-samples "${N_SAMPLES}"
    --max-tokens "${token_length}"
    --min-tokens "${token_length}"
    --output_path "${output_path}"
  )

  if (( token_length > 500 )); then
    cmd+=(--max-generation-chunk-size 500)
  fi

  "${cmd[@]}"
done

if [[ "${RUN_PERPLEXITY}" == "1" ]]; then
  python scripts/evaluate_perplexity.py \
    --input_path "output/llm_completions/${BASE_OUTPUT_PATH}" \
    --batch_size "${PPL_BATCH_SIZE}"
fi
