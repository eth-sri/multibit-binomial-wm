#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Usage: scripts/reproduce_results/main_experiment.sh

Runs the fixed-length main sweep behind the detectability-quality trade-off
figures of the paper (Fig. 2: message accuracy, Fig. 4: BA@1%FPR, and the
bit-accuracy figure in the appendix), then evaluates perplexity on the
generated outputs.

Environment:
  OUTPUT_TAG      Output subdirectory under output/llm_completions
                  (default: main-v2)
  PAYLOAD_SIZES   Space-separated bitstring lengths (default: "16 32 64")
  N_SAMPLES       Number of prompts per configuration (default: 1000)
  MAX_TOKENS      Maximum generation length (default: 350)
  MIN_TOKENS      Minimum generation length (default: 250)
  RUN_PERPLEXITY  If 1, run evaluate_perplexity.py after generation (default: 1)
  PPL_BATCH_SIZE  Batch size for evaluate_perplexity.py (default: script default)

Example (split the payload sizes over two GPUs, then compute perplexity):
  CUDA_VISIBLE_DEVICES=0 PAYLOAD_SIZES="16 64" RUN_PERPLEXITY=0 bash scripts/reproduce_results/main_experiment.sh &
  CUDA_VISIBLE_DEVICES=1 PAYLOAD_SIZES="32"    RUN_PERPLEXITY=0 bash scripts/reproduce_results/main_experiment.sh &
  wait
  python scripts/evaluate_perplexity.py --input_path output/llm_completions/main-v2
EOF
  exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

# Use the repository virtualenv if present; otherwise rely on the active env.
if [[ -f .venv/bin/activate ]]; then source .venv/bin/activate; fi

OUTPUT_TAG="${OUTPUT_TAG:-main-v2}"
read -r -a PAYLOAD_SIZES <<< "${PAYLOAD_SIZES:-16 32 64}"
N_SAMPLES="${N_SAMPLES:-1000}"
MAX_TOKENS="${MAX_TOKENS:-350}"
MIN_TOKENS="${MIN_TOKENS:-250}"
RUN_PERPLEXITY="${RUN_PERPLEXITY:-1}"
PPL_BATCH_SIZE="${PPL_BATCH_SIZE:-}"

WATERMARKS=(
  MirrorMark
  ArcMark
  AAR
  KGW-mpac
  PPLMark-bino
  PPLMark-dynobino
  RSBHWatermark
  StealthInk
  BiMark
  M2Mark
  SynthID
)

run_args=(python scripts/run.py)
for watermark in "${WATERMARKS[@]}"; do
  run_args+=(-w "${watermark}")
done
run_args+=(
  --disable-metrics
  --payload-size "${PAYLOAD_SIZES[@]}"
  --n-samples "${N_SAMPLES}"
  --output_path "${OUTPUT_TAG}"
  --max-tokens "${MAX_TOKENS}"
  --min-tokens "${MIN_TOKENS}"
)

"${run_args[@]}"

if [[ "${RUN_PERPLEXITY}" == "1" ]]; then
  ppl_args=(
    python scripts/evaluate_perplexity.py
    --input_path "output/llm_completions/${OUTPUT_TAG}"
  )
  if [[ -n "${PPL_BATCH_SIZE}" ]]; then
    ppl_args+=(--batch_size "${PPL_BATCH_SIZE}")
  fi

  "${ppl_args[@]}"
fi
