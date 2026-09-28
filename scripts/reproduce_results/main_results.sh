#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Usage: scripts/reproduce_results/main_results.sh

Generates the data behind the main results of the paper:
  1. main_experiment.sh          Figs. 2 and 4 (trade-off vs. perplexity)
  2. token_length_experiment.sh  Figs. 3 and 5 (scaling with the number of tokens)
  3. robustness_experiment.sh    Table 1 (robustness to text modifications)

All environment variables of the individual scripts are forwarded (e.g.
N_SAMPLES, PAYLOAD_SIZES, OPENAI_API_KEY). Positional arguments are forwarded
to token_length_experiment.sh as the list of token lengths.
EOF
  exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bash "${SCRIPT_DIR}/main_experiment.sh"
bash "${SCRIPT_DIR}/token_length_experiment.sh" "$@"
bash "${SCRIPT_DIR}/robustness_experiment.sh"
