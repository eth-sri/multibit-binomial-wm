# Every Bit, Everywhere, All at Once: A Binomial Multibit LLM Watermark

Official code for the NeurIPS 2026 paper
[**Every Bit, Everywhere, All at Once: A Binomial Multibit LLM Watermark**](https://openreview.net/forum?id=E6Jp4SaVf6)
by Thibaud Gloaguen, Robin Staab, Mark Vero, and Martin Vechev.

We propose a multibit LLM watermark based on *binomial encoding*: every bit of the
payload is encoded at every token position. We complement it with a *stateful
encoder* that, during generation, redirects the encoding pressure towards
under-encoded bits. This repository contains:

- our watermark (**Ours**: binomial encoder; **Ours+**: stateful binomial encoder),
- re-implementations of the multibit baselines we compare against, all as
  [vLLM](https://github.com/vllm-project/vllm) logits processors,
- scripts to generate the data behind the main results of the paper.

## Table of Contents

- [Installation](#installation)
- [Quickstart](#quickstart)
- [Reproducing the Paper](#reproducing-the-paper)
- [Generating Watermarked Text](#generating-watermarked-text)
- [Repository Structure](#repository-structure)
- [License](#license)
- [Citation](#citation)

## Installation

The code was tested with Python 3.10, CUDA 12.8, `vllm==0.11.2`, and `torch==2.9.0`
on NVIDIA RTX PRO 6000 Blackwell GPUs (96 GB). We use [uv](https://docs.astral.sh/uv/):

```bash
uv venv --python 3.10 .venv
source .venv/bin/activate
uv pip install -r requirements.txt   # also installs this package (lm_wm_tools) in editable mode
```

The experiments use gated Hugging Face models
([Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)),
so request access and log in first:

```bash
hf auth login
```

## Quickstart

Embed a random 32-bit message with Ours+ and decode it back:

```bash
python examples/quickstart.py
```

```text
The sky appears blue because of a phenomenon called Rayleigh scattering, [...]

Embedded message: 00101001000010101000001110011001
Decoded message:  00001011111110101010001011000011
Bit accuracy: 0.625
Zero-bit detection p-value: 2.280e-01
```

The message is random and the output varies between runs. The example uses the
low-distortion setting (`epsilon` = 0) on a single 350-token reply; increase `epsilon`
or the number of tokens for a higher bit accuracy.

The watermark is a vLLM logits processor configured per request through
`SamplingParams.extra_args`. Decoding only requires the watermark configuration
(the key and the payload length) and the generated token ids; see
[`examples/quickstart.py`](examples/quickstart.py).

## Reproducing the Paper

All reproduction scripts live in [`scripts/reproduce_results/`](scripts/reproduce_results).
They generate the watermarked completions, decode them, and write everything
(generated text, decoded message, per-bit p-values, perplexity, ...) to one
`completions.jsonl` file per configuration; the `--help` of each script gives its
results directory. Every script prints its options with `--help`, and is resumable: rerunning it only
generates the samples that are still missing.

### Main Results

```bash
bash scripts/reproduce_results/main_results.sh
```

runs the three experiments below.

| Paper | Script |
|---|---|
| Fig. 2 (message accuracy vs. PPL), Fig. 4 (BA@1%FPR vs. PPL), bit accuracy vs. PPL | `main_experiment.sh` |
| Fig. 3 (message accuracy vs. #tokens), Fig. 5 (BA@1%FPR vs. #tokens) | `token_length_experiment.sh` |
| Table 1 (robustness) | `robustness_experiment.sh` (requires the outputs of `main_experiment.sh`) |

Notes:

- **Setup.** Llama-3.1-8B-Instruct, temperature 0.7, top-k 50, 1000 prompts
  from ELI5 per configuration, and bitstrings of 16, 32, and 64 bits sampled
  uniformly at random for each prompt. Perplexity is computed with
  Qwen3-30B-A3B-Instruct-2507-FP8.
- **Unsupported payloads.** ArcMark and CycleShift do not support 32- and 64-bit
  payloads: their configurations fail with an `AssertionError` and the sweep moves on
  to the next configuration.
- **Paraphrasing.** The paraphrasing column of Table 1 uses GPT-5-mini and
  requires `OPENAI_API_KEY`. Without it, only the word deletion and synonym
  substitution attacks are run.
- **GPUs.** Generation runs on a single GPU; select it with `CUDA_VISIBLE_DEVICES`.
  To spread the main sweep over several GPUs, split the payload sizes and compute
  the perplexity once at the end (`<results_dir>` is the results directory of
  `main_experiment.sh`):

  ```bash
  CUDA_VISIBLE_DEVICES=0 PAYLOAD_SIZES="16 64" RUN_PERPLEXITY=0 bash scripts/reproduce_results/main_experiment.sh &
  CUDA_VISIBLE_DEVICES=1 PAYLOAD_SIZES="32"    RUN_PERPLEXITY=0 bash scripts/reproduce_results/main_experiment.sh &
  wait
  python scripts/evaluate_perplexity.py --input_path <results_dir>
  ```

- **Runtime.** On one RTX PRO 6000, one configuration (1000 samples) takes between
  3 minutes (KGW-based schemes) and 35 minutes (SynthID), so the main sweep takes
  roughly 10 GPU-hours per bitstring length. The token-length sweep goes up to 5000
  tokens and takes several GPU-days; pass a subset of lengths to shorten it,
  e.g. `bash scripts/reproduce_results/token_length_experiment.sh 50 100 200 300 500`.
- **Quick check.** Set `N_SAMPLES` (e.g. `N_SAMPLES=10`) to run any experiment
  on fewer prompts.

### Metrics

The metrics reported in the paper are computed from the `completions.jsonl`
files by `lm_wm_tools.evaluation_utils.EvaluationMetrics`. Each line of a
`completions.jsonl` file holds one sample, including:

- `expected_message` / `pred_message`: embedded and decoded bitstrings,
- `p_values_per_bit`: per-bit p-values (used for BA@1%FPR),
- `pvalue`: zero-bit detection p-value,
- `perplexity`: added by `scripts/evaluate_perplexity.py`,
- `*_bit_accuracy` / `*_pvalue`: results after each text modification (robustness only).

For example, to compute the metrics of the main experiment for every configuration:

```python
import glob

import polars as pl

from lm_wm_tools.evaluation_utils import EvaluationMetrics

schema = {  # columns missing for a watermark (e.g. per-bit p-values for RSBH) are null
    "watermark_class": pl.Utf8, "multibit_algorithm": pl.Utf8, "payload_size": pl.Int64,
    "epsilon": pl.Float64, "pred_message": pl.List(pl.Int8), "expected_message": pl.List(pl.Int8),
    "p_values_per_bit": pl.List(pl.Float64), "pvalue": pl.Float64, "perplexity": pl.Float64,
}
results_dir = "<results_dir>"  # results directory of main_experiment.sh
paths = glob.glob(f"{results_dir}/**/completions.jsonl", recursive=True)
df = pl.concat([pl.read_ndjson(path, schema=schema) for path in paths]).with_columns(
    pl.col("multibit_algorithm").fill_null("none"),
    (pl.col("pvalue") < 1e-3).alias("tpr"),  # zero-bit detection
)
summary = EvaluationMetrics().calculate_metrics(
    df, group_by=["watermark_class", "multibit_algorithm", "payload_size"], quality_metric="perplexity"
)
print(summary[["watermark_class", "multibit_algorithm", "payload_size", "epsilon",
               "perplexity", "bit_accuracy", "message_accuracy", "ba_at_1pct_fpr"]].to_string())
```

## Generating Watermarked Text

[`scripts/run.py`](scripts/run.py) sweeps watermark configurations: it defines,
for each watermark, the strength (`epsilon`) values used in the paper and calls
[`scripts/generate_vllm.py`](scripts/generate_vllm.py) once per configuration.

```bash
# List the sweeps
python scripts/run.py --list-mapping

# Ours+ on 100 ELI5 prompts with 32-bit messages
python scripts/run.py -w PPLMark-dynobino --payload-size 32 --n-samples 100 \
    --max-tokens 350 --min-tokens 250 --disable-metrics --output_path my-run

# Add perplexity to the generated completions (<results_dir> is the directory run.py wrote to)
python scripts/evaluate_perplexity.py --input_path <results_dir>
```

To run a single configuration directly:

```bash
python scripts/generate_vllm.py --model meta-llama/Llama-3.1-8B-Instruct \
    --dataset sentence-transformers/eli5 --n_samples 100 \
    --watermark-class PPLMark --payload-size 32 \
    --max-tokens 350 --min-tokens 250 --disable-metrics \
    --watermark-config '{"epsilon": 0.0, "distribution_name": "binomial",
      "distribution_parameters": {"total_count": 32, "probs": 0.5},
      "multibit_algorithm": "dynamic_bino_encoder", "method": "prob", "remaining_bits": [200, 300, 500, 1000, 2000],
      "seeding_scheme": "sumhash", "context_size": 4, "seed": 0, "top_k": 50,
      "rng_device": "cuda", "multibit_seed": 1847389390}'
```

### Watermarks

| Name in the paper | `run.py` key | Implementation |
|---|---|---|
| **Ours** | `PPLMark-bino` | `watermarks/logits_processors/aar_extended/ppl_wm.py` + `watermarks/sampling/bino_encoder.py` |
| **Ours+** | `PPLMark-dynobino` | `watermarks/logits_processors/aar_extended/ppl_wm.py` + `watermarks/sampling/bino_encoder_dynamic.py` |
| ArcMark | `ArcMark` | `watermarks/logits_processors/arcmark/` |
| BiMark | `BiMark` | `watermarks/logits_processors/bimark/` |
| CycleShift | `AAR` | `watermarks/logits_processors/aar_extended/aar.py` + `watermarks/sampling/cycle_shift.py` |
| MC2Mark | `M2Mark` | `watermarks/logits_processors/m2mark/` |
| MirrorMark | `MirrorMark` | `watermarks/logits_processors/mirrormark/` + `watermarks/sampling/cabs.py` |
| MPAC | `KGW-mpac` | `watermarks/logits_processors/kgw/red_green.py` + `watermarks/sampling/mpac.py` |
| RSBH | `RSBHWatermark` | `watermarks/logits_processors/kgw/rsbh_watermark.py` |
| StealthInk | `StealthInk` | `watermarks/logits_processors/l1/stealthink.py` |
| SynthID (with our encoder) | `SynthID` | `watermarks/logits_processors/synthid/` |

Paths are relative to `src/lm_wm_tools/`. The Reed-Solomon code used by RSBH is
adapted from [generalizedReedSolomon](src/lm_wm_tools/watermarks/logits_processors/kgw/generalizedReedSolomon)
(MIT license), and the RSBH token mappings in `data/rsbh_data/` were generated
with the [official RSBH code](https://github.com/randomizedtree/segment-watermark).

## Repository Structure

```text
├── data/rsbh_data/            # Precomputed token mappings for RSBH
├── examples/quickstart.py     # Minimal embed / decode example
├── scripts/
│   ├── reproduce_results/     # One script per main experiment of the paper
│   ├── run.py                 # Watermark sweeps (epsilon values used in the paper)
│   ├── run_canonical_config.py# One canonical configuration per watermark
│   ├── generate_vllm.py       # Generation + decoding for one configuration
│   ├── evaluate_perplexity.py # Perplexity of generated completions
│   ├── run_robustness_eval.py # Text modifications + decoding (robustness)
│   ├── evaluate_robustness.py # (used by run_robustness_eval.py)
│   └── evaluate_detection.py  # Re-run decoding on existing completions
└── src/lm_wm_tools/
    ├── watermarks/            # vLLM logits processors and multibit encoders
    ├── robustness/            # Text modifications (deletion, substitution, paraphrasing)
    ├── metrics/               # Perplexity
    └── evaluation_utils.py    # Bit/message accuracy, BA@1%FPR, TPR, ...
```

## License

This code is released under the [MIT License](LICENSE).

## Citation

```bibtex
@inproceedings{gloaguen2026every,
  title={Every Bit, Everywhere, All At Once: A Binomial Multibit {LLM} Watermark},
  author={Thibaud Gloaguen and Robin Staab and Mark Vero and Martin Vechev},
  booktitle={The Fortieth Annual Conference on Neural Information Processing Systems},
  year={2026},
  url={https://openreview.net/forum?id=E6Jp4SaVf6}
}
```
