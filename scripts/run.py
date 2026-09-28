#!/usr/bin/env python
"""Run vLLM watermark sweeps with per-distribution epsilon defaults."""

import argparse
import json
import shlex
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Iterator
import traceback

REPO_ROOT = Path(__file__).resolve().parents[1]
GENERATE_VLLM = REPO_ROOT / "scripts" / "generate_vllm.py"


@dataclass(frozen=True)
class DistributionSpec:
    name: str
    parameters: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class DistributionSweep:
    epsilons: list[float]
    distribution: DistributionSpec | None = None


@dataclass(frozen=True)
class WatermarkSweep:
    distribution_sweeps: list[DistributionSweep] = field(default_factory=list)
    multibit_configs: list[dict[str, object]] = field(
        default_factory=lambda: [{}]
    )


WATERMARK_SWEEPS: dict[str, WatermarkSweep] = {

    "RSBHWatermark": WatermarkSweep(
        distribution_sweeps=[
            DistributionSweep(
                epsilons=[  1.0, 2.0, 3.0, 4.0, 6.0],
                distribution=DistributionSpec("binomial", {"total_count": 1, "probs": 0.5}),
            ),
        ],
        multibit_configs=[{"multibit_algorithm": "none"}],
    ),

    "StealthInk": WatermarkSweep(
        distribution_sweeps=[
            DistributionSweep(
                epsilons=[0.0],
                distribution=DistributionSpec("uniform", {"low": 0.0, "high": 1.0}),
            ),
        ],
        multibit_configs=[{"multibit_algorithm": "mpac"}],
    ),

    "KGW-mpac": WatermarkSweep(
        distribution_sweeps=[
            DistributionSweep(
                epsilons=[1.0,2.0,3.0,4.0, 5.0, 6.0],
                distribution=DistributionSpec("binomial", {"total_count": 1, "probs": 0.5}),
            ),
        ],
        multibit_configs=[{"multibit_algorithm": "mpac"}],
    ),

    "BiMark": WatermarkSweep(
        distribution_sweeps=[
            DistributionSweep(
                epsilons=[1.0],
            ),
        ],
    ),

    "M2Mark": WatermarkSweep(
        distribution_sweeps=[
            DistributionSweep(
                # Paper default number of layers m=10. The M2Mark processor maps this via epsilon
                # when num_layers is not set in the config.
                epsilons=[10.0],
            ),
        ],
    ),

    "PPLUnconstrained-bino": WatermarkSweep(
        distribution_sweeps=[
            DistributionSweep(
                epsilons=[0.5, 0.75, 1.0],
                distribution=DistributionSpec("binomial", {"total_count": 32, "probs": 0.5}),
            ),
        ],
        multibit_configs=[{"multibit_algorithm": "bino_encoder"}],
    ),

    "PPLMark-bino": WatermarkSweep(
        distribution_sweeps=[
            DistributionSweep(
                epsilons=[0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9],
                distribution=DistributionSpec("binomial", {"total_count": 32, "probs": 0.5}),
            ),
        ],
        multibit_configs=[{"multibit_algorithm": "bino_encoder"}],
    ),

    "PPLMark-dynobino": WatermarkSweep(
        distribution_sweeps=[
            DistributionSweep(
                epsilons=[0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9],
                distribution=DistributionSpec("binomial", {"total_count": 32, "probs": 0.5}),
            ),
        ],
        # Stateful scores averaged over the horizons T in {200, 300, 500, 1000, 2000}.
        multibit_configs=[{"multibit_algorithm": "dynamic_bino_encoder", "method": "prob", "remaining_bits": [200, 300, 500, 1000, 2000]}],
    ),

    "AAR": WatermarkSweep(
        distribution_sweeps=[
            DistributionSweep(
                epsilons=[0.0,0.1,0.2,0.3,0.4,0.5, 0.75, 1.0],
                distribution=DistributionSpec("gumbel", {"loc": 0, "scale": 1}),
            ),
        ],
        multibit_configs=[{"multibit_algorithm": "cycle_shift"}],
    ),

    "SynthID": WatermarkSweep(
        distribution_sweeps=[
            DistributionSweep(
                epsilons=[100],
                distribution=DistributionSpec("binomial", {"total_count": 30, "probs": 0.5}),
            ),
        ],
        multibit_configs=[
            {"multibit_algorithm": "bino_encoder"},
            {"multibit_algorithm": "dynamic_bino_encoder"},
        ],
    ),

    "MirrorMark": WatermarkSweep(
        distribution_sweeps=[
            DistributionSweep(
                # Paper default number of tournament layers.
                epsilons=[30.0],
                distribution=DistributionSpec("uniform", {"low": 0.0, "high": 1.0}),
            ),
        ],
        # Paper-best robust configuration uses CABS with m=2.
        multibit_configs=[{"multibit_algorithm": "cabs"}],
    ),

    "ArcMark": WatermarkSweep(
        distribution_sweeps=[
            DistributionSweep(
                # Dummy epsilon
                epsilons=[0],
                distribution=DistributionSpec("uniform", {"low": 0.0, "high": 1.0}),
            ),
        ],
    ),
}

DEFAULT_MODELS = ["meta-llama/Llama-3.1-8B-Instruct"]
DEFAULT_DATASETS = ["sentence-transformers/eli5"]
DEFAULT_N_SAMPLES = 1000
DEFAULT_CONTEXT_SIZES = [4]
DEFAULT_TOP_KS = [50]
DEFAULT_SEEDS = [0]
DEFAULT_RNG_DEVICES = ["cuda"]
DEFAULT_SEEDING_SCHEMES = ["sumhash"]


def available_mapping(watermark_types: Iterable[str]) -> dict[str, dict]:
    """Return a serializable mapping of watermarks to sweeps."""
    mapping = {}
    for wm in watermark_types:
        sweep = WATERMARK_SWEEPS[wm]
        mapping[wm] = {
            "distribution_sweeps": [
                {
                    "epsilons": dist_sweep.epsilons,
                    "distribution": (
                        asdict(dist_sweep.distribution)
                        if dist_sweep.distribution
                        else None
                    ),
                }
                for dist_sweep in sweep.distribution_sweeps
            ],
            "multibit_configs": sweep.multibit_configs,
        }
    return mapping


def resolve_watermark_class(watermark_key: str) -> str:
    return watermark_key.split("-", 1)[0]


def parse_payload_size(value: str) -> int:
    try:
        size = int(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("payload size must be an integer.") from exc
    if size < 0:
        raise argparse.ArgumentTypeError("payload size must be non-negative.")
    return size


def build_watermark_configs(
    watermark_type: str,
    sweep: WatermarkSweep,
    rng_devices: list[str],
    seeding_schemes: list[str],
    context_sizes: list[int],
    top_ks: list[int | None],
    seeds: list[int],
    payload_size: int | None,
    multibit_seed: int | None,
) -> Iterator[dict]:
    """Yield concrete watermark configs for each sweep combination."""
    if not sweep.distribution_sweeps:
        raise ValueError(f"No distribution sweeps configured for watermark {watermark_type}")

    multibit_configs = sweep.multibit_configs or [{}]
    for dist_sweep in sweep.distribution_sweeps:
        dist = dist_sweep.distribution
        for epsilon in dist_sweep.epsilons:
            for top_k in top_ks:
                for rng_device in rng_devices:
                    for seeding_scheme in seeding_schemes:
                        for context_size in context_sizes:
                            for seed in seeds:
                                for multibit_config in multibit_configs:
                                    config = {
                                        "epsilon": float(epsilon),
                                        "rng_device": rng_device,
                                        "seeding_scheme": seeding_scheme,
                                        "context_size": int(context_size),
                                        "seed": int(seed),
                                    }
                                    if top_k is not None:
                                        config["top_k"] = int(top_k)
                                    if dist:
                                        config["distribution_name"] = dist.name
                                        config["distribution_parameters"] = dist.parameters
                                    if multibit_config:
                                        config.update(multibit_config)
                                    if payload_size is not None:
                                        config["payload_size"] = int(payload_size)
                                    if (
                                        multibit_seed is not None
                                        and "multibit_seed" not in config
                                    ):
                                        config["multibit_seed"] = int(multibit_seed)
                                    yield config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run generate_vllm.py with watermark sweeps."
    )
    parser.add_argument(
        "--watermark-type",
        "-w",
        action="append",
        dest="watermark_types",
        choices=sorted(WATERMARK_SWEEPS),
        help=(
            "Watermark types to sweep. Defaults to all known types. "
            "If a type contains a dash (e.g., KGW-1), the base class is the prefix before the dash."
        ),
    )
    parser.add_argument(
        "--model",
        "-m",
        action="append",
        dest="models",
        help="Model(s) to run. Defaults to meta-llama/Llama-3.1-8B-Instruct.",
    )
    parser.add_argument(
        "--dataset",
        "-d",
        action="append",
        dest="datasets",
        help="Dataset(s) to run. Defaults to sentence-transformers/eli5.",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=DEFAULT_N_SAMPLES,
        help=f"Number of samples per run (default: {DEFAULT_N_SAMPLES}).",
    )
    parser.add_argument(
        "--context-size",
        action="append",
        dest="context_sizes",
        type=int,
        help="Context size(s) to use (default: 4).",
    )
    parser.add_argument(
        "--top-k",
        action="append",
        dest="top_ks",
        type=int,
        help="top_k override(s). Use --omit-top-k to drop this field entirely.",
    )
    parser.add_argument(
        "--omit-top-k",
        action="store_true",
        help="If set, top_k is not passed to generate_vllm.py.",
    )
    parser.add_argument(
        "--seed",
        action="append",
        dest="seeds",
        type=int,
        help="Seed(s) to use (default: 0).",
    )
    parser.add_argument(
        "--rng-device",
        action="append",
        dest="rng_devices",
        help="RNG device(s) to use (default: cuda).",
    )
    parser.add_argument(
        "--seeding-scheme",
        action="append",
        dest="seeding_schemes",
        help="Seeding scheme(s) to use (default: sumhash).",
    )
    parser.add_argument(
        "--payload-size",
        "--payload_size",
        "--paylod_size",
        action="append",
        nargs="+",
        dest="payload_sizes",
        type=parse_payload_size,
        help="Payload size(s) in bits; provide one or more values (e.g., --payload-size 10 16 32).",
    )
    parser.add_argument(
        "--multibit-seed",
        dest="multibit_seed",
        default=1847389390,
        type=int,
        help="Seed used for multibit payload positioning.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them.",
    )
    parser.add_argument(
        "--list-mapping",
        action="store_true",
        help="Print the epsilon/distribution mapping and exit.",
    )
    parser.add_argument(
        "--disable-metrics",
        action="store_true",
        help="Disable additional metrics computation",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="",
        help="Suffix directory to save generated outputs",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        help=(
            "Forwarded to generate_vllm.py as --max-tokens. "
            "If omitted, generate_vllm.py defaults are used."
        ),
    )
    parser.add_argument(
        "--min-tokens",
        type=int,
        help=(
            "Forwarded to generate_vllm.py as --min-tokens. "
            "If omitted, generate_vllm.py defaults are used."
        ),
    )
    parser.add_argument(
        "--max-generation-chunk-size",
        type=int,
        help=(
            "Forwarded to generate_vllm.py as --max-generation-chunk-size. "
            "Use this to split long fixed-length generations into balanced chunks."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    watermark_types = args.watermark_types or sorted(WATERMARK_SWEEPS)
    if args.list_mapping:
        print(json.dumps(available_mapping(watermark_types), indent=2))
        return

    models = args.models or DEFAULT_MODELS
    datasets = args.datasets or DEFAULT_DATASETS
    context_sizes = args.context_sizes or DEFAULT_CONTEXT_SIZES
    seeds = args.seeds or DEFAULT_SEEDS
    rng_devices = args.rng_devices or DEFAULT_RNG_DEVICES
    seeding_schemes = args.seeding_schemes or DEFAULT_SEEDING_SCHEMES
    if args.omit_top_k:
        top_ks: list[int | None] = [None]
    else:
        top_ks = args.top_ks or DEFAULT_TOP_KS
    payload_sizes = (
        [size for group in (args.payload_sizes or []) for size in group] or [None]
    )
    multibit_seed = args.multibit_seed

    for model in models:
        for dataset in datasets:
            for wm_type in watermark_types:
                sweep = WATERMARK_SWEEPS[wm_type]
                wm_class = resolve_watermark_class(wm_type)
                for payload_size in payload_sizes:
                    for config in build_watermark_configs(
                        wm_type,
                        sweep,
                        rng_devices,
                        seeding_schemes,
                        context_sizes,
                        top_ks,
                        seeds,
                        payload_size,
                        multibit_seed,
                    ):
                        cmd = [
                            "python",
                            str(GENERATE_VLLM),
                            "--model",
                            model,
                            "--dataset",
                            dataset,
                            "--n_samples",
                            str(args.n_samples),
                            "--watermark-class",
                            wm_class,
                            "--watermark-config",
                            json.dumps(config),
                            "--output_path",
                            args.output_path,
                            "--temperature",
                            str(args.temperature),
                        ]

                        if args.disable_metrics:
                            cmd.append("--disable-metrics")
                        if args.max_tokens is not None:
                            cmd.extend(["--max-tokens", str(args.max_tokens)])
                        if args.min_tokens is not None:
                            cmd.extend(["--min-tokens", str(args.min_tokens)])
                        if args.max_generation_chunk_size is not None:
                            cmd.extend(
                                [
                                    "--max-generation-chunk-size",
                                    str(args.max_generation_chunk_size),
                                ]
                            )

                        printable_cmd = " ".join(shlex.quote(part) for part in cmd)
                        print(f"Running: {printable_cmd}")
                        if args.dry_run:
                            continue
                        try:
                            subprocess.run(cmd, cwd=REPO_ROOT, check=True)
                        except Exception:
                            print(f"Command failed: {printable_cmd}")
                            traceback.print_exc()


if __name__ == "__main__":
    main()
