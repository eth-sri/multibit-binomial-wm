#!/usr/bin/env python
"""Run robustness evaluation on canonical watermark configs only."""

from __future__ import annotations

import argparse
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import evaluate_robustness
import run as sweep_run
from run_canonical_config import CANONICAL_EPSILONS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate robustness on canonical watermark configs only.",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Tokenizer to use for encoding completions.",
    )
    parser.add_argument(
        "--input_path",
        type=str,
        default=None,
        help="Path to a directory containing completion jsonl files.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Number of completions to process per batch.",
    )
    parser.add_argument(
        "--paraphrase",
        action="store_true",
        help="If set, will paraphrase the completions before computing p-values.",
    )
    parser.add_argument(
        "--translate",
        action="store_true",
        help="If set, will back-translate the completions before computing p-values.",
    )
    parser.add_argument(
        "--synonym_substitution",
        action="store_true",
        help="If set, will perform synonym substitution before computing p-values.",
    )
    parser.add_argument(
        "--ca_synonym_substitution",
        action="store_true",
        help="If set, will perform context-aware synonym substitution before computing p-values.",
    )
    parser.add_argument(
        "--word_deletion",
        action="store_true",
        help="If set, will perform word deletion before computing p-values.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-5-mini-2025-08-07",
        help="Model name to use for text editing.",
    )
    parser.add_argument(
        "--openai_url",
        type=str,
        default="https://api.openai.com/v1",
        help="Base URL for the OpenAI API.",
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default=None,
        help="API key for the OpenAI API.",
    )
    parser.add_argument(
        "--payload-size",
        "--payload_size",
        dest="payload_size",
        type=int,
        default=None,
        help="If set, only evaluate files generated with this payload size.",
    )
    parser.add_argument(
        "--watermark-type",
        "-w",
        action="append",
        dest="watermark_types",
        choices=sorted(CANONICAL_EPSILONS),
        help=(
            "Canonical watermark type(s) to evaluate (see scripts/run_canonical_config.py). "
            "Defaults to all canonical types."
        ),
    )
    return parser.parse_args()


@dataclass(frozen=True)
class CanonicalSpec:
    watermark_type: str
    watermark_class: str
    epsilon: float
    distribution_name: str | None
    distribution_parameters: dict[str, Any] | None
    multibit_config: dict[str, Any]

    def matches(self, row: dict[str, Any]) -> bool:
        if row.get("watermark_class") != self.watermark_class:
            return False
        if not _float_equal(row.get("epsilon"), self.epsilon):
            return False

        if self.distribution_name is not None:
            if row.get("distribution_name") != self.distribution_name:
                return False
            if not _values_equal(
                row.get("distribution_parameters"),
                self.distribution_parameters,
            ):
                return False

        for key, expected_value in self.multibit_config.items():
            if not _values_equal(row.get(key), expected_value):
                return False

        return True


def _float_equal(left: Any, right: Any, *, tolerance: float = 1e-9) -> bool:
    try:
        return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=tolerance)
    except (TypeError, ValueError):
        return False


def _normalize_json_like(value: Any) -> Any:
    if not isinstance(value, str):
        return value

    stripped = value.strip()
    if not stripped:
        return value
    if stripped[0] not in "{[":
        return value
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return value


def _values_equal(observed: Any, expected: Any) -> bool:
    observed = _normalize_json_like(observed)
    expected = _normalize_json_like(expected)

    if isinstance(expected, dict) and isinstance(observed, dict):
        if set(expected) != set(observed):
            return False
        return all(_values_equal(observed[key], expected[key]) for key in expected)

    if isinstance(expected, list) and isinstance(observed, list):
        if len(expected) != len(observed):
            return False
        return all(
            _values_equal(observed_item, expected_item)
            for observed_item, expected_item in zip(observed, expected)
        )

    numeric_types = (int, float)
    if isinstance(expected, numeric_types) and isinstance(observed, (int, float, str)):
        try:
            return _float_equal(observed, expected)
        except (TypeError, ValueError):
            return False

    return observed == expected


def _build_canonical_specs(
    watermark_types: list[str] | None = None,
) -> list[CanonicalSpec]:
    missing_from_sweep = [
        wm_type for wm_type in CANONICAL_EPSILONS if wm_type not in sweep_run.WATERMARK_SWEEPS
    ]
    if missing_from_sweep:
        raise ValueError(
            "Canonical map references watermark types that are missing from scripts/run.py: "
            f"{missing_from_sweep}"
        )

    specs: list[CanonicalSpec] = []
    for watermark_type, epsilon in CANONICAL_EPSILONS.items():
        if watermark_types and watermark_type not in watermark_types:
            continue
        sweep = sweep_run.WATERMARK_SWEEPS[watermark_type]
        selected_dist_sweep: sweep_run.DistributionSweep | None = None
        for dist_sweep in sweep.distribution_sweeps:
            if any(_float_equal(candidate, epsilon) for candidate in dist_sweep.epsilons):
                selected_dist_sweep = dist_sweep
                break

        if selected_dist_sweep is None:
            available_epsilons = [
                float(candidate)
                for dist_sweep in sweep.distribution_sweeps
                for candidate in dist_sweep.epsilons
            ]
            raise ValueError(
                f"Canonical epsilon {epsilon} for {watermark_type} is unavailable. "
                f"Available epsilons: {available_epsilons}"
            )

        distribution_name: str | None = None
        distribution_parameters: dict[str, Any] | None = None
        if selected_dist_sweep.distribution is not None:
            distribution_name = selected_dist_sweep.distribution.name
            distribution_parameters = dict(selected_dist_sweep.distribution.parameters)

        multibit_config = dict((sweep.multibit_configs or [{}])[0])

        specs.append(
            CanonicalSpec(
                watermark_type=watermark_type,
                watermark_class=sweep_run.resolve_watermark_class(watermark_type),
                epsilon=float(epsilon),
                distribution_name=distribution_name,
                distribution_parameters=distribution_parameters,
                multibit_config=multibit_config,
            )
        )

    return specs


def _read_first_record(path: Path) -> dict[str, Any] | None:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                return json.loads(stripped)
    return None


def _select_canonical_completion_files(
    input_root: Path,
    specs: list[CanonicalSpec],
    payload_size: int | None,
) -> list[Path]:
    selected: list[Path] = []
    for completion_file in sorted(input_root.glob("**/completions.jsonl")):
        row = _read_first_record(completion_file)
        if row is None:
            continue
        if payload_size is not None and _get_payload_size(row) != payload_size:
            continue
        if any(spec.matches(row) for spec in specs):
            selected.append(completion_file)
    return selected


def _get_payload_size(row: dict[str, Any]) -> int | None:
    if "payload_size" in row and row["payload_size"] is not None:
        try:
            return int(float(row["payload_size"]))
        except (TypeError, ValueError):
            return None

    payload = _normalize_json_like(row.get("payload"))
    if isinstance(payload, list):
        return len(payload)
    return None


def _prepare_output_files(
    input_root: Path,
    output_root: Path,
    completion_files: list[Path],
) -> list[Path]:
    prepared_files: list[Path] = []
    copied_files = 0
    reused_files = 0

    for completion_file in completion_files:
        relative_path = completion_file.relative_to(input_root)
        output_file = output_root / relative_path
        output_file.parent.mkdir(parents=True, exist_ok=True)

        if output_file.exists():
            if not output_file.is_file():
                raise ValueError(
                    f"Output path exists but is not a file: {output_file}"
                )
            reused_files += 1
        else:
            shutil.copy2(completion_file, output_file)
            copied_files += 1

        prepared_files.append(output_file)

    print(
        f"Prepared {len(prepared_files)} files in {output_root} "
        f"({copied_files} copied, {reused_files} reused)."
    )
    return prepared_files


def main() -> None:
    args = parse_args()
    if args.input_path is None:
        raise ValueError("--input_path is required.")
    if args.payload_size is not None and args.payload_size < 0:
        raise ValueError("--payload-size must be non-negative.")

    input_root = Path(args.input_path)
    if not input_root.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_root}")
    if not input_root.is_dir():
        raise ValueError(f"Input path must be a directory: {input_root}")

    canonical_specs = _build_canonical_specs(args.watermark_types)
    all_completion_files = sorted(input_root.glob("**/completions.jsonl"))
    selected_files = _select_canonical_completion_files(
        input_root,
        canonical_specs,
        args.payload_size,
    )

    print(
        "Canonical robustness filter selected "
        f"{len(selected_files)} / {len(all_completion_files)} completion files."
    )

    if not selected_files:
        raise ValueError(
            f"No canonical completion files found under {input_root}. "
            "Check --input_path and canonical mapping."
        )

    output_root = input_root.parent / f"{input_root.name}_robustness_eval"
    if output_root.exists():
        if not output_root.is_dir():
            raise ValueError(f"Output path exists but is not a directory: {output_root}")
    else:
        output_root.mkdir(parents=True, exist_ok=False)

    selected_output_files = _prepare_output_files(
        input_root,
        output_root,
        selected_files,
    )

    args.input_path = str(output_root)
    args.input_files = [str(path) for path in selected_output_files]
    evaluate_robustness.main(args)


if __name__ == "__main__":
    main()
