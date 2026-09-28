from __future__ import annotations

import argparse
import copy
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import SamplingParams

from lm_wm_tools.watermarks import get_watermark

UNEXPECTED_KWARG_PATTERN = re.compile(r"unexpected keyword argument '([^']+)'")

# Non-watermark fields that can appear in completion rows.
NON_CONFIG_FIELDS = {
    "watermark_config",
    "prompt",
    "dataset",
    "output_text",
    "output_ids",
    "output_length",
    "perplexity",
    "self_bleu",
    "statistic",
    "pvalue",
    "pvalue_original",
    "p_value_original",
    "z_score",
    "bit_match_statistic",
    "R",
    "mu_R",
    "sigma_R",
    "pred_message",
    "expected_message",
    "bit_accuracy",
    "voting_matrix",
    "valid_count",
    "n_used_contexts",
    "tie_count",
    "elapsed_ms",
    "entropies",
    "expected_scores",
    "temperature",
    "top_p",
    "max_tokens",
    "min_tokens",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rerun watermark detection for completions.jsonl files, using the "
            "row-level watermark config."
        )
    )
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to a directory containing completion jsonl files.",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="Tokenizer override used for all rows.",
    )
    parser.add_argument(
        "--default_model",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Fallback tokenizer/model name when a row does not include model_name.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Fallback sampling temperature for detection.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=-1,
        help="Fallback top_k for detection if not present in row config.",
    )
    return parser.parse_args()


def is_null(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, (dict, list, tuple)):
        return False
    try:
        return bool(pd.isna(value))
    except Exception:
        return False


def normalize_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()

    if isinstance(value, str):
        stripped = value.strip()
        if stripped and stripped[0] in {"{", "["}:
            try:
                return json.loads(stripped)
            except json.JSONDecodeError:
                return value
    return value


def parse_watermark_config_field(value: Any) -> dict[str, Any]:
    if isinstance(value, str):
        parsed = json.loads(value)
    else:
        parsed = value
    if not isinstance(parsed, dict):
        raise ValueError("watermark_config must decode to a dict.")
    return {key: normalize_value(val) for key, val in parsed.items() if not is_null(val)}


def extract_watermark_config(row: dict[str, Any]) -> dict[str, Any]:
    raw_config = row.get("watermark_config")
    if not is_null(raw_config):
        config = parse_watermark_config_field(raw_config)
    else:
        config = {}
        for key, value in row.items():
            if key in NON_CONFIG_FIELDS:
                continue
            if key.endswith("_pvalue") or key.endswith("_bit_accuracy"):
                continue
            if is_null(value):
                continue
            config[key] = normalize_value(value)

    # Ensure common fields survive even if watermark_config is partial.
    for key in ("watermark_class", "model_name", "vocab_size", "top_k"):
        if key not in config and not is_null(row.get(key)):
            config[key] = normalize_value(row[key])

    if "watermark_class" not in config:
        raise ValueError("Missing watermark_class in row watermark config.")

    return config


def coerce_token_ids(value: Any) -> list[int] | None:
    if is_null(value):
        return None

    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return None

    if isinstance(value, np.ndarray):
        value = value.tolist()

    if not isinstance(value, list):
        return None

    try:
        return [int(token) for token in value]
    except (TypeError, ValueError):
        return None


def detection_output_to_dict(detection_output: Any) -> dict[str, Any]:
    if isinstance(detection_output, dict):
        return dict(detection_output)

    to_dict = getattr(detection_output, "to_dict", None)
    if callable(to_dict):
        parsed = to_dict()
        if isinstance(parsed, dict):
            return dict(parsed)

    if hasattr(detection_output, "statistic") or hasattr(detection_output, "pvalue"):
        out: dict[str, Any] = {}
        statistic = getattr(detection_output, "statistic", None)
        pvalue = getattr(detection_output, "pvalue", None)
        if statistic is not None:
            out["statistic"] = statistic
        if pvalue is not None:
            out["pvalue"] = pvalue
        additional_info = getattr(detection_output, "additional_info", None)
        if isinstance(additional_info, dict):
            out.update(additional_info)
        return out

    if np.isscalar(detection_output):
        return {"statistic": float(detection_output)}

    return {}


def compute_bit_accuracy(result: dict[str, Any]) -> float | None:
    direct = result.get("bit_accuracy")
    if direct is not None:
        try:
            direct_value = float(direct)
            if np.isfinite(direct_value):
                return direct_value
        except (TypeError, ValueError):
            pass

    pred_message = result.get("pred_message")
    expected_message = result.get("expected_message")
    if pred_message is None or expected_message is None:
        return None

    try:
        pred_bits = list(pred_message)
        expected_bits = list(expected_message)
    except TypeError:
        return None

    if not expected_bits or len(pred_bits) != len(expected_bits):
        return None

    correct = sum(int(pred == expected) for pred, expected in zip(pred_bits, expected_bits))
    return correct / len(expected_bits)


def instantiate_detector(
    raw_config: dict[str, Any],
    *,
    temperature: float,
    top_k: int,
):
    config = copy.deepcopy(raw_config)
    while True:
        sampling_params = SamplingParams(
            temperature=temperature,
            top_k=top_k,
            extra_args=copy.deepcopy(config),
        )
        try:
            detector = get_watermark(config, sampling_params)
            return detector, config
        except TypeError as exc:
            match = UNEXPECTED_KWARG_PATTERN.search(str(exc))
            if match is None:
                raise
            bad_key = match.group(1)
            if bad_key not in config:
                raise
            config.pop(bad_key, None)


def choose_tokenizer_name(
    row: dict[str, Any],
    *,
    tokenizer_override: str | None,
    default_model: str,
) -> str:
    if tokenizer_override is not None:
        return tokenizer_override
    row_model_name = row.get("model_name")
    if row_model_name is not None and not is_null(row_model_name):
        return str(row_model_name)
    return default_model


def main(args: argparse.Namespace) -> None:
    input_path = Path(args.input_path)
    completion_files = sorted(input_path.glob("**/completions.jsonl"))
    print(f"Found {len(completion_files)} completion files to process.")
    if not completion_files:
        raise ValueError(f"No completion files found in {input_path}.")

    tokenizer_cache: dict[str, Any] = {}
    detector_cache: dict[str, Any] = {}

    for completion_file in completion_files:
        df = pd.read_json(completion_file, lines=True)

        if "output_ids" not in df.columns and "output_text" not in df.columns:
            raise KeyError(
                f"Expected 'output_ids' or 'output_text' in {completion_file}, "
                "but neither was found."
            )

        updates: dict[int, dict[str, Any]] = {}

        rows = df.to_dict(orient="records")
        iterator = tqdm(
            enumerate(rows),
            total=len(rows),
            desc=f"Detecting {completion_file.name}",
        )
        for row_index, row in iterator:
            token_ids = coerce_token_ids(row.get("output_ids"))
            if token_ids is None:
                if "output_text" not in row or is_null(row.get("output_text")):
                    raise ValueError(
                        f"Row {row_index} in {completion_file} has no output_ids and no output_text."
                    )
                tokenizer_name = choose_tokenizer_name(
                    row,
                    tokenizer_override=args.tokenizer,
                    default_model=args.default_model,
                )
                tokenizer = tokenizer_cache.get(tokenizer_name)
                if tokenizer is None:
                    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
                    tokenizer_cache[tokenizer_name] = tokenizer
                token_ids = tokenizer(
                    row["output_text"],
                    add_special_tokens=False,
                )["input_ids"]

            config = extract_watermark_config(row)
            temperature = float(
                normalize_value(row.get("temperature"))
                if not is_null(row.get("temperature"))
                else args.temperature
            )
            top_k = int(
                normalize_value(config.get("top_k"))
                if not is_null(config.get("top_k"))
                else args.top_k
            )

            cache_key = json.dumps(
                {"config": config, "temperature": temperature, "top_k": top_k},
                sort_keys=True,
                default=str,
            )
            detector = detector_cache.get(cache_key)
            if detector is None:
                detector, _ = instantiate_detector(
                    config,
                    temperature=temperature,
                    top_k=top_k,
                )
                detector_cache[cache_key] = detector

            try:
                detection_output = detector.detect(tokens=token_ids)
                result = detection_output_to_dict(detection_output)
            except Exception as exc:
                print(f"Detection failed for row {row_index} in {completion_file}: {exc}")
                result = {"statistic": np.nan, "pvalue": np.nan}

            bit_accuracy = compute_bit_accuracy(result)
            if bit_accuracy is not None:
                result["bit_accuracy"] = bit_accuracy

            updates[row_index] = result

        all_detection_keys = sorted({key for result in updates.values() for key in result})
        for key in all_detection_keys:
            if key not in df.columns:
                df[key] = np.nan

        for row_index, result in updates.items():
            for key, value in result.items():
                df.at[row_index, key] = value

        df.to_json(completion_file, orient="records", lines=True)

        pvalues = pd.to_numeric(df.get("pvalue"), errors="coerce")
        n_valid = int(pvalues.notna().sum())
        mean_pvalue = float(pvalues.mean()) if n_valid > 0 else float("nan")
        median_pvalue = float(pvalues.median()) if n_valid > 0 else float("nan")
        print(
            f"Saved updated detections to {completion_file} "
            f"(valid pvalues={n_valid}, mean={mean_pvalue:.6g}, median={median_pvalue:.6g})."
        )


if __name__ == "__main__":
    main(parse_args())
