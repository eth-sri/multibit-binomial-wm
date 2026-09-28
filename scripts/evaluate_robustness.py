import argparse
import json
from pathlib import Path
from typing import Iterable

import pandas as pd
import numpy as np
from transformers import AutoTokenizer, BertForMaskedLM, BertTokenizer

from vllm import SamplingParams

import evaluate_detection
from lm_wm_tools.robustness.text_editor import (
    TextParaphraser,
    TextEditor,
    TextBackTranslation,
    SynonymSubstitution,
    ContextAwareSynonymSubstitution,
    WordDeletion,
)
from lm_wm_tools.watermarks import get_watermark
from lm_wm_tools import WatermarkProtocol


DELETIONS = [0.1, 0.2, 0.3, 0.4, 0.5]
SUBSTITUTIONS = [0.1, 0.2, 0.3, 0.4, 0.5]
ROBUSTNESS_TEMPERATURE = 0.7
ROBUSTNESS_TOP_K = 50


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the watermark robustness.")
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
    return parser.parse_args()


def prepare_edits(args) -> dict[str, TextEditor]:
    text_editors = {}

    if args.paraphrase:
        text_editors["paraphrase_pvalue"] = TextParaphraser(
            model_name=args.model,
            openai_url=args.openai_url,
            api_key=args.api_key,
        )
    if args.translate:
        text_editors["backtranslation_pvalue"] = TextBackTranslation(
            model_name=args.model,
            openai_url=args.openai_url,
            api_key=args.api_key,
            language="French",
        )
    if args.synonym_substitution:
        for ratio in SUBSTITUTIONS:
            text_editors[f"synonym_substitution_{int(ratio*100)}_pvalue"] = SynonymSubstitution(
                ratio=ratio
            )
    if args.ca_synonym_substitution:
        for ratio in SUBSTITUTIONS:
            text_editors[f"ca_synonym_substitution_{int(ratio*100)}_pvalue"] = ContextAwareSynonymSubstitution(
                    ratio=ratio,
                    tokenizer=BertTokenizer.from_pretrained("bert-large-uncased"),
                    model=BertForMaskedLM.from_pretrained("bert-large-uncased", device_map="auto"),
                )
    if args.word_deletion:
        for ratio in DELETIONS:
            text_editors[f"word_deletion_{int(ratio*100)}_pvalue"] = WordDeletion(
                ratio=ratio
            )

    return text_editors


def get_mask(
    df: pd.DataFrame, fields: Iterable[str] | str
) -> tuple[pd.Series, Iterable[str]]:
    """Return mask of rows that need editing and the corresponding completions."""
    if isinstance(fields, str):
        fields = [fields]

    existing_fields = [field for field in fields if field in df.columns]
    if existing_fields:
        mask = df[existing_fields].isnull().any(axis=1)
    else:
        mask = pd.Series(True, index=df.index)

    completions = df.loc[mask, "output_text"].tolist()
    return mask, completions


def get_detection_value(detection_scores, key: str):
    if isinstance(detection_scores, dict):
        return detection_scores.get(key)
    if hasattr(detection_scores, key):
        return getattr(detection_scores, key)
    additional_info = getattr(detection_scores, "additional_info", None)
    if isinstance(additional_info, dict):
        return additional_info.get(key)
    return None


def compute_bit_accuracy(pred_message, expected_message):
    if pred_message is None or expected_message is None:
        return np.nan
    try:
        pred_bits = list(pred_message)
        expected_bits = list(expected_message)
    except TypeError:
        return np.nan
    if not expected_bits or len(pred_bits) != len(expected_bits):
        return np.nan
    correct = sum(int(p == e) for p, e in zip(pred_bits, expected_bits))
    return correct / len(expected_bits)


def _is_missing_config_value(value) -> bool:
    if value is None:
        return True
    if isinstance(value, (dict, list, tuple, set)):
        return False

    try:
        missing = pd.isna(value)
    except TypeError:
        return False

    if isinstance(missing, (bool, np.bool_)):
        return bool(missing)
    return False


def _extract_watermark_config(
    row: pd.Series, fields: Iterable[str]
) -> dict[str, object]:
    config: dict[str, object] = {}
    for field in fields:
        value = row[field]
        if _is_missing_config_value(value):
            continue
        config[field] = evaluate_detection.normalize_value(value)
    return config


def get_watermark_groups(
    df: pd.DataFrame,
) -> tuple[list[pd.DataFrame], list[WatermarkProtocol], list[str]]:
    required_watermark_parameter_fields = [
        "watermark_class",
        "epsilon",
        "rng_device",
        "seeding_scheme",
        "context_size",
        "seed",
        "vocab_size",
        "payload_size",
        "payload",
        "model_name",
        "multibit_seed",
    ]
    optional_watermark_parameter_fields = [
        "distribution_name",
        "distribution_parameters",
        "multibit_algorithm",
    ]
    watermark_parameter_fields = (
        required_watermark_parameter_fields + optional_watermark_parameter_fields
    )

    missing_required_fields = [
        field
        for field in required_watermark_parameter_fields
        if field not in df.columns
    ]
    if missing_required_fields:
        missing_fields = ", ".join(missing_required_fields)
        raise KeyError(
            f"Missing required watermark config columns: {missing_fields}"
        )

    present_watermark_parameter_fields = [
        field for field in watermark_parameter_fields if field in df.columns
    ]

    sampling_parameters = SamplingParams(
        temperature=ROBUSTNESS_TEMPERATURE,
        top_k=ROBUSTNESS_TOP_K,
    )

    # Group by the shared watermark setup, but keep payload row-specific so we can
    # still detect against the correct expected message after batch editing.
    grouping_fields = [
        field for field in present_watermark_parameter_fields if field != "payload"
    ]
    grouping_columns = df.loc[:, grouping_fields].astype(str)
    grouped_dfs = df.groupby(
        by=[grouping_columns[field] for field in grouping_fields],
        sort=False,
    )

    dfs: list[pd.DataFrame] = []
    watermarks: list[WatermarkProtocol] = []
    for _, group in grouped_dfs:
        watermark_config = _extract_watermark_config(
            group.iloc[0], present_watermark_parameter_fields
        )
        watermarks.append(
            get_watermark(
                watermark_config=watermark_config,
                sampling_params=sampling_parameters,
            )
        )
        dfs.append(group)

    return dfs, watermarks, present_watermark_parameter_fields


def get_detector_for_row(
    row: pd.Series,
    watermark_parameter_fields: Iterable[str],
    detector_cache: dict[str, WatermarkProtocol],
) -> WatermarkProtocol:
    watermark_config = _extract_watermark_config(row, watermark_parameter_fields)
    cache_key = json.dumps(
        {
            "config": watermark_config,
            "temperature": ROBUSTNESS_TEMPERATURE,
            "top_k": ROBUSTNESS_TOP_K,
        },
        sort_keys=True,
        default=str,
    )

    detector = detector_cache.get(cache_key)
    if detector is None:
        detector, _ = evaluate_detection.instantiate_detector(
            watermark_config,
            temperature=ROBUSTNESS_TEMPERATURE,
            top_k=ROBUSTNESS_TOP_K,
        )
        detector_cache[cache_key] = detector

    return detector


def batched_indices(indices: list[int], batch_size: int) -> Iterable[list[int]]:
    if batch_size < 1:
        raise ValueError("--batch_size must be a positive integer.")
    for start in range(0, len(indices), batch_size):
        yield indices[start : start + batch_size]


def resolve_input_paths(args) -> list[Path]:
    input_files = getattr(args, "input_files", None)
    if input_files is not None:
        return [Path(path) for path in input_files]

    if args.input_path is None:
        raise ValueError("--input_path is required.")

    return sorted(Path(args.input_path).glob("**/completions.jsonl"))


def main(args):
    tokenizer_name = args.tokenizer or args.model

    input_path = Path(args.input_path) if args.input_path is not None else None
    input_paths = resolve_input_paths(args)
    n_inputs = len(input_paths)

    print(f"Found {n_inputs} completion files to process.")

    if n_inputs == 0:
        if input_path is not None:
            raise ValueError(f"No completion files found in {input_path}.")
        raise ValueError("No completion files were provided.")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    text_editors = prepare_edits(args)
    if args.batch_size < 1:
        raise ValueError("--batch_size must be a positive integer.")

    for completion_file in input_paths:
        df = pd.read_json(completion_file, lines=True)
        if "output_text" not in df.columns:
            raise KeyError(
                f"Expected column 'output_text' in {completion_file}, "
                "but it was not found."
            )

        for field in text_editors:
            pvalue_field = field
            bit_accuracy_field = (
                field.replace("_pvalue", "_bit_accuracy")
                if field.endswith("_pvalue")
                else f"{field}_bit_accuracy"
            )
            for col in (pvalue_field, bit_accuracy_field):
                if col not in df.columns:
                    df[col] = np.nan

        print(f"Processing {completion_file} with {len(df)} completions.")
        try:
            detector_cache: dict[str, WatermarkProtocol] = {}
            dfs, watermarks, watermark_parameter_fields = get_watermark_groups(df)
            for watermark, group in zip(watermarks, dfs):
                print(
                    f"Processing watermark: {watermark.get_name()} with {len(group)} completions."
                )

                for field, text_editor in text_editors.items():
                    pvalue_field = field
                    bit_accuracy_field = (
                        field.replace("_pvalue", "_bit_accuracy")
                        if field.endswith("_pvalue")
                        else f"{field}_bit_accuracy"
                    )
                    group_indices = group.index
                    group_slice = df.loc[group_indices]
                    mask, _ = get_mask(
                        group_slice, [pvalue_field, bit_accuracy_field]
                    )
                    if not mask.any():
                        print(f"No completions to edit for field '{field}'. Skipping.")
                        continue

                    target_indices = group_slice.index[mask.to_numpy()].tolist()
                    for batch_indices in batched_indices(target_indices, args.batch_size):
                        completions = df.loc[batch_indices, "output_text"].tolist()
                        edited_completions = text_editor.edit_batch(completions)
                        tokenized_outputs = tokenizer(edited_completions)
                        pvalues = []
                        bit_accuracies = []
                        for row_index, completion in zip(
                            batch_indices, tokenized_outputs["input_ids"]
                        ):
                            detector = get_detector_for_row(
                                df.loc[row_index],
                                watermark_parameter_fields,
                                detector_cache,
                            )
                            try:
                                detection_scores = detector.detect(completion)
                            except Exception as e:
                                print(f"Detection failed with error: {e}")
                                detection_scores = {
                                    "pvalue": np.nan,
                                    "pred_message": None,
                                    "expected_message": None,
                                }

                            pvalue = get_detection_value(detection_scores, "pvalue")
                            if pvalue is None:
                                pvalue = np.nan
                            pvalues.append(pvalue)
                            pred_message = get_detection_value(
                                detection_scores, "pred_message"
                            )
                            expected_message = get_detection_value(
                                detection_scores, "expected_message"
                            )
                            bit_accuracies.append(
                                compute_bit_accuracy(pred_message, expected_message)
                            )

                        df.loc[batch_indices, pvalue_field] = pvalues
                        df.loc[batch_indices, bit_accuracy_field] = bit_accuracies

            save_path = completion_file
            df.to_json(save_path, orient="records", lines=True)
            print(f"Saved results to {save_path}.")
        except AssertionError as e:
            print(f"Skipping {completion_file} due to error: {e}")


if __name__ == "__main__":
    main(parse_args())
