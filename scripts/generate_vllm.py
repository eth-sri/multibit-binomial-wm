import os
from vllm import LLM, SamplingParams
from lm_wm_tools.watermarks import WatermarkLogitsProcessor, get_watermark
from lm_wm_tools.watermarks.logits_processors.metrics import summarize_metrics
from transformers import AutoTokenizer
from datasets import load_dataset, Dataset
import argparse
import json
import random
import sys
import uuid
import tempfile
import time
from typing import Any

from loguru import logger

OUTPUT_PATH = "output/llm_completions"
SEGMENT_SEPARATOR = "##<>##"
DYNAMIC_BINO_ENCODER_ALGORITHM = "dynamic_bino_encoder"
DYNAMIC_BINO_ENCODER_MULTIGEN_ALGORITHM = "dynamic_bino_encoder_multigen"

logger.add(sys.stdout, colorize=True, format="<green>{time}</green> <level>{message}</level>")

class DatasetNotFoundError(Exception):
    """Raised when the dataset name is not recognized."""


class DatasetLoader:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def _process_chat_dataset(self, dataset: Dataset, n_samples: int, n_existing_samples: int, question_key: str = "question") -> list:
        prompts_ds = []
        skipped_prompts = 0
        for row in dataset:
            if skipped_prompts < n_existing_samples:
                skipped_prompts += 1
                continue
            text = row[question_key]
            prompts_ds.append([{"role": "user", "content": text}])
            if len(prompts_ds) >= n_samples:
                break
        return prompts_ds

    def _process_completion_dataset(self, dataset: Dataset, n_samples: int, n_existing_samples: int, text_key: str = "text") -> list:
        prompts_ds = []
        skipped_prompts = 0
        for row in dataset:
            text = row[text_key]
            tokenized_text = self.tokenizer(text)
            if len(tokenized_text["input_ids"]) < 200:
                continue
            if skipped_prompts < n_existing_samples:
                skipped_prompts += 1
                continue
            text = self.tokenizer.decode(tokenized_text["input_ids"][:200])
            prompts_ds.append(text)
            if len(prompts_ds) >= n_samples:
                break
        return prompts_ds

    def load(self, dataset_name: str, n_samples: int, n_existing_samples: int):
        if "c4" in dataset_name:
            is_chat = False
            ds = load_dataset(
                dataset_name, name="realnewslike", split="validation", streaming=False
            )
            prompts_ds = self._process_completion_dataset(ds, n_samples, n_existing_samples)
            return prompts_ds, is_chat
        if "eli5" in dataset_name:
            is_chat = True
            ds = load_dataset(
                dataset_name, split="train", streaming=False
            )
            prompts_ds = self._process_chat_dataset(ds, n_samples, n_existing_samples)
            return prompts_ds, is_chat

        if dataset_name == "diversity_eval":
            is_chat = True

            logger.warning("Diversity evaluation overrides the number of samples.")

            
            ds = load_dataset(
                "sentence-transformers/eli5", split="train", streaming=False
            )
            prompts_ds = self._process_chat_dataset(ds, n_samples=100, n_existing_samples=0)

            # Duplicate each prompt 100 times to get 10,000 samples
            expanded_prompts_ds = []
            for prompt in prompts_ds:
                for _ in range(100):
                    expanded_prompts_ds.append(prompt)
            
            prompts_ds = expanded_prompts_ds
            return prompts_ds, is_chat


        raise DatasetNotFoundError(f"Unknown dataset: {dataset_name}")


def parse_watermark_config(value: str) -> dict:
    """Allow passing either an inline JSON string or a path to a JSON file."""
    if os.path.exists(value):
        try:
            with open(value, "r") as f:
                return json.load(f)
        except json.JSONDecodeError as exc:
            raise argparse.ArgumentTypeError(
                f"Failed to decode JSON watermark config from file {value}: {exc}"
            ) from exc
    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(
            f"Failed to decode JSON watermark config from string: {exc}"
        ) from exc


def parse_payload_size(value: str) -> int:
    try:
        payload_size = int(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("payload_size must be an integer.") from exc
    if payload_size < 0:
        raise argparse.ArgumentTypeError("payload_size must be non-negative.")
    return payload_size


def sample_payload_bits(
    payload_size: int,
    *,
    payload_seed: int,
    sample_index: int,
) -> list[int]:
    if payload_size == 0:
        return []
    # Derive a deterministic per-sample seed so resuming from existing outputs
    # preserves the same payload assignment for each prompt index.
    derived_seed = (int(payload_seed) + 0x9E3779B97F4A7C15 * int(sample_index)) & (
        (1 << 64) - 1
    )
    rng = random.Random(derived_seed)
    return [rng.getrandbits(1) for _ in range(payload_size)]


def split_token_budget(total_tokens: int, max_chunk_size: int) -> list[int]:
    if total_tokens <= max_chunk_size:
        return [total_tokens]

    # Use the fewest chunks possible, then balance them so adjacent chunks differ
    # by at most one token.
    n_chunks = (total_tokens + max_chunk_size - 1) // max_chunk_size
    base_chunk_size = total_tokens // n_chunks
    remainder = total_tokens % n_chunks
    return [base_chunk_size + 1] * remainder + [base_chunk_size] * (n_chunks - remainder)


def get_generation_segment_lengths(
    *,
    max_tokens: int,
    min_tokens: int,
    max_generation_chunk_size: int | None,
) -> list[int]:
    if max_generation_chunk_size is None or max_tokens <= max_generation_chunk_size:
        return [max_tokens]

    if min_tokens != max_tokens:
        raise ValueError(
            "Chunked generation only supports fixed-length requests when "
            "--max-tokens exceeds --max-generation-chunk-size."
        )

    return split_token_budget(max_tokens, max_generation_chunk_size)


def prompt_to_text(prompt: Any) -> str:
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, list):
        parts = []
        for message in prompt:
            if isinstance(message, dict) and "content" in message:
                parts.append(str(message["content"]))
            else:
                parts.append(json.dumps(message, sort_keys=True))
        return "\n".join(parts)
    if isinstance(prompt, dict):
        return json.dumps(prompt, sort_keys=True)
    return str(prompt)


def merge_metrics(metrics_list: list[dict[str, Any]]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for metrics in metrics_list:
        for key, value in metrics.items():
            if isinstance(value, list):
                merged.setdefault(key, [])
                merged[key].extend(value)
            elif isinstance(value, (int, float)) and isinstance(merged.get(key), (int, float)):
                merged[key] += value
            elif key not in merged:
                merged[key] = value
            else:
                merged[key] = value
    return merged


def normalize_algorithm_name(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip().lower()


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate watermarking detection")
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Model to use for generation",
    )
    parser.add_argument(
        "--watermark_class",
        "--watermark-class",
        type=str,
        required=True,
        help="Watermark class to use",
    )
    parser.add_argument(
        "--watermark-config",
        "--watermark_config",
        dest="watermark_config",
        type=parse_watermark_config,
        required=True,
        help="Configuration parameters for the watermarking scheme",
    )
    parser.add_argument(
        "--multibit-algorithm",
        dest="multibit_algorithm",
        type=str,
        help="Random utility to use for sampling (e.g., mpac).",
    )
    parser.add_argument(
        "--payload-size",
        "--payload_size",
        dest="payload_size",
        type=parse_payload_size,
        help="Payload size in bits; payload bits are sampled uniformly per prompt.",
    )
    parser.add_argument(
        "--multibit-seed",
        dest="multibit_seed",
        type=int,
        help="Seed used for multibit payload positioning.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="allenai/c4",
        help="Dataset to use for evaluation",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=1000,
        help="Number of samples to use for evaluation",
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
        default=200,
        help="Maximum number of generated tokens per sample (default: 200).",
    )
    parser.add_argument(
        "--min-tokens",
        type=int,
        default=10,
        help="Minimum number of generated tokens per sample (default: 10).",
    )
    parser.add_argument(
        "--max-generation-chunk-size",
        type=int,
        default=None,
        help=(
            "If set, fixed-length requests larger than this value are split into "
            "balanced chunks, generated with distinct prompts, concatenated, and "
            "detected as a single sample."
        ),
    )
    args = parser.parse_args()
    if args.max_tokens <= 0:
        parser.error("--max-tokens must be > 0.")
    if args.min_tokens < 0:
        parser.error("--min-tokens must be >= 0.")
    if args.min_tokens > args.max_tokens:
        parser.error("--min-tokens must be <= --max-tokens.")
    if args.max_generation_chunk_size is not None:
        if args.max_generation_chunk_size <= 0:
            parser.error("--max-generation-chunk-size must be > 0.")
        if (
            args.max_tokens > args.max_generation_chunk_size
            and args.min_tokens != args.max_tokens
        ):
            parser.error(
                "--max-generation-chunk-size only supports fixed-length runs when "
                "--max-tokens exceeds the chunk size."
            )
    return args


def main(args):
    global OUTPUT_PATH

    if args.output_path:
        OUTPUT_PATH = f"{OUTPUT_PATH}/{args.output_path}"

    model_to_load = args.model
    tokenizer = AutoTokenizer.from_pretrained(model_to_load)

    watermark_parameters = dict(args.watermark_config)

    # Add the model_name to the watermark parameters
    # Useful for RSBH to load the BH mapping
    if "model_name" not in watermark_parameters:
        watermark_parameters["model_name"] = model_to_load

    if args.multibit_algorithm is not None:
        watermark_parameters["multibit_algorithm"] = args.multibit_algorithm
    if args.multibit_seed is not None:
        watermark_parameters["multibit_seed"] = args.multibit_seed
    if args.payload_size is not None:
        watermark_parameters["payload_size"] = args.payload_size

    fixed_payload = watermark_parameters.get("payload")
    dynamic_payload_mode = "payload_size" in watermark_parameters
    payload_size = None
    if dynamic_payload_mode:
        payload_size = int(watermark_parameters["payload_size"])
        watermark_parameters["payload_size"] = payload_size
        watermark_parameters.pop("payload", None)
    elif fixed_payload is not None:
        payload_size = len(fixed_payload)

    payload_seed = int(
        watermark_parameters.get(
            "payload_seed",
            watermark_parameters.get("multibit_seed", 0),
        )
    )

    watermark_parameters["watermark_class"] = args.watermark_class
    watermark_parameters.setdefault("vocab_size", len(tokenizer.get_vocab()))
    for key, value in watermark_parameters.items():
        if not hasattr(args, key):
            setattr(args, key, value)

    logits_processor = WatermarkLogitsProcessor

    sampling_parameters = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        min_tokens=args.min_tokens,
        top_k=watermark_parameters.get("top_k", -1),
        extra_args=watermark_parameters,
    )

    watermark_name_params = dict(watermark_parameters)
    if payload_size is not None and "payload" not in watermark_name_params:
        watermark_name_params["payload"] = [0] * payload_size
    watermark = get_watermark(watermark_name_params, sampling_parameters)
    segment_lengths = get_generation_segment_lengths(
        max_tokens=args.max_tokens,
        min_tokens=args.min_tokens,
        max_generation_chunk_size=args.max_generation_chunk_size,
    )
    split_generation = len(segment_lengths) > 1
    use_dynamic_bino_multigen = (
        split_generation
        and normalize_algorithm_name(watermark_parameters.get("multibit_algorithm"))
        == DYNAMIC_BINO_ENCODER_ALGORITHM
    )


    # Override the output path
    if args.dataset == "diversity_eval":
        OUTPUT_PATH = f"{OUTPUT_PATH}-diversity"


    # Check how many samples have already been generated
    # We count the number of rows with exactly the same watermark parameters + dataset
    path = f"{OUTPUT_PATH}/{watermark.get_name()}/completions.jsonl"

    logger.info(f"Checking for existing samples in {path}...")

    n_existing_samples = 0
    if os.path.exists(path):
        
        with open(path, "r") as f:
            for line in f:
                example = json.loads(line)
                match = True
                for key, value in watermark_parameters.items():

                    if key == "payload": # Ignore payload differences
                        continue

                    if example.get(key) != value:
                        match = False
                        break
                if example.get("dataset") != args.dataset:
                    match = False
                if split_generation and example.get("generation_segment_lengths") != segment_lengths:
                    match = False
                if match:
                    n_existing_samples += 1
    logger.info(f"Found {n_existing_samples} existing samples with the same config.")
    if n_existing_samples >= args.n_samples:
        logger.info("No new samples to generate. Exiting.")
        return
    args.n_samples -= n_existing_samples
    logger.info(f"Generating {args.n_samples} new samples.")

    lm = LLM(
        model_to_load,
        logits_processors=[logits_processor],
        max_num_seqs=64,
    )
    dataset_loader = DatasetLoader(tokenizer)
    prompts_per_sample = len(segment_lengths)
    requested_prompt_count = args.n_samples * prompts_per_sample
    prompts_ds, is_chat = dataset_loader.load(
        args.dataset, requested_prompt_count, n_existing_samples * prompts_per_sample
    )
    if len(prompts_ds) < requested_prompt_count:
        raise ValueError(
            f"Requested {requested_prompt_count} prompts but only found {len(prompts_ds)}."
        )

    top_k = watermark_parameters.get("top_k", -1)
   
    with tempfile.TemporaryDirectory() as temp_dir:
        dynamic_bino_multigen_state_dir = None
        sample_state_ids = None
        if use_dynamic_bino_multigen:
            dynamic_bino_multigen_state_dir = os.path.join(
                temp_dir, "dynamic_bino_multigen_state"
            )
            os.makedirs(dynamic_bino_multigen_state_dir, exist_ok=True)
            sample_state_ids = [str(uuid.uuid4()) for _ in range(args.n_samples)]

        sample_watermark_params = []

        for sample_offset in range(args.n_samples):
            req_wm_params = watermark_parameters.copy()
            sample_index = n_existing_samples + sample_offset
            if dynamic_payload_mode and payload_size is not None:
                req_wm_params["payload"] = sample_payload_bits(
                    payload_size,
                    payload_seed=payload_seed,
                    sample_index=sample_index,
                )
            elif fixed_payload is not None:
                req_wm_params["payload"] = [int(bit) for bit in fixed_payload]
            sample_watermark_params.append(req_wm_params)

        # Split generations run segment-by-segment so stateful watermark samplers can
        # carry state across prompt boundaries via the filesystem-backed multigen path.
        grouped_samples = [
            {
                "prompts": [],
                "output_parts": [],
                "token_ids": [],
                "metrics": [],
                "detector_wm_params": None,
            }
            for _ in range(args.n_samples)
        ]

        generation_start_time = time.perf_counter()
        for segment_index, segment_length in enumerate(segment_lengths):
            batch_requests = []
            for sample_offset, req_wm_params in enumerate(sample_watermark_params):
                prompt_index = sample_offset * prompts_per_sample + segment_index
                req_id = str(uuid.uuid4())
                generation_wm_params = req_wm_params.copy()

                if use_dynamic_bino_multigen:
                    generation_wm_params["multibit_algorithm"] = (
                        DYNAMIC_BINO_ENCODER_MULTIGEN_ALGORITHM
                    )
                    generation_wm_params["multigen_state_dir"] = (
                        dynamic_bino_multigen_state_dir
                    )
                    generation_wm_params["multigen_state_id"] = sample_state_ids[
                        sample_offset
                    ]
                    generation_wm_params["multigen_segment_id"] = req_id

                if not args.disable_metrics:
                    generation_wm_params["request_id"] = req_id
                    generation_wm_params["metrics_dir"] = temp_dir

                generation_min_tokens = (
                    segment_length if split_generation else args.min_tokens
                )
                sp = SamplingParams(
                    temperature=args.temperature,
                    max_tokens=segment_length,
                    min_tokens=generation_min_tokens,
                    top_k=top_k,
                    extra_args=generation_wm_params,
                )
                batch_requests.append(
                    {
                        "logical_sample_index": sample_offset,
                        "prompt": prompts_ds[prompt_index],
                        "request_id": req_id,
                        "sampling_params": sp,
                        "detector_wm_params": req_wm_params.copy(),
                    }
                )

            request_prompts = [request["prompt"] for request in batch_requests]
            batch_sampling_params = [
                request["sampling_params"] for request in batch_requests
            ]
            if is_chat:
                outputs = lm.chat(
                    messages=request_prompts, sampling_params=batch_sampling_params
                )
            else:
                outputs = lm.generate(request_prompts, sampling_params=batch_sampling_params)

            for request, output in zip(batch_requests, outputs):
                metrics = summarize_metrics(request["request_id"], temp_dir)
                candidate = output.outputs[0]
                sample = grouped_samples[request["logical_sample_index"]]
                sample["prompts"].append(request["prompt"])
                sample["output_parts"].append(candidate.text)
                sample["token_ids"].extend(candidate.token_ids)
                sample["metrics"].append(metrics)
                if sample["detector_wm_params"] is None:
                    sample["detector_wm_params"] = request["detector_wm_params"]

        generation_time = time.perf_counter() - generation_start_time


        # Collect prompts, generated text, and detection scores
        examples = []
        detection_time = 0.0
        for sample in grouped_samples:
            detector_wm_params = sample["detector_wm_params"]
            detector_sampling_params = SamplingParams(
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                min_tokens=args.min_tokens,
                top_k=top_k,
                extra_args=detector_wm_params,
            )
            detector = get_watermark(detector_wm_params, detector_sampling_params)
            detection_start_time = time.perf_counter()
            detection_scores = detector.detect(tokens=sample["token_ids"])
            detection_time += time.perf_counter() - detection_start_time
            prompt_value = sample["prompts"][0]
            output_text = sample["output_parts"][0]
            if split_generation:
                prompt_value = SEGMENT_SEPARATOR.join(
                    prompt_to_text(prompt) for prompt in sample["prompts"]
                )
                output_text = SEGMENT_SEPARATOR.join(sample["output_parts"])
            example = {
                "prompt": prompt_value,
                "dataset": args.dataset,
                "output_text": output_text,
                "output_length": len(sample["token_ids"]),
                **detection_scores,
                **merge_metrics(sample["metrics"]),
            }
            example.update(detector_wm_params)
            if split_generation:
                example["generation_segment_lengths"] = segment_lengths
            examples.append(example)

        for example in examples:
            example["generation_time"] = generation_time
            example["detection_time"] = detection_time
        # If the file exists, append to it
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if not os.path.exists(path):
            with open(path, "w") as f:
                pass
        with open(path, "a") as f:
            for example in examples:
                f.write(json.dumps(example) + "\n")


if __name__ == "__main__":
    args = parse_args()
    main(args)
