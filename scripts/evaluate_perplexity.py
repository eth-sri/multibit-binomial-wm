from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from lm_wm_tools.metrics.quality_metrics import compute_perplexity

def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute the perplexity of generated completions."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-30B-A3B-Instruct-2507-FP8",
        help="Language model to use for perplexity computation.",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="Tokenizer to use for encoding completions. Defaults to --model.",
    )
    parser.add_argument(
        "--input_path",
        type=str,
        default=None,
        help="Path to a directory containing completion jsonl files."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Number of completions to process per batch.",
    )
    return parser.parse_args() 


def chunk_sequence(sequence: list[str], batch_size: int) -> Iterable[list[str]]:
    for start in range(0, len(sequence), batch_size):
        yield sequence[start : start + batch_size]


def main(args):
    tokenizer_name = args.tokenizer or args.model

    input_path = Path(args.input_path)
    input_paths = input_path.glob("**/completions.jsonl")

    n_inputs = len(list(input_paths))

    print(f"Found {n_inputs} completion files to process.")

    if n_inputs == 0:
        raise ValueError(f"No completion files found in {input_path}.")

    input_paths = input_path.glob("**/completions.jsonl")

    model = AutoModelForCausalLM.from_pretrained(args.model, device_map="auto")
    device = model.device
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    for completion_file in input_paths:

        df = pd.read_json(completion_file, lines=True)
        if "output_text" not in df.columns:
            raise KeyError(
                f"Expected column 'output_text' in {completion_file}, "
                "but it was not found."
            )

        if "perplexity" in df.columns:
            # Reorder the dataset to have computed perplexities first.
            mask = df["perplexity"].isnull()
            completions = df.loc[mask, "output_text"].tolist()
            perplexities_new = []
        else:
            mask = [True] * len(df)
            mask = np.array(mask, dtype=bool)
            completions = df["output_text"].tolist()
            perplexities_new: list[float] = []

        for batch in tqdm(chunk_sequence(completions, max(1, args.batch_size))):
            encodings = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=False,
            )
            input_ids = encodings["input_ids"].to(device)
            attention_mask = encodings["attention_mask"].to(device)

            perplexity = compute_perplexity(model, input_ids, attention_mask)
            perplexities_new.extend(perplexity)

        assert len(perplexities_new) == mask.sum(), "Mismatch between rows needing PPL and computed values."

        df.loc[mask, "perplexity"] = perplexities_new

        mean_perplexity = df["perplexity"].mean()
        median_perplexity = df["perplexity"].median()
        print(
            f"Computed perplexity for {len(perplexities_new)} completions "
            f"(mean={mean_perplexity:.4f}, median={median_perplexity:.4f})."
        )

        save_path = completion_file
        df.to_json(save_path, orient="records", lines=True)
        print(f"Saved results with perplexity to {save_path}.")


if __name__ == "__main__":
    main(parse_args())
