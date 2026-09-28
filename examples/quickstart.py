"""Embed a 32-bit message with our watermark (Ours+) in vLLM and decode it back.

Usage:
    python examples/quickstart.py [--model meta-llama/Llama-3.1-8B-Instruct]
"""

import argparse
import random

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from lm_wm_tools.watermarks import WatermarkLogitsProcessor, get_watermark


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--payload-size", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=350)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    message = [random.getrandbits(1) for _ in range(args.payload_size)]
    # Canonical low-distortion configuration of Ours+ (stateful binomial encoder).
    # Use "multibit_algorithm": "bino_encoder" (and drop "method"/"remaining_bits")
    # for the stateless version (Ours).
    watermark_config = {
        "watermark_class": "PPLMark",
        "epsilon": 0.0,
        "distribution_name": "binomial",
        "distribution_parameters": {"total_count": 32, "probs": 0.5},
        "multibit_algorithm": "dynamic_bino_encoder",
        "method": "prob",
        "remaining_bits": [200, 300, 500, 1000, 2000],
        "payload": message,
        "seeding_scheme": "sumhash",
        "context_size": 4,
        "seed": 0,
        "multibit_seed": 1847389390,
        "rng_device": "cuda",
        "top_k": 50,
        "model_name": args.model,
        "vocab_size": len(tokenizer.get_vocab()),
    }
    sampling_params = SamplingParams(
        temperature=0.7,
        top_k=50,
        max_tokens=args.max_tokens,
        min_tokens=args.max_tokens,
        extra_args=watermark_config,  # read by WatermarkLogitsProcessor
    )

    llm = LLM(args.model, logits_processors=[WatermarkLogitsProcessor], max_num_seqs=8)
    prompt = [{"role": "user", "content": "Why is the sky blue?"}]
    output = llm.chat(messages=[prompt], sampling_params=sampling_params)[0].outputs[0]

    # Decoding only needs the key (the config) and the generated token ids.
    detector = get_watermark(watermark_config, sampling_params)
    result = detector.detect(tokens=list(output.token_ids))

    decoded = [int(bit) for bit in result["pred_message"]]
    bit_accuracy = sum(d == m for d, m in zip(decoded, message)) / len(message)
    print(output.text)
    print(f"\nEmbedded message: {''.join(map(str, message))}")
    print(f"Decoded message:  {''.join(map(str, decoded))}")
    print(f"Bit accuracy: {bit_accuracy:.3f}")
    print(f"Zero-bit detection p-value: {result['pvalue']:.3e}")


if __name__ == "__main__":
    main()
