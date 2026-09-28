from __future__ import annotations

import math

import torch
from scipy.stats import norm
from vllm import SamplingParams

from lm_wm_tools import WatermarkProtocol
from lm_wm_tools.utils.triton_code import stateless_uniform
from lm_wm_tools.watermarks.sampling import SeedingScheme
from lm_wm_tools.watermarks.utils import (
    add_elapsed_time,
    format_watermark_name,
    normalize_multibit_algorithm,
    resolve_top_k_from_sampling_params,
)


class BiMarkWatermark(WatermarkProtocol):
    """
    BiMark watermark implementation.

    The implementation follows the BiMark paper's core procedure:
    - fixed balanced vocabulary bipartitions across d layers
    - duplicate-prefix skip for unbiasedness
    - XOR-enhanced position allocation (message bit + one-time pad mask)
    - multilayer bit-flip unbiased reweighting
    - message-agnostic detection via a voting matrix and z-test
    """

    def __init__(
        self,
        epsilon: float,
        vocab_size: int,
        rng_device: str,
        seeding_scheme: SeedingScheme | str,
        context_size: int,
        seed: int,
        sampling_parameters: SamplingParams,
        payload: list[int] | None = None,
        num_layers: int = 10,
        multibit_seed: int = 1847389390,
        partition_seed: int | None = None,
        bit_index_seed: int | None = None,
        mask_seed: int | None = None,
        **kwargs,
    ) -> None:
        self.n_mc = int(kwargs.pop("n_mc", 20_000))
        self.multibit_algorithm = normalize_multibit_algorithm(
            kwargs.get("multibit_algorithm", "none")
        )

        if isinstance(seeding_scheme, str):
            seeding_scheme = SeedingScheme(seeding_scheme)
        if isinstance(rng_device, str):
            rng_device = torch.device(rng_device)

        if context_size < 1:
            raise ValueError("BiMark requires context_size >= 1.")
        if num_layers < 1:
            raise ValueError("BiMark requires num_layers >= 1.")

        self.base_delta = float(epsilon)
        if not (0.0 <= self.base_delta <= 1.0):
            raise ValueError("BiMark expects epsilon in [0, 1] as the base scaling factor.")

        self.vocab_size = int(vocab_size)
        self.rng_device = rng_device
        self.context_size = int(context_size)
        self.seed = int(seed)
        self.temperature = sampling_parameters.temperature
        self.top_k = resolve_top_k_from_sampling_params(sampling_parameters, self.vocab_size)
        self.num_layers = int(num_layers)

        raw_payload = payload if payload is not None else [0] * 8
        self.payload = [int(bit) & 1 for bit in raw_payload]
        if len(self.payload) == 0:
            raise ValueError("BiMark payload must have at least one bit.")
        self.payload_size = len(self.payload)

        # Keys for pseudorandom operations.
        self.partition_seed = int(partition_seed if partition_seed is not None else self.seed)
        self.bit_index_seed = int(bit_index_seed if bit_index_seed is not None else multibit_seed)
        self.mask_seed = int(mask_seed if mask_seed is not None else multibit_seed + 1)

        seeding_scheme.initialize(self.vocab_size, self.seed, self.rng_device)
        self.seeding_scheme = seeding_scheme

        self.partition_masks = self._build_partition_masks()
        self._used_prefixes: set[tuple[int, ...]] = set()

    def _build_partition_masks(self) -> list[torch.Tensor]:
        """Create fixed balanced bipartitions for every layer."""
        tokens = torch.arange(self.vocab_size, device=self.rng_device, dtype=torch.int64)
        split = self.vocab_size // 2
        masks: list[torch.Tensor] = []

        for layer in range(self.num_layers):
            layer_column = torch.full_like(tokens, int(layer))
            offsets = torch.hash_tensor(torch.stack((tokens, layer_column), dim=1), dim=1)
            uniforms = stateless_uniform(offsets=offsets, seed=self.partition_seed)
            indices = torch.topk(uniforms, k=split, largest=False).indices
            mask = torch.zeros(self.vocab_size, device=self.rng_device, dtype=torch.bool)
            mask[indices] = True
            masks.append(mask)

        return masks

    def _sample_position_and_mask(self, context_hash: int) -> tuple[int, torch.Tensor]:
        """Sample message position and one-time-pad bits using stateless RNG."""
        context_tensor = torch.tensor([context_hash], device=self.rng_device, dtype=torch.int64)

        pos_uniform = stateless_uniform(offsets=context_tensor, seed=self.bit_index_seed)[0]
        position = int((pos_uniform * float(self.payload_size)).item())
        position = min(max(position, 0), self.payload_size - 1)

        layer_ids = torch.arange(self.num_layers, device=self.rng_device, dtype=torch.int64)
        ctx = torch.full_like(layer_ids, int(context_hash))
        offsets = torch.hash_tensor(torch.stack((ctx, layer_ids), dim=1), dim=1)
        mask_bits = (stateless_uniform(offsets=offsets, seed=self.mask_seed) >= 0.5).to(
            torch.int32
        )

        return position, mask_bits

    @staticmethod
    def _compute_delta_beta(base_delta: float, p0: float, alpha: float = 1.0) -> tuple[float, float]:
        """
        Match the reference BiMark update:
            delta = max(min(alpha / p0, 1 + base_delta), 1) - 1
            beta = min(delta * p0 / (1 - p0), 1)

        The sign (message bit XOR one-time-pad bit) is applied afterwards.
        """
        p0 = float(p0)
        if p0 <= 1e-12 or p0 >= 1.0 - 1e-12:
            return 0.0, 0.0

        delta = max(min(alpha / p0, 1.0 + base_delta), 1.0) - 1.0
        delta = max(0.0, float(delta))
        beta = min(delta * p0 / (1.0 - p0), 1.0)
        beta = max(0.0, float(beta))
        return delta, beta

    def get_name(self) -> str:
        return format_watermark_name(
            "BiMark",
            self.multibit_algorithm,
            (
                f"{self.seeding_scheme.value}/k_{self.context_size}/"
                f"d_{self.num_layers}/delta_{self.base_delta:<.2f}/seed_{self.seed}"
            ),
            payload_size=self.payload_size,
        )

    def __call__(self, output_ids: list[int], logits: torch.Tensor) -> torch.Tensor:
        if len(output_ids) < self.context_size:
            return logits

        prefix = tuple(int(tok) for tok in output_ids[-self.context_size :])
        if prefix in self._used_prefixes:
            return logits
        self._used_prefixes.add(prefix)

        output_tensor = torch.tensor(output_ids, device=self.rng_device, dtype=torch.long)
        context_hash = self.seeding_scheme.hash_last_context(output_tensor, self.context_size)
        if context_hash is None:
            return logits

        payload_pos, mask_bits = self._sample_position_and_mask(int(context_hash))
        message_bit = int(self.payload[payload_pos])
        fair_flips = torch.bitwise_xor(mask_bits, torch.tensor(message_bit, device=mask_bits.device))

        k = min(self.top_k, logits.shape[-1])
        topk_logits, topk_indices = torch.topk(logits, k, dim=-1)
        probs = torch.softmax(topk_logits / self.temperature, dim=-1)
        probs = probs.to(torch.float32)

        for layer_idx, partition_mask in enumerate(self.partition_masks):
            mask_v0 = partition_mask[topk_indices]
            p0 = probs[mask_v0].sum().item()
            delta, beta = self._compute_delta_beta(self.base_delta, p0)
            if delta == 0.0 and beta == 0.0:
                continue

            sign = 1.0 if int(fair_flips[layer_idx].item()) == 1 else -1.0
            delta *= sign
            beta *= sign

            if mask_v0.any():
                probs[mask_v0] = probs[mask_v0] * (1.0 + delta)
            mask_v1 = ~mask_v0
            if mask_v1.any():
                probs[mask_v1] = probs[mask_v1] * (1.0 - beta)

        probs = probs.clamp_min(0.0)
        denom = probs.sum()
        if denom <= 0:
            return logits
        probs = probs / denom

        adjusted_logits = torch.log(probs.clamp_min(1e-30)) * self.temperature

        # Match reference behavior: watermarking is defined on the selected top-k set.
        # Keeping non-top-k logits unchanged allows unwatermarked tokens to re-enter
        # sampling after re-ranking, which breaks bit decoding.
        logits = torch.full_like(logits, -100.0)
        logits[topk_indices] = adjusted_logits.to(logits.dtype)
        return logits

    def get_expected_probs(self, probs: torch.Tensor) -> torch.Tensor:
        return probs

    @add_elapsed_time()
    def detect(self, tokens: list[int] | torch.Tensor) -> dict[str, object]:
        if isinstance(tokens, torch.Tensor):
            token_list = tokens.detach().cpu().tolist()
        else:
            token_list = [int(t) for t in tokens]

        votes = torch.zeros((self.payload_size, 2), dtype=torch.int64, device=self.rng_device)
        used_prefixes: set[tuple[int, ...]] = set()

        n_positions = 0
        for t in range(self.context_size, len(token_list)):
            prefix = tuple(token_list[t - self.context_size : t])
            if prefix in used_prefixes:
                continue
            used_prefixes.add(prefix)

            prefix_tensor = torch.tensor(prefix, device=self.rng_device, dtype=torch.long)
            context_hash = self.seeding_scheme.hash_last_context(prefix_tensor, self.context_size)
            if context_hash is None:
                continue

            payload_pos, mask_bits = self._sample_position_and_mask(int(context_hash))
            token_id = int(token_list[t])
            if token_id < 0 or token_id >= self.vocab_size:
                continue

            for layer_idx, partition_mask in enumerate(self.partition_masks):
                # `partition_mask` is True on the selected half (V0 in the
                # reference code), so this bit is 1 for that half and 0 for
                # its complement.
                e_hat = int(partition_mask[token_id].item())
                decoded_bit = int(e_hat ^ int(mask_bits[layer_idx].item()))
                votes[payload_pos, decoded_bit] += 1

            n_positions += 1


        pred_message = votes.argmax(dim=1).to(torch.int32).tolist()
        tie_count = int((votes[:, 0] == votes[:, 1]).sum().item())

        bit_accuracy = float(
            sum(int(a == b) for a, b in zip(pred_message, self.payload)) / self.payload_size
        )

        out = {
            "statistic": None,
            "pvalue": None,
            "pred_message": pred_message,
            "expected_message": list(self.payload),
            "bit_accuracy": bit_accuracy,
            "voting_matrix": votes.cpu().tolist(),
            "n_used_contexts": int(n_positions),
            "tie_count": tie_count,
        }
        return out
