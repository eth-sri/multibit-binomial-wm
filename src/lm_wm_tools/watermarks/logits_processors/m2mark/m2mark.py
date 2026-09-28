from __future__ import annotations

import itertools
import math
from typing import Any

import torch
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


class M2MarkWatermark(WatermarkProtocol):
    """
    MC2MARK-style multi-bit watermark.

    This implementation follows the paper's high-level construction:
    - segmented payload with per-step random segment selection
    - per-layer random mask xor whitening
    - multi-layer sequential reweighting (MLSR)
    - evidence-accumulation detector with normalized hit rates

    Notes:
    - For efficiency, reweighting is applied over the current top-k set.
    - Deterministic random components use shared stateless RNG utilities.
    """

    def __init__(
        self,
        vocab_size: int,
        rng_device: str | torch.device,
        seeding_scheme: SeedingScheme | str,
        context_size: int,
        seed: int,
        sampling_parameters: SamplingParams,
        payload: list[int] | None = None,
        epsilon: float | None = None,
        num_layers: int | None = None,
        segment_size: int | None = None,
        num_segments: int | None = None,
        multibit_seed: int = 1847389390,
        segment_seed: int | None = None,
        mask_seed: int | None = None,
        partition_seed: int | None = None,
        **kwargs: Any,
    ) -> None:
        self.max_permutations = int(kwargs.pop("max_permutations", 100_000))
        self.n_mc = int(kwargs.pop("n_mc", 20_000))
        self.multibit_algorithm = normalize_multibit_algorithm(
            kwargs.get("multibit_algorithm", "none")
        )
        del kwargs  # Unused compatibility args from generic config pipelines.

        if isinstance(seeding_scheme, str):
            seeding_scheme = SeedingScheme(seeding_scheme)
        if isinstance(rng_device, str):
            rng_device = torch.device(rng_device)

        if context_size < 1:
            raise ValueError("M2Mark requires context_size >= 1.")

        payload = [0] * 8 if payload is None else payload
        if len(payload) == 0:
            raise ValueError("M2Mark payload must contain at least one bit.")
        if any(bit not in (0, 1) for bit in payload):
            raise ValueError("M2Mark payload must be binary (0/1).")

        if num_layers is not None:
            resolved_layers = int(num_layers)
        elif epsilon is not None:
            resolved_layers = int(round(float(epsilon)))
        else:
            # Paper experimental default.
            resolved_layers = 10
        if resolved_layers < 1:
            raise ValueError("M2Mark requires num_layers >= 1.")

        payload_size = len(payload)
        resolved_segment_size, resolved_num_segments = self._resolve_segment_layout(
            payload_size,
            segment_size=segment_size,
            num_segments=num_segments,
        )

        self.vocab_size = int(vocab_size)
        self.rng_device = rng_device
        self.context_size = int(context_size)
        self.seed = int(seed)
        self.temperature = sampling_parameters.temperature
        if self.temperature is None or self.temperature <= 0:
            raise ValueError("M2Mark requires sampling temperature > 0.")
        self.top_k = resolve_top_k_from_sampling_params(sampling_parameters, self.vocab_size)

        self.num_layers = resolved_layers
        self.segment_size = resolved_segment_size
        self.num_segments = resolved_num_segments

        self.payload_size = payload_size
        self.payload = torch.tensor(payload, device=self.rng_device, dtype=torch.int32)
        self.payload_segments = self.payload.view(self.num_segments, self.segment_size)

        self.multibit_seed = int(multibit_seed)
        self.segment_seed = int(segment_seed if segment_seed is not None else self.multibit_seed)
        self.mask_seed = int(mask_seed if mask_seed is not None else self.multibit_seed + 1)
        self.partition_seed = int(
            partition_seed if partition_seed is not None else self.multibit_seed + 2
        )

        self._permutation_cache: dict[tuple[int, str, torch.dtype], torch.Tensor] = {}

        seeding_scheme.initialize(self.vocab_size, self.seed, self.rng_device)
        self.seeding_scheme = seeding_scheme

    @staticmethod
    def _default_segment_size(payload_size: int) -> int:
        target = min(8, payload_size)
        for size in range(target, 0, -1):
            if payload_size % size == 0:
                return size
        return 1

    @classmethod
    def _resolve_segment_layout(
        cls,
        payload_size: int,
        *,
        segment_size: int | None,
        num_segments: int | None,
    ) -> tuple[int, int]:
        if payload_size <= 0:
            raise ValueError("payload_size must be positive.")

        if segment_size is not None and num_segments is not None:
            if int(segment_size) * int(num_segments) != payload_size:
                raise ValueError(
                    "segment_size * num_segments must match payload length when both are set."
                )
            return int(segment_size), int(num_segments)

        if segment_size is not None:
            seg = int(segment_size)
            if seg < 1:
                raise ValueError("segment_size must be >= 1.")
            if payload_size % seg != 0:
                raise ValueError("payload length must be divisible by segment_size.")
            return seg, payload_size // seg

        if num_segments is not None:
            seg_count = int(num_segments)
            if seg_count < 1:
                raise ValueError("num_segments must be >= 1.")
            if payload_size % seg_count != 0:
                raise ValueError("payload length must be divisible by num_segments.")
            return payload_size // seg_count, seg_count

        seg = cls._default_segment_size(payload_size)
        return seg, payload_size // seg

    def _build_layer_keys(self, context_hash: int, device: torch.device) -> torch.Tensor:
        layer_ids = torch.arange(self.num_layers, device=device, dtype=torch.int64)
        contexts = torch.full_like(layer_ids, int(context_hash))
        pairs = torch.stack((contexts, layer_ids), dim=1)
        return torch.hash_tensor(pairs, dim=1).to(torch.int64)

    def _sample_segment_indices(self, layer_keys: torch.Tensor) -> torch.Tensor:
        uniforms = stateless_uniform(offsets=layer_keys, seed=self.segment_seed)
        segment_ids = (uniforms * float(self.num_segments)).to(torch.long)
        return segment_ids.clamp_(0, self.num_segments - 1)

    def _sample_mask_bits(self, layer_keys: torch.Tensor) -> torch.Tensor:
        local_ids = torch.arange(
            self.segment_size, device=layer_keys.device, dtype=torch.int64
        )
        key_grid = layer_keys.view(-1, 1).expand(-1, self.segment_size)
        id_grid = local_ids.view(1, -1).expand(layer_keys.numel(), -1)
        pair_offsets = torch.hash_tensor(torch.stack((key_grid, id_grid), dim=2), dim=2).to(
            torch.int64
        )
        uniforms = stateless_uniform(offsets=pair_offsets.reshape(-1), seed=self.mask_seed)
        return (uniforms >= 0.5).to(torch.int32).view(layer_keys.numel(), self.segment_size)

    def _partition_token_ids(
        self, layer_keys: torch.Tensor, token_ids: torch.Tensor
    ) -> torch.Tensor:
        token_ids = token_ids.to(device=layer_keys.device, dtype=torch.int64).reshape(-1)
        key_grid = layer_keys.view(-1, 1).expand(-1, token_ids.numel())
        tok_grid = token_ids.view(1, -1).expand(layer_keys.numel(), -1)
        pair_offsets = torch.hash_tensor(torch.stack((key_grid, tok_grid), dim=2), dim=2).to(
            torch.int64
        )
        uniforms = stateless_uniform(offsets=pair_offsets.reshape(-1), seed=self.partition_seed)
        subset_ids = (uniforms * float(self.segment_size)).to(torch.long)
        return subset_ids.view(layer_keys.numel(), token_ids.numel()).clamp_(
            0, self.segment_size - 1
        )

    def _get_weight_permutations(
        self, hamming_weight: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        cache_key = (int(hamming_weight), str(device), dtype)
        if cache_key in self._permutation_cache:
            return self._permutation_cache[cache_key]

        n_permutations = math.comb(self.segment_size, hamming_weight)
        if n_permutations > self.max_permutations:
            raise ValueError(
                "M2Mark permutation space is too large for exact Eq.(14) evaluation. "
                f"Got C({self.segment_size}, {hamming_weight})={n_permutations}, "
                f"limit={self.max_permutations}. Reduce segment_size or increase max_permutations."
            )

        combinations = list(itertools.combinations(range(self.segment_size), hamming_weight))
        permutations = torch.zeros(
            (len(combinations), self.segment_size),
            device=device,
            dtype=dtype,
        )
        for row_idx, cols in enumerate(combinations):
            permutations[row_idx, list(cols)] = 1.0

        self._permutation_cache[cache_key] = permutations
        return permutations

    def _compute_group_masses(
        self, probs: torch.Tensor, subset_ids: torch.Tensor
    ) -> torch.Tensor:
        masses = torch.zeros(
            self.segment_size, device=probs.device, dtype=probs.dtype
        )
        masses.index_add_(0, subset_ids, probs)
        return masses

    def _compute_scaling_factors(
        self, group_masses: torch.Tensor, local_payload: torch.Tensor
    ) -> torch.Tensor:
        payload_bits = local_payload.to(device=group_masses.device, dtype=group_masses.dtype)
        hamming_weight = int(payload_bits.sum().item())

        if hamming_weight <= 0 or hamming_weight >= self.segment_size:
            return torch.ones_like(group_masses)

        target_green_scale = float(self.segment_size) / float(hamming_weight)
        permutations = self._get_weight_permutations(
            hamming_weight=hamming_weight,
            device=group_masses.device,
            dtype=group_masses.dtype,
        )

        beta_all = permutations @ group_masses
        inv_beta = torch.where(
            beta_all > 1e-12,
            1.0 / beta_all,
            torch.full_like(beta_all, float("inf")),
        )
        target = torch.full_like(beta_all, target_green_scale)
        actual_green_scale = torch.minimum(target, inv_beta)
        overflow_scale = target - actual_green_scale

        overflow_weights = permutations.transpose(0, 1) @ overflow_scale
        overflow_masses = group_masses * overflow_weights

        beta_payload = torch.dot(payload_bits, group_masses)
        if float(beta_payload.item()) <= 1e-12:
            return torch.ones_like(group_masses)

        sa_payload = min(target_green_scale, 1.0 / float(beta_payload.item()))
        residual_mass = max(0.0, 1.0 - sa_payload * float(beta_payload.item()))

        new_group_masses = payload_bits * sa_payload * group_masses
        overflow_sum = float(overflow_masses.sum().item())
        if residual_mass > 0.0 and overflow_sum > 1e-12:
            new_group_masses = new_group_masses + (
                residual_mass * overflow_masses / overflow_masses.sum()
            )

        total = new_group_masses.sum()
        if (not torch.isfinite(total)) or float(total.item()) <= 1e-12:
            return torch.ones_like(group_masses)
        new_group_masses = new_group_masses / total

        scales = torch.ones_like(group_masses)
        valid = group_masses > 1e-12
        scales[valid] = new_group_masses[valid] / group_masses[valid]
        return scales

    def _apply_single_layer(
        self, probs: torch.Tensor, subset_ids: torch.Tensor, local_payload: torch.Tensor
    ) -> torch.Tensor:
        group_masses = self._compute_group_masses(probs, subset_ids)
        scales = self._compute_scaling_factors(group_masses, local_payload)
        updated = probs * scales[subset_ids]

        total = updated.sum()
        if (not torch.isfinite(total)) or float(total.item()) <= 1e-12:
            return probs
        return updated / total

    def _sample_layer_components(
        self, context_hash: int, token_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        layer_keys = self._build_layer_keys(context_hash, token_ids.device)
        segment_ids = self._sample_segment_indices(layer_keys)
        mask_bits = self._sample_mask_bits(layer_keys)
        subsets = self._partition_token_ids(layer_keys, token_ids)
        return layer_keys, segment_ids, mask_bits, subsets

    def get_name(self) -> str:
        return format_watermark_name(
            "M2Mark",
            self.multibit_algorithm,
            (
                f"{self.seeding_scheme.value}/k_{self.context_size}/"
                f"layers_{self.num_layers}/seg_{self.segment_size}/seed_{self.seed}"
            ),
            payload_size=self.payload_size,
        )

    @torch.no_grad()
    def __call__(self, output_ids: list[int], logits: torch.Tensor) -> torch.Tensor:
        squeeze_batch = False
        if logits.dim() == 2 and logits.size(0) == 1:
            logits = logits.squeeze(0)
            squeeze_batch = True
        elif logits.dim() != 1:
            raise ValueError(
                f"M2Mark expects 1-D logits (or [1, V]), got shape {tuple(logits.shape)}."
            )

        context_hash = self.seeding_scheme.hash_last_context(
            torch.tensor(output_ids, device=self.rng_device, dtype=torch.long),
            self.context_size,
        )
        if context_hash is None:
            return logits.unsqueeze(0) if squeeze_batch else logits

        logits_scaled = logits / self.temperature
        k = min(self.top_k, logits.shape[-1])
        topk_logits, topk_indices = torch.topk(logits_scaled, k, dim=-1)
        probs = torch.softmax(topk_logits, dim=-1).to(torch.float32)

        _, segment_ids, mask_bits, subsets = self._sample_layer_components(
            int(context_hash), topk_indices
        )

        payload_segments = self.payload_segments.to(device=probs.device, dtype=torch.int32)
        for layer_idx in range(self.num_layers):
            segment_idx = int(segment_ids[layer_idx].item())
            local_payload = torch.bitwise_xor(payload_segments[segment_idx], mask_bits[layer_idx])
            probs = self._apply_single_layer(probs, subsets[layer_idx], local_payload)

        updated_log_probs = torch.log(probs.clamp_min(1e-30))
        original_log_probs = torch.log_softmax(topk_logits, dim=-1)
        delta = updated_log_probs - original_log_probs

        logits_scaled = logits_scaled.clone()
        logits_scaled[topk_indices] = logits_scaled[topk_indices] + delta.to(logits_scaled.dtype)
        updated_logits = logits_scaled * self.temperature

        if squeeze_batch:
            return updated_logits.unsqueeze(0)
        return updated_logits

    def get_expected_probs(self, probs: torch.Tensor) -> torch.Tensor:
        return probs

    def _empty_detection_output(self) -> dict[str, object]:
        expected = self.payload.detach().cpu().tolist()
        return {
            "statistic": None,
            "pvalue": None,
            "pred_message": [0] * self.payload_size,
            "expected_message": expected,
            "bit_accuracy": 0.0,
            "n_trials": 0,
            "n_success": 0,
        }


    @add_elapsed_time()
    def detect(self, tokens: list[int] | torch.Tensor) -> dict[str, object]:
        if isinstance(tokens, list):
            tokens = torch.tensor(tokens, device=self.rng_device, dtype=torch.long)
        else:
            tokens = tokens.to(self.rng_device, dtype=torch.long)

        if tokens.numel() <= self.context_size:
            return self._empty_detection_output()

        context_hashes, observed_tokens = self.seeding_scheme.hash_context(tokens, self.context_size)
        if context_hashes.numel() == 0:
            return self._empty_detection_output()

        hit_0 = torch.zeros(self.payload_size, device=self.rng_device, dtype=torch.int64)
        hit_1 = torch.zeros(self.payload_size, device=self.rng_device, dtype=torch.int64)
        total_0 = torch.zeros(self.payload_size, device=self.rng_device, dtype=torch.int64)
        total_1 = torch.zeros(self.payload_size, device=self.rng_device, dtype=torch.int64)

        for step_idx in range(context_hashes.numel()):
            context_hash = int(context_hashes[step_idx].item())
            token_id = int(observed_tokens[step_idx].item())
            token_tensor = torch.tensor([token_id], device=self.rng_device, dtype=torch.long)

            layer_keys = self._build_layer_keys(context_hash, self.rng_device)
            segment_ids = self._sample_segment_indices(layer_keys)
            mask_bits = self._sample_mask_bits(layer_keys)
            active_subsets = self._partition_token_ids(layer_keys, token_tensor).squeeze(-1)

            for layer_idx in range(self.num_layers):
                segment_idx = int(segment_ids[layer_idx].item())
                mask = mask_bits[layer_idx]
                active_subset = int(active_subsets[layer_idx].item())

                active_global = segment_idx * self.segment_size + active_subset
                if 0 <= active_global < self.payload_size:
                    if int(mask[active_subset].item()) == 1:
                        hit_0[active_global] += 1
                    else:
                        hit_1[active_global] += 1

                start = segment_idx * self.segment_size
                end = min(start + self.segment_size, self.payload_size)
                if end <= start:
                    continue

                indices = torch.arange(start, end, device=self.rng_device, dtype=torch.long)
                local_mask = mask[: end - start].to(torch.int64)
                total_0[indices] += local_mask
                total_1[indices] += 1 - local_mask

        hit_rate_0 = hit_0.to(torch.float32) / total_0.clamp_min(1).to(torch.float32)
        hit_rate_1 = hit_1.to(torch.float32) / total_1.clamp_min(1).to(torch.float32)
        pred = (hit_rate_1 > hit_rate_0).to(torch.int32)

        payload = self.payload.to(device=self.rng_device, dtype=torch.int32)
        bit_accuracy = float((pred == payload).to(torch.float32).mean().item())

        return {
            "statistic": None,
            "pvalue": None,
            "pred_message": pred.detach().cpu().tolist(),
            "expected_message": payload.detach().cpu().tolist(),
            "bit_accuracy": bit_accuracy,
            "hit_rate_0": hit_rate_0.detach().cpu().tolist(),
            "hit_rate_1": hit_rate_1.detach().cpu().tolist(),
            "hit_count_0": hit_0.detach().cpu().tolist(),
            "hit_count_1": hit_1.detach().cpu().tolist(),
            "total_count_0": total_0.detach().cpu().tolist(),
            "total_count_1": total_1.detach().cpu().tolist(),
        }
