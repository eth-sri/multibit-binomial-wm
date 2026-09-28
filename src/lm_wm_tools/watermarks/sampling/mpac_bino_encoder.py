import torch
from scipy.stats import binomtest

from lm_wm_tools.utils.triton_code import stateless_uniform
from lm_wm_tools.watermarks.utils import add_elapsed_time
from .random_utility import RandomGenerator, SeedingScheme


class MPACBinoEncoder(RandomGenerator):
    """
    MPAC-style position allocation combined with BinoEncoder-style payload embedding.

    The payload is split into `n_segments`, and each context hash is assigned a segment.
    For each selected segment we sample `segment_size` Bernoulli bits and embed the local
    payload bits with the same XOR-style transform used by BinoEncoder.
    """

    def __init__(
        self,
        distribution_name: str = "binomial",
        distribution_parameters: dict = {"total_count": 8, "probs": 0.5},
        payload: list[int] = [0, 0, 0, 0, 0, 0, 0, 0],
        multibit_seed: int = 1847389390,
        n_segments: int | None = 4,
        segment_size: int | None = None,
        **kwargs,
    ) -> None:
        if len(payload) == 0:
            raise ValueError("payload must contain at least one bit.")
        if any(bit not in (0, 1) for bit in payload):
            raise ValueError("payload must be a list of bits (0/1).")

        num_segments = kwargs.pop("num_segments", None)
        if n_segments is None and num_segments is not None:
            n_segments = int(num_segments)
        elif (
            n_segments is not None
            and num_segments is not None
            and int(n_segments) != int(num_segments)
        ):
            raise ValueError("n_segments and num_segments must match when both are set.")

        payload_size = len(payload)
        segment_size, n_segments = self._resolve_segment_layout(
            payload_size=payload_size,
            segment_size=segment_size,
            n_segments=n_segments,
        )

        distribution_name, distribution_parameters = self.get_distribution(
            distribution_name,
            distribution_parameters,
            segment_size,
        )
        super().__init__(distribution_name, distribution_parameters)

        self.payload = torch.tensor(payload, dtype=torch.int32)
        self.payload_size = payload_size
        self.segment_size = segment_size
        self.n_segments = n_segments
        self.payload_segments = self.payload.view(self.n_segments, self.segment_size)
        self.multibit_seed = int(
            1847389390 if multibit_seed is None else multibit_seed
        )

    @staticmethod
    def _resolve_segment_layout(
        payload_size: int,
        *,
        segment_size: int | None,
        n_segments: int | None,
    ) -> tuple[int, int]:
        if payload_size <= 0:
            raise ValueError("payload_size must be positive.")

        if segment_size is not None and n_segments is not None:
            if int(segment_size) * int(n_segments) != payload_size:
                raise ValueError(
                    "segment_size * n_segments must match payload length when both are set."
                )
            return int(segment_size), int(n_segments)

        if segment_size is not None:
            segment_size = int(segment_size)
            if segment_size < 1:
                raise ValueError("segment_size must be >= 1.")
            if payload_size % segment_size != 0:
                raise ValueError("payload length must be divisible by segment_size.")
            return segment_size, payload_size // segment_size

        if n_segments is not None:
            n_segments = int(n_segments)
            if n_segments < 1:
                raise ValueError("n_segments must be >= 1.")
            if payload_size % n_segments != 0:
                raise ValueError("payload length must be divisible by n_segments.")
            return payload_size // n_segments, n_segments

        # Default: one segment means pure BinoEncoder behavior.
        return payload_size, 1

    def get_distribution(
        self,
        distribution_name: str,
        distribution_parameters: dict,
        segment_size: int,
    ) -> tuple[str, dict]:
        if (
            distribution_name != "binomial"
            or distribution_parameters.get("total_count") != segment_size
        ):
            pass  # Overridden to match per-segment Binomial semantics.

        distribution_name = "binomial"
        distribution_parameters = {"total_count": segment_size, "probs": 0.5}
        return distribution_name, distribution_parameters

    def _compute_position(self, context_hashes: torch.Tensor) -> torch.Tensor:
        offsets_flat = context_hashes.reshape(-1)
        uniform = stateless_uniform(seed=self.multibit_seed, offsets=offsets_flat)
        positions = (uniform * self.n_segments).to(torch.long).clamp_(0, self.n_segments - 1)
        return positions.reshape(context_hashes.shape)

    def _sample_from_offsets(
        self,
        seed: int,
        offsets: torch.Tensor,
        positions: torch.Tensor,
        *,
        embed_message: bool = True,
    ) -> torch.Tensor:
        offsets = offsets.reshape(-1)
        positions = positions.reshape(-1).to(torch.long)
        if offsets.numel() != positions.numel():
            raise ValueError("offsets and positions must have the same number of elements.")

        seeds = [seed + i for i in range(self.segment_size)]
        scores = torch.stack(
            [
                self.distribution.icdf_bernoulli(
                    stateless_uniform(seed=s, offsets=offsets)
                )
                for s in seeds
            ],
            dim=1,
        )  # shape: (N, segment_size)

        if not embed_message:
            return scores

        payload_segments = self.payload_segments.to(scores.device)
        payload_for_positions = payload_segments[positions]
        encoded_scores = (scores + (1 - payload_for_positions)) % 2
        return encoded_scores.sum(dim=1)

    def sample(
        self,
        seed: int,
        context_hashes: torch.Tensor | int,
        tokens: torch.Tensor,
        embed_message: bool = True,
    ) -> torch.Tensor:
        if isinstance(context_hashes, int):
            context_hashes = torch.tensor(
                [context_hashes], device=tokens.device, dtype=torch.int32
            )

        num_ctx = context_hashes.size(0)
        num_tok = tokens.size(0)

        offsets = self._compute_offsets(context_hashes, tokens)
        positions = self._compute_position(context_hashes).view(num_ctx, 1).expand(
            num_ctx, num_tok
        )

        return self._sample_from_offsets(
            seed=seed,
            offsets=offsets,
            positions=positions,
            embed_message=embed_message,
        )

    @add_elapsed_time()
    def detect(
        self,
        tokens: list[int] | torch.Tensor,
        seeding_scheme: SeedingScheme,
        rng_device: torch.device,
        context_size: int,
        seed: int,
    ) -> dict:
        if isinstance(tokens, list):
            tokens = torch.tensor(tokens, device=rng_device, dtype=torch.int32)

        tokens = tokens.to(rng_device)
        context, tokens = seeding_scheme.hash_context(tokens, context_size)
        pairs = torch.stack((context, tokens), dim=1)
        unique_pairs, inverse_indices = torch.unique(
            pairs, dim=0, sorted=False, return_inverse=True, return_counts=False
        )

        if unique_pairs.size(0) == 0:
            return {
                "pred_message": [0] * self.payload_size,
                "expected_message": self.payload.tolist(),
                "p_values_per_bit": [1.0] * self.payload_size,
                "segment_size": self.segment_size,
                "n_segments": self.n_segments,
                "scores_by_segment": torch.zeros(
                    (self.n_segments, self.segment_size), dtype=torch.float32
                ).tolist(),
                "counts_by_segment": [0] * self.n_segments,
                "positions": [],
                "index_pairs": [],
                "statistic": 0.0,
                "pvalue": 1.0,
            }

        index_pairs = [-1] * unique_pairs.size(0)
        for token_idx, pair_idx in enumerate(
            inverse_indices.detach().cpu().tolist(), start=context_size
        ):
            if index_pairs[pair_idx] == -1:
                index_pairs[pair_idx] = token_idx

        offsets = torch.hash_tensor(unique_pairs, dim=1)
        context_hashes = unique_pairs[:, 0]
        positions = self._compute_position(context_hashes).to(torch.long)
        raw_scores = self._sample_from_offsets(
            seed=seed,
            offsets=offsets,
            positions=positions,
            embed_message=False,
        )  # (N, segment_size)

        scores_by_segment = torch.zeros(
            (self.n_segments, self.segment_size),
            device=rng_device,
            dtype=torch.float32,
        )
        scores_by_segment.index_add_(0, positions, raw_scores.to(torch.float32))

        counts_by_segment = torch.bincount(positions, minlength=self.n_segments).to(
            device=rng_device, dtype=torch.float32
        )
        means_by_segment = scores_by_segment / counts_by_segment.clamp_min(1.0).unsqueeze(1)
        pred_segments = (means_by_segment >= 0.5).to(torch.int32)

        zero_mask = counts_by_segment == 0
        if zero_mask.any():
            pred_segments[zero_mask] = 0

        pred_message = pred_segments.reshape(-1)

        pred_bits_for_tokens = pred_segments[positions]
        predicted_scores = (raw_scores == pred_bits_for_tokens).to(torch.int32).sum(dim=1)
        out_stats = super().statistical_test(predicted_scores)

        payload_segments = self.payload_segments.to(rng_device, dtype=torch.int32)
        expected_bits_for_tokens = payload_segments[positions]
        per_token_bit_accuracy = (
            (raw_scores == expected_bits_for_tokens).to(torch.float32).mean(dim=1)
        )

        scores_by_segment_cpu = scores_by_segment.detach().cpu()
        counts_by_segment_cpu = counts_by_segment.detach().cpu()
        p_values_per_bit = []
        for segment_idx in range(self.n_segments):
            n_trials = int(counts_by_segment_cpu[segment_idx].item())
            for local_bit_idx in range(self.segment_size):
                if n_trials <= 0:
                    p_values_per_bit.append(1.0)
                    continue

                k_success = int(
                    round(float(scores_by_segment_cpu[segment_idx, local_bit_idx].item()))
                )
                p_values_per_bit.append(
                    float(
                        binomtest(
                            k=k_success,
                            n=n_trials,
                            p=0.5,
                            alternative="two-sided",
                        ).pvalue
                    )
                )

        out = {
            "pred_message": pred_message.tolist(),
            "expected_message": self.payload.tolist(),
            "p_values_per_bit": p_values_per_bit,
            "segment_size": self.segment_size,
            "n_segments": self.n_segments,
            "scores_by_segment": scores_by_segment.cpu().tolist(),
            "counts_by_segment": counts_by_segment.cpu().tolist(),
            "positions": positions.cpu().tolist(),
            "index_pairs": index_pairs,
            "per_token_bit_accuracy": per_token_bit_accuracy.cpu().tolist(),
        }
        out.update(out_stats)
        return out
