"""SynthID watermarking utilities and vLLM logits processor."""

import torch
from vllm import SamplingParams

from lm_wm_tools import WatermarkProtocol
from lm_wm_tools.watermarks.sampling import (
    RandomUtilityType,
    SeedingScheme,
    create_random_utility,
)
from lm_wm_tools.watermarks.utils import (
    format_watermark_name,
    normalize_multibit_algorithm,
    resolve_top_k_from_sampling_params,
)

@torch.jit.script
def _depth_update_probs_eager(
    probs: torch.Tensor,
    scores: torch.Tensor,
    topk_logits: torch.Tensor,
) -> torch.Tensor:
    """Update top-k probabilities across all SynthID depth steps."""
    fallback_probs = torch.softmax(topk_logits, dim=-1)
    active = torch.tensor(True, device=probs.device, dtype=torch.bool)

    for t in range(scores.shape[1]):
        scores_topk = scores[:, t].to(dtype=probs.dtype)
        mu = -torch.sum(probs * scores_topk)

        max_score, min_score = scores_topk.max(), scores_topk.min()
        sigma = 1 / (max_score - min_score + 1e-12) # Condition for distortion-free 1 layer tournament

        candidate = probs * (1 + sigma * (scores_topk + mu))
        candidate = torch.clamp(candidate, min=0)
        candidate_sum = candidate.sum()
        safe_candidate = candidate / torch.clamp(candidate_sum, min=1e-12)
        invalid = torch.logical_not(torch.isfinite(candidate_sum)) | (candidate_sum <= 0)
        active = active & torch.logical_not(invalid)
        probs = torch.where(active, safe_candidate, fallback_probs)

    return probs

class SynthIDWatermark(WatermarkProtocol):
    """SynthID watermarking implementation for vLLM logits processors."""

    def __init__(
        self,
        context_size: int,
        seed: int,
        vocab_size: int,
        sampling_parameters: SamplingParams,
        rng_device: str | torch.device,
        seeding_scheme: SeedingScheme | str,
        epsilon: int = 30,  # This is the depth
        distribution_name: str = "binomial",
        distribution_parameters: dict | None = None,
        multibit_algorithm: RandomUtilityType | str = "none",
        payload: list[int] | None = None,
        multibit_seed: int = 0,
        **kwargs,
    ):
        """Initializes the logits processor.

        Args:
          context_size: N-gram length.
          seed: Watermark seed.
          depth: Depth of the watermark tree.
          rng_device: Device used for RNG state.
          skip_first_ngram_calls: Whether to skip the first ngram_len - 1 calls.
          apply_top_k: Whether to restrict to top-k tokens when watermarking.
          num_leaves: Number of leaves per node in the tournament tree.
        """

        algorithm_name = (
            multibit_algorithm.value
            if isinstance(multibit_algorithm, RandomUtilityType)
            else str(multibit_algorithm).strip().lower()
        )
        self.multibit_algorithm = normalize_multibit_algorithm(algorithm_name)

        payload = [0] if payload is None else payload
        if len(payload) == 0:
            raise ValueError("payload must contain at least one bit.")
        if any(bit not in (0, 1) for bit in payload):
            raise ValueError("payload must be a list of bits (0/1).")

        if distribution_parameters is None:
            distribution_parameters = {"total_count": len(payload), "probs": 0.5}
        else:
            distribution_parameters = dict(distribution_parameters)
        distribution_parameters["total_count"] = len(payload)

        if isinstance(seeding_scheme, str):
            seeding_scheme = SeedingScheme(seeding_scheme)

        if isinstance(rng_device, str):
            rng_device = torch.device(rng_device)

        self.payload_size = len(payload)
        self.payload = torch.tensor(payload, device=rng_device, dtype=torch.int32)
        self.seed = int(seed)

        # Scheme parameters
        self.depth = int(epsilon)
        if self.depth <= 0:
            raise ValueError("epsilon (depth) must be a positive integer.")
        self._depth_seed_cache = tuple(self._depth_seeds())

        # Sampling parameters
        self.temperature = sampling_parameters.temperature
        if self.temperature is None or self.temperature <= 0:
            raise ValueError("SynthIDWatermark requires sampling temperature > 0.")
        self.vocab_size = vocab_size
        self.rng_device = rng_device
        self.top_k = resolve_top_k_from_sampling_params(
            sampling_parameters, vocab_size
        )

        # Initialize the seeding scheme
        seeding_scheme.initialize(vocab_size, seed, rng_device)
        self.seeding_scheme = seeding_scheme
        self.context_size = context_size

        # Intialize the RNG
        self.random_generator = create_random_utility(
            algorithm_name,
            distribution_name,
            distribution_parameters,
            payload=payload,
            multibit_seed=multibit_seed,
            vocab_size=vocab_size,
            **kwargs,
        )
        self.distribution_name = distribution_name
        self.distribution_parameters = distribution_parameters
        # The context hash used to score the previous step; consumed on the next call.
        self._pending_state_context_hash: int | None = None
        

    def _depth_seeds(self) -> list[int]:
        """Use non-overlapping windows so depth draws stay independent across bits."""
        stride = max(1, self.payload_size)
        return [self.seed + i * stride for i in range(self.depth)]

    def get_name(self) -> str:
        return format_watermark_name(
            "SynthID",
            self.multibit_algorithm,
            f"k_{self.context_size}/seed_{self.seed}/scheme_{self.seeding_scheme.value}/topk_{self.top_k}",
            payload_size=self.payload_size,
        )
    
    def _sample_all_depths(
        self,
        context_hashes: torch.Tensor,
        tokens: torch.Tensor,
        *,
        embed_message: bool,
    ) -> torch.Tensor:
        tokens = tokens.reshape(-1)
        n_tokens = int(tokens.numel())

        scores = self.random_generator.sample(
            seed=self.seed,
            context_hashes=context_hashes,
            tokens=tokens,
            embed_message=embed_message,
            depth=self.depth,
        )

        if embed_message:
            flat_scores = scores.reshape(-1)
            expected = n_tokens * self.depth
            if flat_scores.numel() != expected:
                raise RuntimeError(
                    "SynthID embed-time scores have unexpected shape: "
                    f"got {tuple(scores.shape)} ({flat_scores.numel()} elems), "
                    f"expected {expected} elems for ({n_tokens}, {self.depth})."
                )
            # BinoEncoder depth sampling is produced in (depth, T) layout.
            return flat_scores.reshape(self.depth, n_tokens).transpose(0, 1).contiguous()

        expected = n_tokens * self.depth * self.payload_size
        if scores.numel() != expected:
            raise RuntimeError(
                "SynthID detection-time scores have unexpected shape: "
                f"got {tuple(scores.shape)} ({scores.numel()} elems), "
                f"expected {expected} elems for ({n_tokens}, {self.payload_size}, {self.depth})."
            )
        return (
            scores.reshape(self.depth, n_tokens, self.payload_size)
            .permute(1, 2, 0)
            .contiguous()
        )


    def get_gvalues(self, context_hash: int, tokens: torch.Tensor) -> torch.Tensor:
        """Get the G values for all tokens given a context hash."""

        tokens = tokens.reshape(-1)
        context_hashes = torch.tensor(
            [context_hash], device=tokens.device, dtype=torch.int32
        )

        distribution = self._sample_all_depths(
            context_hashes=context_hashes,
            tokens=tokens,
            embed_message=True,
        )

        return distribution

    @torch.no_grad()
    def __call__(
        self,
        output_ids: list[int],
        logits: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """Applies the SynthID watermark to the provided logits."""

        if self._pending_state_context_hash is not None and len(output_ids) > 0:
            self.random_generator.update_state(
                context_hash=self._pending_state_context_hash,
                token=output_ids[-1],
                rng_device=self.rng_device,
                seed=self.seed,
            )
            self._pending_state_context_hash = None

        squeeze_batch = False
        if logits.dim() == 2 and logits.size(0) == 1:
            logits = logits.squeeze(0)
            squeeze_batch = True
        elif logits.dim() != 1:
            raise ValueError(
                f"SynthIDWatermark expects 1-D logits (or [1, V]), got shape {tuple(logits.shape)}."
            )

        context_hash = self.seeding_scheme.hash_last_context(
            torch.tensor(output_ids, device=self.rng_device), self.context_size
        )
        if context_hash is not None:

            logits_scaled = logits / self.temperature

            k = min(self.top_k, logits.shape[-1])
            topk_logits, topk_indices = torch.topk(logits_scaled, k, dim=-1)
            probs = torch.softmax(topk_logits, dim=-1)
            scores = self.get_gvalues(context_hash, topk_indices)
            
            probs = _depth_update_probs_eager(probs, scores, topk_logits)

            updated_log_probs = torch.log(probs)
            updated_log_probs = torch.where(
                torch.isfinite(updated_log_probs),
                updated_log_probs,
                torch.full_like(updated_log_probs, -1e12),
            )

            original_log_probs = torch.log_softmax(topk_logits, dim=-1)
            deltas = updated_log_probs - original_log_probs

            logits_scaled = logits_scaled.clone()
            logits_scaled[topk_indices] = logits_scaled[topk_indices] + deltas

            updated_logits = logits_scaled * self.temperature
            self._pending_state_context_hash = int(context_hash)
            if squeeze_batch:
                return updated_logits.unsqueeze(0)
            return updated_logits
        self._pending_state_context_hash = None
        if squeeze_batch:
            return logits.unsqueeze(0)
        return logits
    
    def get_gvalues_detection(self, context: int, token: int) -> torch.Tensor:
        """Get the G values for a single (context, token) pair without the embedded message."""
        context_hashes = torch.tensor([context], device=self.rng_device, dtype=torch.int32)
        tokens = torch.tensor([token], device=self.rng_device, dtype=torch.int32)

        g_values = self._sample_all_depths(
            context_hashes=context_hashes,
            tokens=tokens,
            embed_message=False,
        )

        return g_values.squeeze(0)
    
        
    def detect(self, tokens: list[int] | torch.Tensor) -> dict:

        self.random_generator._add_rng_device(self.rng_device)

        def _empty_detection_output() -> dict:
            return {
                "pred_message": [0] * self.payload_size,
                "expected_message": self.payload.tolist(),
                "statistic_chi2": 0.0,
                "p_value_chi2": 1.0,
                "statistic": 0.0,
                "pvalue": 1.0,
            }

        if isinstance(tokens, list):
            tokens = torch.tensor(tokens, device=self.rng_device, dtype=torch.int32)

        tokens = tokens.to(self.rng_device)
        if tokens.numel() <= self.context_size:
            return _empty_detection_output()

        context, tokens = self.seeding_scheme.hash_context(tokens, self.context_size)
        pairs = torch.stack((context, tokens), dim=1)  # shape: [N, 2]
        unique_pairs = torch.unique(
            pairs, dim=0, sorted=False, return_inverse=False, return_counts=False
        )
        if unique_pairs.size(0) == 0:
            return _empty_detection_output()
        
        weights = torch.linspace(start=10, end=1, steps=self.depth, device=self.rng_device) / self.depth
        weights = (weights / weights.sum()) * self.depth  

        messages = torch.zeros(self.payload_size, device=self.rng_device)
        weighted_messages = torch.zeros(self.payload_size, device=self.rng_device)
        for context_hash, token in unique_pairs:
            detection_g_values = self.get_gvalues_detection(
                context_hash.item(), token.item()
            )  # shape: (payload_size, depth)
            pair_scores = detection_g_values.sum(dim=1)  #shape: (payload_size,)

            weighted_pair_scores = ( detection_g_values * weights.unsqueeze(0) ).sum(dim=1)

            messages += pair_scores.to(messages.dtype)
            weighted_messages += weighted_pair_scores.to(weighted_messages.dtype)

        n_trials = int(unique_pairs.size(0) * self.depth)
        out_chi2 = self.random_generator.test_chi2_approx(messages, n_trials)
        out_mc = self.random_generator.test_exact_mc(messages, n_trials)

        messages = messages / n_trials  # Normalize by the number of samples
        messages = messages > 0.5
        messages = messages.int().tolist()

        weighted_messages = weighted_messages / n_trials 
        weighted_messages = weighted_messages > 0.5
        weighted_messages = weighted_messages.int().tolist()

        out = {}
        out["pred_message"] = messages
        out["pred_message_weighted"] = weighted_messages
        out["expected_message"] = self.payload.tolist()
        out.update(out_chi2)
        out.update(out_mc)

        return out
