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


class RedGreenWatermark(WatermarkProtocol):
    """Red-Green watermark implementation for vLLM logits processor."""

    def __init__(
        self,
        epsilon: float,
        vocab_size: int,
        rng_device: str,
        seeding_scheme: SeedingScheme | str,
        context_size: int,
        seed: int,
        sampling_parameters: SamplingParams,
        distribution_name: str = "binomial",
        distribution_parameters: dict = {"total_count": 1, "probs": 0.5},
        multibit_algorithm: RandomUtilityType | str = "none",
        payload: list[int] = [0],
        multibit_seed: int = 0,
        **kwargs,
    ) -> None:
        if isinstance(seeding_scheme, str):
            seeding_scheme = SeedingScheme(seeding_scheme)
            
        if isinstance(rng_device, str):
            rng_device = torch.device(rng_device)

        # Scheme parameters
        self.multibit_algorithm = normalize_multibit_algorithm(multibit_algorithm)
        self.payload_size = len(payload)
        self.delta = epsilon
        if self.multibit_algorithm == "bino_encoder":
            self.delta = epsilon / len(payload)  # Scale epsilon by payload size for multibit schemes 

        # Sampling parameters
        self.vocab_size = vocab_size
        self.rng_device = rng_device
        self.top_k = resolve_top_k_from_sampling_params(
            sampling_parameters, vocab_size
        )
        self.temperature = sampling_parameters.temperature

        # Initialize the seeding scheme
        seeding_scheme.initialize(vocab_size, seed, rng_device)
        self.seeding_scheme = seeding_scheme
        self.context_size = context_size

        # Intialize the RNG
        self.seed = seed
        self.random_generator = create_random_utility(
            multibit_algorithm,
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

    def get_name(self) -> str:
        return format_watermark_name(
            "KGW",
            self.multibit_algorithm,
            f"{self.seeding_scheme.value}/k_{self.context_size}/delta_{self.delta:<.2f}/seed_{self.seed}",
            payload_size=self.payload_size,
        )

    def call_with_scores(
        self,
        output_ids: list[int],
        logits: torch.Tensor,  # (vocab_size)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self._pending_state_context_hash is not None and len(output_ids) > 0:
            self.random_generator.update_state(
                context_hash=self._pending_state_context_hash,
                token=output_ids[-1],
                rng_device=self.rng_device,
                seed=self.seed,
            )
            self._pending_state_context_hash = None

        context_hash = self.seeding_scheme.hash_last_context(torch.tensor(output_ids, device=self.rng_device), self.context_size)
        og_scores = torch.zeros_like(logits)
        if context_hash is not None:

            k = min(self.top_k, logits.shape[-1])
            topk_logits, topk_indices = torch.topk(logits, k, dim=-1)

            scores = self.random_generator.sample(
                self.seed,
                context_hash,
                topk_indices,
            ).squeeze(0) * self.delta
            logits[topk_indices] = topk_logits + scores.to(logits.device)
            og_scores[topk_indices] = scores.to(logits.device).to(logits.dtype)
            self._pending_state_context_hash = int(context_hash)
        else:
            self._pending_state_context_hash = None
        return logits, og_scores

    def __call__(
        self,
        output_ids: list[int],
        logits: torch.Tensor,  # (vocab_size)
    ) -> torch.Tensor:
        modified_logits, _ = self.call_with_scores(output_ids, logits)
        return modified_logits


    def get_expected_probs(self, probs: torch.Tensor) -> torch.Tensor:
        """
        Given a prob vector p [V], return the expected probs under the watermarking
        selection policy.
        """       

        # Only consider top-k
        k = min(self.top_k, probs.shape[-1])
        topk_probs, topk_indices = torch.topk(probs, k, dim=-1)
        
        V = topk_probs.numel()
        n_mc = 128
        device = probs.device

        topk_logits = torch.log(topk_probs + 1e-30) * self.temperature

        scores = self.random_generator.random_sample(V, n_mc, device)  # shape [n_mc, V]
        scores = scores * self.delta

        topk_logits = topk_logits.unsqueeze(0).expand(n_mc, -1)  + scores  # shape [n_mc, V]
        expected_probs = torch.softmax(topk_logits / self.temperature, dim=-1).mean(dim=0)
        # Expand back to full vocab size
        expanded_expected_probs = torch.zeros_like(probs)
        expanded_expected_probs[topk_indices] = expected_probs

        return expanded_expected_probs

    def detect(self, tokens: list[int] | torch.Tensor) -> float:
        
        return self.random_generator.detect(
            tokens,
            self.seeding_scheme,
            self.rng_device,
            self.context_size,
            self.seed,
        )
