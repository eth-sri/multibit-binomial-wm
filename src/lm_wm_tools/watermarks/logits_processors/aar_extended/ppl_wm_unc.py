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


class PPLUnconstrainedWatermark(WatermarkProtocol):
    """PPL unconstrained watermark implementation for vLLM logits processor."""

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
        distribution_parameters: dict = {"total_count": 30, "probs": 0.5},
        multibit_algorithm: RandomUtilityType | str = "none",
        payload: list[int] = [0],
        multibit_seed: int = 0,
        **kwargs,
    ) -> None:
        if isinstance(seeding_scheme, str):
            seeding_scheme = SeedingScheme(seeding_scheme)

        assert context_size >= 1, "context_size must be at least 1"

        if isinstance(rng_device, str):
            rng_device = torch.device(rng_device)

        # Scheme Parameters
        self.multibit_algorithm = normalize_multibit_algorithm(multibit_algorithm)
        self.payload_size = len(payload)
        self.epsilon = epsilon
        if self.multibit_algorithm == "bino_encoder":
            self.epsilon = epsilon / len(payload)  # Scale epsilon by payload size for multibit schemes
        self.tau = 1/ (1e-10 + epsilon)

        # Sampling Parameters
        self.temperature = sampling_parameters.temperature
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


    def get_name(self) -> str:
        return format_watermark_name(
            "PPLUnconstrained",
            self.multibit_algorithm,
            f"{self.distribution_name}/{self.seeding_scheme.value}/k_{self.context_size}/epsilon_{self.epsilon}/seed_{self.seed}",
            payload_size=self.payload_size,
        )

    def call_with_scores(
        self,
        output_ids: list[int],
        logits: torch.Tensor,  # (vocab_size)
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        context_hash = self.seeding_scheme.hash_last_context(
            torch.tensor(output_ids, device=self.rng_device), self.context_size
        )
        og_scores = torch.zeros_like(logits)
        if context_hash is not None:

            k = min(self.top_k, logits.shape[-1])
            topk_logits, topk_indices = torch.topk(logits, k, dim=-1)
            with torch.no_grad():
                probs = torch.softmax(topk_logits / self.temperature, dim=-1)

            
            scores = self.random_generator.sample(self.seed, context_hash, topk_indices)

            og_scores[topk_indices] = scores.to(logits.device).to(logits.dtype)

            scores = scores + self.tau * torch.log(probs.clamp_min(1e-12))
            argmax_idx = torch.argmax(scores).item()
            argmax_idx = topk_indices[argmax_idx].item()

            # Create a dirac distribution at the argmax index
            logits = torch.full_like(logits, -100.0)
            logits[argmax_idx] = 0.0

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

        if self.tau is None:
            return probs

        # Only consider top-k
        k = min(self.top_k, probs.shape[-1])
        topk_probs, topk_indices = torch.topk(probs, k, dim=-1)
        
        V = topk_probs.numel()
        n_mc = 128
        device = probs.device
        
        scores = self.random_generator.random_sample(V, n_mc, device)  # shape [n_mc, V]
        scores = scores + self.tau * torch.log(topk_probs.clamp_min(1e-12))
        argmax_idxs = torch.argmax(scores, dim=1)  # [n_mc]
        counts = torch.bincount(argmax_idxs, minlength=V).float()  # [V]
        expected_topk_probs = counts / n_mc  # [V]

        # Reconstruct full prob vector
        expected_probs = torch.zeros_like(probs)
        expected_probs[topk_indices] = expected_topk_probs
        
        return expected_probs

    def detect(self, tokens: list[int] | torch.Tensor) -> float:
        
        return self.random_generator.detect(
            tokens,
            self.seeding_scheme,
            self.rng_device,
            self.context_size,
            self.seed,
        )
