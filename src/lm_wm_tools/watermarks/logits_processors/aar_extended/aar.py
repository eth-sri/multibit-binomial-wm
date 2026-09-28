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

class AARWatermark(WatermarkProtocol):
    """AAR (extended) watermark implementation for vLLM logits processor."""

    def __init__(
        self,
        epsilon: float,
        vocab_size: int,
        rng_device: str,
        seeding_scheme: SeedingScheme | str,
        context_size: int,
        seed: int,
        sampling_parameters: SamplingParams,
        distribution_name: str = "gumbel",
        distribution_parameters: dict = {"loc": 0.0, "scale": 1.0},
        multibit_algorithm: RandomUtilityType | str = "none",
        payload: list[int] = [0],
        multibit_seed: int = 0,
        **kwargs,
    ) -> None:
        if isinstance(seeding_scheme, str):
            seeding_scheme = SeedingScheme(seeding_scheme)

        assert context_size >= 1, "context_size must be at least 1"
        assert distribution_name == "gumbel", "AARWatermark only supports gumbel distribution"

        if isinstance(rng_device, str):
            rng_device = torch.device(rng_device)

        self.delta = epsilon
        self.beta = distribution_parameters["scale"]
        self.tau = self.beta / (1 + self.beta * self.delta)

        self.temperature = sampling_parameters.temperature
        self.vocab_size = vocab_size
        self.rng_device = rng_device

        # Initialize the seeding scheme
        seeding_scheme.initialize(vocab_size, seed, rng_device)
        self.seeding_scheme = seeding_scheme
        self.context_size = context_size
        self.top_k = resolve_top_k_from_sampling_params(sampling_parameters, vocab_size)

        # Intialize the RNG
        self.seed = seed
        self.multibit_algorithm = normalize_multibit_algorithm(multibit_algorithm)
        self.payload_size = len(payload)
        self.random_generator = create_random_utility(
            multibit_algorithm,
            distribution_name,
            distribution_parameters,
            payload=payload,
            multibit_seed=multibit_seed,
            vocab_size=vocab_size,
            **kwargs,
        )

    def get_name(self) -> str:
        return format_watermark_name(
            "AAR",
            self.multibit_algorithm,
            f"{self.seeding_scheme.value}/k_{self.context_size}/delta_{self.delta:<.2f}/beta_{self.beta:<.2f}/seed_{self.seed}",
            payload_size=self.payload_size,
        )

    def __call__(
        self,
        output_ids: list[int],
        logits: torch.Tensor,  # (vocab_size)
    ) -> torch.Tensor:
        context_hash = self.seeding_scheme.hash_last_context(
            torch.tensor(output_ids, device=self.rng_device), self.context_size
        )
        if context_hash is not None:

            k = min(self.top_k, logits.shape[-1])
            topk_logits, topk_indices = torch.topk(logits, k, dim=-1)


            gumbel = self.random_generator.sample(
                self.seed,
                context_hash,
                topk_indices
            ).squeeze(0)
            gumbel = gumbel.to(logits.device)

            scores = gumbel + self.tau * topk_logits / self.temperature
            argmax_idx = torch.argmax(scores).item()
            argmax_idx = topk_indices[argmax_idx].item()

            # Create a dirac distribution at the argmax index
            logits = torch.full_like(logits, -100.0)
            logits[argmax_idx] = 0.0

        return logits

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
        
        scores = self.random_generator.random_sample(V, n_mc, device)  # shape [n_mc, V]
        scores = scores + self.tau * torch.log(topk_probs.clamp_min(1e-12))

        argmax_idxs = torch.argmax(scores, dim=1)  # [n_mc]
        counts = torch.bincount(argmax_idxs, minlength=V).float()  # [V]
        expected_topk_probs = counts / n_mc  # [V]

        # Expand back to full vocab size
        expanded_expected_probs = torch.zeros_like(probs)
        expanded_expected_probs[topk_indices] = expected_topk_probs

        return expanded_expected_probs

    def detect(self, tokens: list[int] | torch.Tensor) -> float:
        
        return self.random_generator.detect(
            tokens,
            self.seeding_scheme,
            self.rng_device,
            self.context_size,
            self.seed,
        )
