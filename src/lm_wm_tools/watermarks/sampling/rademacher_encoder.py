import torch
from scipy.stats import binomtest

from lm_wm_tools.utils.triton_code import stateless_uniform, stateless_bernoulli_bits
from lm_wm_tools.watermarks.utils import add_elapsed_time
from .random_utility import RandomGenerator, SeedingScheme


def int_to_bits(n: int, width: int) -> list[int]:
    """Convert an integer to a list of bits with the given width."""
    return [(n >> i) & 1 for i in range(width)][::-1]


def bits_to_int(bits: list[int]) -> int:
    """Convert a list of bits to an integer."""
    n = 0
    for bit in bits:
        n = (n << 1) | bit
    return n


class RademacherEncoder(RandomGenerator):
    def __init__(
        self,
        distribution_name: str = "binomial",
        distribution_parameters: dict = {"total_count": 8, "probs": 0.5},
        payload: list[int] = [0, 0, 0, 0, 0, 0, 0, 0],
        **kwargs,
    ) -> None:

        payload_size = len(payload)
        distribution_name, distribution_parameters = self.get_distribution(
            distribution_name, distribution_parameters, payload_size
        )

        super().__init__(distribution_name, distribution_parameters)
        self.payload = torch.tensor(payload, dtype=torch.int32)
        self.rademacher_payload = 2 * self.payload - 1

        self.payload_size = payload_size
        self.payload_int = bits_to_int(payload)

    def get_distribution(
        self, distribution_name: str, distribution_parameters: dict, payload_size: int
    ) -> tuple[str, dict]:
        if (
            distribution_name != "binomial"
            or distribution_parameters["total_count"] != payload_size
        ):
            pass  # We override the distribution parameters to match the payload size, but we keep the distribution name for logging purposes.

        distribution_name = "binomial"
        distribution_parameters = {"total_count": payload_size, "probs": 0.5}
        return distribution_name, distribution_parameters
    
    def embed_message(self, scores: torch.Tensor) -> torch.Tensor:
        assert scores.shape[1] == self.payload_size, "Scores shape does not match payload size."
        scores = scores + (1 - self.payload.unsqueeze(0).to(scores.device))  # shape: (k, m)
        scores = scores % 2
        return scores
    
    def _add_rng_device(self, device: torch.device) -> None:
        self.rng_device = device

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

        # Create all (context_hash, token) pairs as a grid
        ctx_grid = context_hashes.view(num_ctx, 1).expand(num_ctx, num_tok)  # (C, T)
        tok_grid = tokens.view(1, num_tok).expand(num_ctx, num_tok)  # (C, T)

        # Stack into pairs and hash along the last dimension
        pairs = torch.stack((ctx_grid, tok_grid), dim=-1)  # (C, T, 2)
        offsets = torch.hash_tensor(pairs, dim=-1)  # (C, T)

        offsets = offsets.reshape(-1)  # (C * T,)
        scores = stateless_bernoulli_bits(seed=seed, offsets=offsets, k=self.payload_size) # shape: (C * T, m)
        scores = 2 * scores - 1 # Turn into Rademacher

        if embed_message:
            scores = scores * self.rademacher_payload.view(1,-1).to(tokens.device)
            scores = scores.sum(dim = -1)

        return scores

    @staticmethod
    def _lrt_deviance_from_rademacher_sums(
        R: torch.Tensor, n_trials: int
    ) -> torch.Tensor:
        R = R.to(dtype=torch.float64)
        n = torch.tensor(float(n_trials), device=R.device, dtype=torch.float64)
        S = (R + n) / 2.0
        half_n = n / 2.0

        term1 = torch.where(S > 0, S * torch.log(S / half_n), torch.zeros_like(S))
        term2 = torch.where(
            S < n, (n - S) * torch.log((n - S) / half_n), torch.zeros_like(S)
        )

        T = 2.0 * (term1 + term2).sum(dim=-1)
        return T

    def test_exact_mc(
        self, message_counts: torch.Tensor, n_trials: int, n_mc: int = 20000
    ) -> dict:
        """
        Monte Carlo calibration under H0 by simulating R_k = 2*B_k - n_trials,
        with B_k ~ Bin(n_trials, 0.5) iid across k.
        """
        R_obs = message_counts.to(self.rng_device)
        m = int(R_obs.numel())
        T_obs = self._lrt_deviance_from_rademacher_sums(R_obs, n_trials)

        binomial = torch.distributions.Binomial(total_count=n_trials, probs=0.5)
        R_sim = 2 * binomial.sample((n_mc, m)) - n_trials
        R_sim = R_sim.to(self.rng_device)
        T_sim = self._lrt_deviance_from_rademacher_sums(R_sim, n_trials)

        ge = (T_sim >= T_obs).sum().to(torch.float64)
        p_mc = (ge + 1.0) / (float(n_mc) + 1.0)

        return {
            "statistic": float(T_obs.item()),
            "pvalue": float(p_mc.item()),
        }

    @add_elapsed_time()
    def detect(
        self,
        tokens: list[int] | torch.Tensor,
        seeding_scheme: SeedingScheme,
        rng_device: torch.device,
        context_size: int,
        seed: int,
    ) -> float:
        self.rng_device = rng_device

        scores = []

        if isinstance(tokens, list):
            tokens = torch.tensor(tokens, device=self.rng_device, dtype=torch.int32)

        tokens = tokens.to(self.rng_device)

        context, tokens = seeding_scheme.hash_context(tokens, context_size)
        pairs = torch.stack((context, tokens), dim=1)  # shape: [N, 2]
        unique_pairs = torch.unique(
            pairs, dim=0, sorted=False, return_inverse=False, return_counts=False
        )
        n_trials = int(unique_pairs.size(0))

        scores = []
        messages = torch.zeros(self.payload_size, device=self.rng_device)
        for context, token in unique_pairs:
            scores = self.sample(
                seed=seed,
                context_hashes=context.item(),
                tokens=token.unsqueeze(0),
                embed_message=False,
            ).squeeze(0)
            messages += scores

        if n_trials > 0:
            out_mc = self.test_exact_mc(messages, n_trials)
        else:
            out_mc = {"statistic": 0.0, "pvalue": 1.0}

        unormalized_message = messages.clone()
        p_values_per_bit = []
        for bit_idx in range(self.payload_size):
            if n_trials <= 0:
                p_values_per_bit.append(1.0)
                continue

            rademacher_sum = float(unormalized_message[bit_idx].item())
            k_success = int(round((rademacher_sum + n_trials) / 2.0))
            k_success = max(0, min(k_success, n_trials))
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

        messages = messages > 0
        messages = messages.int().tolist()

        out = {}
        out["pred_message"] = messages
        out["expected_message"] = self.payload.tolist()
        out["p_values_per_bit"] = p_values_per_bit
        out["unormalized_message"] = unormalized_message.tolist()
        out.update(out_mc)

        return out
