import torch
from scipy.stats import binomtest

from lm_wm_tools.utils.triton_code import stateless_bernoulli_bits
from lm_wm_tools.watermarks.utils import add_elapsed_time
from .random_utility import RandomGenerator, SeedingScheme

_VOCAB_SIZE_UPPER_BOUND = 512000

def int_to_bits(n: int, width: int) -> list[int]:
    """Convert an integer to a list of bits with the given width."""
    return [(n >> i) & 1 for i in range(width)][::-1]


def bits_to_int(bits: list[int]) -> int:
    """Convert a list of bits to an integer."""
    n = 0
    for bit in bits:
        n = (n << 1) | bit
    return n


class BinoEncoder(RandomGenerator):
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
    
    
    def _add_rng_device(self, device: torch.device) -> None:
        self.rng_device = device

    def _sample_from_offsets(
        self,
        seed: int,
        offsets: torch.Tensor,
        embed_message: bool = True,
    ) -> torch.Tensor:
        offsets = offsets.reshape(-1)

        scores = stateless_bernoulli_bits(seed=seed, offsets=offsets, k=self.payload_size) # shape: (N, m)

        if embed_message:
            scores = scores + (
                1 - self.payload.unsqueeze(0).to(scores.device).to(scores.dtype)
            )  # shape: (N, m)
            scores = scores % 2
            scores = scores.sum(dim=1)  # shape: (N,)

        return scores

    def sample(
        self,
        seed: int,
        context_hashes: torch.Tensor | int,
        tokens: torch.Tensor,
        embed_message: bool = True,
        depth: int = 1,
    ) -> torch.Tensor:
        if isinstance(context_hashes, int):
            context_hashes = torch.tensor(
                [context_hashes], device=tokens.device, dtype=torch.int32
            )

        offsets = self._compute_offsets(context_hashes, tokens)
        # torch.hash_tensor may return uint64 depending on the torch build.
        # Use a signed integer dtype so depth arithmetic is always valid.
        offsets = offsets.to(dtype=torch.int64)

        if depth > 1:
            offsets = offsets.expand(depth, -1)  # shape: (depth, N)
            depth_stride = torch.arange(
                depth, device=offsets.device, dtype=offsets.dtype
            ).unsqueeze(1)
            offsets = offsets + depth_stride * int(_VOCAB_SIZE_UPPER_BOUND)  # shape: (depth, N)

        samples = self._sample_from_offsets(
            seed=seed,
            offsets=offsets,
            embed_message=embed_message,
        ) # shape: (depth, N) if depth > 1 else (N,)

        return samples 

    @staticmethod
    def _lrt_deviance_from_counts(S: torch.Tensor, n_trials: int) -> torch.Tensor:
        S = S.to(dtype=torch.float64)
        n = torch.tensor(float(n_trials), device=S.device, dtype=torch.float64)
        half_n = n / 2.0

        term1 = torch.where(S > 0, S * torch.log(S / half_n), torch.zeros_like(S))
        term2 = torch.where(
            S < n, (n - S) * torch.log((n - S) / half_n), torch.zeros_like(S)
        )

        T = 2.0 * (term1 + term2).sum()
        return T

    @staticmethod
    def _chi2_survival(x: torch.Tensor, df: int) -> torch.Tensor:
        x = x.to(dtype=torch.float64)
        a = torch.tensor(df / 2.0, device=x.device, dtype=torch.float64)
        z = x / 2.0
        return torch.special.gammaincc(a, z)

    def test_chi2_approx(self, message_counts: torch.Tensor, n_trials: int) -> dict:
        """
        Wilks chi-square approximation: T ~ Chi^2_{df=m} under H0 (approx; good when n_trials is not tiny).
        """
        S = message_counts.to(self.rng_device)
        m = int(S.numel())
        T = self._lrt_deviance_from_counts(S, n_trials)
        p = self._chi2_survival(T, df=m)
        return {
            "statistic_chi2": float(T.item()),
            "p_value_chi2": float(p.item()),
        }

    def test_exact_mc(
        self, message_counts: torch.Tensor, n_trials: int, n_mc: int = 20000
    ) -> dict:
        """
        Monte Carlo 'exact' calibration under H0 by simulating S_k ~ Bin(n_trials, 0.5) iid across k.
        """
        S_obs = message_counts.to(self.rng_device)
        m = int(S_obs.numel())
        T_obs = self._lrt_deviance_from_counts(S_obs, n_trials)

        binomial = torch.distributions.Binomial(total_count=n_trials, probs=0.5)
        S_sim = binomial.sample((n_mc, m))
        S_sim = S_sim.to(self.rng_device)

        S_sim_f = S_sim.to(torch.float64)
        n = torch.tensor(float(n_trials), device=self.rng_device, dtype=torch.float64)
        half_n = n / 2.0

        term1 = torch.where(
            S_sim_f > 0,    
            S_sim_f * torch.log(S_sim_f / half_n),
            torch.zeros_like(S_sim_f),
        )
        term2 = torch.where(
            S_sim_f < n,
            (n - S_sim_f) * torch.log((n - S_sim_f) / half_n),
            torch.zeros_like(S_sim_f),
        )
        T_sim = 2.0 * (term1 + term2).sum(dim=1)

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
        unique_pairs, inverse_indices = torch.unique(
            pairs, dim=0, sorted=False, return_inverse=True, return_counts=False
        )

        # Map each unique (context_hash, token) pair to the first token index
        # in the original input token list.
        index_pairs = [-1] * unique_pairs.size(0)
        for token_idx, pair_idx in enumerate(
            inverse_indices.detach().cpu().tolist(), start=context_size
        ):
            if index_pairs[pair_idx] == -1:
                index_pairs[pair_idx] = token_idx

        scores = []
        messages = torch.zeros(self.payload_size, device=self.rng_device)
        payload = self.payload.to(self.rng_device, dtype=torch.int32)
        per_token_bit_accuracy = []
        for step_idx, (context, token) in enumerate(unique_pairs, start=1):
            scores = self.sample(
                seed=seed,
                context_hashes=context.item(),
                tokens=token.unsqueeze(0),
                embed_message=False,
            ).squeeze(0)
            messages += scores

            # Per token bit accuracy
            bit_acc = (scores == payload).to(torch.float32).mean().item()
            per_token_bit_accuracy.append(float(bit_acc))

        out_chi2 = self.test_chi2_approx(messages, len(unique_pairs))
        out_mc = self.test_exact_mc(messages, len(unique_pairs))

        unormalized_message = messages.clone()

        # Decoding the message
        messages = messages / len(unique_pairs)
        messages = messages > 0.5
        messages = messages.int().tolist()

        # Compute one-sided p-value per bit under H0: Bernoulli(0.5),
        # with tail selected by the expected payload bit.
        p_values_per_bit = []
        for bit_idx in range(self.payload_size):
            k = int(unormalized_message[bit_idx])
            n = len(unique_pairs)
            test_result = binomtest(k=k, n=n, p=0.5, alternative="two-sided") # We don't know the GT
            p_values_per_bit.append(test_result.pvalue)
            
        out = {}
        out["p_values_per_bit"] = p_values_per_bit
        out["unormalized_message"] = unormalized_message.tolist()
        
        out["pred_message"] = messages
        out["expected_message"] = self.payload.tolist()
        out["per_token_bit_accuracy"] = per_token_bit_accuracy
        out["index_pairs"] = index_pairs
        out.update(out_chi2)
        out.update(out_mc)

        return out
