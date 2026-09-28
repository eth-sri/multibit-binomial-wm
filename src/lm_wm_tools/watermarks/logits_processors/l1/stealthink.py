import math
from cudnn.wrapper import torch
import torch
from vllm import SamplingParams

from lm_wm_tools import WatermarkProtocol
from lm_wm_tools.watermarks.sampling import (
    RandomUtilityType,
    SeedingScheme,
    create_random_utility,
)
from lm_wm_tools.watermarks.utils import (
    add_elapsed_time,
    format_watermark_name,
    normalize_multibit_algorithm,
)

def pos(x: torch.Tensor) -> torch.Tensor:
    return torch.maximum(x, torch.tensor(0.0, device=x.device))

def neg(x: torch.Tensor) -> torch.Tensor:
    return -torch.minimum(x, torch.tensor(0.0, device=x.device)) 

# Calibrated on 10k human samples (allenai/c4 realnewslike),
# generation_length=200, payload_size=32, m=1, sumhash k=4, seed=0.
MU_R = 66.6485
SIGMA_R = 4.658148532410705


def _normal_cdf(z: float) -> float:
    return 0.5 * math.erfc(-z / math.sqrt(2.0))


class StealthInkWatermark(WatermarkProtocol):
    """StealthInk watermark implementation for vLLM logits processor."""

    def __init__(
        self,
        vocab_size: int,
        rng_device: str,
        seeding_scheme: SeedingScheme | str,
        context_size: int,
        seed: int,
        sampling_parameters: SamplingParams,
        distribution_name: str = "uniform",
        distribution_parameters: dict = {"low": 0.0, "high": 1.0},
        multibit_algorithm: RandomUtilityType | str = "none",
        payload: list[int] = [1,0,1,0,1,0,1,0,1],
        multibit_seed: int = 0,
        m: int = 1,
        **kwargs,
    ) -> None:

        self.multibit_algorithm = normalize_multibit_algorithm(multibit_algorithm)

        assert distribution_name == "uniform", "StealthInkWatermark only supports uniform distribution"
        assert self.multibit_algorithm == "mpac", "StealthInkWatermark only supports 'mpac' multibit algorithm"

        if isinstance(seeding_scheme, str):
            seeding_scheme = SeedingScheme(seeding_scheme)

        if isinstance(rng_device, str):
            rng_device = torch.device(rng_device)

        self.gamma = 2**(-m)
        self.m = m

        self.payload = payload
        self.payload_size = len(self.payload)
        self.n_mc = int(kwargs.get("n_mc", 10000))

        self.temperature = sampling_parameters.temperature
        self.vocab_size = vocab_size
        self.rng_device = rng_device

        # Initialize the seeding scheme
        seeding_scheme.initialize(vocab_size, seed, rng_device)
        self.seeding_scheme = seeding_scheme
        self.context_size = context_size
        self.top_k = vocab_size  # StealthInk uses full vocab

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

    def get_name(self) -> str:
        return format_watermark_name(
            "StealthInk",
            self.multibit_algorithm,
            f"{self.seeding_scheme.value}/k_{self.context_size}/m_{self.m}/seed_{self.seed}",
            payload_size=self.payload_size,
        )

    def _compute_alpha(self, permuted_probs: torch.tensor, gamma_M: float) -> float:
        idx = int(gamma_M * self.vocab_size) 
        alpha = permuted_probs[:idx].sum().item()
        return alpha

    def _compute_beta(self, permuted_probs: torch.tensor, gamma_M: float) -> float:
        idx = int((gamma_M + self.gamma)* self.vocab_size) 
        beta = permuted_probs[:idx].sum().item()
        return beta

    def _reweight_cdf(
        self,
        cumulative_probs: torch.Tensor,
        alpha: float,
        beta: float,
    ) -> torch.Tensor:

        beta_overline = 1.0 - beta
        alpha_overline = 1.0 - alpha

        case_1_or_3 = (beta <= 0.5) | ((alpha < 0.5) & (beta > 0.5) & (alpha + beta <= 1.0))

        X = cumulative_probs

        if case_1_or_3:
            term1 = pos(X - beta)
            term2 = pos(X - beta_overline)
            term3 = neg(X - alpha) 
            term4 = pos(X - alpha_overline)
            return term1 + term2 - term3 - term4
        else:
            term1 = pos(X - beta)
            term2 = neg(X - beta_overline)
            term3 = neg(X - alpha)
            term4 = neg(X - alpha_overline)
            return term1 + term2 - term3 - term4

    @torch.no_grad()
    def __call__(
        self,
        output_ids: list[int],
        logits: torch.Tensor,  # (vocab_size)
    ) -> torch.Tensor:
        context_hash = self.seeding_scheme.hash_last_context(
            torch.tensor(output_ids, device=self.rng_device), self.context_size
        )
        if context_hash is not None:

            logits_scaled = logits / self.temperature
            probs = torch.softmax(logits_scaled, dim=-1)

            # Generate a random permutation based on the context hash
            uniform, positions = self.random_generator.sample_with_position(
                self.seed,
                context_hash,
                torch.arange(logits.size(-1), device=logits.device)
            )
        
            uniform = uniform.to(logits.device).squeeze(0)
            permutation = torch.argsort(uniform, dim=-1, stable=True)
            position = positions.item()
            M = self.payload[position]

            permuted_probs = torch.gather(probs, 0, permutation)
            gamma_M = M * self.gamma
            alpha = self._compute_alpha(permuted_probs, gamma_M)
            beta = self._compute_beta(permuted_probs, gamma_M)

            cumsum_probs = torch.cumsum(permuted_probs, dim=0)
            cumulative_probs = torch.cat([torch.zeros(1, device=logits.device), cumsum_probs], dim=0)
            reweighted_cdf = self._reweight_cdf(cumulative_probs, alpha, beta)
            reweighted_permuted_probs = reweighted_cdf[1:] - reweighted_cdf[:-1]

            # Inverse permutation
            inverse_permutation = torch.argsort(permutation, dim=-1, stable=True)
            reweighted_probs = torch.gather(reweighted_permuted_probs, 0, inverse_permutation)
            reweighted_probs = reweighted_probs.clamp(min=1e-12)

            logits = torch.log(reweighted_probs) * self.temperature
       
        return logits

    def get_expected_probs(self, probs: torch.Tensor) -> torch.Tensor:
        return probs

    def _mc_pvalue(
        self,
        observed_z_score: float,
        positions: torch.Tensor,
        n_mc: int,
    ) -> float:
        device = positions.device
        if n_mc <= 0 or positions.numel() == 0 or SIGMA_R <= 0:
            return 1.0

        positions = positions.long()
        token_count = positions.numel()

        try:
            generator = torch.Generator(device=device)
            uniform = torch.rand((n_mc, token_count), generator=generator, device=device)
        except RuntimeError:
            generator = torch.Generator(device="cpu")
            uniform = torch.rand((n_mc, token_count), generator=generator, device="cpu").to(device)
        uniform = uniform.clamp_(min=0.0, max=1.0 - 1e-12)

        ranks = torch.floor(uniform * float(self.vocab_size)).long()  # (n_mc, token_count)

        num_M = 2 ** self.m
        M = torch.arange(num_M, device=device, dtype=torch.float32)  # [num_M]
        vocab_size_f = float(self.vocab_size)

        starts = torch.floor(M * (self.gamma * vocab_size_f)).long()  # [num_M]
        ends = torch.floor((M * self.gamma + self.gamma) * vocab_size_f).long()  # [num_M]

        hits = (ranks[..., None] >= starts) & (ranks[..., None] < ends)  # (n_mc, token_count, num_M)
        hits = hits.to(torch.int32)

        counts = torch.zeros(
            (n_mc, num_M, self.payload_size), device=device, dtype=torch.int32
        )
        counts.index_add_(2, positions, hits.permute(0, 2, 1))
        decoded_counts = counts.permute(0, 2, 1)  # (n_mc, payload_size, num_M)

        decoded_message = torch.argmin(decoded_counts, dim=-1)  # (n_mc, payload_size)
        selected_counts = decoded_counts.gather(2, decoded_message.unsqueeze(-1)).squeeze(-1)
        red_hit_counts = selected_counts.sum(dim=1).to(torch.float64)
        z_scores_mc = (red_hit_counts - MU_R) / SIGMA_R

        # One-sided left-tail test: watermark evidence is smaller R / smaller z.
        le = (z_scores_mc <= observed_z_score).sum().item()
        pvalue = (le + 1) / (n_mc + 1)
        return float(pvalue)

    @torch.no_grad()
    def _compute_decoding_state(
        self,
        tokens: list[int] | torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
        if isinstance(tokens, list):
            tokens = torch.tensor(tokens, device=self.rng_device, dtype=torch.int32)

        tokens = tokens.to(self.rng_device)

        context, tokens = self.seeding_scheme.hash_context(tokens, self.context_size)
        pairs = torch.stack((context, tokens), dim=1)  # [N, 2]

        # Deduplicate (context, token) pairs
        unique_pairs = torch.unique(pairs, dim=0, sorted=False)

        context_hashes = unique_pairs[:, 0]  # [N]
        unique_tokens = unique_pairs[:, 1].long()  # [N]

        vocab = torch.arange(self.vocab_size, device=self.rng_device, dtype=torch.int32)
        offsets = self.random_generator._compute_offsets(context_hashes, vocab)
        scores = self.random_generator._sample_with_offsets(
            self.seed, offsets, unique_pairs.size(0), self.vocab_size
        )  # [N, V]

        permutations = torch.argsort(scores, dim=-1, stable=True)  # [N, V]
        positions = self.random_generator._compute_position(context_hashes).long()  # [N]

        inv_perm = torch.empty_like(permutations)
        ranks = torch.arange(self.vocab_size, device=self.rng_device).long()
        inv_perm.scatter_(1, permutations, ranks.unsqueeze(0).expand_as(permutations))

        token_ranks = inv_perm.gather(1, unique_tokens.view(-1, 1)).squeeze(1)  # [N]

        num_M = 2 ** self.m

        M = torch.arange(num_M, device=self.rng_device, dtype=torch.float32)  # [num_M]
        V = float(self.vocab_size)

        starts = torch.floor(M * (self.gamma * V)).long()                  # [num_M]
        ends = torch.floor((M * self.gamma + self.gamma) * V).long()       # [num_M]

        hits = (token_ranks[:, None] >= starts[None, :]) & (token_ranks[:, None] < ends[None, :])
        hits = hits.to(torch.int32)  # [N, num_M]

        decoded_message_counts = torch.zeros(
            (self.payload_size, num_M), dtype=torch.int32, device=self.rng_device
        )

        pos_rep = positions[:, None].expand(-1, num_M).reshape(-1)  # [N*num_M]
        m_rep = torch.arange(num_M, device=self.rng_device).repeat(positions.numel())  # [N*num_M]
        lin_idx = pos_rep * num_M + m_rep  # [N*num_M]

        flat = decoded_message_counts.view(-1)
        flat.scatter_add_(0, lin_idx, hits.reshape(-1))
        decoded_message_counts = flat.view(self.payload_size, num_M)

        decoded_message = torch.argmin(decoded_message_counts, dim=-1)

        payload_tensor = torch.tensor(
            self.payload, device=self.rng_device, dtype=torch.long
        )
        statistic = int((decoded_message == payload_tensor).sum().item())

        # R = sum_pos R_pos^(decoded_message_pos)
        selected_counts = decoded_message_counts.gather(
            1, decoded_message.view(-1, 1)
        )
        red_hit_count = int(selected_counts.sum().item())

        return decoded_message_counts, positions, decoded_message, statistic, red_hit_count

    @torch.no_grad()
    def estimate_mu_sigma_R(
        self,
        token_sequences: list[list[int] | torch.Tensor],
    ) -> tuple[float, float]:
        red_hit_counts: list[float] = []
        for tokens in token_sequences:
            _, _, _, _, red_hit_count = self._compute_decoding_state(tokens)
            red_hit_counts.append(float(red_hit_count))

        if not red_hit_counts:
            return 0.0, 0.0

        counts_tensor = torch.tensor(red_hit_counts, dtype=torch.float64)
        mu_r = float(counts_tensor.mean().item())
        sigma_r = float(counts_tensor.std(unbiased=False).item())
        return mu_r, sigma_r

    @add_elapsed_time()
    def detect(self, tokens: list[int] | torch.Tensor):
        decoded_message_counts, positions, decoded_message, bit_match_statistic, red_hit_count = (
            self._compute_decoding_state(tokens)
        )

        if SIGMA_R > 0:
            z_score = float((red_hit_count - MU_R) / SIGMA_R)
            p_value_original = float(_normal_cdf(z_score))
        else:
            z_score = 0.0
            p_value_original = 1.0

        p_value = self._mc_pvalue(
            observed_z_score=z_score,
            positions=positions,
            n_mc=self.n_mc,
        )

        out = {
            "pred_message": decoded_message.cpu().tolist(),
            "expected_message": self.payload,
            "positions": positions.cpu().tolist(),
            "decoded_message_counts": decoded_message_counts.cpu().tolist(),
            "statistic": z_score,
            "bit_match_statistic": bit_match_statistic,
            "R": red_hit_count,
            "pvalue": p_value,
            "mu_R": MU_R,
            "sigma_R": SIGMA_R,
            "z_score": z_score,
            "pvalue_original": p_value_original,
            "p_value_original": p_value_original,
        }
        
        return out
