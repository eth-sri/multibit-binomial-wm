import torch
import numpy as np
import scipy
from scipy.stats import binomtest
import math
import random

from lm_wm_tools.utils.triton_code import stateless_uniform
from lm_wm_tools.watermarks.utils import add_elapsed_time
from .random_utility import RandomGenerator, SeedingScheme

class MPAC(RandomGenerator):
    """Multi-Bit Watermark vis Position Allocation (MPAC) random generator."""

    def __init__(
        self,
        distribution_name: str = "binomial",
        distribution_parameters: dict = {"total_count": 1, "probs": 0.5},
        payload: list[int] = [0,0,0,0,0,0,0,0,0],
        multibit_seed: int = 1847389390,
        **kwargs,
    ) -> None:
        super().__init__(distribution_name, distribution_parameters)
        self.payload = torch.tensor(payload, dtype=torch.int32)
        self._is_distribution_supported()

        self.multibit_seed = multibit_seed
        self.payload_size = len(self.payload)

        self.n_mc = 10000

    def _is_distribution_supported(self):
        error_msg = "MPAC only supports Bernoulli distribution or Uniform distribution. {distribution_name} with {distribution_parameters} is not supported.".format(
            distribution_name=self.distribution_name,
            distribution_parameters=self.distribution_parameters,
        )
        if self.distribution_name == "binomial":
            assert self.distribution_parameters["total_count"] == 1, error_msg
        else:
            assert self.distribution_name == "uniform", error_msg

    def _compute_position(self, offsets: torch.Tensor) -> torch.Tensor:
        """Compute the position in the payload for each offset.

        Args:
            offsets: (N,) Tensor of offsets
        Returns:
            positions: (N,) Tensor of positions in the payload
        """


        offsets_flat = offsets.reshape(-1)
        uniform = stateless_uniform(seed=self.multibit_seed, offsets=offsets_flat)
        positions = (uniform * self.payload_size).to(torch.long).clamp_(0, self.payload_size - 1)

        return positions.reshape(offsets.shape)

    def sample(
        self, seed: int, context_hashes: torch.Tensor | int, tokens: torch.Tensor
    ) -> torch.Tensor:
        if isinstance(context_hashes, int):
            context_hashes = torch.tensor(
                [context_hashes], device=tokens.device, dtype=torch.int32
            )

        num_ctx = context_hashes.size(0)
        num_tok = tokens.size(0)

        offsets = self._compute_offsets(context_hashes, tokens)
        samples = self._sample_with_offsets(seed, offsets, num_ctx, num_tok)

        positions = self._compute_position(context_hashes).unsqueeze(1).expand(num_ctx, num_tok)  # (C, T)

        payload_gathered = self.payload.to(tokens.device)[positions]
        samples = 2 * payload_gathered * samples + (1 - payload_gathered - samples)

        return samples

    def sample_with_position(
        self, seed: int, context_hashes: torch.Tensor | int, tokens: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(context_hashes, int):
            context_hashes = torch.tensor(
                [context_hashes], device=tokens.device, dtype=torch.int32
            )

        num_ctx = context_hashes.size(0)
        num_tok = tokens.size(0)

        offsets = self._compute_offsets(context_hashes, tokens)
        samples = self._sample_with_offsets(seed, offsets, num_ctx, num_tok)
        positions = self._compute_position(context_hashes) # (C, )

        return samples, positions

    def _compute_max_multinomial_p_val(self, observed_count, T, k = 2):
        """
        Compute the p-value by subtracting the cdf(observed_count -1) of multinomial~(T, 1/base, ... 1/base),
        which is the probability of observing a sample as extreme or more as the observed_count
        The computation follows from Levin, Bruce. "A representation for multinomial cumulative distribution functions."
        The Annals of Statistics (1981): 1123-1126.
        """
        if T <= 0:
            return 1
        poiss = scipy.stats.poisson
        normal = scipy.stats.norm
        s = T
        a = observed_count 
        poiss_cdf_X = poiss.cdf(a, T / k)
        normal_approx_W = normal.cdf(0.5 / np.sqrt(T)) - normal.cdf(-0.5 / np.sqrt(T))
        log_max_multi_cdf = math.log(np.sqrt(2 * math.pi * T)) + k * math.log(poiss_cdf_X) + math.log(normal_approx_W)
        max_multi_cdf = math.exp(log_max_multi_cdf)
        p_val = 1 - min(1, max_multi_cdf)
        return p_val
    
    def _mc_pvalue(
        self,
        statistic: float,
        positions: torch.Tensor,
        n_mc: int,
    ):
        device = positions.device
        positions = positions.long()
        V = positions.size(0)

        scores = self.random_sample(V=V, n_mc=n_mc, device=device)

        sums_by_pos = torch.zeros((n_mc, self.payload_size), device=device)
        sums_by_pos.index_add_(1, positions, scores)

        counts_by_pos = torch.bincount(positions, minlength=self.payload_size).to(device)  # (payload_size,)
        counts_by_pos_f = counts_by_pos.clamp_min(1).float()

        means_by_pos = sums_by_pos / counts_by_pos_f # (n_mc, payload_size)

        predicted_bits = (means_by_pos >= 0.5).long()  # (n_mc, payload_size)

        zero_mask = counts_by_pos == 0
        if zero_mask.any():
            predicted_bits[:, zero_mask] = torch.randint(
                0, 2, (n_mc, int(zero_mask.sum().item())), device=device
            )

        bits_for_tokens = predicted_bits[:, positions]
        predicted_scores = torch.where(bits_for_tokens.bool(), scores, 1.0 - scores) 

        mc_statistics = []
        for i in range(n_mc):
            out = super().statistical_test(predicted_scores[i].tolist())
            mc_statistics.append(out["statistic"])

        mc_statistics = np.asarray(mc_statistics)
        pvalue = (np.sum(mc_statistics >= statistic) + 1) / (n_mc + 1)
        return float(pvalue)


    @add_elapsed_time()
    def detect(
        self,
        tokens: list[int] | torch.Tensor,
        seeding_scheme: SeedingScheme,
        rng_device: torch.device,
        context_size: int,
        seed: int,
    ) -> float:

        scores = []

        if isinstance(tokens, list):
            tokens = torch.tensor(tokens, device=rng_device, dtype=torch.int32)

        tokens = tokens.to(rng_device)

        context, tokens = seeding_scheme.hash_context(tokens, context_size)
        pairs = torch.stack((context, tokens), dim=1)  # shape: [N, 2]
        unique_pairs = torch.unique(
            pairs, dim=0, sorted=False, return_inverse=False, return_counts=False
        )
        offsets = torch.hash_tensor(unique_pairs, dim=1)
        context_hashes = unique_pairs[:,0]

        scores = self._sample(seed, offsets)
        positions = self._compute_position(context_hashes)

        scores_by_position = torch.bincount(
            positions,
            weights=scores,
            minlength=self.payload_size,
        ).cpu()
        counts_by_position = torch.bincount(positions, minlength=self.payload_size).cpu()

        p_values_per_bit = []
        for p in range(self.payload_size):
            n_trials = int(counts_by_position[p].item())
            if n_trials <= 0:
                p_values_per_bit.append(1.0)
                continue

            k_success = int(round(float(scores_by_position[p].item())))
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

        # Compute confidence per position
        p_val_per_position = []
        for p in range(self.payload_size):
            multi_pval = self._compute_max_multinomial_p_val(scores_by_position[p], counts_by_position[p])
            p_val_per_position.append(multi_pval)

        # Compute the predicted bit per position
        predicted_bits = []
        for p in range(self.payload_size):
            if counts_by_position[p] == 0:
                predicted_bits.append(random.randint(0,1))
            else:
                if scores_by_position[p] / counts_by_position[p] >= 0.5:
                    predicted_bits.append(1)
                else:
                    predicted_bits.append(0)

        # In the MPAC paper, they compute the z-score with the predicted bits and do a binomial test.
        # This is statistically incorrect, but we include it here for completeness.
        predicted_scores = [ 2 * predicted_bits[position] * samples + (1 - predicted_bits[position] - samples) for position, samples in zip(positions, scores)] 
        og_out = super().statistical_test(predicted_scores)

        p_value = self._mc_pvalue(
            statistic=og_out["statistic"],
            positions=positions,
            n_mc=self.n_mc,
        )

        out = {}
        out["pred_message"] = predicted_bits
        out["expected_message"] = self.payload.tolist()
        out["og_pvalue"] = og_out["pvalue"]
        out["og_statistic"] = og_out["statistic"]
        out["scores"] = scores.cpu().tolist()
        out["positions"] = positions.cpu().tolist()
        out["scores_by_position"] = scores_by_position.tolist()
        out["counts_by_position"] = counts_by_position.tolist()
        out["pvalues_by_position"] = p_val_per_position
        out["p_values_per_bit"] = p_values_per_bit
        out["pvalue"] = p_value


        return out
