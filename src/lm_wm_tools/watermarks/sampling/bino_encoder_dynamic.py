import torch
from math import sqrt
from scipy.stats import binomtest

from lm_wm_tools.utils.triton_code import stateless_bernoulli_bits
from lm_wm_tools.watermarks.utils import add_elapsed_time
from .random_utility import RandomGenerator, SeedingScheme
from .bino_encoder import _VOCAB_SIZE_UPPER_BOUND


def int_to_bits(n: int, width: int) -> list[int]:
    """Convert an integer to a list of bits with the given width."""
    return [(n >> i) & 1 for i in range(width)][::-1]


def bits_to_int(bits: list[int]) -> int:
    """Convert a list of bits to an integer."""
    n = 0
    for bit in bits:
        n = (n << 1) | bit
    return n


def normalize_remaining_bits(remaining_bits: int | list[int]) -> list[int]:
    """Normalize remaining_bits into a non-empty list of positive integers."""
    if isinstance(remaining_bits, int):
        bits = [int(remaining_bits)]
    elif isinstance(remaining_bits, list):
        if len(remaining_bits) == 0:
            raise ValueError("remaining_bits must contain at least one value.")
        bits = [int(bit) for bit in remaining_bits]
    else:
        raise TypeError(
            "remaining_bits must be an int or a list of ints, "
            f"got {type(remaining_bits)}."
        )

    if any(bit <= 0 for bit in bits):
        raise ValueError("remaining_bits values must be strictly positive.")

    return bits


class BinoEncoderState:

    def __init__(
        self,
        payload_size: int,
        payload: torch.Tensor,
        wait: int = 0,
        remaining_bits: int | list[int] = 64,
        method: str = "prob",
    ):
        
        if payload.numel() != payload_size:
            raise ValueError(
                "payload_size must match payload length, "
                f"got payload_size={payload_size} and payload length={payload.numel()}."
            )

        self.payload_size = int(payload_size)
        self.payload = payload.to(dtype=torch.int32).clone()
        self.payload_matches = torch.zeros(
            payload_size, dtype=torch.int32, device=self.payload.device
        )
        self.running_counts = torch.zeros_like(self.payload_matches)

        self.n_trials = 0

        if wait < 0:
            raise ValueError("wait must be non-negative.")
        if method not in {"linear", "exp", "prob", "prob_dynamic", "ba_at_1pct_fpr"}:
            raise ValueError("method must be one of {'linear', 'exp', 'prob', 'prob_dynamic', 'ba_at_1pct_fpr'}.")

        self.wait = int(wait)
        self.remaining_bits = normalize_remaining_bits(remaining_bits)
        self.method = method

    def update(self, scores: torch.Tensor):

        if self.wait > 0:
            self.wait -= 1
            return

        scores = scores.to(dtype=torch.int32)
        if scores.shape != self.payload.shape:
            raise ValueError(
                "Scores must have the same shape as payload when updating state, "
                f"got {tuple(scores.shape)} vs {tuple(self.payload.shape)}."
            )

        if self.payload_matches.device != scores.device:
            self.payload_matches = self.payload_matches.to(scores.device)
            self.running_counts = self.running_counts.to(scores.device)

        payload = self.payload.to(scores.device)
        self.payload_matches += (scores == payload).to(torch.int32)
        self.running_counts += scores.to(torch.int32)
        self.n_trials += 1

    def transform_scores(self, scores: torch.Tensor) -> torch.Tensor:

        if scores.shape[-1] != self.payload_size:
            raise ValueError(
                "scores last dimension must match payload_size for transform_scores, "
                f"got scores shape {tuple(scores.shape)} and payload_size {self.payload_size}."
            )

        if self.method in {"linear", "exp"}:

            weights = self.get_weights().to(device=scores.device, dtype=torch.float32)
            view_shape = (1,) * (scores.dim() - 1) + (scores.shape[-1],)
            return scores.to(dtype=torch.float32) * weights.reshape(view_shape)
        
        elif self.method == "prob":

            dit = 2 *self.payload_matches.to(dtype=torch.float32) - self.n_trials
            view_shape = (1,) * (scores.dim() - 1) + (scores.shape[-1],)
            dit = dit.reshape(view_shape).to(device=scores.device)
            
            new_dit = dit + (2 * scores.to(dtype=torch.float32) - 1)


            score_terms = [
                torch.distributions.Normal(0, sqrt(remaining_bit)).cdf(new_dit)
                for remaining_bit in self.remaining_bits
            ]
            scores = torch.stack(score_terms, dim=0).mean(dim=0)

            return scores
        
        elif self.method == "prob_dynamic":

            dit = 2 *self.payload_matches.to(dtype=torch.float32) - self.n_trials
            view_shape = (1,) * (scores.dim() - 1) + (scores.shape[-1],)
            dit = dit.reshape(view_shape).to(device=scores.device)
            
            new_dit = dit + (2 * scores.to(dtype=torch.float32) - 1)

            score_terms = [
                torch.distributions.Normal(
                    0,
                    sqrt(max(1, remaining_bit - self.n_trials)),
                ).cdf(new_dit)
                for remaining_bit in self.remaining_bits
            ]
            scores = torch.stack(score_terms, dim=0).mean(dim=0)

            return scores

        elif self.method == "ba_at_1pct_fpr":
            dit = 2 * self.payload_matches.to(dtype=torch.float32) - self.n_trials
            view_shape = (1,) * (scores.dim() - 1) + (scores.shape[-1],)
            dit = dit.reshape(view_shape).to(device=scores.device)

            new_dit = dit + (2 * scores.to(dtype=torch.float32) - 1)

            score_terms = []
            for remaining_bit in self.remaining_bits:
                future_var = max(1, remaining_bit)
                total_trials = self.n_trials + 1 + remaining_bit
                threshold = torch.distributions.Normal(0, sqrt(total_trials)).icdf(torch.tensor(0.99))
                term = torch.distributions.Normal(0, sqrt(future_var)).cdf(new_dit - threshold)
                score_terms.append(term)

            scores = torch.stack(score_terms, dim=0).mean(dim=0)
            return scores


    def get_weights(self):
            
        if self.method == "linear":
            if self.n_trials <= 0:
                # No observations yet: keep all bits equally weighted and avoid NaNs.
                return torch.zeros_like(self.payload_matches, dtype=torch.float32)

            weights = 1 - self.payload_matches.to(dtype=torch.float32) / float(self.n_trials)

        elif self.method == "exp":
            running_counts = self.running_counts.to(dtype=torch.float32)
            distance = 2 * running_counts - self.n_trials
            weight_terms = [
                torch.exp(-distance**2 / (2 * remaining_bit))
                for remaining_bit in self.remaining_bits
            ]
            weights = torch.stack(weight_terms, dim=0).mean(dim=0)

        return weights

class DynamicBinoEncoder(RandomGenerator):
    def __init__(
        self,
        distribution_name: str = "binomial",
        distribution_parameters: dict = {"total_count": 8, "probs": 0.5},
        payload: list[int] = [0, 0, 0, 0, 0, 0, 0, 0],
        remaining_bits: int | list[int] = 64,
        wait: int = 0,
        method: str = "prob",
        **kwargs,
    ) -> None:
        assert distribution_name == "binomial", "DynamicBinoEncoder only supports binomial distribution."
        if len(payload) == 0:
            raise ValueError("payload must contain at least one bit.")
        if any(bit not in (0, 1) for bit in payload):
            raise ValueError("payload must be a list of bits (0/1).")

        payload_size = len(payload)
        distribution_name, distribution_parameters = self.get_distribution(
            distribution_name, distribution_parameters, payload_size
        )

        super().__init__(distribution_name, distribution_parameters)
        self.payload = torch.tensor(payload, dtype=torch.int32)

        self.payload_size = payload_size
        self.payload_int = bits_to_int(payload)
        self.remaining_bits = normalize_remaining_bits(remaining_bits)
        self.wait = int(wait)
        self.method = method

        self.state = BinoEncoderState(
            payload_size,
            self.payload,
            wait=self.wait,
            remaining_bits=self.remaining_bits,
            method=self.method,
        )

        self.candidate_indexes = None

    def update_distribution(self):
        # Dynamic effective distribution depends on state at call time.
        self.candidate_indexes = self.state.get_candidate_indexes()

    def update_state(
        self,
        tokens: list[int] | torch.Tensor | None = None,
        seeding_scheme: SeedingScheme | None = None,
        rng_device: torch.device | None = None,
        context_size: int | None = None,
        seed: int | None = None,
        context_hash: int | None = None,
        token: int | None = None,
    ):
        if seed is None:
            return

        if rng_device is not None:
            self.rng_device = rng_device
        elif not hasattr(self, "rng_device"):
            self.rng_device = self.payload.device

        observed_context_hash: int | None = None
        observed_token: int | None = None

        if context_hash is not None:
            observed_context_hash = int(context_hash)
            if token is not None:
                observed_token = int(token)
            elif tokens is not None:
                if isinstance(tokens, torch.Tensor):
                    if tokens.numel() == 0:
                        return
                    observed_token = int(tokens.reshape(-1)[-1].item())
                else:
                    if len(tokens) == 0:
                        return
                    observed_token = int(tokens[-1])
        else:
            if tokens is None or seeding_scheme is None or context_size is None:
                return

            if isinstance(tokens, list):
                tokens_tensor = torch.tensor(
                    tokens,
                    device=self.rng_device,
                    dtype=torch.int32,
                )
            else:
                tokens_tensor = tokens.to(device=self.rng_device, dtype=torch.int32)

            if tokens_tensor.numel() <= context_size:
                return

            observed_context_hash = seeding_scheme._hash_last_context(
                tokens_tensor[:-1], context_size
            )
            if observed_context_hash is None:
                return
            observed_token = int(tokens_tensor[-1].item())

        if observed_context_hash is None or observed_token is None:
            return

        scores = self.sample(
            seed=seed,
            context_hashes=observed_context_hash,
            tokens=torch.tensor(
                [observed_token],
                device=self.rng_device,
                dtype=torch.int32,
            ),
            embed_message=False,
        ).squeeze(0)

        self.state.update(scores)

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

            # Encode the confidence
            # Low confidence scores should be boosted
            scores = scores.to(dtype=torch.float32)
            scores = self.state.transform_scores(scores)  # shape: (N, m)

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

        out = self._sample_from_offsets(
            seed=seed,
            offsets=offsets,
            embed_message=embed_message,
        ) # shape: (depth, N) if depth > 1 else (N,)

        out = out.to(dtype=torch.float32)
        if embed_message:
            # Add small uniform noise for tie-breaking only while ranking
            # candidate tokens. Raw score paths are used for state updates and
            # detection, where exact 0/1 bit values matter.
            out = out + 1e-6 * torch.rand_like(out, dtype=torch.float32)
        return out

    def random_sample(self, V: int, n_mc: int, device: torch.device) -> torch.Tensor:
        """Random sampling for Monte-Carlo estimation. Do not use for watermarking as it is fully random."""

        prob = self.distribution_parameters["probs"]

        bernoullis = torch.distributions.Bernoulli(probs=prob).sample((n_mc, V, self.payload_size)).to(device)
        bernoullis = self.state.transform_scores(bernoullis)  # shape: (n_mc, V, m)
        samples = bernoullis.sum(dim=2)  # shape: (n_mc, V)

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
