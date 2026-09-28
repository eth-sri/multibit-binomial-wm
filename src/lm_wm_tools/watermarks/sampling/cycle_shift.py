import torch

from .random_utility import RandomGenerator, SeedingScheme
from lm_wm_tools.watermarks.utils import add_elapsed_time

def int_to_bits(n: int, width: int) -> list[int]:
    """Convert an integer to a list of bits with the given width."""
    return [(n >> i) & 1 for i in range(width)][::-1]

def bits_to_int(bits: list[int]) -> int:
    """Convert a list of bits to an integer."""
    n = 0
    for bit in bits:
        n = (n << 1) | bit
    return n

class CycleShift(RandomGenerator):
    """Multi-Bit Watermark vis Position Allocation (MPAC) random generator."""

    def __init__(
        self,
        vocab_size: int,
        distribution_name: str = "binomial",
        distribution_parameters: dict = {"total_count": 1, "probs": 0.5},
        payload: list[int] = [0,0,0,0,0,0,0,0,0],
        **kwargs,
    ) -> None:


        # Assert that the payload size is smaller than vocab size
        payload_size = len(payload)
        assert 2**payload_size <= vocab_size, "Payload size must be smaller than or equal to log2(vocab_size)"

        super().__init__(distribution_name, distribution_parameters)
        self.payload = torch.tensor(payload, dtype=torch.int32)

        self.payload_size = len(self.payload)
        self.payload_int = bits_to_int(payload)

        self.vocab_size = vocab_size


    def sample(
        self, seed: int, context_hashes: torch.Tensor | int, tokens: torch.Tensor, shift: bool = True
    ) -> torch.Tensor:

        if shift:
            shifted_tokens = (tokens + self.payload_int) % self.vocab_size
        else:
            shifted_tokens = tokens

        samples = super().sample(seed, context_hashes, shifted_tokens)

        return samples

    @add_elapsed_time()
    def detect(
        self,
        tokens: list[int] | torch.Tensor,
        seeding_scheme: SeedingScheme,
        rng_device: torch.device,
        context_size: int,
        seed: int,
        chunk_size: int = 512,  # Added chunk_size parameter
    ) -> float:

        if isinstance(tokens, list):
            tokens = torch.tensor(tokens, device=rng_device, dtype=torch.int32)

        tokens = tokens.to(rng_device)

        context, tokens = seeding_scheme.hash_context(tokens, context_size)
        pairs = torch.stack((context, tokens), dim=1)
        unique_pairs = torch.unique(
            pairs, dim=0, sorted=False, return_inverse=False, return_counts=False
        )

        num_pairs = unique_pairs.size(0)
        payload_dim = 2**self.payload_size  # M
        
        reduced_scores_list = []
        
        total_scores_sum = torch.zeros(payload_dim, device=rng_device)

        for i in range(0, num_pairs, chunk_size):
            chunk = unique_pairs[i : i + chunk_size]
            batch_scores = self.sample(
                seed,
                chunk[:, 0],
                torch.arange(self.vocab_size, device=rng_device),
                shift=False,
            )
            
            V = batch_scores.size(1)
            shifts = chunk[:, 1].to(torch.long)

            base_indices = torch.arange(payload_dim, device=rng_device).unsqueeze(0) # [1, M]
            gather_indices = (base_indices + shifts.unsqueeze(1)) % V                # [Batch, M]
            
            batch_shifted = batch_scores.gather(1, gather_indices)
            
            total_scores_sum += batch_shifted.sum(dim=0)
            
            reduced_scores_list.append(batch_shifted)
            
            del batch_scores


        final_scores = torch.cat(reduced_scores_list, dim=0)

        payload = torch.argmax(total_scores_sum).item()

        winning_scores = final_scores[:, payload]

        out = self.statistical_test(winning_scores)

        out["expected_payload"] = self.payload_int
        out["pred_payload"] = payload
        out["pred_message"] = int_to_bits(payload, self.payload_size)
        out["expected_message"] = self.payload.tolist()

        # Adjust p-value
        pvalue = out["pvalue"]
        M = payload_dim
        pvalue = (1 - (1 - pvalue)**M) if pvalue > min(1 / M, 1e-5) else M * pvalue
        out["pvalue"] = float(pvalue)

        return out
