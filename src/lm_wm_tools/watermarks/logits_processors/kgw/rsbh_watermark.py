from functools import lru_cache
from pathlib import Path

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

from .generalizedReedSolomon.generalizedreedsolo import Generalized_Reed_Solomon

def extract_model_family(model_name: str) -> str:
    if "Llama-3" in model_name:
        return "llama3"
    elif "Ministral" in model_name:
        return "ministral"
    raise ValueError(f"Unknown model family for model_name: {model_name}")

# Map payload size to RSBH parameters
# Values are from Table 3: https://arxiv.org/pdf/2401.16820
# (segments_num, gf_segments_num, segment_bit)
RSBH_CONFIG = {
    16: (4,6,4),
    20: (4,6,5),
    24: (3,5,8),
    32: (4,6,8),
}

# BH mapping path
# These mappings are generated in an offline manner, see: https://github.com/randomizedtree/segment-watermark
# The key corresponds to the gf_segments_num
BH_MAPPING_PATHS = {
    "llama3": {
        3: "data/rsbh_data/llama3/gf_seg_3_map_freq.pkl",
        4: "data/rsbh_data/llama3/gf_seg_4_map_freq.pkl",
        5: "data/rsbh_data/llama3/gf_seg_5_map_freq.pkl",
        6: "data/rsbh_data/llama3/gf_seg_6_map_freq.pkl",
    },
    "ministral": {
        4: "data/rsbh_data/ministral/gf_seg_4_map_freq.pkl",
        5: "data/rsbh_data/ministral/gf_seg_5_map_freq.pkl",
        6: "data/rsbh_data/ministral/gf_seg_6_map_freq.pkl",
    },
}

# Cache heavy resources to avoid per-example re-init in multiprocessing.
@lru_cache(maxsize=1)
def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "data").exists():
            return parent
    return Path.cwd()


def _resolve_bh_mapping_path(path: str) -> str:
    candidate = Path(path)
    if candidate.is_absolute():
        return str(candidate)
    return str((_repo_root() / candidate).resolve())


def _device_cache_key(device: torch.device | str) -> str:
    if isinstance(device, str):
        device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        return "cuda"
    return str(device)


@lru_cache(maxsize=None)
def _load_bh_mapping(path: str):
    import pickle

    with open(path, "rb") as f:
        return pickle.load(f)


@lru_cache(maxsize=None)
def _get_mapping_tensor(path: str, vocab_size: int, device_key: str) -> torch.Tensor:
    mapping = _load_bh_mapping(path)
    if isinstance(mapping, dict):
        mapping_tensor = torch.empty(vocab_size, dtype=torch.int64)
        keys = torch.tensor(list(mapping.keys()), dtype=torch.int64)
        values = torch.tensor(list(mapping.values()), dtype=torch.int64)
        mapping_tensor[keys] = values
    else:
        mapping_tensor = torch.tensor(mapping, dtype=torch.int64)
    return mapping_tensor.to(device_key)


@lru_cache(maxsize=None)
def _get_rs(message_length: int, payload_length: int, symbol_size: int, p_factor: int):
    return Generalized_Reed_Solomon(
        field_size=2,
        message_length=message_length,
        payload_length=payload_length,
        symbol_size=symbol_size,
        p_factor=p_factor,
        debug=False,
    )



def bits_to_int(bits: list[int]) -> int:
    out = 0
    for i, bit in enumerate(bits):
        out += bit << i
    return out

def int_to_bits(x: int, bitwidth: int) -> list[int]:
    bits = []
    for i in range(bitwidth):
        bits.append((x >> i) & 1)
    return bits


class RSBHWatermark(WatermarkProtocol):
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
        model_name: str | None = None, 
        **kwargs,
    ) -> None:

        self.multibit_algorithm = normalize_multibit_algorithm(multibit_algorithm)

        assert "binomial" in distribution_name, "RSBHWatermark only supports Bernoulli distribution"
        assert distribution_parameters["total_count"] == 1, "RSBHWatermark only supports Bernoulli distribution"
        assert self.multibit_algorithm == "none", "RSBHWatermark only supports 'none' multibit algorithm"
        assert model_name is not None, "model_name must be provided for RSBHWatermark to load BH mapping"


        # RS parameters
        payload_size = len(payload)
        self.payload_size = payload_size
        assert payload_size in RSBH_CONFIG, f"Payload size {payload_size} not supported for RSBH"
        segments_num, gf_segments_num, segment_bit = RSBH_CONFIG[payload_size]

        model_family = extract_model_family(model_name)
        try:
            bh_mapping = BH_MAPPING_PATHS[model_family][gf_segments_num]
        except KeyError:
            raise ValueError(f"BH mapping not found for model family {model_family} with gf_segments_num {gf_segments_num}")

        if isinstance(seeding_scheme, str):
            seeding_scheme = SeedingScheme(seeding_scheme)
            
        if isinstance(rng_device, str):
            rng_device = torch.device(rng_device)

        # Scheme parameters
        self.delta = epsilon

        # Sampling parameters
        self.vocab_size = vocab_size
        self.rng_device = rng_device
        self.top_k = vocab_size # RSBH uses full vocab
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

        self.n_mc = int(kwargs.pop("n_mc", 10000))
        self.mc_batch_size = int(kwargs.pop("mc_batch_size", 128))
        max_mc_work = kwargs.pop("max_mc_work", 50_000_000)
        self.max_mc_work = int(max_mc_work) if max_mc_work is not None else None

        self.original_payload = payload
        self.payload = bits_to_int(payload)

        # total number of segments, or k in RS code
        self.segments_num = segments_num
        # total number of segments after RS, or n in RS code
        self.gf_segments_num = gf_segments_num
        # the # of bits within one segment, m in RS code (GF(q^m))
        self.segment_bit = segment_bit

        # bidwidth = segment_bit * segments_num
        self.bitwidth = self.segments_num * self.segment_bit

        # balanced hash mapping, we generate this in an offline manner
        mapping_path = None
        if isinstance(bh_mapping, str):
            mapping_path = _resolve_bh_mapping_path(bh_mapping)
            bh_mapping = _load_bh_mapping(mapping_path)
        self.mapping = bh_mapping  # dict: token_id -> segment_id
        if mapping_path is not None:
            self.mapping_tensor = _get_mapping_tensor(
                mapping_path, self.vocab_size, _device_cache_key(self.rng_device)
            )
        elif isinstance(self.mapping, dict):
            mapping_tensor = torch.empty(self.vocab_size, dtype=torch.int64)
            for token_id, seg_id in self.mapping.items():
                mapping_tensor[token_id] = seg_id
            self.mapping_tensor = mapping_tensor.to(self.rng_device)
        else:
            self.mapping_tensor = torch.tensor(
                self.mapping, dtype=torch.int64, device=self.rng_device
            )

        # 1. divide original message into segments
        mask = 2 ** self.segment_bit - 1
        self.segments = [
            (self.payload >> (self.segment_bit * i)) & mask for i in range(self.segments_num)
        ]

        # 2. encode segments with RS
        self.rs = _get_rs(
            self.gf_segments_num,
            self.segments_num,
            self.segment_bit,
            1,
        )
        self.gf_segments = [int(i) for i in self.rs.encode(self.segments)]

    def get_name(self) -> str:
        return format_watermark_name(
            "RSBH",
            self.multibit_algorithm,
            f"{self.seeding_scheme.value}/k_{self.context_size}/delta_{self.delta:<.2f}/seed_{self.seed}",
            payload_size=self.payload_size,
        )

    def __call__(
        self,
        output_ids: list[int],
        logits: torch.Tensor,  # (vocab_size)
    ) -> torch.Tensor:
        context_hash = self.seeding_scheme.hash_last_context(torch.tensor(output_ids, device=self.rng_device), self.context_size)
        if context_hash is not None:

            scores = self.random_generator.sample(
                self.seed,
                context_hash,
                torch.arange(self.vocab_size, device=logits.device),
            ).squeeze(0) * self.delta

            # assign current token to one segment based on mapping
            random_int = self.mapping[output_ids[-1]]

            scores = scores.roll(-self.gf_segments[random_int])

            logits = logits + scores.to(logits.device)
        return logits

    def get_expected_probs(self, probs: torch.Tensor) -> torch.Tensor:
        """
        Given a prob vector p [V], return the expected probs under the watermarking
        selection policy.
        """       
        return probs

    @add_elapsed_time()
    def detect(self, tokens: list[int] | torch.Tensor) -> float:
        
        if isinstance(tokens, list):
            tokens = torch.tensor(tokens, device=self.rng_device, dtype=torch.int32)

        full_tokens = tokens.to(self.rng_device)

        context, tokens = self.seeding_scheme.hash_context(full_tokens, self.context_size)
        prev_tokens = full_tokens[self.context_size - 1 : -1]
        triplets = torch.stack((context, tokens, prev_tokens), dim=1)  # shape: [N, 3]
        unique_pairs = torch.unique(
            triplets, dim=0, sorted=False, return_inverse=False, return_counts=False
        )

        segment_count = 2**self.segment_bit
        count = torch.zeros(
            (self.gf_segments_num, segment_count),
            device=tokens.device,
            dtype=torch.int32,
        )
        random_ints = None

        if unique_pairs.numel() > 0:
            context_hashes = unique_pairs[:, 0]
            token_ids = unique_pairs[:, 1]
            prev_token_ids = unique_pairs[:, 2]

            random_ints = self.mapping_tensor[prev_token_ids].to(torch.int64)
            j_offsets = torch.arange(
                segment_count, device=tokens.device, dtype=torch.int64
            )
            token_offsets = (token_ids.to(torch.int64).unsqueeze(1) + j_offsets) % self.vocab_size
            context_grid = context_hashes.to(torch.int64).unsqueeze(1).expand(-1, segment_count)
            pair_grid = torch.stack((context_grid, token_offsets), dim=-1)
            offsets = torch.hash_tensor(pair_grid, dim=-1)
            scores = self.random_generator._sample(self.seed, offsets.reshape(-1)).reshape(
                -1, segment_count
            )
            if scores.dtype != count.dtype:
                scores = scores.to(count.dtype)
            count.index_add_(0, random_ints, scores)

        decoded_segments = torch.argmax(count, dim=1).cpu().numpy().tolist()
        token_num = int(unique_pairs.size(0))

        try:
            payload_in_segs = self.rs.decode(decoded_segments)
        except (ZeroDivisionError, IndexError) as e:
            print(f"{e} in RS decode!")
            payload_in_segs = decoded_segments[:self.segments_num]
        decoded_message_segments = payload_in_segs
        
        # reconstruct the decoded message in base 2
        decoded_message = 0
        for i, segment in enumerate(decoded_message_segments):
            decoded_message += segment << (i * self.segment_bit)
        decoded_message_bits = int_to_bits(decoded_message, self.bitwidth)

        out = {}

        out["gf_segments"] = self.gf_segments
        out["decoded_segments"] = decoded_segments
        out["pred_message"] = decoded_message_bits
        out["expected_message"] = self.original_payload
        out["pvalue"] = None
        out["n_trials"] = token_num

        return out
