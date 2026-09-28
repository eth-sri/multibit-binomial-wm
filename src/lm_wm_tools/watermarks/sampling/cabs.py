from __future__ import annotations

from collections import deque
from math import ceil

import torch

from lm_wm_tools.utils.triton_code import stateless_uniform

from .random_utility import RandomGenerator


def bits_to_int(bits: list[int]) -> int:
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def int_to_bits(value: int, width: int) -> list[int]:
    return [(value >> i) & 1 for i in range(width)][::-1]


class CABS(RandomGenerator):
    """
    Context-Anchored Balanced Scheduler (CABS) with mod-1 mirroring.

    The generator provides:
    - Stateful balanced position allocation with context-anchored frame resets.
    - Mod-1 mirroring for embedding symbols without changing the base U(0, 1)
      distribution.
    """

    def __init__(
        self,
        distribution_name: str = "uniform",
        distribution_parameters: dict | None = None,
        payload: list[int] | None = None,
        symbol_size: int = 2,
        context_size: int = 4,
        frame_window: int = 4,
        frame_bits: int = 3,
        min_len: int | None = None,
        max_factor: float = 1.5,
        multibit_seed: int = 1847389390,
        one_bit_symmetric: bool = True,
        **kwargs,
    ) -> None:
        if distribution_parameters is None:
            distribution_parameters = {"low": 0.0, "high": 1.0}
        else:
            distribution_parameters = dict(distribution_parameters)

        if payload is None:
            payload = [0] * 36
        payload_bits = [int(bit) for bit in payload]
        if len(payload_bits) == 0:
            raise ValueError("payload must contain at least one bit.")
        if any(bit not in (0, 1) for bit in payload_bits):
            raise ValueError("payload must be binary (0/1).")

        m_from_kwargs = kwargs.pop("m", None)
        if m_from_kwargs is not None:
            symbol_size = int(m_from_kwargs)

        w_from_kwargs = kwargs.pop("W", None)
        if w_from_kwargs is not None:
            frame_window = int(w_from_kwargs)

        f_from_kwargs = kwargs.pop("f", None)
        if f_from_kwargs is not None:
            frame_bits = int(f_from_kwargs)

        if symbol_size < 1:
            raise ValueError("symbol_size must be >= 1.")
        if context_size < 1:
            raise ValueError("context_size must be >= 1.")
        if frame_window < 1:
            raise ValueError("frame_window must be >= 1.")
        if frame_bits < 1:
            raise ValueError("frame_bits must be >= 1.")
        if max_factor < 1.0:
            raise ValueError("max_factor must be >= 1.0.")

        low = float(distribution_parameters.get("low", 0.0))
        high = float(distribution_parameters.get("high", 1.0))
        if high <= low:
            raise ValueError("Uniform distribution requires high > low.")

        # MirrorMark uses U(0,1); keep external name for logging compatibility.
        super().__init__(
            distribution_name="uniform",
            distribution_parameters={"low": low, "high": high},
        )

        self.low = low
        self.high = high
        self.payload = torch.tensor(payload_bits, dtype=torch.int32)
        self.payload_size = len(payload_bits)

        self.symbol_size = int(symbol_size)
        self.symbol_cardinality = 1 << self.symbol_size
        self.context_size = int(context_size)
        self.frame_window = int(frame_window)
        self.frame_bits = int(frame_bits)
        self.max_factor = float(max_factor)
        self.multibit_seed = int(multibit_seed)
        self.one_bit_symmetric = bool(one_bit_symmetric)

        self.num_positions = (self.payload_size + self.symbol_size - 1) // self.symbol_size
        pad = self.num_positions * self.symbol_size - self.payload_size
        payload_padded = payload_bits + ([0] * pad)
        symbols = [
            bits_to_int(payload_padded[i * self.symbol_size : (i + 1) * self.symbol_size])
            for i in range(self.num_positions)
        ]
        self.message_symbols = torch.tensor(symbols, dtype=torch.long)

        self.min_len = int(min_len) if min_len is not None else self.num_positions
        if self.min_len < 1:
            raise ValueError("min_len must be >= 1.")
        self.max_len = max(self.min_len, int(ceil(self.max_factor * self.num_positions)))

        self.reset_scheduler_state()

    def reset_scheduler_state(self) -> None:
        self._counts = [0 for _ in range(self.num_positions)]
        self._queue: deque[int] = deque([], maxlen=self.frame_window)
        self._frame_len = 0
        self._seen_contexts: set[tuple[int, ...]] = set()

    def _reset_frame_state(self) -> None:
        self._counts = [0 for _ in range(self.num_positions)]
        self._queue.clear()
        self._frame_len = 0

    def _context_key(self, output_ids: list[int] | torch.Tensor) -> tuple[int, ...] | None:
        if isinstance(output_ids, torch.Tensor):
            if output_ids.numel() < self.context_size:
                return None
            return tuple(int(x) for x in output_ids[-self.context_size :].tolist())

        if len(output_ids) < self.context_size:
            return None
        return tuple(int(x) for x in output_ids[-self.context_size :])

    @staticmethod
    def _stable_queue_hash(values: deque[int]) -> int:
        # Deterministic 32-bit FNV-1a hash to avoid Python hash randomization.
        h = 2166136261
        for value in values:
            h ^= int(value) & 0xFFFFFFFF
            h = (h * 16777619) & 0xFFFFFFFF
        return int(h)

    def _sample_choice(self, context_hash: int, num_choices: int) -> int:
        if num_choices <= 1:
            return 0

        offsets = torch.tensor([int(context_hash)], dtype=torch.int64)
        draw = float(
            stateless_uniform(seed=self.multibit_seed, offsets=offsets)[0].item()
        )
        idx = int(draw * float(num_choices))
        return min(max(idx, 0), num_choices - 1)

    def assign_position(
        self, output_ids: list[int] | torch.Tensor, context_hash: int
    ) -> int | None:
        context_key = self._context_key(output_ids)
        if context_key is None:
            return None
        if context_key in self._seen_contexts:
            return None
        self._seen_contexts.add(context_key)

        frame_hash = self._stable_queue_hash(self._queue)
        last_token = int(context_key[-1])
        self._queue.append(last_token)

        min_count = min(self._counts)
        min_positions = [idx for idx, count in enumerate(self._counts) if count == min_count]
        position = min_positions[self._sample_choice(context_hash, len(min_positions))]

        self._counts[position] += 1
        self._frame_len += 1

        cut = (
            (self._frame_len >= self.min_len and frame_hash % (1 << self.frame_bits) == 0)
            or self._frame_len >= self.max_len
        )
        if cut:
            self._reset_frame_state()

        return position

    def update_state(self, **kwargs):
        output_ids = kwargs.get("output_ids")
        context_hash = kwargs.get("context_hash")
        if output_ids is None or context_hash is None:
            return None
        return self.assign_position(output_ids, int(context_hash))

    def position_from_context_hash(self, context_hashes: torch.Tensor) -> torch.Tensor:
        offsets = context_hashes.reshape(-1).to(torch.int64)
        draws = stateless_uniform(seed=self.multibit_seed, offsets=offsets)
        positions = (draws * float(self.num_positions)).to(torch.long).clamp_(
            0, self.num_positions - 1
        )
        return positions.reshape(context_hashes.shape)

    def _to_unit_interval(self, samples: torch.Tensor) -> torch.Tensor:
        samples = samples.to(torch.float32)
        if self.low == 0.0 and self.high == 1.0:
            return samples
        normalized = (samples - self.low) / (self.high - self.low)
        return normalized.clamp(0.0, 1.0 - 1e-7)

    def mirror(self, values: torch.Tensor, symbols: torch.Tensor | int) -> torch.Tensor:
        squeeze_ctx = False
        if values.dim() == 1:
            values = values.unsqueeze(0)
            squeeze_ctx = True

        if isinstance(symbols, int):
            symbol_tensor = torch.tensor([symbols], device=values.device, dtype=torch.long)
        else:
            symbol_tensor = symbols.to(values.device, dtype=torch.long).reshape(-1)

        if symbol_tensor.numel() == 1 and values.size(0) > 1:
            symbol_tensor = symbol_tensor.expand(values.size(0))
        if symbol_tensor.numel() != values.size(0):
            raise ValueError(
                "symbols must provide one value per context. "
                f"Got {symbol_tensor.numel()} for {values.size(0)} contexts."
            )

        values = values.to(torch.float32)

        if self.symbol_size == 1 and self.one_bit_symmetric:
            mirrored = values.clone()
            symbol_zero_mask = symbol_tensor == 0
            if symbol_zero_mask.any():
                mirrored[symbol_zero_mask] = 1.0 - values[symbol_zero_mask]
            return mirrored.squeeze(0) if squeeze_ctx else mirrored

        psi = symbol_tensor.to(values.dtype) / float(1 << (self.symbol_size + 1))
        mirrored = torch.remainder((2.0 * psi).unsqueeze(1) - values, 1.0)
        return mirrored.squeeze(0) if squeeze_ctx else mirrored

    def sample(
        self,
        seed: int,
        context_hashes: torch.Tensor | int,
        tokens: torch.Tensor,
        embed_message: bool = True,
        positions: torch.Tensor | int | None = None,
        symbols: torch.Tensor | int | None = None,
    ) -> torch.Tensor:
        if isinstance(context_hashes, int):
            context_hashes = torch.tensor(
                [context_hashes], device=tokens.device, dtype=torch.int32
            )

        num_ctx = context_hashes.size(0)
        num_tok = tokens.size(0)

        offsets = self._compute_offsets(context_hashes, tokens)
        raw = self._sample_with_offsets(seed, offsets, num_ctx, num_tok)
        raw = self._to_unit_interval(raw)
        if not embed_message:
            return raw

        if symbols is None:
            if positions is None:
                positions = self.position_from_context_hash(context_hashes)
            elif isinstance(positions, int):
                positions = torch.tensor(
                    [positions], device=tokens.device, dtype=torch.long
                )
            else:
                positions = positions.to(tokens.device, dtype=torch.long).reshape(-1)

            if positions.numel() == 1 and num_ctx > 1:
                positions = positions.expand(num_ctx)

            if positions.numel() != num_ctx:
                raise ValueError(
                    "positions must provide one value per context. "
                    f"Got {positions.numel()} for {num_ctx} contexts."
                )
            symbols = self.message_symbols.to(tokens.device)[positions]
        else:
            if isinstance(symbols, int):
                symbols = torch.tensor([symbols], device=tokens.device, dtype=torch.long)
            else:
                symbols = symbols.to(tokens.device, dtype=torch.long).reshape(-1)
            if symbols.numel() == 1 and num_ctx > 1:
                symbols = symbols.expand(num_ctx)

        return self.mirror(raw, symbols)

    def symbols_to_bits(self, symbols: list[int]) -> list[int]:
        bits: list[int] = []
        for symbol in symbols:
            bits.extend(int_to_bits(int(symbol), self.symbol_size))
        return bits[: self.payload_size]
