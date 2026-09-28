from __future__ import annotations

import math
from typing import Any


import ot
import torch
from vllm import SamplingParams

from lm_wm_tools import WatermarkProtocol
from lm_wm_tools.utils.triton_code import stateless_uniform
from lm_wm_tools.watermarks.sampling import RandomUtilityType, SeedingScheme
from lm_wm_tools.watermarks.utils import (
    add_elapsed_time,
    format_watermark_name,
    normalize_multibit_algorithm,
    resolve_top_k_from_sampling_params,
)


class ArcMarkWatermark(WatermarkProtocol):
    """
    ArcMark multi-bit watermark using OT-based token coupling and random linear coding.

    Practical notes:
    - Randomness is fully stateless and shared between encoder/decoder via context hash + seeds.
    - The random permutation is implemented as an affine permutation over token ids.
    - OT is solved with POT Sinkhorn on the selected top-k token set.
    """

    _SIDE_SALT_BASE = 11
    _PERM_SALT_BASE = 23
    _CODE_SALT_BASE = 101
    _MAX_TORCH_INT64 = int(torch.iinfo(torch.int64).max)

    def __init__(
        self,
        vocab_size: int,
        rng_device: str | torch.device,
        seeding_scheme: SeedingScheme | str,
        context_size: int,
        seed: int,
        sampling_parameters: SamplingParams,
        payload: list[int] | None = None,
        p: int | None = None,
        r: int | None = None,
        phi: float = 0.0,
        sinkhorn_iters: int = 80,
        sinkhorn_stabilizer: float = 1e-8,
        sinkhorn_reg: float = 0.1,
        top_k: int | None = None,
        ot_top_k: int | None = None,
        decode_loss: str = "identity",
        max_decode_messages: int = 1 << 16,
        decode_chunk_size: int = 4096,
        multibit_seed: int = 1847389390,
        multibit_algorithm: RandomUtilityType | str = "none",
        distribution_name: str = "uniform",
        distribution_parameters: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        del kwargs  # Compatibility with generic config pipelines.

        if isinstance(seeding_scheme, str):
            seeding_scheme = SeedingScheme(seeding_scheme)
        if isinstance(rng_device, str):
            rng_device = torch.device(rng_device)

        if context_size < 1:
            raise ValueError("ArcMark requires context_size >= 1.")

        payload_bits = [0] * 8 if payload is None else [int(bit) for bit in payload]
        if len(payload_bits) == 0:
            raise ValueError("ArcMark payload must contain at least one bit.")
        if any(bit not in (0, 1) for bit in payload_bits):
            raise ValueError("ArcMark payload must be binary (0/1).")

        payload_size = len(payload_bits)

        assert payload_size <= 16, "Decoding requires enumeration of all possible messages, which is infeasible for too large payloads."

        if p is None:
            # ArcMark arithmetic is implemented with torch.int64 tensors.
            # Cap the default modulus to avoid uint64 promotion / overflow for large payloads.
            p = min(1 << payload_size, self._MAX_TORCH_INT64)
        p = int(p)
        if p < 2:
            raise ValueError("ArcMark requires p >= 2.")
        if p > self._MAX_TORCH_INT64:
            raise ValueError(
                f"ArcMark requires p <= {self._MAX_TORCH_INT64} "
                f"for torch.int64 compatibility, got p={p}."
            )

        if r is None:
            r = min(4 * p, 256)
        r = int(r)
        if r < 2:
            raise ValueError("ArcMark requires r >= 2.")

        if sinkhorn_iters < 1:
            raise ValueError("sinkhorn_iters must be >= 1.")

        sinkhorn_reg = float(sinkhorn_reg)
        if sinkhorn_reg <= 0.0:
            raise ValueError("ArcMark expects epsilon > 0 as Sinkhorn regularization.")

        decode_loss_normalized = str(decode_loss).strip().lower()
        if decode_loss_normalized not in {"identity", "log"}:
            raise ValueError("decode_loss must be one of {'identity', 'log'}.")

        self.vocab_size = int(vocab_size)
        self.rng_device = rng_device
        self.context_size = int(context_size)
        self.seed = int(seed)

        self.temperature = sampling_parameters.temperature
        if self.temperature is None or self.temperature <= 0:
            raise ValueError("ArcMark requires sampling temperature > 0.")

        resolved_top_k = (
            min(int(top_k), self.vocab_size)
            if top_k is not None
            else resolve_top_k_from_sampling_params(sampling_parameters, self.vocab_size)
        )
        if resolved_top_k < 1:
            raise ValueError("top_k must be >= 1.")

        if ot_top_k is not None:
            if int(ot_top_k) < 1:
                raise ValueError("ot_top_k must be >= 1 when provided.")
            resolved_top_k = min(resolved_top_k, int(ot_top_k), self.vocab_size)
        elif resolved_top_k == self.vocab_size:
            # Full-vocab OT is too expensive for large vocabularies.
            resolved_top_k = min(256, self.vocab_size)

        self.top_k = int(resolved_top_k)

        self.payload_bits = payload_bits
        self.payload_size = payload_size
        self.payload_tensor = torch.tensor(
            payload_bits, device=self.rng_device, dtype=torch.int64
        )

        self.p = p
        self.r = r
        self.phi = float(phi)

        self.sinkhorn_reg = sinkhorn_reg
        self.sinkhorn_iters = int(sinkhorn_iters)
        self.sinkhorn_stabilizer = float(max(1e-12, sinkhorn_stabilizer))

        self.decode_loss = decode_loss_normalized
        self.max_decode_messages = int(max_decode_messages)
        self.decode_chunk_size = int(max(1, decode_chunk_size))

        self.multibit_seed = int(multibit_seed)
        self.side_seed = self.multibit_seed
        self.permutation_seed = self.multibit_seed + 1
        self.code_seed = self.multibit_seed + 2

        algorithm_name = (
            multibit_algorithm.value
            if isinstance(multibit_algorithm, RandomUtilityType)
            else multibit_algorithm
        )
        self.multibit_algorithm = normalize_multibit_algorithm(algorithm_name)

        self.distribution_name = str(distribution_name)
        self.distribution_parameters = (
            {} if distribution_parameters is None else dict(distribution_parameters)
        )

        self._two_pi = float(2.0 * math.pi)
        self._token_angle_scale = self._two_pi / float(self.vocab_size)
        self._symbol_angle_scale = self._two_pi / float(self.p)
        self._side_angle_scale = self._two_pi / float(self.r)
        self._dmax = float(math.pi - (math.pi / (2.0 * max(1, self.vocab_size))))

        self._uniform_target_cache: dict[str, torch.Tensor] = {}
        self._z_angle_cache: dict[str, torch.Tensor] = {}

        self.num_messages = 1 << self.payload_size
        self._decode_enabled = self.num_messages <= self.max_decode_messages

        self._decode_matmul_dtype = (
            torch.float64
            if self.payload_size * max(1, self.p - 1) > (1 << 23)
            else torch.float32
        )

        self._candidate_bits_cpu: torch.Tensor | None = None
        self._candidate_bits_device: dict[str, torch.Tensor] = {}
        self._candidate_mat_device: dict[str, torch.Tensor] = {}

        if self._decode_enabled:
            self._candidate_bits_cpu = self._enumerate_candidate_messages(
                self.payload_size
            )

        seeding_scheme.initialize(self.vocab_size, self.seed, self.rng_device)
        self.seeding_scheme = seeding_scheme

    @staticmethod
    def _enumerate_candidate_messages(payload_size: int) -> torch.Tensor:
        num_messages = 1 << payload_size
        values = torch.arange(num_messages, dtype=torch.int64)
        shifts = torch.arange(payload_size - 1, -1, -1, dtype=torch.int64)
        bits = ((values.unsqueeze(1) >> shifts.unsqueeze(0)) & 1).to(torch.int64)
        return bits

    def _get_candidate_bits(self, device: torch.device) -> torch.Tensor | None:
        if self._candidate_bits_cpu is None:
            return None

        key = str(device)
        if key not in self._candidate_bits_device:
            self._candidate_bits_device[key] = self._candidate_bits_cpu.to(
                device=device, dtype=torch.int64
            )
        return self._candidate_bits_device[key]

    def _get_candidate_mat(self, device: torch.device) -> torch.Tensor | None:
        bits = self._get_candidate_bits(device)
        if bits is None:
            return None

        key = f"{device}_{self._decode_matmul_dtype}"
        if key not in self._candidate_mat_device:
            self._candidate_mat_device[key] = bits.to(
                device=device, dtype=self._decode_matmul_dtype
            )
        return self._candidate_mat_device[key]

    def _shared_uniform_draws(
        self,
        context_hash: int,
        *,
        count: int,
        seed: int,
        salt_base: int,
        device: torch.device,
    ) -> torch.Tensor:
        salts = torch.arange(
            salt_base,
            salt_base + count,
            device=device,
            dtype=torch.int64,
        )
        contexts = torch.full((count,), int(context_hash), device=device, dtype=torch.int64)
        pairs = torch.stack((contexts, salts), dim=1)
        offsets = torch.hash_tensor(pairs, dim=1).to(torch.int64)
        return stateless_uniform(offsets=offsets, seed=seed)

    def _coprime_multiplier(self, raw_value: int) -> int:
        if self.vocab_size <= 1:
            return 0

        start = int(raw_value) % self.vocab_size
        if start == 0:
            start = 1

        for shift in range(self.vocab_size):
            candidate = (start + shift) % self.vocab_size
            if candidate == 0:
                continue
            if math.gcd(candidate, self.vocab_size) == 1:
                return candidate

        return 1

    def _sample_side_information(
        self,
        context_hash: int,
        *,
        device: torch.device,
    ) -> tuple[int, int, int, torch.Tensor]:
        side_draw = self._shared_uniform_draws(
            context_hash,
            count=1,
            seed=self.side_seed,
            salt_base=self._SIDE_SALT_BASE,
            device=device,
        )[0]
        v_value = int(float(side_draw.item()) * float(self.r))
        v_value = min(max(v_value, 0), self.r - 1)

        perm_draws = self._shared_uniform_draws(
            context_hash,
            count=2,
            seed=self.permutation_seed,
            salt_base=self._PERM_SALT_BASE,
            device=device,
        )
        a_raw = int(float(perm_draws[0].item()) * float(self.vocab_size))
        b_shift = int(float(perm_draws[1].item()) * float(self.vocab_size))
        b_shift = min(max(b_shift, 0), max(0, self.vocab_size - 1))

        a_mult = self._coprime_multiplier(a_raw)

        code_draws = self._shared_uniform_draws(
            context_hash,
            count=self.payload_size,
            seed=self.code_seed,
            salt_base=self._CODE_SALT_BASE,
            device=device,
        )
        g_column = torch.floor(code_draws.to(torch.float64) * float(self.p)).to(
            torch.int64
        )
        g_column = torch.remainder(g_column, self.p)

        if int(g_column.sum().item()) == 0 and self.p > 1:
            g_column[int(context_hash) % self.payload_size] = 1

        return v_value, a_mult, b_shift, g_column

    def _encode_code_symbol(self, g_column: torch.Tensor) -> int:
        payload = self.payload_tensor.to(device=g_column.device, dtype=torch.int64)
        symbol = torch.remainder((payload * g_column).sum(), self.p)
        return int(symbol.item())

    def _token_angles(
        self, token_ids: torch.Tensor, a_mult: int, b_shift: int
    ) -> torch.Tensor:
        token_ids = token_ids.to(torch.int64)
        permuted = torch.remainder(
            token_ids * int(a_mult) + int(b_shift), self.vocab_size
        )
        angles = torch.remainder(
            permuted.to(torch.float32) * self._token_angle_scale,
            self._two_pi,
        )
        return angles

    def _get_z_angles(self, device: torch.device) -> torch.Tensor:
        key = str(device)
        if key not in self._z_angle_cache:
            values = torch.arange(self.r, device=device, dtype=torch.float32)
            self._z_angle_cache[key] = torch.remainder(
                values * self._side_angle_scale,
                self._two_pi,
            )
        return self._z_angle_cache[key]

    def _get_uniform_target(
        self,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        key = f"{device}_{dtype}"
        if key not in self._uniform_target_cache:
            self._uniform_target_cache[key] = torch.full(
                (self.r,),
                1.0 / float(self.r),
                device=device,
                dtype=dtype,
            )
        return self._uniform_target_cache[key]

    def _angular_distance(self, lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
        delta = torch.abs(lhs - rhs)
        return torch.minimum(delta, self._two_pi - delta)

    def _distance_to_loss(self, distances: torch.Tensor) -> torch.Tensor:
        if self.decode_loss == "identity":
            return distances

        if self._dmax <= 0:
            return distances

        normalized = 1.0 - (distances / self._dmax)
        normalized = normalized.clamp_min(1e-8)
        return -torch.log(normalized)

    def _build_cost_matrix(
        self,
        token_ids: torch.Tensor,
        *,
        a_mult: int,
        b_shift: int,
        code_symbol: int,
    ) -> torch.Tensor:
        token_angles = self._token_angles(token_ids, a_mult=a_mult, b_shift=b_shift)

        base_angle = torch.remainder(
            torch.tensor(
                float(code_symbol) * self._symbol_angle_scale + self.phi,
                device=token_ids.device,
                dtype=torch.float32,
            ),
            self._two_pi,
        )
        z_angles = torch.remainder(base_angle + self._get_z_angles(token_ids.device), self._two_pi)

        costs = self._angular_distance(token_angles.unsqueeze(1), z_angles.unsqueeze(0))
        return costs

    def _solve_ot_conditional(
        self,
        source_probs: torch.Tensor,
        cost_matrix: torch.Tensor,
        *,
        side_value: int,
    ) -> torch.Tensor:
        source = source_probs.to(dtype=torch.float64)
        source = source / source.sum().clamp_min(self.sinkhorn_stabilizer)
        target = self._get_uniform_target(
            device=source.device,
            dtype=source.dtype,
        )
        cost = cost_matrix.to(dtype=source.dtype)

        try:
            transport_plan = ot.bregman.sinkhorn(
                source,
                target,
                cost,
                reg=self.sinkhorn_reg,
                numItermax=self.sinkhorn_iters,
                stopThr=self.sinkhorn_stabilizer,
                warn=False,
            )
        except Exception:
            return source.to(dtype=torch.float32)
        if not isinstance(transport_plan, torch.Tensor):
            transport_plan = torch.as_tensor(
                transport_plan, device=source.device, dtype=source.dtype
            )
        elif (
            transport_plan.device != source.device
            or transport_plan.dtype != source.dtype
        ):
            transport_plan = transport_plan.to(device=source.device, dtype=source.dtype)

        if transport_plan.shape != cost.shape:
            return source.to(dtype=torch.float32)

        column = transport_plan[:, side_value]
        column_sum = column.sum()

        if (not torch.isfinite(column_sum)) or float(column_sum.item()) <= 0.0:
            return source.to(dtype=torch.float32)

        conditional = column / column_sum
        if not torch.isfinite(conditional).all():
            return source.to(dtype=torch.float32)

        return conditional.to(dtype=torch.float32)

    def _context_hash(self, output_ids: list[int]) -> int | None:
        if len(output_ids) < self.context_size:
            return None

        output_tensor = torch.tensor(
            output_ids,
            device=self.rng_device,
            dtype=torch.long,
        )
        return self.seeding_scheme.hash_last_context(output_tensor, self.context_size)

    def get_name(self) -> str:
        return format_watermark_name(
            "ArcMark",
            self.multibit_algorithm,
            (
                f"{self.seeding_scheme.value}/k_{self.context_size}/"
                f"p_{self.p}/r_{self.r}/"
                f"otk_{self.top_k}/eps_{self.sinkhorn_reg:.3f}/"
                f"iters_{self.sinkhorn_iters}/seed_{self.seed}"
            ),
            payload_size=self.payload_size,
        )

    @torch.no_grad()
    def __call__(self, output_ids: list[int], logits: torch.Tensor) -> torch.Tensor:
        squeeze_batch = False
        if logits.dim() == 2 and logits.size(0) == 1:
            logits = logits.squeeze(0)
            squeeze_batch = True
        elif logits.dim() != 1:
            raise ValueError(
                f"ArcMark expects 1-D logits (or [1, V]), got shape {tuple(logits.shape)}."
            )

        context_hash = self._context_hash(output_ids)
        if context_hash is None:
            return logits.unsqueeze(0) if squeeze_batch else logits

        logits_scaled = logits / self.temperature

        k = min(self.top_k, logits.shape[-1])
        topk_logits, topk_indices = torch.topk(logits_scaled, k, dim=-1)
        source_probs = torch.softmax(topk_logits, dim=-1)

        side_value, a_mult, b_shift, g_column = self._sample_side_information(
            int(context_hash),
            device=topk_indices.device,
        )
        code_symbol = self._encode_code_symbol(g_column)

        cost_matrix = self._build_cost_matrix(
            topk_indices,
            a_mult=a_mult,
            b_shift=b_shift,
            code_symbol=code_symbol,
        )
        conditional_probs = self._solve_ot_conditional(
            source_probs,
            cost_matrix,
            side_value=side_value,
        )

        updated_logits = torch.full_like(logits, -100.0)
        updated_logits[topk_indices] = (
            torch.log(conditional_probs.clamp_min(1e-30)) * self.temperature
        ).to(logits.dtype)

        if squeeze_batch:
            return updated_logits.unsqueeze(0)
        return updated_logits

    def get_expected_probs(self, probs: torch.Tensor) -> torch.Tensor:
        return probs

    def _decode_message(
        self,
        g_matrix: torch.Tensor,
        observed_angles: torch.Tensor,
    ) -> tuple[list[int], int, float, float, float, float]:
        candidate_bits = self._get_candidate_bits(observed_angles.device)
        candidate_mat = self._get_candidate_mat(observed_angles.device)
        if candidate_bits is None or candidate_mat is None:
            return [0] * self.payload_size, 0, float("inf"), 0.0, 1.0, 0.0

        observed = observed_angles.to(dtype=self._decode_matmul_dtype).unsqueeze(0)
        g_matrix = g_matrix.to(dtype=self._decode_matmul_dtype)

        scores = torch.empty(
            self.num_messages,
            device=observed_angles.device,
            dtype=self._decode_matmul_dtype,
        )

        for start in range(0, self.num_messages, self.decode_chunk_size):
            end = min(start + self.decode_chunk_size, self.num_messages)
            msg_chunk = candidate_mat[start:end]

            code_symbols = torch.remainder(msg_chunk @ g_matrix, float(self.p))
            code_angles = torch.remainder(
                code_symbols * self._symbol_angle_scale + self.phi,
                self._two_pi,
            )

            distances = self._angular_distance(code_angles, observed)
            loss = self._distance_to_loss(distances).sum(dim=1)
            scores[start:end] = loss

        topk = torch.topk(scores, k=min(2, self.num_messages), largest=False)
        best_idx = int(topk.indices[0].item())
        best_score = float(topk.values[0].item())

        if self.num_messages > 1:
            second_score = float(topk.values[1].item())
            gap = max(0.0, second_score - best_score)
        else:
            second_score = best_score
            gap = 0.0

        tie_count = int((scores <= (topk.values[0] + 1e-7)).sum().item())
        pvalue = float((tie_count + 1) / (self.num_messages + 1))

        pred_message = candidate_bits[best_idx].to(torch.int32).tolist()
        return pred_message, best_idx, best_score, second_score, pvalue, gap

    @add_elapsed_time()
    def detect(self, tokens: list[int] | torch.Tensor) -> dict[str, object]:
        def _empty() -> dict[str, object]:
            return {
                "pred_message": [0] * self.payload_size,
                "expected_message": self.payload_bits,
                "pred_message_idx": 0,
                "best_distance": float("inf"),
                "second_distance": float("inf"),
                "distance_gap": 0.0,
                "statistic": 0.0,
                "pvalue": 1.0,
                "n_positions": 0,
                "decode_enabled": self._decode_enabled,
            }

        if isinstance(tokens, list):
            token_tensor = torch.tensor(tokens, device=self.rng_device, dtype=torch.long)
        else:
            token_tensor = tokens.to(self.rng_device, dtype=torch.long)

        if token_tensor.numel() <= self.context_size:
            return _empty()

        context_hashes, observed_tokens = self.seeding_scheme.hash_context(
            token_tensor,
            self.context_size,
        )
        if observed_tokens.numel() == 0:
            return _empty()

        valid_g_columns: list[torch.Tensor] = []
        valid_observed_angles: list[float] = []

        for ctx_hash, token_id in zip(context_hashes.tolist(), observed_tokens.tolist()):
            token_int = int(token_id)
            if token_int < 0 or token_int >= self.vocab_size:
                continue

            side_value, a_mult, b_shift, g_column = self._sample_side_information(
                int(ctx_hash),
                device=self.rng_device,
            )

            token_angle = self._token_angles(
                torch.tensor([token_int], device=self.rng_device, dtype=torch.int64),
                a_mult=a_mult,
                b_shift=b_shift,
            )[0]

            observed_code_angle = torch.remainder(
                token_angle - (float(side_value) * self._side_angle_scale),
                self._two_pi,
            )

            valid_g_columns.append(g_column)
            valid_observed_angles.append(float(observed_code_angle.item()))

        if len(valid_g_columns) == 0:
            return _empty()

        if not self._decode_enabled:
            out = _empty()
            out["n_positions"] = len(valid_g_columns)
            out["decode_reason"] = (
                f"message space too large: 2^{self.payload_size}={self.num_messages} "
                f"> max_decode_messages={self.max_decode_messages}"
            )
            return out

        g_matrix = torch.stack(valid_g_columns, dim=1).to(
            device=self.rng_device,
            dtype=self._decode_matmul_dtype,
        )
        observed_angles = torch.tensor(
            valid_observed_angles,
            device=self.rng_device,
            dtype=self._decode_matmul_dtype,
        )

        (
            pred_message,
            pred_idx,
            best_distance,
            second_distance,
            pvalue,
            gap,
        ) = self._decode_message(g_matrix, observed_angles)

        bit_matches = sum(
            int(int(a) == int(b))
            for a, b in zip(pred_message, self.payload_bits)
        )

        return {
            "pred_message": pred_message,
            "expected_message": self.payload_bits,
            "pred_message_idx": pred_idx,
            "best_distance": best_distance,
            "second_distance": second_distance,
            "distance_gap": gap,
            "bit_accuracy": bit_matches / float(self.payload_size),
            "statistic": gap,
            "pvalue": None,
            "n_positions": len(valid_g_columns),
            "decode_enabled": self._decode_enabled,
        }
