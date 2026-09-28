from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

import torch
from scipy.stats import norm
from vllm import SamplingParams

from lm_wm_tools import WatermarkProtocol
from lm_wm_tools.watermarks.sampling import (
    RandomUtilityType,
    SeedingScheme,
    create_random_utility,
)
from lm_wm_tools.watermarks.sampling.cabs import CABS
from lm_wm_tools.watermarks.utils import (
    add_elapsed_time,
    format_watermark_name,
    normalize_multibit_algorithm,
)


class MirrorMarkWatermark(WatermarkProtocol):
    """
    MirrorMark watermark with CABS scheduling.

    The implementation follows the paper's core structure:
    - context-anchored balanced position assignment (CABS)
    - mod-1 mirroring around message-dependent pivots
    - distortion-free sequential reweighting for embedding
    - CABS replay + per-position symbol decoding for detection
    """

    def __init__(
        self,
        vocab_size: int,
        rng_device: str | torch.device,
        seeding_scheme: SeedingScheme | str,
        context_size: int,
        seed: int,
        sampling_parameters: SamplingParams,
        payload: list[int] | None = None,
        symbol_size: int = 2,
        epsilon: float = 30.0,
        distribution_name: str = "uniform",
        distribution_parameters: dict | None = None,
        multibit_algorithm: RandomUtilityType | str = "cabs",
        multibit_seed: int = 1847389390,
        decoder: str = "gumbel",
        scorer: str | None = None,
        frame_window: int = 4,
        frame_bits: int = 3,
        min_len: int | None = None,
        max_factor: float = 1.5,
        one_bit_symmetric: bool = True,
        top_k: int | None = None,
        **kwargs: Any,
    ) -> None:
        m_from_kwargs = kwargs.pop("m", None)
        if m_from_kwargs is not None:
            symbol_size = int(m_from_kwargs)

        num_layers = kwargs.pop("num_layers", None)
        if num_layers is not None:
            epsilon = float(num_layers)

        w_from_kwargs = kwargs.pop("W", None)
        if w_from_kwargs is not None:
            frame_window = int(w_from_kwargs)

        f_from_kwargs = kwargs.pop("f", None)
        if f_from_kwargs is not None:
            frame_bits = int(f_from_kwargs)

        if isinstance(seeding_scheme, str):
            seeding_scheme = SeedingScheme(seeding_scheme)
        if isinstance(rng_device, str):
            rng_device = torch.device(rng_device)

        if context_size < 1:
            raise ValueError("MirrorMark requires context_size >= 1.")
        if symbol_size < 1:
            raise ValueError("MirrorMark requires symbol_size >= 1.")

        payload = [0] * 36 if payload is None else payload
        payload_bits = [int(bit) for bit in payload]
        if len(payload_bits) == 0:
            raise ValueError("MirrorMark payload must contain at least one bit.")
        if any(bit not in (0, 1) for bit in payload_bits):
            raise ValueError("MirrorMark payload must be binary (0/1).")

        self.vocab_size = int(vocab_size)
        self.rng_device = rng_device
        self.context_size = int(context_size)
        self.seed = int(seed)
        self.temperature = sampling_parameters.temperature
        if self.temperature is None or self.temperature <= 0:
            raise ValueError("MirrorMark requires sampling temperature > 0.")

        if top_k is None:
            sampling_top_k = getattr(sampling_parameters, "top_k", None)
            if sampling_top_k is None or sampling_top_k == -1:
                # Paper default is top-100 sampling.
                self.top_k = min(100, self.vocab_size)
            elif sampling_top_k <= 0:
                raise ValueError("top_k must be a positive integer.")
            else:
                self.top_k = min(int(sampling_top_k), self.vocab_size)
        else:
            if top_k <= 0:
                raise ValueError("top_k must be a positive integer.")
            self.top_k = min(int(top_k), self.vocab_size)

        self.depth = int(round(float(epsilon)))
        if self.depth < 1:
            raise ValueError("MirrorMark requires epsilon/num_layers >= 1.")
        self._depth_seed_cache = tuple(self.seed + 10007 * i for i in range(self.depth))

        self.payload_size = len(payload_bits)
        self.payload = torch.tensor(payload_bits, device=self.rng_device, dtype=torch.int32)
        self.symbol_size = int(symbol_size)
        self.symbol_cardinality = 1 << self.symbol_size
        self.sigma = 1.0

        algorithm_name = (
            multibit_algorithm.value
            if isinstance(multibit_algorithm, RandomUtilityType)
            else str(multibit_algorithm)
        )
        self.multibit_algorithm = normalize_multibit_algorithm(algorithm_name)
        if self.multibit_algorithm != RandomUtilityType.CABS.value:
            raise ValueError("MirrorMark only supports multibit_algorithm='cabs'.")

        self.decoder = str(decoder).strip().lower()
        if self.decoder not in {"gumbel", "wmean"}:
            raise ValueError("decoder must be one of: {'gumbel', 'wmean'}.")
        self.scorer = self.decoder if scorer is None else str(scorer).strip().lower()
        if self.scorer not in {"gumbel", "wmean"}:
            raise ValueError("scorer must be one of: {'gumbel', 'wmean'}.")
        self.n_mc = int(kwargs.pop("n_mc", 2000))
        self.mc_batch_size = int(kwargs.pop("mc_batch_size", 256))
        if self.n_mc < 0:
            raise ValueError("n_mc must be >= 0.")
        if self.mc_batch_size < 1:
            raise ValueError("mc_batch_size must be >= 1.")

        # CABS + mirroring random generator.
        if distribution_parameters is None:
            distribution_parameters = {"low": 0.0, "high": 1.0}
        else:
            distribution_parameters = dict(distribution_parameters)

        self.random_generator = create_random_utility(
            self.multibit_algorithm,
            distribution_name,
            distribution_parameters,
            payload=payload_bits,
            multibit_seed=multibit_seed,
            symbol_size=self.symbol_size,
            context_size=self.context_size,
            frame_window=frame_window,
            frame_bits=frame_bits,
            min_len=min_len,
            max_factor=max_factor,
            one_bit_symmetric=one_bit_symmetric,
            **kwargs,
        )
        if not isinstance(self.random_generator, CABS):
            raise TypeError(
                "MirrorMark requires CABS random generator; "
                f"got {type(self.random_generator).__name__}."
            )

        self.frame_window = int(self.random_generator.frame_window)
        self.frame_bits = int(self.random_generator.frame_bits)
        self.max_factor = float(self.random_generator.max_factor)
        self.min_len = int(self.random_generator.min_len)
        self.max_len = int(self.random_generator.max_len)
        self.num_positions = int(self.random_generator.num_positions)

        seeding_scheme.initialize(self.vocab_size, self.seed, self.rng_device)
        self.seeding_scheme = seeding_scheme

        # Weighted-mean decoder/scorer default from SynthID-style depth weighting.
        weights = torch.linspace(
            start=10.0, end=1.0, steps=self.depth, device=self.rng_device
        )
        self.layer_weights = weights / weights.sum()

    def get_name(self) -> str:
        return format_watermark_name(
            "MirrorMark",
            self.multibit_algorithm,
            (
                f"{self.seeding_scheme.value}/k_{self.context_size}/m_{self.symbol_size}/"
                f"layers_{self.depth}/f_{self.frame_bits}/W_{self.frame_window}/"
                f"mf_{self.max_factor:.2f}/seed_{self.seed}/topk_{self.top_k}"
            ),
            payload_size=self.payload_size,
        )

    def _sample_all_depths(
        self,
        context_hash: int,
        tokens: torch.Tensor,
        *,
        position: int,
        embed_message: bool,
    ) -> torch.Tensor:
        context_hashes = torch.tensor(
            [context_hash], device=tokens.device, dtype=torch.int32
        )
        position_tensor = torch.tensor([position], device=tokens.device, dtype=torch.long)

        depth_values = []
        for depth_seed in self._depth_seed_cache:
            samples = self.random_generator.sample(
                seed=depth_seed,
                context_hashes=context_hashes,
                tokens=tokens,
                embed_message=embed_message,
                positions=position_tensor,
            )
            if samples.dim() == 2 and samples.shape[0] == 1:
                samples = samples.squeeze(0)
            depth_values.append(samples.to(torch.float32))

        stacked = torch.stack(depth_values, dim=0)  # (depth, top_k)
        return stacked.transpose(0, 1).contiguous()  # (top_k, depth)

    @torch.no_grad()
    def __call__(self, output_ids: list[int], logits: torch.Tensor) -> torch.Tensor:
        squeeze_batch = False
        if logits.dim() == 2 and logits.size(0) == 1:
            logits = logits.squeeze(0)
            squeeze_batch = True
        elif logits.dim() != 1:
            raise ValueError(
                f"MirrorMark expects 1-D logits (or [1, V]), got shape {tuple(logits.shape)}."
            )

        if len(output_ids) < self.context_size:
            return logits.unsqueeze(0) if squeeze_batch else logits

        context_hash = self.seeding_scheme.hash_last_context(
            torch.tensor(output_ids, device=self.rng_device, dtype=torch.long),
            self.context_size,
        )
        if context_hash is None:
            return logits.unsqueeze(0) if squeeze_batch else logits

        position = self.random_generator.assign_position(output_ids, int(context_hash))
        if position is None:
            return logits.unsqueeze(0) if squeeze_batch else logits

        logits_scaled = logits / self.temperature
        k = min(self.top_k, logits.shape[-1])
        topk_logits, topk_indices = torch.topk(logits_scaled, k, dim=-1)
        probs = torch.softmax(topk_logits, dim=-1).to(torch.float32)

        mirrored_scores = self._sample_all_depths(
            int(context_hash),
            topk_indices,
            position=position,
            embed_message=True,
        )

        for depth_idx in range(self.depth):
            layer_scores = mirrored_scores[:, depth_idx].to(dtype=probs.dtype)
            mu = -torch.sum(probs * layer_scores)
            probs = probs * (1.0 + self.sigma * (layer_scores + mu))
            probs = torch.clamp(probs, min=0.0)

            probs_sum = probs.sum()
            if (not torch.isfinite(probs_sum)) or probs_sum <= 0:
                probs = torch.softmax(topk_logits, dim=-1).to(torch.float32)
                break
            probs = probs / probs_sum

        updated_log_probs = torch.log(probs.clamp_min(1e-30))
        original_log_probs = torch.log_softmax(topk_logits, dim=-1)
        deltas = updated_log_probs - original_log_probs

        logits_scaled = logits_scaled.clone()
        logits_scaled[topk_indices] = logits_scaled[topk_indices] + deltas.to(
            dtype=logits_scaled.dtype
        )
        updated_logits = logits_scaled * self.temperature

        if squeeze_batch:
            return updated_logits.unsqueeze(0)
        return updated_logits

    def get_expected_probs(self, probs: torch.Tensor) -> torch.Tensor:
        return probs

    def _mirror_values(
        self, values: torch.Tensor, symbols: torch.Tensor | int
    ) -> torch.Tensor:
        """
        Mirror values around message-dependent pivots.

        `values` has shape [N, D] or [B, K, D]. The first dimension is the batch
        dimension when `symbols` is a tensor.
        """
        values = values.to(torch.float32)
        first_dim = values.size(0)

        if isinstance(symbols, int):
            symbol_tensor = torch.full(
                (first_dim,),
                int(symbols),
                device=values.device,
                dtype=torch.long,
            )
        else:
            symbol_tensor = symbols.to(values.device, dtype=torch.long).reshape(-1)
            if symbol_tensor.numel() == 1 and first_dim > 1:
                symbol_tensor = symbol_tensor.expand(first_dim)
            if symbol_tensor.numel() != first_dim:
                raise ValueError(
                    "symbols must match the first dimension of values. "
                    f"Got {symbol_tensor.numel()} for {first_dim}."
                )

        symbol_broadcast = symbol_tensor.to(values.dtype)
        while symbol_broadcast.dim() < values.dim():
            symbol_broadcast = symbol_broadcast.unsqueeze(-1)

        if self.symbol_size == 1 and self.random_generator.one_bit_symmetric:
            return torch.where(symbol_broadcast == 0, 1.0 - values, values)

        psi = symbol_broadcast / float(1 << (self.symbol_size + 1))
        return torch.remainder(2.0 * psi - values, 1.0)

    def _decode_symbol(self, values: torch.Tensor) -> int:
        # values shape: (n_tokens_for_position, depth)
        if values.numel() == 0:
            return 0

        scores = torch.empty(self.symbol_cardinality, device=values.device, dtype=torch.float32)
        for symbol in range(self.symbol_cardinality):
            mirrored = self._mirror_values(values, symbol)
            if self.decoder == "gumbel":
                score = -torch.log1p(-mirrored.clamp(max=1.0 - 1e-7)).sum()
            else:
                weighted = mirrored * self.layer_weights.view(1, -1)
                score = weighted.sum(dim=1).mean()
            scores[symbol] = score.to(torch.float32)

        return int(torch.argmax(scores).item())

    def _symbols_to_bits(self, symbols: list[int]) -> list[int]:
        return self.random_generator.symbols_to_bits(symbols)

    def _analytic_global_test(
        self, mirrored_values: torch.Tensor
    ) -> tuple[float, float, float, int]:
        if mirrored_values.numel() == 0:
            return 0.0, 0.0, 1.0, 0

        n_trials = int(mirrored_values.numel())

        if self.scorer == "gumbel":
            transformed = -torch.log1p(-mirrored_values.clamp(max=1.0 - 1e-7))
            statistic = float(transformed.sum().item())
            z_score = float((statistic - n_trials) / math.sqrt(float(n_trials)))
            pvalue = float(norm.sf(z_score))
            return statistic, z_score, pvalue, n_trials

        # WeightedMeanScore-like global aggregation.
        mean_value = float(mirrored_values.mean().item())
        std_error = math.sqrt((1.0 / 12.0) / float(n_trials))
        z_score = float((mean_value - 0.5) / std_error)
        pvalue = float(norm.sf(z_score))
        return mean_value, z_score, pvalue, n_trials

    def _mc_pvalue(
        self,
        observed_statistic: float,
        position_counts: list[int],
    ) -> float:
        """
        Monte-Carlo calibration under H0 with full decode+score replay.

        This mirrors the MPAC/BinoEncoder philosophy: simulate null draws and
        measure how often the resulting statistic is at least as large as the
        observed one.
        """
        if self.n_mc <= 0:
            return 1.0

        active_counts = [count for count in position_counts if count > 0]
        if not active_counts:
            return 1.0

        total_values = int(sum(active_counts) * self.depth)
        if total_values <= 0:
            return 1.0

        ge = 0
        total = 0
        batch_size = max(1, self.mc_batch_size)
        layer_weights = self.layer_weights.view(1, 1, -1)

        for start in range(0, self.n_mc, batch_size):
            bsz = min(batch_size, self.n_mc - start)
            if self.scorer == "gumbel":
                stats_batch = torch.zeros(bsz, device=self.rng_device, dtype=torch.float32)
            else:
                score_sum_batch = torch.zeros(
                    bsz, device=self.rng_device, dtype=torch.float32
                )

            for count in active_counts:
                values = torch.rand(
                    (bsz, count, self.depth),
                    device=self.rng_device,
                    dtype=torch.float32,
                )

                decoder_scores = []
                scorer_values = []
                for symbol in range(self.symbol_cardinality):
                    mirrored = self._mirror_values(values, symbol)

                    if self.decoder == "gumbel":
                        dec_score = -torch.log1p(-mirrored.clamp(max=1.0 - 1e-7)).sum(
                            dim=(1, 2)
                        )
                    else:
                        dec_score = (mirrored * layer_weights).sum(dim=2).mean(dim=1)
                    decoder_scores.append(dec_score)

                    if self.scorer == "gumbel":
                        scorer_metric = -torch.log1p(
                            -mirrored.clamp(max=1.0 - 1e-7)
                        ).sum(dim=(1, 2))
                    else:
                        scorer_metric = mirrored.sum(dim=(1, 2))
                    scorer_values.append(scorer_metric)

                decoder_scores_t = torch.stack(decoder_scores, dim=1)  # (bsz, S)
                decoded_symbol = torch.argmax(decoder_scores_t, dim=1)  # (bsz,)

                scorer_values_t = torch.stack(scorer_values, dim=1)  # (bsz, S)
                chosen_scores = scorer_values_t.gather(
                    1, decoded_symbol.unsqueeze(1)
                ).squeeze(1)

                if self.scorer == "gumbel":
                    stats_batch = stats_batch + chosen_scores
                else:
                    score_sum_batch = score_sum_batch + chosen_scores

            if self.scorer == "wmean":
                stats_batch = score_sum_batch / float(total_values)

            ge += int((stats_batch >= float(observed_statistic)).sum().item())
            total += int(bsz)

        return float((ge + 1.0) / (float(total) + 1.0))

    def _empty_detection_output(self) -> dict[str, object]:
        expected_bits = self.payload.detach().cpu().tolist()
        expected_symbols = self.random_generator.message_symbols.detach().cpu().tolist()
        return {
            "statistic": 0.0,
            "pvalue": 1.0,
            "pvalue_original": 1.0,
            "z_score": 0.0,
            "pred_message": [0] * self.payload_size,
            "expected_message": expected_bits,
            "pred_symbols": [0] * self.num_positions,
            "expected_symbols": expected_symbols,
            "bit_accuracy": 0.0,
            "p_values_per_bit": [1.0] * self.payload_size,
            "n_trials": 0,
        }

    @add_elapsed_time()
    def detect(self, tokens: list[int] | torch.Tensor) -> dict[str, object]:
        if isinstance(tokens, list):
            tokens = torch.tensor(tokens, device=self.rng_device, dtype=torch.long)
        else:
            tokens = tokens.to(self.rng_device, dtype=torch.long)

        if tokens.numel() <= self.context_size:
            return self._empty_detection_output()

        self.random_generator.reset_scheduler_state()

        per_position: dict[int, list[torch.Tensor]] = defaultdict(list)
        token_records: list[tuple[int, torch.Tensor]] = []

        for t in range(self.context_size, int(tokens.numel())):
            prefix = tokens[:t]
            context_hash = self.seeding_scheme.hash_last_context(prefix, self.context_size)
            if context_hash is None:
                continue

            position = self.random_generator.assign_position(prefix, int(context_hash))
            if position is None:
                continue

            token_id = int(tokens[t].item())
            if token_id < 0 or token_id >= self.vocab_size:
                continue

            token_tensor = torch.tensor([token_id], device=self.rng_device, dtype=torch.long)
            context_hashes = torch.tensor(
                [int(context_hash)], device=self.rng_device, dtype=torch.int32
            )

            depth_draws = []
            for depth_seed in self._depth_seed_cache:
                draw = self.random_generator.sample(
                    seed=depth_seed,
                    context_hashes=context_hashes,
                    tokens=token_tensor,
                    embed_message=False,
                )
                depth_draws.append(draw.reshape(-1)[0].to(torch.float32))
            raw_values = torch.stack(depth_draws, dim=0)  # (depth,)

            per_position[position].append(raw_values)
            token_records.append((position, raw_values))

        if not token_records:
            return self._empty_detection_output()

        pred_symbols = [0] * self.num_positions
        for position in range(self.num_positions):
            values = per_position.get(position)
            if not values:
                pred_symbols[position] = 0
                continue
            position_tensor = torch.stack(values, dim=0)  # (K, depth)
            pred_symbols[position] = self._decode_symbol(position_tensor)

        position_counts = [len(per_position.get(position, [])) for position in range(self.num_positions)]
        mirrored_all = []
        for position, raw_values in token_records:
            mirrored = self._mirror_values(raw_values.unsqueeze(0), pred_symbols[position]).squeeze(0)
            mirrored_all.append(mirrored)

        mirrored_values = torch.stack(mirrored_all, dim=0).reshape(-1)
        statistic, z_score, pvalue_original, n_trials = self._analytic_global_test(
            mirrored_values
        )
        pvalue = self._mc_pvalue(
            observed_statistic=statistic,
            position_counts=position_counts,
        )

        pred_message = self._symbols_to_bits(pred_symbols)
        expected_message = self.payload.detach().cpu().tolist()
        bit_accuracy = float(
            sum(int(a == b) for a, b in zip(pred_message, expected_message))
            / max(1, len(expected_message))
        )

        # MirrorMark decodes per-position symbols; compute a local p-value for each
        # decoded position and map it to its constituent payload bits.
        p_values_per_position = [1.0] * self.num_positions
        for position in range(self.num_positions):
            values = per_position.get(position)
            if not values:
                continue
            position_tensor = torch.stack(values, dim=0)  # (K, depth)
            mirrored_position = self._mirror_values(
                position_tensor, pred_symbols[position]
            ).reshape(-1)
            position_statistic, _, analytic_pvalue, _ = self._analytic_global_test(
                mirrored_position
            )
            # Use MC calibration under decoded-symbol selection to keep
            # per-position p-values statistically aligned with the global test.
            if self.n_mc > 0:
                position_pvalue = self._mc_pvalue(
                    observed_statistic=position_statistic,
                    position_counts=[int(position_tensor.size(0))],
                )
            else:
                position_pvalue = float(analytic_pvalue)
            p_values_per_position[position] = float(position_pvalue)

        p_values_per_bit = [
            p_values_per_position[bit_idx // self.symbol_size]
            for bit_idx in range(self.payload_size)
        ]

        return {
            "statistic": statistic,
            "pvalue": pvalue,
            "pvalue_original": pvalue_original,
            "z_score": z_score,
            "pred_message": pred_message,
            "expected_message": expected_message,
            "pred_symbols": pred_symbols,
            "expected_symbols": self.random_generator.message_symbols.detach()
            .cpu()
            .tolist(),
            "bit_accuracy": bit_accuracy,
            "p_values_per_bit": p_values_per_bit,
            "n_trials": n_trials,
        }
