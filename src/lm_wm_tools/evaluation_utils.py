import math

import numpy as np
import pandas as pd
import polars as pl


class EvaluationMetrics:
    CSS_ALPHA = 0.99
    BA_AT_1PCT_FPR_ALPHA = 0.01
    CI_Z = 1.959963984540054

    def __init__(
        self,
        pred_col: str = "pred_message",
        expected_col: str = "expected_message",
        strength_column: str = "epsilon",
        n_bootstrap: int = 1000,
        bootstrap_seed: int = 0,
    ):
        self.pred_col = pred_col
        self.expected_col = expected_col
        self.strength_column = strength_column
        self.n_bootstrap = n_bootstrap
        self.bootstrap_seed = bootstrap_seed

    def ber_expr(
        self,
        hamming_total_col: str = "__hamming_total",
        n_bits_total_col: str = "__n_bits_total",
    ) -> pl.Expr:
        return pl.col(hamming_total_col) / pl.col(n_bits_total_col)

    def mer_expr(self, hamming_col: str = "__hamming") -> pl.Expr:
        return (pl.col(hamming_col) == 0).cast(pl.Float64).mean()

    def mutual_information_expr(
        self,
        payload_size_col: str = "payload_size",
        mi_sum_col: str = "__mi_sum",
    ) -> pl.Expr:
        return pl.col(payload_size_col) * math.log(2.0) + pl.col(mi_sum_col)

    @classmethod
    def _normal_proportion_ci(
        cls,
        proportion: float | int | None,
        total: float | int | None,
    ) -> tuple[float | None, float | None]:
        if proportion is None or total is None:
            return (None, None)
        try:
            p = float(proportion)
            n = float(total)
        except (TypeError, ValueError):
            return (None, None)
        if not math.isfinite(p) or not math.isfinite(n) or n <= 0.0:
            return (None, None)

        p = min(max(p, 0.0), 1.0)
        delta = cls.CI_Z * math.sqrt(max(p * (1.0 - p), 0.0) / n)
        return (max(0.0, p - delta), min(1.0, p + delta))

    @classmethod
    def _normal_mean_ci(
        cls,
        values: np.ndarray,
        *,
        lower_bound: float | None = None,
    ) -> tuple[float | None, float | None]:
        values = values[np.isfinite(values)]
        if len(values) == 0:
            return (None, None)
        mean = float(np.mean(values))
        if len(values) == 1:
            lower = upper = mean
        else:
            delta = cls.CI_Z * float(np.std(values, ddof=1)) / math.sqrt(len(values))
            lower = mean - delta
            upper = mean + delta

        if lower_bound is not None:
            lower = max(lower_bound, lower)
        return (lower, upper)

    @staticmethod
    def _mi_from_hamming_counts(
        counts: np.ndarray,
        *,
        payload_size: int,
    ) -> np.ndarray:
        totals = counts.sum(axis=1, keepdims=True)
        probabilities = np.divide(
            counts,
            totals,
            out=np.zeros_like(counts, dtype=float),
            where=totals > 0,
        )
        hamming_values = np.arange(counts.shape[1])
        log_binom = np.array(
            [
                math.lgamma(payload_size + 1)
                - math.lgamma(int(hamming) + 1)
                - math.lgamma(payload_size - int(hamming) + 1)
                for hamming in hamming_values
            ],
            dtype=float,
        )
        log_probabilities = np.zeros_like(probabilities, dtype=float)
        np.log(
            probabilities,
            out=log_probabilities,
            where=probabilities > 0,
        )
        terms = np.where(
            probabilities > 0,
            probabilities * (log_probabilities - log_binom),
            0.0,
        )
        return payload_size * math.log(2.0) + terms.sum(axis=1)

    def _bootstrap_mi_ci(
        self,
        hamming: np.ndarray,
        *,
        payload_size: int,
        rng: np.random.Generator,
    ) -> tuple[float | None, float | None]:
        if len(hamming) == 0 or payload_size <= 0:
            return (None, None)

        counts = np.bincount(hamming.astype(int), minlength=payload_size + 1).astype(float)
        n_messages = int(counts.sum())
        if n_messages <= 1 or self.n_bootstrap <= 0:
            mi_nats = float(
                self._mi_from_hamming_counts(
                    counts.reshape(1, -1),
                    payload_size=payload_size,
                )[0]
            )
            return (max(0.0, mi_nats), min(payload_size * math.log(2.0), mi_nats))

        probabilities = counts / n_messages
        bootstrap_counts = rng.multinomial(
            n_messages,
            probabilities,
            size=self.n_bootstrap,
        )
        mi_samples = self._mi_from_hamming_counts(
            bootstrap_counts,
            payload_size=payload_size,
        )
        lower, upper = np.quantile(mi_samples, [0.025, 0.975])
        return (
            max(0.0, float(lower)),
            min(payload_size * math.log(2.0), float(upper)),
        )

    def _add_confidence_intervals(
        self,
        output: pd.DataFrame,
        per_message: pl.DataFrame,
        group_cols: list[str],
    ) -> pd.DataFrame:
        if output.empty or per_message.height == 0:
            return output

        required_cols = [
            *group_cols,
            "__hamming",
            "__n_bits",
            "tpr",
            "__css_at_99",
            "__ba_at_1pct_fpr_hits",
            "__ba_at_1pct_fpr_total",
        ]
        per_message_pd = per_message.select(required_cols).to_pandas()
        rng = np.random.default_rng(self.bootstrap_seed)
        interval_rows = []

        group_key = group_cols[0] if len(group_cols) == 1 else group_cols
        for key, group in per_message_pd.groupby(group_key, dropna=False, sort=False):
            if not isinstance(key, tuple):
                key = (key,)

            row = dict(zip(group_cols, key))
            hamming = group["__hamming"].to_numpy(dtype=int)
            n_bits = group["__n_bits"].to_numpy(dtype=int)
            n_messages = len(group)
            n_bits_total = int(n_bits.sum())
            payload_size = int(n_bits[0]) if len(n_bits) else 0

            bit_accuracy = (
                1.0 - (float(hamming.sum()) / n_bits_total)
                if n_bits_total > 0
                else None
            )
            bit_low, bit_high = self._normal_proportion_ci(
                bit_accuracy,
                n_bits_total,
            )
            row["bit_accuracy_ci_low"] = bit_low
            row["bit_accuracy_ci_high"] = bit_high
            row["ber_ci_low"] = None if bit_high is None else 1.0 - bit_high
            row["ber_ci_high"] = None if bit_low is None else 1.0 - bit_low

            message_accuracy = float(np.mean(hamming == 0)) if n_messages else None
            mer_low, mer_high = self._normal_proportion_ci(
                message_accuracy,
                n_messages,
            )
            row["mer_ci_low"] = mer_low
            row["mer_ci_high"] = mer_high
            row["message_accuracy_ci_low"] = mer_low
            row["message_accuracy_ci_high"] = mer_high
            row["message_error_rate_ci_low"] = (
                None if mer_high is None else 1.0 - mer_high
            )
            row["message_error_rate_ci_high"] = (
                None if mer_low is None else 1.0 - mer_low
            )

            tpr_values = group["tpr"].to_numpy(dtype=float)
            tpr_mean = (
                float(np.mean(tpr_values[np.isfinite(tpr_values)]))
                if np.isfinite(tpr_values).any()
                else None
            )
            tpr_low, tpr_high = self._normal_proportion_ci(tpr_mean, n_messages)
            row["tpr_ci_low"] = tpr_low
            row["tpr_ci_high"] = tpr_high

            ba_hits = group["__ba_at_1pct_fpr_hits"].dropna().to_numpy(dtype=float)
            ba_total = group["__ba_at_1pct_fpr_total"].dropna().to_numpy(dtype=float)
            ba_hits_total = float(ba_hits.sum()) if len(ba_hits) else 0.0
            ba_total_bits = float(ba_total.sum()) if len(ba_total) else 0.0
            ba_value = ba_hits_total / ba_total_bits if ba_total_bits > 0.0 else None
            ba_low, ba_high = self._normal_proportion_ci(ba_value, ba_total_bits)
            row["ba_at_1pct_fpr_ci_low"] = ba_low
            row["ba_at_1pct_fpr_ci_high"] = ba_high

            css_values = group["__css_at_99"].dropna().to_numpy(dtype=float)
            css_low, css_high = self._normal_mean_ci(css_values, lower_bound=0.0)
            row["css_at_99_ci_low"] = css_low
            row["css_at_99_ci_high"] = css_high

            mi_low, mi_high = self._bootstrap_mi_ci(
                hamming,
                payload_size=payload_size,
                rng=rng,
            )
            row["mi_nats_ci_low"] = mi_low
            row["mi_nats_ci_high"] = mi_high
            if mi_low is None or mi_high is None:
                row["mi_bits_ci_low"] = None
                row["mi_bits_ci_high"] = None
                row["normalized_mi_ci_low"] = None
                row["normalized_mi_ci_high"] = None
            else:
                row["mi_bits_ci_low"] = mi_low / math.log(2.0)
                row["mi_bits_ci_high"] = mi_high / math.log(2.0)
                normalizer = payload_size * math.log(2.0)
                row["normalized_mi_ci_low"] = max(0.0, mi_low / normalizer)
                row["normalized_mi_ci_high"] = min(1.0, mi_high / normalizer)

            interval_rows.append(row)

        if not interval_rows:
            return output
        intervals = pd.DataFrame(interval_rows)
        return output.merge(intervals, on=group_cols, how="left")

    def _mutual_information_term_expr(
        self,
        p_k_col: str = "__p_k",
        payload_size_col: str = "payload_size",
        hamming_col: str = "__hamming",
    ) -> pl.Expr:
        log_binom_expr = pl.struct([payload_size_col, hamming_col]).map_elements(
            lambda row: math.lgamma(int(row[payload_size_col]) + 1)
            - math.lgamma(int(row[hamming_col]) + 1)
            - math.lgamma(int(row[payload_size_col] - row[hamming_col]) + 1),
            return_dtype=pl.Float64,
        )
        return pl.col(p_k_col) * (pl.col(p_k_col).log() - log_binom_expr)

    def _normalize_quality_metric(self, quality_metric: str | None) -> str | None:
        if quality_metric is None:
            return None
        if not isinstance(quality_metric, str):
            raise TypeError("quality_metric must be a string or None")

        quality_metric = quality_metric.strip()
        return quality_metric or None

    @staticmethod
    def _css_upper_bound_from_p_values(
        p_values_per_bit, alpha: float = CSS_ALPHA
    ) -> float | None:
        if p_values_per_bit is None:
            return None
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be in (0, 1)")

        try:
            p_values = [float(value) for value in p_values_per_bit]
        except (TypeError, ValueError):
            return None

        if not p_values:
            return 1.0
        if any(math.isnan(value) for value in p_values):
            return None

        budget = 1.0 - alpha
        running = 0.0
        fixed_bits = 0

        # Keep the most confident bits fixed while the union-bound failure
        # budget stays below 1 - alpha. Every remaining bit doubles the set size.
        for p_value in sorted(min(max(value, 0.0), 1.0) for value in p_values):
            if running + p_value <= budget + 1e-12:
                running += p_value
                fixed_bits += 1
            else:
                break

        return float(2 ** (len(p_values) - fixed_bits))

    @staticmethod
    def _bit_accuracy_at_pvalue_threshold_stats(
        pred_bits,
        expected_bits,
        p_values_per_bit,
        alpha: float,
    ) -> dict[str, int | None]:
        if pred_bits is None or expected_bits is None or p_values_per_bit is None:
            return {"hits": None, "total": None}
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be in (0, 1)")

        try:
            pred = [int(bit) for bit in pred_bits]
            expected = [int(bit) for bit in expected_bits]
            p_values = [float(value) for value in p_values_per_bit]
        except (TypeError, ValueError):
            return {"hits": None, "total": None}

        if len(pred) != len(expected) or len(pred) != len(p_values):
            return {"hits": None, "total": None}
        if any(math.isnan(value) for value in p_values):
            return {"hits": None, "total": None}

        hits = sum(
            int(
                pred_bit == expected_bit
                and min(max(p_value, 0.0), 1.0) < alpha
            )
            for pred_bit, expected_bit, p_value in zip(pred, expected, p_values)
        )
        return {"hits": hits, "total": len(pred)}

    def calculate_metrics(
        self,
        df: pl.DataFrame,
        group_by: list[str],
        quality_metric: str | None = None,
    ) -> pd.DataFrame:
        """Calculate grouped BER/BA/MER/MI/CSS with Polars and export the final result as pandas."""
        if not isinstance(df, pl.DataFrame):
            raise TypeError("df must be a polars.DataFrame")

        quality_metric = self._normalize_quality_metric(quality_metric)
        has_p_values_per_bit = "p_values_per_bit" in df.columns

        group_cols = list(group_by)
        if self.strength_column not in group_cols:
            group_cols.append(self.strength_column)

        required_columns = {self.pred_col, self.expected_col, "tpr", *group_cols}
        if quality_metric:
            required_columns.add(quality_metric)
        missing_cols = sorted(required_columns.difference(df.columns))
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        selected_cols = [*group_cols, self.pred_col, self.expected_col, "tpr"]
        if has_p_values_per_bit:
            selected_cols.append("p_values_per_bit")
        if quality_metric:
            selected_cols.append(quality_metric)

        payload_size_in_group = "payload_size" in group_cols
        payload_size_col = "payload_size" if not payload_size_in_group else "__payload_size_calc"

        data = df.select(selected_cols).with_row_index("__row_id")
        if has_p_values_per_bit:
            data = data.with_columns(
                pl.col("p_values_per_bit")
                .map_elements(
                    lambda values: self._css_upper_bound_from_p_values(
                        values, alpha=self.CSS_ALPHA
                    ),
                    return_dtype=pl.Float64,
                )
                .alias("__css_at_99"),
                pl.struct([self.pred_col, self.expected_col, "p_values_per_bit"])
                .map_elements(
                    lambda row: self._bit_accuracy_at_pvalue_threshold_stats(
                        pred_bits=row[self.pred_col],
                        expected_bits=row[self.expected_col],
                        p_values_per_bit=row["p_values_per_bit"],
                        alpha=self.BA_AT_1PCT_FPR_ALPHA,
                    ),
                    return_dtype=pl.Struct(
                        [
                            pl.Field("hits", pl.Int64),
                            pl.Field("total", pl.Int64),
                        ]
                    ),
                )
                .alias("__ba_at_1pct_fpr_stats"),
            )
            data = data.with_columns(
                pl.col("__ba_at_1pct_fpr_stats")
                .struct.field("hits")
                .alias("__ba_at_1pct_fpr_hits"),
                pl.col("__ba_at_1pct_fpr_stats")
                .struct.field("total")
                .alias("__ba_at_1pct_fpr_total"),
            ).drop("__ba_at_1pct_fpr_stats")
        else:
            data = data.with_columns(
                pl.lit(None).cast(pl.Float64).alias("__css_at_99"),
                pl.lit(None).cast(pl.Int64).alias("__ba_at_1pct_fpr_hits"),
                pl.lit(None).cast(pl.Int64).alias("__ba_at_1pct_fpr_total"),
            )

        valid = data.filter(
            pl.col(self.pred_col).is_not_null() & pl.col(self.expected_col).is_not_null()
        )

        per_message = (
            valid
            .explode([self.pred_col, self.expected_col])
            .with_columns(
                (pl.col(self.pred_col) != pl.col(self.expected_col))
                .cast(pl.Int64)
                .alias("__bit_error")
            )
            .group_by(["__row_id", *group_cols], maintain_order=True)
            .agg(
                pl.sum("__bit_error").alias("__hamming"),
                pl.len().alias("__n_bits"),
                pl.first("tpr").alias("tpr"),
                pl.first("__css_at_99").alias("__css_at_99"),
                pl.first("__ba_at_1pct_fpr_hits").alias("__ba_at_1pct_fpr_hits"),
                pl.first("__ba_at_1pct_fpr_total").alias("__ba_at_1pct_fpr_total"),
            )
        )

        grouped = (
            per_message
            .group_by(group_cols, maintain_order=True)
            .agg(
                pl.len().alias("n_messages"),
                pl.first("__n_bits").alias(payload_size_col),
                pl.sum("__hamming").alias("__hamming_total"),
                pl.sum("__n_bits").alias("__n_bits_total"),
                self.mer_expr("__hamming").alias("mer"),
                pl.mean("__css_at_99").alias("css_at_99"),
                pl.sum("__ba_at_1pct_fpr_hits").alias("__ba_at_1pct_fpr_hits_total"),
                pl.sum("__ba_at_1pct_fpr_total").alias("__ba_at_1pct_fpr_total_bits"),
                pl.mean("tpr").alias("tpr")
            )
            .with_columns(
                self.ber_expr("__hamming_total", "__n_bits_total").alias("ber"),
                pl.col("mer").alias("message_accuracy"),
                (1.0 - pl.col("mer")).alias("message_error_rate"),
                pl.when(pl.col("__ba_at_1pct_fpr_total_bits") > 0)
                .then(
                    pl.col("__ba_at_1pct_fpr_hits_total")
                    / pl.col("__ba_at_1pct_fpr_total_bits")
                )
                .otherwise(None)
                .alias("ba_at_1pct_fpr"),
            )
            .with_columns((1.0 - pl.col("ber")).alias("bit_accuracy"))
        )

        hamming_hist = (
            per_message
            .group_by([*group_cols, "__hamming"], maintain_order=True)
            .agg(pl.len().alias("__count_k"))
        )

        mi_terms = (
            hamming_hist
            .join(
                grouped.select([*group_cols, "n_messages", payload_size_col]),
                on=group_cols,
                how="left",
            )
            .with_columns((pl.col("__count_k") / pl.col("n_messages")).alias("__p_k"))
            .with_columns(
                self._mutual_information_term_expr(
                    p_k_col="__p_k",
                    payload_size_col=payload_size_col,
                    hamming_col="__hamming",
                ).alias("__mi_term"),
            )
            .group_by(group_cols, maintain_order=True)
            .agg(pl.sum("__mi_term").alias("__mi_sum"))
        )

        result = (
            grouped
            .join(mi_terms, on=group_cols, how="left")
            .with_columns(self.mutual_information_expr(payload_size_col, "__mi_sum").alias("mi_nats"))
            .with_columns((pl.col("mi_nats") / math.log(2.0)).alias("mi_bits"))
        )

        result = result.with_columns(
            (pl.col("mi_nats") / (pl.col(payload_size_col) * math.log(2.0)) ).alias("normalized_mi")   
        )

        if payload_size_in_group:
            result = result.drop("__payload_size_calc")

        if quality_metric:
            quality = (
                data
                .group_by(group_cols, maintain_order=True)
                .agg(
                    pl.col(quality_metric)
                    .cast(pl.Float64, strict=False)
                    .mean()
                    .alias(quality_metric)
                )
            )
            result = result.join(quality, on=group_cols, how="left")

        output_cols = [
            *group_cols,
            "n_messages",
            *([] if payload_size_in_group else ["payload_size"]),
            "ber",
            "bit_accuracy",
            "ba_at_1pct_fpr",
            "mer",
            "message_accuracy",
            "message_error_rate",
            "css_at_99",
            "mi_nats",
            "mi_bits",
            "normalized_mi",
            "tpr",
            *([quality_metric] if quality_metric else []),
        ]
        result = result.select(output_cols).sort(group_cols)

        output = result.to_pandas()
        if "css_at_99" in output.columns:
            css_at_99 = output["css_at_99"]
            if css_at_99.isna().any():
                output["css_at_99"] = css_at_99.astype(object).where(
                    pd.notna(css_at_99), None
                )

        output = self._add_confidence_intervals(output, per_message, group_cols)
        return output
