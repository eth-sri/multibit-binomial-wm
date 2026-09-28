#!/usr/bin/env python
"""Run generate_vllm.py with one canonical config per watermark type."""

from __future__ import annotations

from typing import Final

import run as sweep_run

# One canonical epsilon per supported watermark type.
# The selected epsilon must be present in that type's sweep definition in scripts/run.py.
CANONICAL_EPSILONS: Final[dict[str, float]] = {
    "RSBHWatermark": 1.0,
    "StealthInk": 0.0,
    "KGW-mpac": 1.0,
    "BiMark": 1.0,
    "M2Mark": 10.0,
    "PPLMark-dynobino": 0.0,
    "PPLMark-bino": 0.0,
    "AAR": 0.0,
    "MirrorMark": 30.0,
    "ArcMark": 0.0,
}


def _copy_distribution(
    distribution: sweep_run.DistributionSpec | None,
) -> sweep_run.DistributionSpec | None:
    if distribution is None:
        return None
    return sweep_run.DistributionSpec(
        name=distribution.name,
        parameters=dict(distribution.parameters),
    )


def build_canonical_sweeps() -> dict[str, sweep_run.WatermarkSweep]:
    base_sweeps = sweep_run.WATERMARK_SWEEPS
    base_types = set(base_sweeps)
    configured_types = set(CANONICAL_EPSILONS)

    extra = sorted(configured_types - base_types)
    if extra:
        raise ValueError(
            "Canonical epsilon map references unknown watermark types in scripts/run.py. "
            f"Extra: {extra}."
        )

    canonical_sweeps: dict[str, sweep_run.WatermarkSweep] = {}
    for watermark_type, sweep in base_sweeps.items():
        if watermark_type not in CANONICAL_EPSILONS:
            continue

        epsilon = float(CANONICAL_EPSILONS[watermark_type])

        selected_dist_sweep: sweep_run.DistributionSweep | None = None
        for dist_sweep in sweep.distribution_sweeps:
            if any(float(candidate) == epsilon for candidate in dist_sweep.epsilons):
                selected_dist_sweep = dist_sweep
                break

        if selected_dist_sweep is None:
            available = [
                float(candidate)
                for dist_sweep in sweep.distribution_sweeps
                for candidate in dist_sweep.epsilons
            ]
            raise ValueError(
                f"Canonical epsilon {epsilon} for {watermark_type} is not available. "
                f"Available: {available}"
            )

        multibit_configs = sweep.multibit_configs or [{}]
        canonical_sweeps[watermark_type] = sweep_run.WatermarkSweep(
            distribution_sweeps=[
                sweep_run.DistributionSweep(
                    epsilons=[epsilon],
                    distribution=_copy_distribution(selected_dist_sweep.distribution),
                )
            ],
            multibit_configs=[dict(multibit_configs[0])],
        )

    return canonical_sweeps


def main() -> None:
    sweep_run.WATERMARK_SWEEPS = build_canonical_sweeps()
    sweep_run.main()


if __name__ == "__main__":
    main()
