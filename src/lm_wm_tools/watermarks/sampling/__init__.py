from __future__ import annotations

from enum import Enum

from .mpac import MPAC
from .cycle_shift import CycleShift
from .bino_encoder import BinoEncoder
from .bino_encoder_dynamic import DynamicBinoEncoder
from .bino_encoder_dynamic_multigen import DynamicBinoEncoderMultiGen
from .mpac_bino_encoder import MPACBinoEncoder
from .random_utility import RandomGenerator, SeedingScheme
from .rademacher_encoder import RademacherEncoder
from .cabs import CABS


class RandomUtilityType(Enum):
    RANDOM_GENERATOR = "random_generator"
    CABS = "cabs"
    MPAC = "mpac"
    CYCLE_SHIFT = "cycle_shift"
    BINO_ENCODER = "bino_encoder"
    BINO_ENCODER_DYNAMIC = "bino_encoder_dynamic"
    BINO_ENCODER_DYNAMIC_MULTIGEN = "bino_encoder_dynamic_multigen"
    MPAC_BINO_ENCODER = "mpac_bino_encoder"
    RADEMACHER_ENDOCER = "rademacher_encoder"

    @classmethod
    def from_string(
        cls, value: "RandomUtilityType | str | None"
    ) -> "RandomUtilityType":
        if value is None:
            return cls.RANDOM_GENERATOR

        if isinstance(value, cls):
            return value

        if not isinstance(value, str):
            raise TypeError(
                "random_utility must be a string or RandomUtilityType, "
                f"got {type(value)}"
            )

        normalized = value.strip().lower()
        if normalized in {"none", "random", "random_generator", "randomgenerator", "default"}:
            return cls.RANDOM_GENERATOR
        if normalized in {"cabs", "mirrormark", "mirror_mark"}:
            return cls.CABS
        if normalized == "mpac":
            return cls.MPAC
        if normalized in {"mpac_bino", "mpac_binoencoder"}:
            return cls.MPAC_BINO_ENCODER
        if normalized in {"dynamic_bino_encoder", "dynamicbinoencoder"}:
            return cls.BINO_ENCODER_DYNAMIC
        if normalized in {
            "dynamic_bino_encoder_multigen",
            "dynamicbinomultigen",
            "bino_encoder_dynamic_multigen",
        }:
            return cls.BINO_ENCODER_DYNAMIC_MULTIGEN

        try:
            return cls(normalized)
        except ValueError as exc:
            expected = ", ".join(sorted({member.value for member in cls}))
            raise ValueError(
                f"Unknown random utility: {value}. "
                f"Expected one of: {expected}."
            ) from exc


def create_random_utility(
    random_utility: RandomUtilityType | str | None,
    distribution_name: str,
    distribution_parameters: dict,
    payload: list[int] | None = None,
    multibit_seed: int | None = None,
    vocab_size: int | None = None,
    n_segments: int | None = None,
    **kwargs,
) -> RandomGenerator:
    utility_type = RandomUtilityType.from_string(random_utility)

    utility_kwargs = {
        "distribution_name": distribution_name,
        "distribution_parameters": distribution_parameters,
        "vocab_size": vocab_size,
        "payload": payload,
        "multibit_seed": multibit_seed,
        "n_segments": n_segments,
    }
    utility_kwargs.update(kwargs)

    if utility_type is RandomUtilityType.MPAC:
        return MPAC(**utility_kwargs)
    elif utility_type is RandomUtilityType.CABS:
        return CABS(**utility_kwargs)
    elif utility_type is RandomUtilityType.CYCLE_SHIFT:
        return CycleShift(**utility_kwargs)
    elif utility_type is RandomUtilityType.BINO_ENCODER:
        return BinoEncoder(**utility_kwargs)
    elif utility_type is RandomUtilityType.BINO_ENCODER_DYNAMIC:
        return DynamicBinoEncoder(**utility_kwargs)
    elif utility_type is RandomUtilityType.BINO_ENCODER_DYNAMIC_MULTIGEN:
        return DynamicBinoEncoderMultiGen(**utility_kwargs)
    elif utility_type is RandomUtilityType.MPAC_BINO_ENCODER:
        return MPACBinoEncoder(**utility_kwargs)
    elif utility_type is RandomUtilityType.RADEMACHER_ENDOCER:
        return RademacherEncoder(**utility_kwargs)

    return RandomGenerator(**utility_kwargs)


__all__ = [
    "MPAC",
    "CABS",
    "MPACBinoEncoder",
    "DynamicBinoEncoder",
    "DynamicBinoEncoderMultiGen",
    "RandomGenerator",
    "RandomUtilityType",
    "SeedingScheme",
    "create_random_utility",
]
