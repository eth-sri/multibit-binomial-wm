import time
from functools import wraps
from typing import Any, Callable, Dict, ParamSpec, TypeVar, cast
from vllm import SamplingParams

P = ParamSpec("P")
R = TypeVar("R", bound=Dict[str, Any])

_MULTIBIT_NONE_ALIASES = {
    "",
    "none",
    "random",
    "default",
    "random_generator",
    "randomgenerator",
}


def normalize_multibit_algorithm(multibit_algorithm: Any) -> str:
    """Return a stable string used in watermark names for multibit algorithms."""
    if multibit_algorithm is None:
        return "none"

    value = getattr(multibit_algorithm, "value", multibit_algorithm)
    normalized = str(value).strip().lower()
    if normalized in _MULTIBIT_NONE_ALIASES:
        return "none"
    return normalized


def format_watermark_name(
    watermark_class: str,
    multibit_algorithm: Any,
    suffix: str = "",
    payload_size: int | None = None,
) -> str:
    """Build names as: watermarkClass/multibitAlgorithm/payload_size_{m}/{previous_suffix}."""
    name = f"{watermark_class}/{normalize_multibit_algorithm(multibit_algorithm)}"
    parts: list[str] = []
    if payload_size is not None:
        parts.append(f"payload_size_{int(payload_size)}")
    clean_suffix = suffix.strip("/")
    if clean_suffix:
        parts.append(clean_suffix)
    if parts:
        return f"{name}/{'/'.join(parts)}"
    return name

def resolve_top_k_from_sampling_params(
    sampling_parameters: SamplingParams, vocab_size: int
) -> int:
    """
    Determine the effective top-k to use for watermarking from vLLM sampling parameters.
    vLLM stores "no top-k" as -1, so default to the full vocabulary in that case.
    """
    sp_top_k = getattr(sampling_parameters, "top_k", None)
    if sp_top_k is None or sp_top_k == -1:
        return vocab_size
    if sp_top_k <= 0:
        raise ValueError("top_k must be a positive integer")
    return min(sp_top_k, vocab_size)

def add_elapsed_time(
    *,
    key: str = "elapsed_ms",
    timer: Callable[[], float] = time.perf_counter,
    overwrite: bool = True,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """
    Decorator for functions that return a dict.
    Adds elapsed time to the returned dict under `key`.

    Args:
        key: Dict key to store elapsed time.
        timer: Timing function (default: time.perf_counter).
        overwrite: If False and key already exists, raises KeyError.
    """
    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            start = timer()
            result = func(*args, **kwargs)
            end = timer()

            if not isinstance(result, dict):
                raise TypeError(f"{func.__name__} must return a dict, got {type(result).__name__}")

            if (not overwrite) and (key in result):
                raise KeyError(f"Key '{key}' already exists in result dict from {func.__name__}")

            result[key] = (end - start) * 1000.0  # milliseconds
            return cast(R, result)

        return wrapper
    return decorator
