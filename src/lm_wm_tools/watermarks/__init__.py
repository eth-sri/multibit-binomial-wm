import copy
import importlib
from typing import Optional
import json
import random

from loguru import logger
from vllm import SamplingParams
from vllm.v1.sample.logits_processor import (
    AdapterLogitsProcessor,
    RequestLogitsProcessor,
)

from lm_wm_tools import WatermarkProtocol
from lm_wm_tools.watermarks.logits_processors.metrics import LogitsMetricWrapper

_WATERMARK_MAPPING = {
    "KGW": "lm_wm_tools.watermarks.logits_processors.kgw.red_green.RedGreenWatermark",
    "AAR": "lm_wm_tools.watermarks.logits_processors.aar_extended.aar.AARWatermark",
    "PPLMark": "lm_wm_tools.watermarks.logits_processors.aar_extended.ppl_wm.PPLWatermark",
    "PPLUnconstrained": "lm_wm_tools.watermarks.logits_processors.aar_extended.ppl_wm_unc.PPLUnconstrainedWatermark",
    "SynthID": "lm_wm_tools.watermarks.logits_processors.synthid.synthid.SynthIDWatermark",
    "MirrorMark": "lm_wm_tools.watermarks.logits_processors.mirrormark.mirrormark.MirrorMarkWatermark",
    "ArcMark": "lm_wm_tools.watermarks.logits_processors.arcmark.arcmark.ArcMarkWatermark",
    "BiMark": "lm_wm_tools.watermarks.logits_processors.bimark.bimark.BiMarkWatermark",
    "M2Mark": "lm_wm_tools.watermarks.logits_processors.m2mark.m2mark.M2MarkWatermark",
    "MC2Mark": "lm_wm_tools.watermarks.logits_processors.m2mark.m2mark.M2MarkWatermark",
    "MC2MARK": "lm_wm_tools.watermarks.logits_processors.m2mark.m2mark.M2MarkWatermark",
    "StealthInk": "lm_wm_tools.watermarks.logits_processors.l1.stealthink.StealthInkWatermark",
    "RSBHWatermark": "lm_wm_tools.watermarks.logits_processors.kgw.rsbh_watermark.RSBHWatermark",
}
 

def _warn_if_top_k_unset(
    *,
    watermark_class: str,
    sampling_params: SamplingParams,
) -> None:
    top_k = getattr(sampling_params, "top_k", None)
    if top_k not in (None, 0, -1):
        return

    logger.warning(
        "top_k is not set for watermark_class={} in WatermarkLogitsProcessor "
        "(SamplingParams.top_k={}); full-vocabulary watermarking may be very slow.",
        watermark_class,
        top_k,
    )


def _materialize_payload_config(config: dict) -> None:
    """Resolve payload_size-only configs into an explicit fixed payload.

    Callers that need per-request randomized payloads should still provide
    `payload` directly. This helper only prevents watermark constructors from
    silently falling back to their single-bit default payload.
    """
    payload = config.get("payload", None)
    payload_size = config.pop("payload_size", None)

    if payload is not None:
        payload_bits = [int(bit) for bit in payload]
        if payload_size is not None and int(payload_size) != len(payload_bits):
            raise ValueError(
                "payload_size must match payload length when both are provided."
            )
        config["payload"] = payload_bits
        return

    if payload_size is None:
        return

    payload_size = int(payload_size)
    if payload_size < 0:
        raise ValueError("payload_size must be non-negative.")

    config["payload"] = [random.randint(0, 1) for _ in range(payload_size)]


def _decode_json_config_values(config: dict) -> None:
    for key in ("distribution_parameters", "payload", "remaining_bits"):
        value = config.get(key)
        if not isinstance(value, str):
            continue
        try:
            config[key] = json.loads(value)
        except json.JSONDecodeError:
            pass


# Wrapping the request-level logits processor:
class WatermarkLogitsProcessor(AdapterLogitsProcessor):
    def is_argmax_invariant(self) -> bool:
        return False

    def new_req_logits_processor(
        self,
        params: SamplingParams,
    ) -> Optional[RequestLogitsProcessor]:
        
        # Extract config
        watermark_config = copy.deepcopy(params.extra_args)

        # No watermark config for this request: leave the logits untouched.
        if watermark_config is None:
            return None

        # Extract our special side-channel keys
        request_id = watermark_config.pop("request_id", None)
        metrics_dir = watermark_config.pop("metrics_dir", None)
        _decode_json_config_values(watermark_config)

        # Parse distribution_parameters
        distribution_parameters = watermark_config.pop("distribution_parameters", {})
        watermark_config["distribution_parameters"] = distribution_parameters
        _materialize_payload_config(watermark_config)
        
        watermark_class = watermark_config.pop("watermark_class", None)
        assert watermark_class is not None, "watermark_class missing"
        _warn_if_top_k_unset(
            watermark_class=watermark_class,
            sampling_params=params,
        )

        watermark_processor = get_watermark_class(watermark_class)(sampling_parameters=params, **watermark_config)
        if request_id is not None and metrics_dir is not None:
            return LogitsMetricWrapper(watermark_processor, request_id, metrics_dir, params.temperature)
        else:
            return watermark_processor

def get_watermark_class(spec: str) -> type[WatermarkProtocol]:

    full_path = _WATERMARK_MAPPING.get(spec, spec)
    try:
        module_name, class_name = full_path.rsplit(".", 1)
        module = importlib.import_module(module_name)
        return getattr(module, class_name)
    except (ValueError, ImportError, AttributeError) as e:
        msg = f"Unknown watermark type: {spec} (resolved to {full_path}, available: {_WATERMARK_MAPPING})"
        raise ValueError(msg) from e


def get_watermark(watermark_config: dict, sampling_params: SamplingParams) -> WatermarkProtocol:
    config = copy.deepcopy(watermark_config)
    watermark_class = config.pop("watermark_class", None)
    _decode_json_config_values(config)

    # Parse distribution_parameters
    distribution_parameters = config.pop("distribution_parameters", {})
    config["distribution_parameters"] = distribution_parameters
    _materialize_payload_config(config)


    # Do some sanity checks
    seed = watermark_config.get("seed", None)
    multibit_seed = watermark_config.get("multibit_seed", None)
    if seed is not None and multibit_seed is not None:
        assert seed != multibit_seed, "seed and multibit_seed must be different values"

    assert watermark_class is not None, "watermark_class must be specified in watermark_config"
    return get_watermark_class(watermark_class)(sampling_parameters=sampling_params, **config)
