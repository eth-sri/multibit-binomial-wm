from typing import Any, Protocol, List
import torch
from dataclasses import dataclass
from vllm import SamplingParams

@dataclass
class DetectionOutput:

    statistic: float
    pvalue: float
    additional_info: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        out = {
            "statistic": self.statistic,
            "pvalue": self.pvalue,
        }
        out.update(self.additional_info)
        return out

class WatermarkProtocol(Protocol):

    def __init__(self,  sampling_params: SamplingParams, **kwargs) -> None:
        """Initialize the watermark with the given configuration and sampling parameters."""
        ...

    def __call__(self, output_ids: List[int], logits: torch.Tensor) -> torch.Tensor:
        """Called by vLLM during sampling to modify the logits."""
        ...

    def get_expected_probs(self, probs: torch.Tensor) -> torch.Tensor:
        """Given original probabilities, return the expected modified probabilities after watermarking.
        
        This is used for metric calculations and is optional.
        """
        return probs

    def detect(self, tokens: List[int]) -> DetectionOutput:
        """Called to detect watermark in a sequence of tokens."""
        ...
        
    def get_name(self) -> str:
        """Return the name of the watermarking scheme and its config as a string."""
        ...