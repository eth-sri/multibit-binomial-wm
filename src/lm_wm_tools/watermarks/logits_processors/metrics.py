import torch
from typing import Dict, List
import os
import json
from dataclasses import dataclass, field


@dataclass
class WmStats:
    entropies: List[float] = field(default_factory=list)
    expected_scores: List[float] = field(default_factory=list)

    def compute_metrics(
        self,
        original_probs: torch.Tensor,
        modified_probs: torch.Tensor,
        expected_probs: torch.Tensor,
        scores: torch.Tensor,
    ) -> None:

        original_probs = original_probs.clamp_min(1e-12)  # Avoid log(0)
        original_probs = original_probs / original_probs.sum(
            dim=-1, keepdim=True
        )  # Normalize to get valid probabilities

        token_entropies = -(original_probs * original_probs.log()).sum(dim=-1)
        self.entropies.extend(token_entropies.detach().cpu().reshape(-1).tolist())

        # Compute the expected score under the watermarking policy
        expected_score = (modified_probs * scores).sum().item()
        self.expected_scores.append(expected_score)


    def to_dict(self) -> Dict:
        return {
            "entropies": self.entropies,
            "expected_scores": self.expected_scores,
        }

    def update(self, other: "WmStats") -> None:
        self.entropies.extend(other.entropies)
        self.expected_scores.extend(other.expected_scores)


class LogitsMetricWrapper:
    def __init__(
        self,
        watermark_processor,
        request_id: str,
        metrics_dir: str,
        temperature: float,
        top_k: int = -1,
    ):

        self.processor = watermark_processor
        self.filepath = get_metrics_filepath(request_id, metrics_dir)
        self.temperature = temperature
        self.top_k = top_k

    @torch.no_grad()
    def __call__(self, token_ids: List[int], logits: torch.Tensor) -> torch.Tensor:
        original_logits = logits.clone()
        if hasattr(self.processor, "call_with_scores"):
            modified_logits, scores = self.processor.call_with_scores(token_ids, logits)
        else:
            modified_logits = self.processor(token_ids, logits)
            scores = torch.zeros_like(modified_logits)  # No scores available, use zeros

        if original_logits is modified_logits:
            return modified_logits  # No modification done by the processor, skip metrics calculation

        orginal_probs = torch.nn.functional.softmax(
            original_logits / self.temperature, dim=-1
        )
        modified_probs = torch.nn.functional.softmax(
            modified_logits / self.temperature, dim=-1
        )
        expected_probs = torch.zeros_like(orginal_probs)

        # Metric calculation
        wm_stats = WmStats()
        wm_stats.compute_metrics(orginal_probs, modified_probs, expected_probs, scores)
        data = wm_stats.to_dict()

        # Write to the specific file passed in the constructor
        with open(self.filepath, "a") as f:
            f.write(json.dumps(data) + "\n")

        return modified_logits


def get_metrics_filepath(request_id: str, metrics_dir: str) -> str:
    file_path = os.path.join(metrics_dir, f"{request_id}.jsonl")
    return file_path


def summarize_metrics(request_id: str, metrics_dir: str) -> Dict:
    """
    Summarizes the collected metrics for a given request_id.
    Returns the full list of token entropies.
    """

    filepath = get_metrics_filepath(request_id, metrics_dir)
    if not os.path.exists(filepath):
        return {}

    aggregated_stats = WmStats()
    for line in open(filepath, "r"):
        data = json.loads(line)
        wm_stats = WmStats(
            entropies=data.get("entropies", []),
            expected_scores=data.get("expected_scores", []),
        )
        aggregated_stats.update(wm_stats)
    if not aggregated_stats.entropies:
        return {}
    return aggregated_stats.to_dict()
