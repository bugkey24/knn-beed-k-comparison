"""Evaluation layer module."""
from src.evaluation.metrics import compute_metrics
from src.evaluation.comparator import aggregate_k_results

__all__ = ["compute_metrics", "aggregate_k_results"]
