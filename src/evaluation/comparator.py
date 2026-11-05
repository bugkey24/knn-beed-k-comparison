"""Comparator module aggregating evaluation metrics across K values."""
from typing import Any, Dict, List
import pandas as pd


def aggregate_k_results(results_list: List[Dict[str, Any]]) -> pd.DataFrame:
    """Aggregates per-K evaluation metrics into a consolidated pandas DataFrame.

    Args:
        results_list: List of metric dicts containing K and evaluation values.

    Returns:
        Structured pd.DataFrame sorted by K.
    """
    records = []
    for res in results_list:
        records.append({
            "k": res["k"],
            "train_accuracy": round(res["train_accuracy"], 4),
            "test_accuracy": round(res["test_accuracy"], 4),
            "macro_precision": round(res["macro_precision"], 4),
            "macro_recall": round(res["macro_recall"], 4),
            "macro_f1": round(res["macro_f1"], 4),
            "inference_time_ms": round(res["inference_time_sec"] * 1000, 2),
        })

    df = pd.DataFrame(records)
    return df.sort_values("k").reset_index(drop=True)
