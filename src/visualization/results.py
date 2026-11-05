"""Model performance and evaluation visualization."""
from pathlib import Path
from typing import Sequence
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_accuracy_vs_k(
    comparison_df: pd.DataFrame,
    save_path: str | Path | None = None,
    dpi: int = 150
) -> plt.Figure:
    """Plots training vs testing accuracy across tested K values."""
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(
        comparison_df["k"],
        comparison_df["train_accuracy"],
        marker="o",
        linewidth=2,
        label="Train Accuracy",
        color="#2b5c8f"
    )
    ax.plot(
        comparison_df["k"],
        comparison_df["test_accuracy"],
        marker="s",
        linewidth=2,
        label="Test Accuracy",
        color="#d95f02"
    )

    # Highlight best test accuracy
    best_row = comparison_df.loc[comparison_df["test_accuracy"].idxmax()]
    best_k = int(best_row["k"])
    best_acc = float(best_row["test_accuracy"])

    ax.scatter(
        [best_k], [best_acc],
        color="crimson",
        s=120,
        zorder=5,
        label=f"Optimal K={best_k} ({best_acc:.2%})"
    )
    ax.annotate(
        f"Best K={best_k}\nAcc={best_acc:.2%}",
        xy=(best_k, best_acc),
        xytext=(best_k, best_acc - 0.04),
        ha="center",
        arrowprops=dict(facecolor="black", shrink=0.05, width=1, headwidth=6),
        fontweight="bold"
    )

    ax.set_title("KNN Hyperparameter Analysis: Train vs Test Accuracy vs K", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Number of Neighbors (K)", fontsize=11)
    ax.set_ylabel("Classification Accuracy", fontsize=11)
    ax.set_xticks(comparison_df["k"])
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(frameon=True, facecolor="white", loc="best")

    plt.tight_layout()
    if save_path:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=dpi)

    return fig


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: Sequence[str] = ("Healthy (0)", "Generalized (1)", "Focal (2)", "Seizure (3)"),
    k_value: int = 5,
    save_path: str | Path | None = None,
    dpi: int = 150
) -> plt.Figure:
    """Plots confusion matrix heatmap for a specific K value."""
    fig, ax = plt.subplots(figsize=(7, 6))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
        ax=ax
    )

    ax.set_title(f"Confusion Matrix (KNN with K={k_value})", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted Class", fontsize=11)
    ax.set_ylabel("True Class", fontsize=11)

    plt.tight_layout()
    if save_path:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=dpi)

    return fig
