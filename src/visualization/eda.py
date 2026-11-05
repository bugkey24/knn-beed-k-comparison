"""Exploratory data analysis visualizations."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")  # Headless backend
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_class_distribution(
    df: pd.DataFrame,
    target_column: str = "y",
    save_path: str | Path | None = None,
    dpi: int = 150
) -> plt.Figure:
    """Plots and optionally saves target class distribution bar chart."""
    fig, ax = plt.subplots(figsize=(8, 5))
    class_labels = {0: "Healthy (0)", 1: "Generalized (1)", 2: "Focal (2)", 3: "Seizure (3)"}
    counts = df[target_column].value_counts().sort_index()
    names = [class_labels.get(i, str(i)) for i in counts.index]

    sns.barplot(x=names, y=counts.values, palette="mako", ax=ax)
    ax.set_title("BEED Dataset Target Class Distribution", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Clinical Neurological Class", fontsize=11)
    ax.set_ylabel("Sample Count", fontsize=11)
    ax.set_ylim(0, max(counts.values) * 1.15)

    for i, v in enumerate(counts.values):
        ax.text(i, v + 25, f"{v:,} ({v/len(df):.1%})", ha="center", va="bottom", fontsize=10, fontweight="semibold")

    plt.tight_layout()
    if save_path:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=dpi)

    return fig


def plot_feature_distributions(
    df: pd.DataFrame,
    feature_cols: list[str] | None = None,
    save_path: str | Path | None = None,
    dpi: int = 150
) -> plt.Figure:
    """Plots boxplots of feature amplitude distributions across channels."""
    if feature_cols is None:
        feature_cols = [c for c in df.columns if c.startswith("X")]

    fig, ax = plt.subplots(figsize=(12, 6))
    df_features = df[feature_cols]
    sns.boxplot(data=df_features, palette="crest", ax=ax, showfliers=False)
    ax.set_title("EEG Feature Amplitude Distributions Across Channels (Excl. Outliers)", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("EEG Feature / Channel", fontsize=11)
    ax.set_ylabel("Signal Amplitude", fontsize=11)
    plt.xticks(rotation=45)
    plt.tight_layout()

    if save_path:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=dpi)

    return fig


def plot_correlation_heatmap(
    df: pd.DataFrame,
    feature_cols: list[str] | None = None,
    save_path: str | Path | None = None,
    dpi: int = 150
) -> plt.Figure:
    """Plots correlation heatmap among EEG features."""
    if feature_cols is None:
        feature_cols = [c for c in df.columns if c.startswith("X")]

    fig, ax = plt.subplots(figsize=(10, 8))
    corr = df[feature_cols].corr()
    sns.heatmap(corr, cmap="vlag", annot=False, vmin=-1, vmax=1, ax=ax)
    ax.set_title("Inter-Feature Correlation Matrix (EEG Channels X1-X16)", fontsize=14, fontweight="bold", pad=12)
    plt.tight_layout()

    if save_path:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=dpi)

    return fig
