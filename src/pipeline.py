"""Pipeline orchestrator executing the end-to-end KNN classification workflow."""
from pathlib import Path
from typing import Any, Dict
import pandas as pd

from src.config import AppConfig, load_config
from src.data.loader import load_beed_dataset
from src.data.validator import validate_beed_dataset
from src.data.splitter import split_dataset
from src.preprocessing.scaler import FeatureScaler
from src.models.trainer import train_and_evaluate_k
from src.evaluation.metrics import compute_metrics
from src.evaluation.comparator import aggregate_k_results
from src.visualization.eda import (
    plot_class_distribution,
    plot_feature_distributions,
    plot_correlation_heatmap,
)
from src.visualization.results import (
    plot_accuracy_vs_k,
    plot_confusion_matrix,
)
from src.utils.io import save_dataframe, save_pickle
from src.utils.logger import setup_logger


def run_pipeline(config: AppConfig | None = None, config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """Executes the complete machine learning workflow.

    Args:
        config: Pre-loaded AppConfig instance. If None, loaded from config_path.
        config_path: Path to YAML config if config is None.

    Returns:
        Dictionary containing comparison DataFrame, best K metrics, and model references.
    """
    if config is None:
        config = load_config(config_path)

    # Initialize Logger
    log_file = Path(config.output.logs_dir) / "pipeline_run.log"
    logger = setup_logger("knn_beed_pipeline", log_file=log_file)
    logger.info("================ Starting BEED KNN Classification Pipeline ================")

    # 1. Ingestion & Validation
    logger.info(f"Loading dataset from: {config.dataset.path}")
    df = load_beed_dataset(config.dataset.path)
    logger.info("Validating dataset integrity...")
    validate_beed_dataset(
        df,
        expected_rows=config.dataset.expected_rows,
        expected_features=config.dataset.num_features,
        target_column=config.dataset.target_column,
        expected_classes=config.dataset.num_classes,
    )
    logger.info("Dataset validated: 8000 rows, 16 features, 4 balanced classes, 0 missing values.")

    # 2. EDA Visualizations
    logger.info("Generating Exploratory Data Analysis figures...")
    plot_class_distribution(
        df,
        target_column=config.dataset.target_column,
        save_path=Path(config.output.figures_dir) / "class_distribution.png"
    )
    plot_feature_distributions(
        df,
        save_path=Path(config.output.figures_dir) / "feature_distribution.png"
    )
    plot_correlation_heatmap(
        df,
        save_path=Path(config.output.figures_dir) / "correlation_heatmap.png"
    )
    logger.info("EDA figures successfully saved.")

    # 3. Stratified Partitioning (Anti-leakage: split BEFORE scaling)
    logger.info(
        f"Partitioning data (test_size={config.split.test_size}, "
        f"stratify={config.split.stratify}, seed={config.split.random_state})..."
    )
    X_train, X_test, y_train, y_test = split_dataset(
        df,
        target_column=config.dataset.target_column,
        test_size=config.split.test_size,
        random_state=config.split.random_state,
        stratify=config.split.stratify,
    )
    logger.info(f"Split complete. Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    # 4. Standard Scaling (Fit strictly on train, transform on test)
    logger.info("Standardizing features (fitting exclusively on X_train)...")
    scaler = FeatureScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    save_pickle(scaler, Path(config.output.models_dir) / "scaler.pkl")
    logger.info("Feature standardization complete. Scaler serialized.")

    # 5. Multi-K Training & Evaluation Loop
    logger.info(f"Iterating KNN hyperparameter grid K in {config.model.k_values}...")
    k_eval_results = []
    models_dict = {}
    confusion_matrices = {}

    for k in config.model.k_values:
        model, y_train_pred, y_test_pred, inf_time = train_and_evaluate_k(
            k=k,
            X_train=X_train_scaled,
            y_train=y_train.values,
            X_test=X_test_scaled,
            metric=config.model.metric,
            weights=config.model.weights,
        )

        train_metrics = compute_metrics(y_train.values, y_train_pred, average=config.evaluation.average)
        test_metrics = compute_metrics(y_test.values, y_test_pred, average=config.evaluation.average)

        k_eval_results.append({
            "k": k,
            "train_accuracy": train_metrics["accuracy"],
            "test_accuracy": test_metrics["accuracy"],
            "macro_precision": test_metrics["precision"],
            "macro_recall": test_metrics["recall"],
            "macro_f1": test_metrics["f1"],
            "inference_time_sec": inf_time,
        })

        models_dict[k] = model
        confusion_matrices[k] = test_metrics["confusion_matrix"]

        logger.info(
            f"K={k:2d} | Train Acc: {train_metrics['accuracy']:.4f} | "
            f"Test Acc: {test_metrics['accuracy']:.4f} | "
            f"Macro F1: {test_metrics['f1']:.4f} | "
            f"Latency: {inf_time*1000:.1f}ms"
        )

    # 6. Aggregating & Saving Results
    comparison_df = aggregate_k_results(k_eval_results)
    results_csv_path = Path(config.output.results_dir) / "k_comparison.csv"
    save_dataframe(comparison_df, results_csv_path)
    logger.info(f"Comparison metrics saved to: {results_csv_path}")

    # 7. Identifying Optimal K & Visualizing Performance
    best_row = comparison_df.loc[comparison_df["test_accuracy"].idxmax()]
    best_k = int(best_row["k"])
    best_acc = float(best_row["test_accuracy"])
    logger.info(f"--> Optimal Model Identified: K={best_k} with Test Accuracy = {best_acc:.4f}")

    # Plot Accuracy vs K
    plot_accuracy_vs_k(
        comparison_df,
        save_path=Path(config.output.figures_dir) / "accuracy_vs_k.png"
    )

    # Plot Confusion Matrix for Best K
    best_cm = confusion_matrices[best_k]
    plot_confusion_matrix(
        best_cm,
        k_value=best_k,
        save_path=Path(config.output.figures_dir) / f"confusion_matrix_k{best_k}.png"
    )

    # Save best model
    save_pickle(models_dict[best_k], Path(config.output.models_dir) / f"best_knn_k{best_k}.pkl")
    logger.info("================ Pipeline Execution Completed Successfully ================")

    return {
        "comparison_df": comparison_df,
        "best_k": best_k,
        "best_accuracy": best_acc,
        "best_confusion_matrix": best_cm,
        "models": models_dict,
        "scaler": scaler,
    }
