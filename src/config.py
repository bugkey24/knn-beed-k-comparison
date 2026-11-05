"""Configuration parser and dataclass representations."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List
import yaml


@dataclass(frozen=True)
class DatasetConfig:
    """Dataset configuration parameters."""
    path: str
    target_column: str = "y"
    feature_prefix: str = "X"
    num_features: int = 16
    num_classes: int = 4
    expected_rows: int = 8000


@dataclass(frozen=True)
class SplitConfig:
    """Train/test split parameters."""
    test_size: float = 0.20
    random_state: int = 42
    stratify: bool = True


@dataclass(frozen=True)
class PreprocessingConfig:
    """Preprocessing parameters."""
    scaler: str = "standard"


@dataclass(frozen=True)
class ModelConfig:
    """KNN Model parameters."""
    algorithm: str = "knn"
    k_values: List[int] = field(default_factory=lambda: [1, 3, 5, 7, 9, 11, 15, 21, 25, 31])
    metric: str = "euclidean"
    weights: str = "uniform"


@dataclass(frozen=True)
class EvaluationConfig:
    """Evaluation parameters."""
    average: str = "macro"
    metrics: List[str] = field(default_factory=lambda: ["accuracy", "precision", "recall", "f1"])


@dataclass(frozen=True)
class OutputConfig:
    """Output directories."""
    dir: str = "outputs"
    figures_dir: str = "outputs/figures"
    results_dir: str = "outputs/results"
    models_dir: str = "outputs/models"
    logs_dir: str = "outputs/logs"


@dataclass(frozen=True)
class AppConfig:
    """Root configuration holding all sub-configurations."""
    dataset: DatasetConfig
    split: SplitConfig
    preprocessing: PreprocessingConfig
    model: ModelConfig
    evaluation: EvaluationConfig
    output: OutputConfig


def load_config(config_path: str | Path = "config/config.yaml") -> AppConfig:
    """Loads a YAML configuration file into a typed AppConfig dataclass.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        AppConfig populated with parsed values.

    Raises:
        FileNotFoundError: If configuration file is missing.
    """
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path.resolve()}")

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    return AppConfig(
        dataset=DatasetConfig(**data.get("dataset", {})),
        split=SplitConfig(**data.get("split", {})),
        preprocessing=PreprocessingConfig(**data.get("preprocessing", {})),
        model=ModelConfig(**data.get("model", {})),
        evaluation=EvaluationConfig(**data.get("evaluation", {})),
        output=OutputConfig(**data.get("output", {})),
    )
