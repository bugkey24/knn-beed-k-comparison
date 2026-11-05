"""IO helpers for serialization and persistence."""
from pathlib import Path
import pickle
from typing import Any
import pandas as pd


def save_dataframe(df: pd.DataFrame, path: str | Path, index: bool = False) -> None:
    """Saves a pandas DataFrame to CSV, ensuring parent directories exist."""
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(file_path, index=index)


def save_pickle(obj: Any, path: str | Path) -> None:
    """Serializes an object using pickle, ensuring parent directories exist."""
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: str | Path) -> Any:
    """Loads an object from a pickle file."""
    file_path = Path(path)
    with open(file_path, "rb") as f:
        return pickle.load(f)
