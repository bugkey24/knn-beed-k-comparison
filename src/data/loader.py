"""Dataset loading module."""
from pathlib import Path
import pandas as pd


def load_beed_dataset(csv_path: str | Path) -> pd.DataFrame:
    """Loads the BEED CSV file into a pandas DataFrame.

    Args:
        csv_path: File path to BEED_Data.csv.

    Returns:
        pd.DataFrame containing the raw dataset.

    Raises:
        FileNotFoundError: If the specified file does not exist.
    """
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"BEED dataset file not found at: {path.resolve()}")

    return pd.read_csv(path)
