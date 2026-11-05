"""Logging and input/output utilities."""
import logging
from pathlib import Path
import sys


def setup_logger(name: str = "knn_pipeline", log_file: str | Path | None = None, level: int = logging.INFO) -> logging.Logger:
    """Configures and returns a structured logger.

    Args:
        name: Name of the logger instance.
        log_file: Optional path to output log file.
        level: Logging severity level.

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers if setup is called multiple times
    if not logger.handlers:
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        if log_file:
            path = Path(log_file)
            path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(path, encoding="utf-8")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger
