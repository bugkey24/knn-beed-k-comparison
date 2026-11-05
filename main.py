"""CLI entry point for running the BEED KNN Classification Pipeline."""
import argparse
import sys
from pathlib import Path

# Add project root to sys.path for direct script execution
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.pipeline import run_pipeline


def main() -> None:
    """Parses CLI arguments and executes the machine learning pipeline."""
    parser = argparse.ArgumentParser(
        description="Run end-to-end KNN classification and K hyperparameter comparison on BEED dataset."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to YAML configuration file (default: config/config.yaml)"
    )
    args = parser.parse_args()

    print(f"Loading configuration from: {args.config}")
    config = load_config(args.config)
    results = run_pipeline(config=config)

    print("\n--- Hyperparameter K Comparison Results ---")
    print(results["comparison_df"].to_string(index=False))
    print(f"\nOptimal K identified: K={results['best_k']} (Test Accuracy: {results['best_accuracy']:.2%})")
    print(f"Artifacts saved in directory: {config.output.dir}/")


if __name__ == "__main__":
    main()
