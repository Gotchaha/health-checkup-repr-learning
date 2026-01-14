# scripts/plot_hpo_ssl.py

"""
Standalone script for regenerating Optuna study visualizations.

Usage example:
    python scripts/plot_hpo_ssl.py \
        --study-name ssl_pretraining_v1_lr_wd \
        --storage sqlite:///outputs/hpo/studies/ssl_pretraining_v1_lr_wd.db

Saves optimization history, param importances, and slice plots under
outputs/hpo/plots/<study_name>/ (or a custom --output-dir).
"""

import argparse
import logging
from pathlib import Path
from typing import Any

import optuna
from optuna.visualization.matplotlib import (
    plot_optimization_history,
    plot_param_importances,
    plot_slice,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Regenerate Optuna study plots")
    parser.add_argument(
        "--study-name",
        required=True,
        help="Optuna study name",
    )
    parser.add_argument(
        "--storage",
        required=True,
        help="Optuna storage URI (e.g., sqlite:///outputs/hpo/studies/foo.db)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to store plots (default: outputs/hpo/plots/<study>)",
    )
    return parser.parse_args()


def save_from_axes(obj: Any, output_path: Path) -> None:
    """Helper to save matplotlib figures returned as Axes or ndarray of Axes."""
    if hasattr(obj, "get_figure"):
        fig = obj.get_figure()
    elif hasattr(obj, "figure"):
        fig = obj.figure
    elif hasattr(obj, "flat"):
        fig = obj.flat[0].figure
    else:
        raise TypeError(f"Cannot obtain figure from object type: {type(obj)}")

    fig.savefig(output_path, dpi=150, bbox_inches="tight")


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    study = optuna.load_study(study_name=args.study_name, storage=args.storage)

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path("outputs/hpo/plots") / args.study_name
    output_dir.mkdir(parents=True, exist_ok=True)

    plots = [
        (plot_optimization_history, "optimization_history.png"),
        (plot_param_importances, "param_importances.png"),
        (plot_slice, "slice.png"),
    ]

    for plot_fn, filename in plots:
        try:
            axes_obj = plot_fn(study)
            save_from_axes(axes_obj, output_dir / filename)
            logging.info("Saved %s", filename)
        except Exception as e:
            logging.warning("Failed to generate %s: %s", filename, e)

    logging.info("Plots saved to %s", output_dir)


if __name__ == "__main__":
    main()
