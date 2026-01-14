# scripts/hpo_ssl.py

"""
Optuna-based Hyperparameter Optimization driver for SSL pretraining.

- Treats scripts/train_ssl.py as a black-box objective.
- For each trial: sample hyperparameters, generate a temporary training config,
  run the training script (subprocess), then read validation metrics to compute the objective.
- Stores Optuna study state and visualization artifacts under outputs/hpo/.
"""

import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import optuna
from optuna.visualization.matplotlib import (
    plot_optimization_history,
    plot_param_importances,
    plot_slice,
)
import yaml

# Make project root importable (scripts run from project root)
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.utils import load_experiment_results, load_experiment_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Optuna HPO for SSL pretraining (outer-loop driver)."
    )
    parser.add_argument(
        "--hpo-config",
        type=str,
        required=True,
        help="Path to HPO meta-config YAML (e.g., config/hpo/ssl_pretraining_v1_lr_wd.yaml)",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=None,
        help="Override number of trials (default: use config value)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Override study timeout in seconds (default: use config value)",
    )
    parser.add_argument(
        "--study-name",
        type=str,
        default=None,
        help="Override study name (default: use config value)",
    )
    parser.add_argument(
        "--storage",
        type=str,
        default=None,
        help="Override Optuna storage URI (default: use config value)",
    )
    return parser.parse_args()


def load_hpo_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"HPO config not found: {path}")
    with open(path, "r") as f:
        cfg = yaml.safe_load(f) or {}

    for key in ("study", "objective", "search_space"):
        if key not in cfg:
            raise ValueError(f"HPO config missing required key: '{key}'")

    base_cfg = Path(cfg["objective"].get("base_config", ""))
    if not base_cfg.exists():
        raise FileNotFoundError(
            f"Base training config not found (objective.base_config): {base_cfg}"
        )

    direction = cfg["study"].get("direction", "minimize")
    if direction not in ("minimize", "maximize"):
        raise ValueError("study.direction must be 'minimize' or 'maximize'")

    return cfg


def sample_trial_params(trial: optuna.Trial, search_space: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Map search space definitions to optuna trial suggestions."""
    params = {}
    for name, spec in search_space.items():
        ptype = spec.get("type")
        if ptype == "float":
            params[name] = trial.suggest_float(
                name,
                spec["low"],
                spec["high"],
                log=spec.get("log", False),
            )
        elif ptype == "int":
            params[name] = trial.suggest_int(
                name,
                spec["low"],
                spec["high"],
                step=spec.get("step"),
            )
        elif ptype == "categorical":
            params[name] = trial.suggest_categorical(name, spec["choices"])
        else:
            raise ValueError(f"Unsupported search space type for '{name}': {ptype}")
    return params


def set_nested(config: Dict[str, Any], dotted_key: str, value: Any) -> None:
    """Set nested dictionary value given a dot-delimited key."""
    keys = dotted_key.split(".")
    current = config
    for key in keys[:-1]:
        if key not in current or not isinstance(current[key], dict):
            current[key] = {}
        current = current[key]
    current[keys[-1]] = value


def build_trial_config(
    hpo_cfg: Dict[str, Any],
    params: Dict[str, Any],
    trial: optuna.Trial,
) -> Tuple[Path, str]:
    """Generate a trial-specific training config file."""
    objective_cfg = hpo_cfg["objective"]
    base_config_path = Path(objective_cfg["base_config"])

    # Load base config without auto timestamp
    train_config = load_experiment_config(base_config_path, auto_timestamp=False)

    # Assign experiment name
    prefix = objective_cfg.get("experiment_name_prefix", "ssl_hpo")
    experiment_name = f"{prefix}_trial_{trial.number:04d}"
    train_config["experiment_name"] = experiment_name

    # Override outputs root if provided
    outputs_root = objective_cfg.get("outputs_root")
    if outputs_root:
        train_config.setdefault("paths", {})
        train_config["paths"]["outputs_root"] = outputs_root

    # Enforce lightweight settings for HPO (defensive even if base config already sets them)
    train_config.setdefault("training", {})
    train_config["training"].setdefault("checkpoint", {})
    train_config["training"]["checkpoint"]["enabled"] = False
    train_config.setdefault("monitoring", {})
    train_config["monitoring"]["enabled"] = False
    train_config["use_wandb"] = False
    train_config.setdefault("experiment_logging", {})
    train_config["experiment_logging"]["enabled"] = True

    # Apply sampled hyperparameters
    for dotted_key, value in params.items():
        set_nested(train_config, dotted_key, value)

    # Write trial config
    study_name = hpo_cfg["study"]["name"]
    configs_dir = Path(outputs_root or "outputs/hpo") / "configs" / study_name
    configs_dir.mkdir(parents=True, exist_ok=True)
    trial_config_path = (configs_dir / f"{experiment_name}.yaml").resolve()

    # Remove metadata before saving (load_experiment_config inserts _meta)
    to_save = {k: v for k, v in train_config.items() if not k.startswith("_")}
    with open(trial_config_path, "w") as f:
        yaml.safe_dump(to_save, f, sort_keys=False)

    return trial_config_path, experiment_name


def extract_best_metric(results: Dict[str, Any], metric_name: str) -> float:
    """Extract best metric (min) from validation history."""
    val_entries = results.get("validation_metrics", [])
    if not val_entries:
        raise ValueError("No validation metrics found in experiment results.")

    values = [
        entry.get(metric_name)
        for entry in val_entries
        if entry.get(metric_name) is not None
    ]
    if not values:
        raise ValueError(f"Metric '{metric_name}' not present in validation history.")
    return min(values)


def make_objective(hpo_cfg: Dict[str, Any], logger: logging.Logger):
    objective_cfg = hpo_cfg["objective"]
    study_cfg = hpo_cfg["study"]
    logs_root = Path(objective_cfg.get("outputs_root", "outputs/hpo")) / "logs"

    def objective(trial: optuna.Trial) -> float:
        params = sample_trial_params(trial, hpo_cfg["search_space"])
        config_path, experiment_name = build_trial_config(hpo_cfg, params, trial)

        cmd = [
            sys.executable,
            "scripts/train_ssl.py",
            "--config",
            str(config_path),
            "--no-timestamp",
        ]

        logger.info("Trial %d: running %s", trial.number, " ".join(cmd))
        start = time.time()
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        elapsed = time.time() - start

        if proc.returncode != 0:
            logger.error(
                "Trial %d failed (returncode=%s). Stdout tail:\n%s\nStderr tail:\n%s",
                trial.number,
                proc.returncode,
                "\n".join(proc.stdout.splitlines()[-10:]),
                "\n".join(proc.stderr.splitlines()[-10:]),
            )
            raise optuna.TrialPruned(f"Training failed: returncode={proc.returncode}")

        try:
            results = load_experiment_results(experiment_name, logs_root=str(logs_root))
            metric = extract_best_metric(results, objective_cfg["metric"])
        except Exception as e:
            logger.error("Failed to extract metric for trial %d: %s", trial.number, e)
            raise optuna.TrialPruned(f"Metric extraction failed: {e}") from e

        logger.info(
            "Trial %d completed in %.2fs | %s = %.4f",
            trial.number,
            elapsed,
            objective_cfg["metric"],
            metric,
        )
        return metric

    return objective


def create_sampler(sampler_cfg: Dict[str, Any]):
    stype = sampler_cfg.get("type", "tpe").lower()
    seed = sampler_cfg.get("seed")
    if stype == "tpe":
        return optuna.samplers.TPESampler(seed=seed)
    if stype == "random":
        return optuna.samplers.RandomSampler(seed=seed)
    raise ValueError(f"Unsupported sampler type: {stype}")


def create_pruner(pruner_cfg: Dict[str, Any]):
    if not pruner_cfg:
        return None
    ptype = pruner_cfg.get("type", "median").lower()
    if ptype == "median":
        return optuna.pruners.MedianPruner(
            n_startup_trials=pruner_cfg.get("n_startup_trials", 5),
            n_warmup_steps=pruner_cfg.get("n_warmup_steps", 0),
            interval_steps=pruner_cfg.get("interval_steps", 1),
        )
    if ptype in ("none", "null"):
        return None
    raise ValueError(f"Unsupported pruner type: {ptype}")


def _save_from_axes(obj: Any, output_path: Path) -> None:
    """Helper to save matplotlib objects returned as Axes/ndarray."""
    if hasattr(obj, "get_figure"):
        fig = obj.get_figure()
    elif hasattr(obj, "figure"):
        fig = obj.figure
    elif hasattr(obj, "flat"):
        fig = obj.flat[0].figure
    else:
        raise TypeError(f"Cannot obtain figure from object type: {type(obj)}")

    fig.savefig(output_path, dpi=150, bbox_inches="tight")


def save_plots(study: optuna.Study, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    plots = [
        (plot_optimization_history, "optimization_history.png"),
        (plot_param_importances, "param_importances.png"),
        (plot_slice, "slice.png"),
    ]
    for plot_fn, filename in plots:
        try:
            axes_obj = plot_fn(study)
            _save_from_axes(axes_obj, output_dir / filename)
        except Exception as e:
            logging.warning("Failed to generate %s: %s", filename, e)

    summary_path = output_dir / "study_summary.json"
    best = study.best_trial
    summary = {
        "best_value": best.value,
        "best_trial_number": best.number,
        "best_params": best.params,
        "study_name": study.study_name,
        "direction": study.direction.name,
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger("hpo_driver")

    hpo_cfg = load_hpo_config(Path(args.hpo_config))

    # Apply CLI overrides
    if args.n_trials is not None:
        hpo_cfg["objective"]["max_trials"] = args.n_trials
    if args.timeout is not None:
        hpo_cfg["objective"]["timeout"] = args.timeout
    if args.study_name:
        hpo_cfg["study"]["name"] = args.study_name
    if args.storage:
        hpo_cfg["study"]["storage"] = args.storage

    study_cfg = hpo_cfg["study"]
    objective_cfg = hpo_cfg["objective"]

    sampler = create_sampler(study_cfg.get("sampler", {}))
    pruner = create_pruner(study_cfg.get("pruner", {}))

    resume_flag = study_cfg.get("resume_study")
    if resume_flag is None:
        resume_flag = objective_cfg.get("resume_study", False)

    study = optuna.create_study(
        study_name=study_cfg["name"],
        direction=study_cfg.get("direction", "minimize"),
        storage=study_cfg.get("storage"),
        sampler=sampler,
        pruner=pruner,
        load_if_exists=resume_flag,
    )

    objective = make_objective(hpo_cfg, logger)

    logger.info(
        "Starting Optuna study '%s' with max_trials=%s, timeout=%s",
        study.study_name,
        objective_cfg.get("max_trials"),
        objective_cfg.get("timeout"),
    )
    study.optimize(
        objective,
        n_trials=objective_cfg.get("max_trials"),
        timeout=objective_cfg.get("timeout"),
        gc_after_trial=True,
    )

    logger.info("Study finished: best value = %.4f", study.best_value)
    logger.info("Best params: %s", study.best_params)

    plots_dir = Path(objective_cfg.get("outputs_root", "outputs/hpo")) / "plots" / study.study_name
    save_plots(study, plots_dir)
    logger.info("Saved study plots and summary to %s", plots_dir)


if __name__ == "__main__":
    main()
