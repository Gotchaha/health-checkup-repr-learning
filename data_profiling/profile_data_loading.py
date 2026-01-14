"""
Main data loading profiling script for SSL pretraining.
Academic research code for identifying and quantifying bottlenecks.
"""

import os
import sys
import time
import math
import glob
import json
import yaml
import torch
import random
import atexit
import signal
import numpy as np
from pathlib import Path
from datetime import datetime
from functools import partial
from typing import Dict, Any, Tuple
import multiprocessing as mp

# Add parent directory to path for imports (assuming this file lives in data_profiling/)
sys.path.append(str(Path(__file__).parent.parent))

# Import profiled components
from profiled_dataset import ProfiledHealthExamDataset
from profiled_sampler import ProfiledInfinitePersonBatchSampler
from profiled_collate import (
    profiled_collate_exams,
    get_collate_profiling_summary,
    get_collate_counter_summary,
    collate_profiler,
)

from src.utils.multiprocessing import worker_init_with_cleanup
from src.models import create_embedders_from_config

# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------

def load_config(config_path: str = str(Path(__file__).parent / "data_profiling_config.yaml")) -> Dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

# ----------------------------------------------------------------------
# Worker seeding, cleanup & thread noise control
# ----------------------------------------------------------------------

def _set_thread_env():
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    try:
        torch.set_num_threads(1)
    except Exception:
        pass

def prof_worker_init(worker_id: int, base_seed: int = 42, finalize_arrow: bool = True) -> None:
    """
    PyTorch DataLoader worker_init_fn. Seeds numpy/random/torch and installs cleanups.
    """
    # 1) per-worker seeds
    seed = int(base_seed) + int(worker_id)
    try:
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        try:
            torch.set_num_threads(1)  # reduce BLAS jitter
        except Exception:
            pass
    except Exception:
        pass

    # 2) proper Arrow cleanup on exit (avoid mmap handle leaks)
    def _arrow_finalize():
        try:
            import pyarrow
            pyarrow.total_allocated_bytes()  # touch to ensure module loaded
        except Exception:
            return
        try:
            import gc
            gc.collect()
        except Exception:
            pass

    atexit.register(_arrow_finalize)

    # 3) handle SIGTERM to flush logs quickly
    def _on_term(signum, frame):
        try:
            _arrow_finalize()
        finally:
            os._exit(0)

    try:
        signal.signal(signal.SIGTERM, _on_term)
    except Exception:
        pass


# ----------------------------------------------------------------------
# JSONL aggregation (multi-worker) + normalization (single-worker)
# ----------------------------------------------------------------------

def _aggregate_jsonl_metrics(log_dir: Path, prefix: str, namespace: str) -> Dict[str, Dict[str, float]]:
    """
    Aggregate per-worker JSONL dumps and emit namespaced metrics.
    `prefix` should be like "dataset-<run_id>" or "collate-<run_id>".
    Output keys are f"{namespace}_<metric>" or f"{namespace}_<cat>_<metric>" for timers,
    and f"{namespace}_{metric}_count" / f"{namespace}_{cat}_{metric}_count" for counters.
    """
    import numpy as _np

    simple_raw, detailed_raw = {}, {}
    simple_cnt_raw, detailed_cnt_raw = {}, {}

    for p in glob.glob(str(log_dir / f"{prefix}*.jsonl")):
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)

                # timers
                for k, arr in payload.get("simple", {}).items():
                    simple_raw.setdefault(k, []).extend(arr or [])
                for cat, m in payload.get("detailed", {}).items():
                    d = detailed_raw.setdefault(cat, {})
                    for mk, arr in m.items():
                        d.setdefault(mk, []).extend(arr or [])

                # counters
                for k, arr in payload.get("simple_counters", {}).items():
                    simple_cnt_raw.setdefault(k, []).extend(arr or [])
                for cat, m in payload.get("detailed_counters", {}).items():
                    d = detailed_cnt_raw.setdefault(cat, {})
                    for mk, arr in m.items():
                        d.setdefault(mk, []).extend(arr or [])

    summary: Dict[str, Dict[str, float]] = {}

    # timers -> *_ms stats
    for k, arr in simple_raw.items():
        arr = _np.asarray(arr, dtype=float)
        if arr.size:
            summary[f"{namespace}_{k}"] = {
                "mean_ms": float(arr.mean()*1000),
                "std_ms":  float(arr.std()*1000),
                "min_ms":  float(arr.min()*1000),
                "max_ms":  float(arr.max()*1000),
                "p50_ms":  float(_np.percentile(arr, 50)*1000),
                "p95_ms":  float(_np.percentile(arr, 95)*1000),
                "p99_ms":  float(_np.percentile(arr, 99)*1000),
                "count":   int(arr.size),
            }
    for cat, d in detailed_raw.items():
        for mname, arr in d.items():
            arr = _np.asarray(arr, dtype=float)
            if arr.size:
                summary[f"{namespace}_{cat}_{mname}"] = {
                    "mean_ms": float(arr.mean()*1000),
                    "std_ms":  float(arr.std()*1000),
                    "min_ms":  float(arr.min()*1000),
                    "max_ms":  float(arr.max()*1000),
                    "p50_ms":  float(_np.percentile(arr, 50)*1000),
                    "p95_ms":  float(_np.percentile(arr, 95)*1000),
                    "p99_ms":  float(_np.percentile(arr, 99)*1000),
                    "count":   int(arr.size),
                }

    # counters -> *_count stats
    def _cnt_stats(a: _np.ndarray) -> Dict[str, float]:
        return {
            "sum":   float(a.sum()),
            "mean":  float(a.mean()),
            "p50":   float(_np.percentile(a, 50)),
            "p95":   float(_np.percentile(a, 95)),
            "p99":   float(_np.percentile(a, 99)),
            "max":   float(a.max()),
            "count": int(a.size),
        }

    for k, arr in simple_cnt_raw.items():
        arr = _np.asarray(arr, dtype=float)
        if arr.size:
            summary[f"{namespace}_{k}_count"] = _cnt_stats(arr)

    for cat, d in detailed_cnt_raw.items():
        for mname, arr in d.items():
            arr = _np.asarray(arr, dtype=float)
            if arr.size:
                summary[f"{namespace}_{cat}_{mname}_count"] = _cnt_stats(arr)

    return summary


def _normalize_single_worker_metrics(metrics: Dict[str, Dict[str, float]], namespace: str) -> Dict[str, Dict[str, float]]:
    """
    Convert in-process fallback keys like 'simple_x' / 'detailed_cat_m' into
    'namespace_x' / 'namespace_cat_m'. Deterministic renaming, no guessing。
    """
    out = {}
    for k, v in (metrics or {}).items():
        if k.startswith("simple_"):
            out[f"{namespace}_{k[len('simple_'):] }"] = v
        elif k.startswith("detailed_"):
            out[f"{namespace}_{k[len('detailed_'):] }"] = v
        else:
            # non-timing stats (e.g., text_duplication_rate) pass through as-is
            out[k] = v
    return out

# ----------------------------------------------------------------------
# Main profiling entry
# ----------------------------------------------------------------------

def profile_data_loading(config: Dict):
    print("="*60)
    print("DATA LOADING PROFILING")
    print("="*60)

    _set_thread_env()

    device = 'cpu'  # data loading happens on CPU
    profiling_config = config['profiling']

    # Create experiment dir
    output_dir = Path(config['output']['results_dir'])
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_dir = output_dir / f"profile_{timestamp}"
    experiment_dir.mkdir(parents=True, exist_ok=True)
    (experiment_dir / "worker_logs").mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {experiment_dir}")

    # Iterate configurations
    batch_sizes = profiling_config.get('batch_sizes', [32])
    num_workers_list = profiling_config.get('num_workers_list', [0, 2, 4])

    all_results = {}

    for batch_size in batch_sizes:
        for num_workers in num_workers_list:
            print(f"\n{'='*40}")
            print(f"Testing: batch_size={batch_size}, num_workers={num_workers}")
            print(f"{'='*40}")

            results = profile_configuration(
                config=config,
                batch_size=batch_size,
                num_workers=num_workers,
                num_iterations=profiling_config.get('num_iterations', 100),
                warmup_iterations=profiling_config.get('warmup_iterations', 10),
                experiment_dir=experiment_dir,
                timestamp=timestamp,
            )

            all_results[f"bs{batch_size}_nw{num_workers}"] = results

            # Save intermediate result
            with open(experiment_dir / f"results_bs{batch_size}_nw{num_workers}.json", 'w') as f:
                json.dump(results, f, indent=2, default=str)

    # Save all
    with open(experiment_dir / "all_results.json", 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    # Summary
    generate_summary_report(all_results, experiment_dir, config)


# ----------------------------------------------------------------------
# DataLoader construction (single loader; warmup via logging gate)
# ----------------------------------------------------------------------

def _build_dataloader(
    config: Dict,
    batch_size: int,
    num_workers: int,
    enable_profiling: bool,
    start_logging_after_batches: int,
    worker_seed_base: int,
) -> Tuple[Any, Any, torch.utils.data.DataLoader]:

    # Dataset with profiling; dataset gate is in items
    start_after_items = start_logging_after_batches * batch_size
    ds = ProfiledHealthExamDataset(
        split_name='train_ssl',
        mcinfo_dir=config['data']['mcinfo_dir'],
        demographics_path=config['data']['demographics_path'],
        use_result=config['data']['use_result'],
        result_path=config['data']['result_path'],
        use_interview=config['data']['use_interview'],
        interview_path=config['data']['interview_path'],
        use_pretokenized_result=config['data'].get('use_pretokenized_result', False),
        result_tokenized_path=config['data'].get('result_tokenized_path'),
        enable_profiling=enable_profiling,
        detailed_mcinfo=config['profiling'].get('mcinfo', {}).get('enabled', True),
        start_logging_after_items=start_after_items,
    )

    # Sampler
    sp = ProfiledInfinitePersonBatchSampler(
        manifest_path=config['data']['manifest_path'],
        batch_size=batch_size,
        drop_last=False,
        enable_profiling=enable_profiling,
        start_logging_after_batches=start_logging_after_batches,
    )

    embedders = create_embedders_from_config(config['model']['embedders'], device='cpu')
    code_embedder = embedders.categorical
    text_embedder = embedders.text

    # Collate closure (inject profiling flags & gate in batches)
    def _collate(batch):
        return profiled_collate_exams(
            batch=batch,
            code_embedder=code_embedder,
            text_embedder=text_embedder,
            config=config,
            device='cpu',
            enable_profiling=enable_profiling,
            detailed_mcc=config['profiling'].get('mcc', {}).get('enabled', True),
            detailed_text=config['profiling'].get('text', {}).get('enabled', True),
            start_logging_after_batches=start_logging_after_batches,
        )

    # Worker init
    init_fn = partial(prof_worker_init, base_seed=worker_seed_base)

    dl = torch.utils.data.DataLoader(
        ds,
        batch_sampler=sp,
        num_workers=num_workers,
        pin_memory=False,
        collate_fn=_collate,
        worker_init_fn=init_fn,
        persistent_workers=(num_workers > 0),
        prefetch_factor=max(2, config['profiling'].get('prefetch_factor', 2)) if num_workers > 0 else None,
    )
    return ds, sp, dl

# ----------------------------------------------------------------------
# One configuration run
# ----------------------------------------------------------------------

def profile_configuration(
    config: Dict,
    batch_size: int,
    num_workers: int,
    num_iterations: int,
    warmup_iterations: int,
    experiment_dir: Path,
    timestamp: str,
) -> Dict[str, Any]:

    # Unique run id and log dir
    run_id = f"bs{batch_size}_nw{num_workers}_{timestamp}"
    os.environ['DL_PROFILE_DIR'] = str(experiment_dir / "worker_logs")
    os.environ['DL_PROFILE_RUN'] = run_id
    os.environ['DL_PROFILE_FLUSH_EVERY'] = str(config['profiling'].get('flush_every', 1))

    # Single DataLoader, but gate logging for warmup
    base_seed = int(config.get('seed', 42))
    # Warmup batches are split across workers; only after this gate do we record.
    start_logging_after_batches = math.ceil(warmup_iterations / max(1, num_workers))

    ds, sp, dl = _build_dataloader(
        config=config,
        batch_size=batch_size,
        num_workers=num_workers,
        enable_profiling=True,
        start_logging_after_batches=start_logging_after_batches,
        worker_seed_base=base_seed,
    )

    # Iterate (warmup + profile) on the same loader
    data_iter = iter(dl)
    total_iters = warmup_iterations + num_iterations

    batch_times = []
    t0_overall = time.perf_counter()
    for i in range(total_iters):
        t0 = time.perf_counter()
        _ = next(data_iter)
        dt = time.perf_counter() - t0

        # Only after warmup do we record batch time
        if i >= warmup_iterations:
            batch_times.append(dt)
        if (i+1) % max(1, total_iters // max(1, config.get('log_every', 10))) == 0:
            phase = "WARMUP" if i < warmup_iterations else "PROFILE"
            print(f"  {phase} {i+1}/{total_iters}: {dt*1000:.2f} ms")

    overall_time = time.perf_counter() - t0_overall

    # ---- Collect results ----
    # Sampler metrics (lives in main process); fetch BEFORE releasing sampler.
    sampler_metrics = sp.get_stats()

    # Take single-process snapshots BEFORE closing ds (closing will clear in-memory buffers).
    ds_timers_snapshot = ds.get_profiling_summary()
    ds_counters_snapshot = getattr(ds, "get_counter_summary", lambda: {})()
    collate_timers_snapshot = get_collate_profiling_summary()
    try:
        collate_counters_snapshot = get_collate_counter_summary()
    except Exception:
        collate_counters_snapshot = {}

    # Force a final flush for collate so tail batches ( < flush_every ) are persisted.
    try:
        from profiled_collate import _flush_collate_metrics_to_file  # local import on demand
        _flush_collate_metrics_to_file(force=True)
    except Exception:
        pass

    # Release iterator and DataLoader; then close dataset to trigger final JSONL flush.
    del data_iter
    try:
        # optional internal iterator clear; safe no-op if missing
        dl._iterator = None  # type: ignore[attr-defined]
    except Exception:
        pass
    del dl, sp
    try:
        ds.close()
    except Exception:
        pass

    # Dataset & collate metrics aggregation:
    # - Prefer multi-worker JSONL aggregation (works for nw=0/2/4 uniformly)
    # - If empty (e.g., gate prevented flush), fall back to in-process snapshots
    log_dir = Path(os.environ['DL_PROFILE_DIR'])

    dataset_metrics = _aggregate_jsonl_metrics(log_dir, prefix=f"dataset-{run_id}", namespace="dataset")
    collate_metrics  = _aggregate_jsonl_metrics(log_dir, prefix=f"collate-{run_id}",  namespace="collate")

    if not dataset_metrics:
        dataset_metrics = _normalize_single_worker_metrics(
            {**ds_timers_snapshot, **ds_counters_snapshot}, namespace="dataset"
        )

    if not collate_metrics:
        collate_metrics = {**collate_timers_snapshot, **collate_counters_snapshot}

    # Overall numbers
    mean_bt = float(np.mean(batch_times)) if batch_times else 0.0
    results = {
        "config": {
            "batch_size": batch_size,
            "num_workers": num_workers,
            "warmup_iterations": warmup_iterations,
            "num_iterations": num_iterations,
            "run_id": run_id,
        },
        "overall": {
            "mean_batch_time_ms": mean_bt * 1000,
            "throughput_bps": (batch_size / mean_bt) if mean_bt > 0 else 0.0,
            "overall_time_s_including_init": overall_time,
        },
        "dataset_metrics": dataset_metrics,
        "collate_metrics": collate_metrics,
        "sampler_metrics": sampler_metrics,
    }

    return results



# ----------------------------------------------------------------------
# Simple textual summary
# ----------------------------------------------------------------------

def generate_summary_report(all_results: Dict, output_dir: Path, config: Dict):
    lines = []
    lines.append("# Data Loading Profiling Summary\n")
    for key, res in all_results.items():
        bs = res["config"]["batch_size"]
        nw = res["config"]["num_workers"]
        mean_bt = res["overall"]["mean_batch_time_ms"]
        thr = res["overall"]["throughput_bps"]
        lines.append(f"- **bs={bs}, nw={nw}**: mean batch {mean_bt:.2f} ms, throughput {thr:.2f} samples/s")
    (output_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

if __name__ == "__main__":
    cfg_path = str(Path(__file__).parent / "data_profiling_config.yaml")
    if not Path(cfg_path).exists():
        print(f"Error: Configuration file '{cfg_path}' not found!")
        sys.exit(1)

    config = load_config(cfg_path)

    # Global seeding (sampler randomness lives in main process)
    base_seed = int(config.get('seed', 42))
    random.seed(base_seed)
    np.random.seed(base_seed)
    torch.manual_seed(base_seed)

    profile_data_loading(config)
