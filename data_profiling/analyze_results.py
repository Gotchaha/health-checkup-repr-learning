"""
Analysis and visualization script for data loading profiling results.
Academic research code for generating plots and insights.

Notes:
- Works with results emitted by profile_data_loading.py
  (which already aggregates multi-worker JSONL into per-config JSON files,
   and optionally an all_results.json that maps config-keys to dicts).
- Optional 2nd CLI arg: config YAML path, used to pick the "default" configuration (not used here).
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple


# ------------------------------
# I/O helpers
# ------------------------------

def _is_single_config_payload(obj: dict) -> bool:
    """Heuristic to decide whether a JSON is a single-config payload or an 'all_results' mapping."""
    return isinstance(obj, dict) and ("config" in obj) and ("overall" in obj)


def load_results(results_path: Path) -> Dict:
    """
    Load either:
      1) all_results.json (mapping from 'bs*_nw*' -> config result dict), or
      2) a directory containing per-config 'results_bs*_nw*.json', or
      3) a single per-config results_*.json (wrap into a mapping with computed key).
    """
    if results_path.is_file() and results_path.name.endswith(".json"):
        with open(results_path, "r") as f:
            obj = json.load(f)
        # Case 3: single config file → wrap
        if _is_single_config_payload(obj):
            key = f"bs{obj['config']['batch_size']}_nw{obj['config']['num_workers']}"
            return {key: obj}
        # Case 1: already an "all_results"-style mapping
        return obj

    # Case 2: aggregate directory
    results = {}
    for p in results_path.glob("results_bs*_nw*.json"):
        with open(p, "r") as f:
            d = json.load(f)
        if not _is_single_config_payload(d):
            # Skip unknown payloads quietly
            continue
        key = f"bs{d['config']['batch_size']}_nw{d['config']['num_workers']}"
        results[key] = d
    return results


def load_default_bs_nw_from_config(cfg_path: Optional[Path]) -> Optional[str]:
    """
    If a config YAML is provided, parse default bs/nw to pick a default result key.
    This script expects the calling shell to pass a JSON, so we just return None here.
    """
    return None


def pick_default_config_key(results: Dict) -> Optional[str]:
    """Pick a default key: prefer the one with the highest steady-state throughput."""
    best = None
    best_sps = -1.0
    for k, d in results.items():
        ov = d.get("overall", {})
        # Prefer 'throughput_bps' (samples/sec); fall back to older key if present.
        sps = ov.get("throughput_bps", ov.get("throughput_samples_per_sec", -1.0))
        if sps is not None and sps > best_sps:
            best_sps = sps
            best = k
    return best


# ------------------------------
# Plots
# ------------------------------

def _compute_including_init_throughput(d: dict) -> float:
    """
    Compute including-init throughput as: (#profiled_samples) / (overall_time_s_including_init).
    We only count profiled samples: batch_size * num_iterations (warmup excluded).
    """
    ov = d.get("overall", {})
    cfg = d.get("config", {})
    total_time = ov.get("overall_time_s_including_init", None)
    bs = cfg.get("batch_size", None)
    iters = cfg.get("num_iterations", None)
    if total_time and total_time > 0 and bs and iters:
        return (bs * iters) / total_time
    return float("nan")


def plot_scaling_analysis(results: Dict, output_dir: Path, show: bool = False):
    """Plot throughput vs (bs, nw), both steady-state and including-init."""
    keys = sorted(results.keys())
    if not keys:
        print("No results to plot.")
        return

    bs_list: List[int] = []
    nw_list: List[int] = []
    throughputs: List[float] = []
    throughputs_inc: List[float] = []

    for k in keys:
        d = results[k]
        cfg = d["config"]
        ov = d["overall"]
        bs_list.append(cfg["batch_size"])
        nw_list.append(cfg["num_workers"])
        throughputs.append(ov.get("throughput_bps", ov.get("throughput_samples_per_sec", float("nan"))))
        throughputs_inc.append(_compute_including_init_throughput(d))

    # Line plots
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.title('Throughput (steady-state)')
    plt.plot(range(len(keys)), throughputs, marker='o')
    plt.xticks(range(len(keys)), keys, rotation=30, ha='right')
    plt.ylabel('samples/sec')

    plt.subplot(2, 1, 2)
    plt.title('Throughput (including init/warmup)')
    plt.plot(range(len(keys)), throughputs_inc, marker='o')
    plt.xticks(range(len(keys)), keys, rotation=30, ha='right')
    plt.ylabel('samples/sec')

    plt.tight_layout()
    plt.savefig(output_dir / "throughput_summary.png", dpi=100)
    if show:
        plt.show()
    plt.close()

    # Heatmap for steady-state throughput
    fig, ax4 = plt.subplots(figsize=(8, 5))
    bs_unique = sorted(set(bs_list))
    nw_unique = sorted(set(nw_list))
    mat = np.zeros((len(bs_unique), len(nw_unique)))
    for i, b in enumerate(bs_unique):
        for j, w in enumerate(nw_unique):
            val = 0.0
            for k in range(len(bs_list)):
                if bs_list[k] == b and nw_list[k] == w:
                    val = throughputs[k]
                    break
            mat[i, j] = val
    im = ax4.imshow(mat, aspect='auto', cmap='YlOrRd')
    ax4.set_xticks(range(len(nw_unique))); ax4.set_xticklabels(nw_unique)
    ax4.set_yticks(range(len(bs_unique))); ax4.set_yticklabels(bs_unique)
    ax4.set_xlabel('Number of Workers'); ax4.set_ylabel('Batch Size')
    ax4.set_title('Throughput Heatmap (samples/sec)')
    plt.colorbar(im, ax=ax4)
    for i in range(len(bs_unique)):
        for j in range(len(nw_unique)):
            ax4.text(j, i, f'{mat[i, j]:.0f}', ha="center", va="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(output_dir / "scaling_analysis.png", dpi=100)
    if show: plt.show()
    plt.close(fig)


def plot_bottleneck_breakdown(results: Dict, output_dir: Path,
                              default_key: str, show: bool = False):
    """Plot component-wise breakdown of bottlenecks using a default config."""
    if default_key not in results:
        print(f"Default configuration '{default_key}' not in results; skipping bottleneck plot.")
        return
    data = results[default_key]

    dm = data.get('dataset_metrics', {})
    cm = data.get('collate_metrics', {})

    dataset_total = dm.get('dataset_total_getitem', {}).get('mean_ms', np.nan)
    collate_total = cm.get('collate_total_collate', {}).get('mean_ms', np.nan)

    # Top dataset details (timers only, show mcinfo sub-steps)
    dataset_details: List[Tuple[str, float]] = []
    for k, v in dm.items():
        if isinstance(v, dict) and ('mean_ms' in v) and k.startswith('dataset_mcinfo_'):
            dataset_details.append((k.replace('dataset_mcinfo_', ''), v.get('mean_ms', np.nan)))
    dataset_details = sorted(
        dataset_details,
        key=lambda x: (x[1] if x[1] == x[1] else -1),
        reverse=True
    )[:8]

    # Top collate details (timers only)
    collate_details: List[Tuple[str, float]] = []
    for k, v in cm.items():
        if isinstance(v, dict) and ('mean_ms' in v):
            if k.startswith('collate_text_') or k.startswith('collate_mcc_') or k.startswith('collate_mcm_') or k.startswith('collate_cvr_'):
                collate_details.append((k.replace('collate_', ''), v.get('mean_ms', np.nan)))
    collate_details = sorted(
        collate_details,
        key=lambda x: (x[1] if x[1] == x[1] else -1),
        reverse=True
    )[:8]

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Dataset side
    ax1.set_title('Dataset breakdown (timers)')
    y = [k for k, _ in dataset_details]
    x = [v for _, v in dataset_details]
    ax1.barh(y, x)
    ax1.set_xlabel('mean ms')
    if dataset_total == dataset_total:
        ax1.axvline(dataset_total, color='r', linestyle='--', alpha=0.4, label='total_getitem')
        ax1.legend()

    # Collate side
    ax2.set_title('Collate breakdown (timers)')
    y2 = [k for k, _ in collate_details]
    x2 = [v for _, v in collate_details]
    ax2.barh(y2, x2)
    ax2.set_xlabel('mean ms')
    if collate_total == collate_total:
        ax2.axvline(collate_total, color='r', linestyle='--', alpha=0.4, label='total_collate')
        ax2.legend()

    plt.tight_layout()
    plt.savefig(output_dir / "bottleneck_breakdown.png", dpi=100)
    if show: plt.show()
    plt.close(fig)


def _add(ax, xs, ys, label):
    ax.plot(xs, ys, marker='o', label=label)


def plot_detailed_mcinfo_analysis(results: Dict, output_dir: Path, show: bool = False):
    """
    Plot mcinfo pipeline sub-steps across all configs (timers only).
    We visualize: filter_construction, dataset_filter, to_dict.
    """
    keys = sorted(results.keys())
    if not keys:
        return

    fig, ax2 = plt.subplots(figsize=(11, 6))
    xs = []
    ys_filter = []
    ys_dataset_filter = []
    ys_to_dict = []
    for k in keys:
        dm = results[k].get('dataset_metrics', {})
        if not dm:
            continue
        xs.append(k)
        f = dm.get('dataset_mcinfo_filter_construction', {}).get('mean_ms', np.nan)
        g = dm.get('dataset_mcinfo_dataset_filter', {}).get('mean_ms', np.nan)
        i = dm.get('dataset_mcinfo_to_dict', {}).get('mean_ms', np.nan)
        ys_filter.append(f)
        ys_dataset_filter.append(g)
        ys_to_dict.append(i)

    idxs = list(range(len(xs)))
    _add(ax2, idxs, ys_filter, 'filter_construction')
    _add(ax2, idxs, ys_dataset_filter, 'dataset_filter')
    _add(ax2, idxs, ys_to_dict, 'to_dict')
    ax2.set_xticks(idxs); ax2.set_xticklabels(xs, rotation=30, ha='right')
    ax2.set_ylabel('mean ms')
    ax2.set_title('mcinfo sub-steps across (bs, nw)')
    ax2.legend()

    # Trend line for dataset_filter if enough points
    xs_np = np.arange(len(xs))
    y_main = np.nan_to_num(ys_dataset_filter)
    if xs_np.size >= 2:
        z = np.polyfit(xs_np, y_main, 1)
        p = np.poly1d(z)
        ax2.plot(xs_np, p(xs_np), "r--", alpha=0.5, label=f'Trend: {z[0]:.2f}x + {z[1]:.1f}')
        ax2.legend()

    plt.tight_layout()
    plt.savefig(output_dir / "mcinfo_analysis.png", dpi=100)
    if show: plt.show()
    plt.close(fig)


def _safe_get(d: dict, *keys, default=None):
    """Fetch nested dict key path with a default fallback."""
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur

def _get_ss_throughput(d: dict) -> float:
    """Steady-state throughput (samples/sec), with backward-compatible fallback."""
    ov = d.get("overall", {})
    val = ov.get("throughput_bps", ov.get("throughput_samples_per_sec", None))
    return float(val) if (val is not None) else float("nan")

def _get_inc_throughput(d: dict) -> float:
    """Including-init throughput derived as (#profiled_samples) / overall_time."""
    ov = d.get("overall", {})
    cfg = d.get("config", {})
    total_time = ov.get("overall_time_s_including_init", None)
    bs = cfg.get("batch_size", None)
    iters = cfg.get("num_iterations", None)
    if total_time and total_time > 0 and bs and iters:
        return (bs * iters) / total_time
    return float("nan")

def save_summary_tables(results: Dict, output_dir: Path) -> None:
    """
    Emit two CSVs:
      - summary_per_config.csv: per-config headline stats（throughput, total time, mean value of key sub-steps）
      - scaling_efficiency.csv: relative to speedup/efficiency (use nw=0 of the same batch_size as basis)
    """
    import csv
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1) per-config summary
    summary_csv = output_dir / "summary_per_config.csv"
    fields = [
        "key", "batch_size", "num_workers",
        "steady_throughput_sps", "including_init_throughput_sps",
        "mean_batch_time_ms",
        # dataset
        "dataset_total_getitem_ms",
        "mcinfo_filter_ms", "mcinfo_dataset_filter_ms", "mcinfo_to_dict_ms",
        # collate
        "collate_total_ms",
        "tests_flattening_ms", "category_mapping_ms", "mcm_masking_ms",
        "mcc_total_ms", "mcc_sampling_ms", "text_tokenization_ms",
        # optional text duplication (single-worker fallback only)
        "text_duplication_rate",
        # sampler headlines
        "sampler_total_persons", "sampler_total_exams", "sampler_avg_exams_per_person",
    ]
    with open(summary_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for k, d in sorted(results.items()):
            cfg, ov = d.get("config", {}), d.get("overall", {})
            dm, cm = d.get("dataset_metrics", {}), d.get("collate_metrics", {})
            row = dict(
                key=k,
                batch_size=cfg.get("batch_size"),
                num_workers=cfg.get("num_workers"),
                steady_throughput_sps=_get_ss_throughput(d),
                including_init_throughput_sps=_get_inc_throughput(d),
                mean_batch_time_ms=ov.get("mean_batch_time_ms", float("nan")),
                dataset_total_getitem_ms=_safe_get(dm, "dataset_total_getitem", "mean_ms", default=float("nan")),
                mcinfo_filter_ms=_safe_get(dm, "dataset_mcinfo_filter_construction", "mean_ms", default=float("nan")),
                mcinfo_dataset_filter_ms=_safe_get(dm, "dataset_mcinfo_dataset_filter", "mean_ms", default=float("nan")),
                mcinfo_to_dict_ms=_safe_get(dm, "dataset_mcinfo_to_dict", "mean_ms", default=float("nan")),
                collate_total_ms=_safe_get(cm, "collate_total_collate", "mean_ms", default=float("nan")),
                tests_flattening_ms=_safe_get(cm, "collate_tests_flattening", "mean_ms", default=float("nan")),
                category_mapping_ms=_safe_get(cm, "collate_category_mapping", "mean_ms", default=float("nan")),
                mcm_masking_ms=_safe_get(cm, "collate_mcm_masking", "mean_ms", default=float("nan")),
                mcc_total_ms=_safe_get(cm, "collate_mcc_total", "mean_ms", default=float("nan")),
                mcc_sampling_ms=_safe_get(cm, "collate_mcc_sampling_loop", "mean_ms", default=float("nan")),
                text_tokenization_ms=_safe_get(cm, "collate_text_tokenization", "mean_ms", default=float("nan")),
                text_duplication_rate=cm.get("text_duplication_rate", float("nan")),
                sampler_total_persons=_safe_get(d, "sampler_metrics", "total_persons", default=""),
                sampler_total_exams=_safe_get(d, "sampler_metrics", "total_exams", default=""),
                sampler_avg_exams_per_person=_safe_get(d, "sampler_metrics", "avg_exams_per_person", default=""),
            )
            w.writerow(row)

    # ---- 2) scaling efficiency per bs
    eff_csv = output_dir / "scaling_efficiency.csv"
    # aggregate by bs
    per_bs = {}
    for k, d in results.items():
        bs = _safe_get(d, "config", "batch_size")
        nw = _safe_get(d, "config", "num_workers")
        sps = _get_ss_throughput(d)
        if bs is None or nw is None or not np.isfinite(sps):
            continue
        per_bs.setdefault(bs, {})[nw] = sps

    # write to table
    fields = ["batch_size", "num_workers", "throughput_sps", "speedup_vs_nw0", "efficiency_speedup_per_worker"]
    with open(eff_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for bs, dmap in sorted(per_bs.items()):
            base = dmap.get(0, np.nan)
            for nw, sps in sorted(dmap.items()):
                speedup = (sps / base) if (np.isfinite(base) and base > 0) else np.nan
                eff = (speedup / nw) if (nw and np.isfinite(speedup)) else (1.0 if nw == 0 else np.nan)
                w.writerow(dict(
                    batch_size=bs, num_workers=nw,
                    throughput_sps=sps,
                    speedup_vs_nw0=speedup,
                    efficiency_speedup_per_worker=eff
                ))

def generate_objective_report(results: Dict, output_dir: Path, topk: int = 8) -> None:
    """
    Emit a concise, objective Markdown report:
      - headline summary table (derived from CSV)
      - top-K dataset/collate timers (mean value, global aggregation by default key）
      - optional counters snapshot (e.g. mcc_num_cells_count.mean)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    md = []

    # 1) Headline per-config
    md.append("# Objective Profiling Report")
    md.append("")
    md.append("This report summarizes key steady-state and component timings. See CSVs for full details:")
    md.append("- `summary_per_config.csv`")
    md.append("- `scaling_efficiency.csv`")
    md.append("")

    # 2) Choose a default config (by max steady throughput) to expand
    def_key = None
    best_sps = -1.0
    for k, d in results.items():
        sps = _get_ss_throughput(d)
        if np.isfinite(sps) and sps > best_sps:
            best_sps, def_key = sps, k
    if def_key is None and results:
        def_key = next(iter(results.keys()))
    md.append(f"**Default config (by max steady throughput):** `{def_key}`")
    md.append("")

    if def_key and def_key in results:
        d = results[def_key]
        dm, cm = d.get("dataset_metrics", {}), d.get("collate_metrics", {})
        # dataset top-K
        ds_items = []
        for k, v in dm.items():
            if isinstance(v, dict) and "mean_ms" in v:
                ds_items.append((k, v["mean_ms"]))
        ds_items.sort(key=lambda x: (x[1] if np.isfinite(x[1]) else -1), reverse=True)
        md.append("## Dataset — Top timers (mean_ms)")
        for k, val in ds_items[:topk]:
            md.append(f"- {k}: {val:.3f}")
        if not ds_items:
            md.append("_no dataset timers found_")
        md.append("")

        # collate top-K
        co_items = []
        for k, v in cm.items():
            if isinstance(v, dict) and "mean_ms" in v:
                co_items.append((k, v["mean_ms"]))
        co_items.sort(key=lambda x: (x[1] if np.isfinite(x[1]) else -1), reverse=True)
        md.append("## Collate — Top timers (mean_ms)")
        for k, val in co_items[:topk]:
            md.append(f"- {k}: {val:.3f}")
        if not co_items:
            md.append("_no collate timers found_")
        md.append("")

        # optional counters snapshot
        cnt_key = "collate_mcc_num_cells_count"
        cnt = cm.get(cnt_key, {})
        if isinstance(cnt, dict) and "mean" in cnt:
            md.append("## Selected Counters")
            md.append(f"- {cnt_key}.mean: {cnt['mean']:.3f}  (count={cnt.get('count','')})")
            md.append("")

        # optional duplication hints (only in single-process fallback)
        dup = cm.get("text_duplication_rate", None)
        if dup is not None and np.isfinite(dup):
            md.append("## Text Duplication (single-worker hint)")
            md.append(f"- text_duplication_rate: {dup:.3f}")
            md.append(f"- text_unique_count: {cm.get('text_unique_count', 'NA')}")
            md.append(f"- text_total_occurrences: {cm.get('text_total_occurrences', 'NA')}")
            md.append("")

    (output_dir / "REPORT.md").write_text("\n".join(md), encoding="utf-8")


def plot_scaling_efficiency(results: Dict, output_dir: Path, show: bool = False) -> None:
    """
    Plot speedup & efficiency change by num_workers (branching by batch_size)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    # Aggregate
    per_bs = {}
    for k, d in results.items():
        bs = _safe_get(d, "config", "batch_size")
        nw = _safe_get(d, "config", "num_workers")
        sps = _get_ss_throughput(d)
        if bs is None or nw is None or not np.isfinite(sps):
            continue
        per_bs.setdefault(bs, {})[nw] = sps

    # speedup
    plt.figure(figsize=(9, 5))
    for bs, dmap in sorted(per_bs.items()):
        nws = sorted(dmap.keys())
        base = dmap.get(0, np.nan)
        ys = [(dmap[n] / base) if (np.isfinite(base) and base > 0) else np.nan for n in nws]
        plt.plot(nws, ys, marker='o', label=f'bs={bs}')
    plt.xlabel('num_workers'); plt.ylabel('speedup vs nw=0'); plt.title('Scaling Speedup')
    plt.legend()
    plt.tight_layout(); plt.savefig(output_dir / "scaling_speedup.png", dpi=100)
    if show: plt.show(); plt.close()

    # efficiency
    plt.figure(figsize=(9, 5))
    for bs, dmap in sorted(per_bs.items()):
        nws = sorted(dmap.keys())
        base = dmap.get(0, np.nan)
        ys = []
        for n in nws:
            if n == 0:
                ys.append(1.0)
            else:
                sp = (dmap[n] / base) if (np.isfinite(base) and base > 0) else np.nan
                ys.append(sp / n if np.isfinite(sp) else np.nan)
        plt.plot(nws, ys, marker='o', label=f'bs={bs}')
    plt.xlabel('num_workers'); plt.ylabel('efficiency (speedup / nw)'); plt.title('Scaling Efficiency')
    plt.ylim(0, 1.2)
    plt.legend()
    plt.tight_layout(); plt.savefig(output_dir / "scaling_efficiency.png", dpi=100)
    if show: plt.show()
    plt.close()

def plot_text_duplication_vs_batchsize(results: Dict, output_dir: Path, show: bool = False) -> None:
    """
    Plot text repetition rate vs batch_size only for single-process fallback (or when we write these keys for multi-process aggregation in the future).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    xs, ys, cs = [], [], []
    for k, d in results.items():
        cm = d.get("collate_metrics", {})
        dup = cm.get("text_duplication_rate", None)
        if dup is None or not np.isfinite(dup):  # only for existing points
            continue
        xs.append(_safe_get(d, "config", "batch_size"))
        ys.append(float(dup))
        cs.append(_safe_get(d, "config", "num_workers"))

    if not xs:
        return

    plt.figure(figsize=(8, 5))
    sc = plt.scatter(xs, ys, c=cs)
    plt.xlabel('batch_size'); plt.ylabel('text_duplication_rate')
    plt.title('Text Duplication vs Batch Size (points colored by num_workers)')
    plt.colorbar(sc, label='num_workers')
    plt.tight_layout(); plt.savefig(output_dir / "text_duplication_vs_bs.png", dpi=100)
    if show: plt.show()
    plt.close()

def plot_mcc_cells_vs_batchsize(results: Dict, output_dir: Path, show: bool = False) -> None:
    """
    Plot the mean number of numeric cells masked by MCC (collate_mcc_num_cells_count.mean) per batch versus bs.
Skip if this count does not exist.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    xs, ys, cs = [], [], []
    for k, d in results.items():
        cm = d.get("collate_metrics", {})
        stat = cm.get("collate_mcc_num_cells_count", {})
        mean_val = stat.get("mean", None) if isinstance(stat, dict) else None
        if mean_val is None or not np.isfinite(mean_val):
            continue
        xs.append(_safe_get(d, "config", "batch_size"))
        ys.append(float(mean_val))
        cs.append(_safe_get(d, "config", "num_workers"))

    if not xs:
        return

    plt.figure(figsize=(8, 5))
    sc = plt.scatter(xs, ys, c=cs)
    plt.xlabel('batch_size'); plt.ylabel('mean MCC-masked numeric cells per batch')
    plt.title('MCC Masked Cells vs Batch Size (points colored by num_workers)')
    plt.colorbar(sc, label='num_workers')
    plt.tight_layout(); plt.savefig(output_dir / "mcc_cells_vs_bs.png", dpi=100)
    if show: plt.show()
    plt.close()

def plot_collate_vs_dataset_breakdown(results: Dict, output_dir: Path, show: bool = False) -> None:
    """
    Make a side-by-side histogram of mean_ms for dataset_total_getitem vs collate_total_collate for all configurations.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    keys = sorted(results.keys())
    if not keys:
        return
    labels, ds_vals, co_vals = [], [], []
    for k in keys:
        d = results[k]
        dm, cm = d.get("dataset_metrics", {}), d.get("collate_metrics", {})
        ds = _safe_get(dm, "dataset_total_getitem", "mean_ms", default=np.nan)
        co = _safe_get(cm, "collate_total_collate", "mean_ms", default=np.nan)
        if not (np.isfinite(ds) or np.isfinite(co)):
            continue
        labels.append(k); ds_vals.append(ds); co_vals.append(co)

    if not labels:
        return

    x = np.arange(len(labels))
    width = 0.40
    plt.figure(figsize=(max(10, 0.5*len(labels)), 5))
    plt.bar(x - width/2, ds_vals, width, label='dataset_total_getitem')
    plt.bar(x + width/2, co_vals, width, label='collate_total_collate')
    plt.xticks(x, labels, rotation=30, ha='right')
    plt.ylabel('mean ms'); plt.title('Dataset vs Collate — Total Times')
    plt.legend()
    plt.tight_layout(); plt.savefig(output_dir / "dataset_vs_collate_totals.png", dpi=100)
    if show: plt.show()
    plt.close()



# ------------------------------
# Entry
# ------------------------------

def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_results.py <path/to/all_results.json or results_dir> [optional_config_yaml]")
        sys.exit(1)

    results_path = Path(sys.argv[1])

    results = load_results(results_path)
    if not results:
        print("No valid results found.")
        sys.exit(1)

    results_dir = results_path.parent if results_path.is_file() else results_path
    # Analysis directory for plots/markdown output
    analysis_dir = results_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)


    default_key = pick_default_config_key(results) or next(iter(results.keys()), None)
    print(f"Default config: {default_key}")

    # Print including-init throughput if derivable
    try:
        if default_key and default_key in results:
            inc_thr = _compute_including_init_throughput(results[default_key])
            if inc_thr == inc_thr:  # not NaN
                print(f"Including-init throughput: {inc_thr:.2f} samples/s")
            ss_bt = results[default_key].get("overall", {}).get("mean_batch_time_ms", float("nan"))
            if ss_bt == ss_bt:
                print(f"Steady-state mean batch time: {ss_bt:.2f} ms")
    except Exception:
        pass

    # Plots
    print("\nGenerating visualizations...")
    plot_scaling_analysis(results, analysis_dir, show=False)
    plot_bottleneck_breakdown(results, analysis_dir, default_key, show=False)
    plot_detailed_mcinfo_analysis(results, analysis_dir, show=False)
    plot_scaling_efficiency(results, analysis_dir, show=False)
    plot_collate_vs_dataset_breakdown(results, analysis_dir, show=False)
    plot_text_duplication_vs_batchsize(results, analysis_dir, show=False)
    plot_mcc_cells_vs_batchsize(results, analysis_dir, show=False)

    # Report
    print("\nGenerating tables & reports")
    save_summary_tables(results, analysis_dir)
    generate_objective_report(results, analysis_dir, topk=8)


    print(f"\nAnalysis complete! Results saved to: {analysis_dir}")


if __name__ == "__main__":
    main()
