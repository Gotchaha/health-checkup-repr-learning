#!/usr/bin/env python
"""
Analyze and visualize profiling results from SSL training.
Generates matplotlib plots and summary reports.
"""

import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List
import pandas as pd


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Analyze SSL profiling results')
    parser.add_argument('profile_dir', type=str,
                       help='Directory containing profiling results')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for plots (default: same as profile_dir)')
    parser.add_argument('--format', type=str, default='png',
                       choices=['png', 'pdf', 'svg'],
                       help='Output format for plots')
    return parser.parse_args()


def load_results(profile_dir: Path) -> Dict:
    """Load profiling results from directory."""
    results_file = profile_dir / 'profile_results.json'
    
    if not results_file.exists():
        # Try loading individual batch results
        results = {}
        for batch_file in profile_dir.glob('profile_batch_*.json'):
            with open(batch_file, 'r') as f:
                batch_size = batch_file.stem.split('_')[-1]
                results[f'batch_{batch_size}'] = json.load(f)
    else:
        with open(results_file, 'r') as f:
            results = json.load(f)
    
    return results


def plot_time_breakdown(results: Dict, output_dir: Path, format: str = 'png'):
    """Create time breakdown visualization."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('SSL Training Performance Breakdown', fontsize=16, fontweight='bold')
    
    # Get all batch sizes
    batch_sizes = sorted(
        [k for k in results.keys() if k.startswith('batch_')],
        key=lambda k: int(k.split('_')[1])
    )
    
    # 1. Component time comparison across batch sizes
    ax = axes[0, 0]
    components = ['data_loading', 'embedding_text', 'forward_cross_attention', 
                 'forward_unified_transformer', 'backward_pass', 'optimizer_step']
    
    x = np.arange(len(components))
    width = 0.25
    
    for i, batch_key in enumerate(batch_sizes[:3]):  # Show up to 3 batch sizes
        batch_data = results[batch_key]['time_breakdown']
        values = [batch_data.get(comp, {}).get('mean_ms', 0) for comp in components]
        batch_size = batch_key.split('_')[1]
        ax.bar(x + i*width, values, width, label=f'Batch {batch_size}')
    
    ax.set_xlabel('Component', fontweight='bold')
    ax.set_ylabel('Time (ms)', fontweight='bold')
    ax.set_title('Component Time by Batch Size')
    ax.set_xticks(x + width)
    ax.set_xticklabels([c.replace('_', '\n') for c in components], rotation=0, ha='center')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Percentage breakdown pie chart (for first batch size)
    ax = axes[0, 1]
    first_batch = results[batch_sizes[0]]
    bottlenecks = first_batch.get('bottlenecks', [])
    
    if bottlenecks:
        labels = [b['component'].replace('_', ' ').title() for b in bottlenecks[:6]]
        sizes = [b['percent'] for b in bottlenecks[:6]]
        
        # Add "Other" category if needed
        total_shown = sum(sizes)
        if total_shown < 95:
            labels.append('Other')
            sizes.append(100 - total_shown)
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
        explode = [0.05] * len(labels)  # Slightly explode all slices
        
        ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90,
               colors=colors, explode=explode)
        ax.set_title(f'Time Distribution (Batch {batch_sizes[0].split("_")[1]})')
    
    # 3. Scaling analysis - step time vs batch size
    ax = axes[1, 0]
    batch_nums = []
    step_times = []
    
    for batch_key in batch_sizes:
        batch_size = int(batch_key.split('_')[1])
        batch_nums.append(batch_size)
        step_time = results[batch_key]['time_breakdown'].get('total_step', {}).get('mean_ms', 0)
        step_times.append(step_time)
    
    ax.plot(batch_nums, step_times, 'o-', linewidth=2, markersize=8, color='#2E86AB')
    ax.set_xlabel('Batch Size', fontweight='bold')
    ax.set_ylabel('Step Time (ms)', fontweight='bold')
    ax.set_title('Step Time Scaling')
    ax.grid(True, alpha=0.3)
    
    # Add throughput on secondary y-axis
    ax2 = ax.twinx()
    throughputs = [batch_size / (time / 1000) for batch_size, time in zip(batch_nums, step_times)]
    ax2.plot(batch_nums, throughputs, 's-', linewidth=2, markersize=8, color='#A23B72', alpha=0.7)
    ax2.set_ylabel('Throughput (samples/sec)', fontweight='bold', color='#A23B72')
    ax2.tick_params(axis='y', labelcolor='#A23B72')
    
    # 4. Component scaling - show how each component scales
    ax = axes[1, 1]
    
    # Select key components to show scaling
    key_components = ['data_loading', 'forward_unified_transformer', 'backward_pass']
    
    for component in key_components:
        component_times = []
        for batch_key in batch_sizes:
            time_ms = results[batch_key]['time_breakdown'].get(component, {}).get('mean_ms', 0)
            component_times.append(time_ms)
        
        ax.plot(batch_nums, component_times, 'o-', linewidth=2, markersize=6,
               label=component.replace('_', ' ').title())
    
    ax.set_xlabel('Batch Size', fontweight='bold')
    ax.set_ylabel('Time (ms)', fontweight='bold')
    ax.set_title('Component Scaling Analysis')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file = output_dir / f'time_breakdown.{format}'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved time breakdown plot to {output_file}")
    plt.close()


def plot_memory_analysis(results: Dict, output_dir: Path, format: str = 'png'):
    """Create memory usage visualization."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Memory Usage Analysis', fontsize=16, fontweight='bold')
    
    batch_sizes = sorted(
        [k for k in results.keys() if k.startswith('batch_')],
        key=lambda k: int(k.split('_')[1])
    )
    
    # 1. Memory usage at different stages
    ax = axes[0]
    
    stages = ['before_batch', 'after_data_transfer', 'after_forward', 'after_backward']
    stage_labels = ['Start', 'After Transfer', 'After Forward', 'After Backward']
    
    for batch_key in batch_sizes[:3]:  # Show up to 3 batch sizes
        memory_stats = results[batch_key].get('memory_stats', {})
        if memory_stats:
            batch_size = batch_key.split('_')[1]
            values = [memory_stats.get(stage, {}).get('mean_mb', 0) for stage in stages]
            ax.plot(stage_labels, values, 'o-', linewidth=2, markersize=8, 
                   label=f'Batch {batch_size}')
    
    ax.set_xlabel('Stage', fontweight='bold')
    ax.set_ylabel('Memory (MB)', fontweight='bold')
    ax.set_title('Memory Usage Through Training Step')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Peak memory vs batch size
    ax = axes[1]
    
    batch_nums = []
    peak_memories = []
    
    for batch_key in batch_sizes:
        batch_size = int(batch_key.split('_')[1])
        batch_nums.append(batch_size)
        
        memory_stats = results[batch_key].get('memory_stats', {})
        # Get maximum memory across all stages
        max_mem = 0
        for stage_stats in memory_stats.values():
            if isinstance(stage_stats, dict):
                max_mem = max(max_mem, stage_stats.get('max_mb', 0))
        peak_memories.append(max_mem)
    
    ax.bar(batch_nums, peak_memories, color='#E63946', alpha=0.7, edgecolor='black')
    ax.set_xlabel('Batch Size', fontweight='bold')
    ax.set_ylabel('Peak Memory (MB)', fontweight='bold')
    ax.set_title('Peak Memory Usage by Batch Size')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add memory per sample on secondary axis
    ax2 = ax.twinx()
    memory_per_sample = [mem / bs for mem, bs in zip(peak_memories, batch_nums)]
    ax2.plot(batch_nums, memory_per_sample, 's-', color='#2A9D8F', linewidth=2, markersize=8)
    ax2.set_ylabel('Memory per Sample (MB)', fontweight='bold', color='#2A9D8F')
    ax2.tick_params(axis='y', labelcolor='#2A9D8F')
    
    plt.tight_layout()
    output_file = output_dir / f'memory_analysis.{format}'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved memory analysis plot to {output_file}")
    plt.close()


def generate_bottleneck_report(results: Dict, output_dir: Path):
    """Generate detailed bottleneck analysis report."""
    report_lines = []
    report_lines.append("="*70)
    report_lines.append("SSL TRAINING PERFORMANCE BOTTLENECK ANALYSIS")
    report_lines.append("="*70)
    
    # Analyze each batch size
    for batch_key in sorted(results.keys(), key=lambda k: int(k.split('_')[1])):
        batch_size = batch_key.split('_')[1]
        batch_results = results[batch_key]
        
        report_lines.append(f"\n{'='*50}")
        report_lines.append(f"BATCH SIZE: {batch_size}")
        report_lines.append(f"{'='*50}")
        
        # Time analysis
        report_lines.append("\n--- TIME ANALYSIS ---")
        time_breakdown = batch_results.get('time_breakdown', {})
        total_time = time_breakdown.get('total_step', {}).get('mean_ms', 0)
        
        report_lines.append(f"Total step time: {total_time:.2f}ms")
        report_lines.append(f"Throughput: {float(batch_size) / (total_time / 1000):.2f} samples/sec")
        
        # Top bottlenecks
        report_lines.append("\nTop 5 Time-Consuming Components:")
        bottlenecks = batch_results.get('bottlenecks', [])
        for i, bottleneck in enumerate(bottlenecks[:5], 1):
            component = bottleneck['component']
            percent = bottleneck['percent']
            time_ms = time_breakdown.get(component, {}).get('mean_ms', 0)
            report_lines.append(f"  {i}. {component:30s}: {time_ms:8.2f}ms ({percent:5.1f}%)")
        
        # Memory analysis
        memory_stats = batch_results.get('memory_stats', {})
        if memory_stats:
            report_lines.append("\n--- MEMORY ANALYSIS ---")
            
            # Find peak memory
            peak_memory = 0
            peak_stage = ""
            for stage, stats in memory_stats.items():
                if isinstance(stats, dict):
                    max_mb = stats.get('max_mb', 0)
                    if max_mb > peak_memory:
                        peak_memory = max_mb
                        peak_stage = stage
            
            report_lines.append(f"Peak memory usage: {peak_memory:.1f}MB at {peak_stage}")
            report_lines.append(f"Memory per sample: {peak_memory / float(batch_size):.2f}MB")
            
            # Memory growth through stages
            report_lines.append("\nMemory growth through stages:")
            stages = ['before_batch', 'after_data_transfer', 'after_forward', 'after_backward']
            prev_mem = 0
            for stage in stages:
                if stage in memory_stats:
                    curr_mem = memory_stats[stage].get('mean_mb', 0)
                    growth = curr_mem - prev_mem
                    report_lines.append(f"  {stage:20s}: {curr_mem:8.1f}MB (+{growth:6.1f}MB)")
                    prev_mem = curr_mem
    
    # Recommendations
    report_lines.append(f"\n{'='*70}")
    report_lines.append("OPTIMIZATION RECOMMENDATIONS")
    report_lines.append("="*70)
    
    # Analyze patterns across batch sizes
    if len(results) > 1:
        # Check scaling efficiency
        batch_nums = []
        step_times = []
        for batch_key in sorted(results.keys()):
            batch_size = int(batch_key.split('_')[1])
            batch_nums.append(batch_size)
            step_time = results[batch_key]['time_breakdown'].get('total_step', {}).get('mean_ms', 0)
            step_times.append(step_time)
        
        if len(batch_nums) > 1:
            # Calculate scaling factor
            scaling_factor = (step_times[-1] - step_times[0]) / (batch_nums[-1] - batch_nums[0])
            
            report_lines.append(f"\n1. BATCH SIZE SCALING:")
            report_lines.append(f"   - Time increases {scaling_factor:.3f}ms per sample")
            
            # Find optimal batch size (best throughput)
            throughputs = [bs / (t / 1000) if t > 0 else 0 for bs, t in zip(batch_nums, step_times)]
            optimal_idx = np.argmax(throughputs)
            report_lines.append(f"   - Optimal batch size for throughput: {batch_nums[optimal_idx]}")
            report_lines.append(f"   - Maximum throughput: {throughputs[optimal_idx]:.2f} samples/sec")
    
    # Component-specific recommendations
    first_batch = list(results.values())[0]
    bottlenecks = first_batch.get('bottlenecks', [])
    
    if bottlenecks:
        top_bottleneck = bottlenecks[0]['component']
        
        report_lines.append(f"\n2. PRIMARY BOTTLENECK: {top_bottleneck}")
        
        recommendations = {
            'data_loading': [
                "   - Increase num_workers in DataLoader",
                "   - Enable pin_memory for faster GPU transfer",
                "   - Use persistent_workers=True",
                "   - Consider caching preprocessed data"
            ],
            'forward_unified_transformer': [
                "   - Consider gradient checkpointing to trade compute for memory",
                "   - Reduce number of transformer layers if possible",
                "   - Use Flash Attention if available",
                "   - Enable torch.compile() for PyTorch 2.0+"
            ],
            'backward_pass': [
                "   - Enable mixed precision training (AMP)",
                "   - Use gradient accumulation for larger effective batch size",
                "   - Consider gradient checkpointing"
            ],
            'embedding_text': [
                "   - Use smaller text encoder if possible",
                "   - Reduce max sequence length",
                "   - Consider distilled models",
                "   - Cache embeddings if text is repeated"
            ]
        }
        
        for key, recs in recommendations.items():
            if key in top_bottleneck:
                report_lines.append("   Recommendations:")
                report_lines.extend(recs)
                break
    
    # Memory recommendations
    report_lines.append("\n3. MEMORY OPTIMIZATION:")
    if memory_stats:
        peak_memory = max(stats.get('max_mb', 0) for stats in memory_stats.values() 
                         if isinstance(stats, dict))
        if peak_memory > 8000:  # More than 8GB
            report_lines.append("   - High memory usage detected!")
            report_lines.append("   - Enable gradient checkpointing")
            report_lines.append("   - Reduce batch size")
            report_lines.append("   - Use mixed precision (fp16/bf16)")
        else:
            report_lines.append("   - Memory usage is reasonable")
            report_lines.append("   - Consider increasing batch size for better GPU utilization")
    
    report_lines.append("\n" + "="*70)
    
    # Write report
    report_file = output_dir / 'bottleneck_analysis.txt'
    with open(report_file, 'w') as f:
        f.write('\n'.join(report_lines))
    
    print(f"Saved bottleneck analysis to {report_file}")
    
    # Also print to console
    print('\n'.join(report_lines))


def generate_csv_summary(results: Dict, output_dir: Path):
    """Generate CSV summary of profiling results."""
    rows = []
    
    for batch_key in sorted(results.keys(), key=lambda k: int(k.split('_')[1])):
        batch_size = int(batch_key.split('_')[1])
        batch_results = results[batch_key]
        time_breakdown = batch_results.get('time_breakdown', {})
        memory_stats = batch_results.get('memory_stats', {})
        
        # Calculate key metrics
        total_time = time_breakdown.get('total_step', {}).get('mean_ms', 0)
        throughput = batch_size / (total_time / 1000) if total_time > 0 else 0
        
        # Get peak memory
        peak_memory = 0
        for stats in memory_stats.values():
            if isinstance(stats, dict):
                peak_memory = max(peak_memory, stats.get('max_mb', 0))
        
        # Create row
        row = {
            'batch_size': batch_size,
            'total_step_time_ms': total_time,
            'throughput_samples_per_sec': throughput,
            'peak_memory_mb': peak_memory,
            'memory_per_sample_mb': peak_memory / batch_size if batch_size > 0 else 0
        }
        
        # Add component times
        for component in ['data_loading', 'embedding_text', 'forward_cross_attention',
                         'forward_unified_transformer', 'backward_pass', 'optimizer_step']:
            row[f'{component}_ms'] = time_breakdown.get(component, {}).get('mean_ms', 0)
        
        rows.append(row)
    
    # Create DataFrame and save
    df = pd.DataFrame(rows)
    csv_file = output_dir / 'profiling_summary.csv'
    df.to_csv(csv_file, index=False)
    print(f"Saved CSV summary to {csv_file}")
    
    # Also display summary
    print("\n" + "="*70)
    print("PROFILING SUMMARY TABLE")
    print("="*70)
    print(df.to_string(index=False))


def main():
    """Main analysis function."""
    args = parse_args()
    
    # Setup paths
    profile_dir = Path(args.profile_dir)
    if not profile_dir.exists():
        print(f"Error: Profile directory {profile_dir} does not exist")
        return
    
    output_dir = Path(args.output_dir) if args.output_dir else profile_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading results from: {profile_dir}")
    print(f"Saving outputs to: {output_dir}")
    
    # Load results
    results = load_results(profile_dir)
    
    if not results:
        print("Error: No profiling results found")
        return
    
    print(f"Found results for {len(results)} batch size(s)")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_time_breakdown(results, output_dir, args.format)
    plot_memory_analysis(results, output_dir, args.format)
    
    # Generate reports
    print("\nGenerating reports...")
    generate_bottleneck_report(results, output_dir)
    generate_csv_summary(results, output_dir)
    
    print(f"\nAnalysis complete! Check {output_dir} for results.")


if __name__ == '__main__':
    main()