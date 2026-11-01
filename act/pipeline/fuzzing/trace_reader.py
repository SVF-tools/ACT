"""
Trace Reader and Analyzer for ACTFuzzer execution traces.

This module provides two main interfaces:

1. **CLI Tool (TraceReader)**: Command-line trace inspection and export
   - Read JSON and HDF5 trace files
   - Display summary statistics
   - List and export specific traces

2. **Jupyter Analyzer (TraceAnalyzer)**: Interactive visual analysis in notebooks
   - Rich visualizations (coverage, strategies, violations)
   - Interactive widgets for trace exploration
   - Input tensor heatmaps and comparisons
   - Pandas DataFrame export for custom analysis

CLI Usage:
    # Show summary
    python -m act.pipeline.fuzzing.trace_reader traces.json --summary

    # List traces
    python -m act.pipeline.fuzzing.trace_reader traces.json --list

    # Export specific trace
    python -m act.pipeline.fuzzing.trace_reader traces.json --export 42 --output trace_42.pt

Jupyter Usage:
    from act.pipeline.fuzzing.trace_reader import TraceAnalyzer

    analyzer = TraceAnalyzer("traces.json")
    analyzer.show_overview()
    analyzer.plot_overview()
    analyzer.interactive_explorer()

Copyright (C) 2025 SVF-tools/ACT
License: AGPLv3+
"""

from __future__ import annotations
import argparse
import os
from pathlib import Path
from typing import Dict, Any, List, Union
import json
import sys
import torch

from act.util.format_utils import rule


class TraceReader:
    """Base class for reading trace files."""

    def __init__(self, path: Path):
        """
        Initialize trace reader.

        Args:
            path: Path to trace file
        """
        self.path = path
        self.traces: List[Dict[str, Any]] = []
        self._load()

    def _load(self):
        """Load traces from file."""
        raise NotImplementedError
    
    def __len__(self) -> int:
        """Return number of traces."""
        return len(self.traces)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get trace by index."""
        return self.traces[idx]

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        if not self.traces:
            return {}

        violations = sum(1 for t in self.traces if t.get("violation_found", False))
        strategies = {}
        for t in self.traces:
            strat = t.get("mutation_strategy", "unknown")
            strategies[strat] = strategies.get(strat, 0) + 1

        coverages = [t.get("coverage", 0.0) for t in self.traces]

        return {
            "total_traces": len(self.traces),
            "violations_found": violations,
            "strategies": strategies,
            "coverage_min": min(coverages) if coverages else 0.0,
            "coverage_max": max(coverages) if coverages else 0.0,
            "coverage_final": coverages[-1] if coverages else 0.0,
            "iterations_range": (
                self.traces[0].get("iteration", 0),
                self.traces[-1].get("iteration", 0),
            )
            if self.traces
            else (0, 0),
        }


class JSONTraceReader(TraceReader):
    """Reader for JSON trace files."""

    def _load(self):
        """Load traces from JSON file."""
        with open(self.path, "r") as f:
            data = json.load(f)
            if isinstance(data, list):
                self.traces = data
            else:
                # Handle single trace object
                self.traces = [data]

        # Convert list representations back to tensors if needed
        for trace in self.traces:
            for key in ["input_before", "input_after"]:
                if key in trace and isinstance(trace[key], list):
                    trace[key] = torch.tensor(trace[key])

            # Convert activations dict
            if "activations" in trace and isinstance(trace["activations"], dict):
                for layer_name, activation in trace["activations"].items():
                    if isinstance(activation, list):
                        trace["activations"][layer_name] = torch.tensor(activation)

            # Convert gradients dict
            if "gradients" in trace and isinstance(trace["gradients"], dict):
                for layer_name, gradient in trace["gradients"].items():
                    if isinstance(gradient, list):
                        trace["gradients"][layer_name] = torch.tensor(gradient)


class HDF5TraceReader(TraceReader):
    """Reader for HDF5 trace files."""

    def _load(self):
        """Load traces from HDF5 file."""
        try:
            import h5py
        except ImportError:
            raise ImportError(
                "h5py is required to read HDF5 traces. Install with: pip install h5py"
            )

        with h5py.File(self.path, "r") as f:
            iterations_group = f["iterations"]

            # Load each iteration
            for iter_name in sorted(
                iterations_group.keys(), key=lambda x: int(x.split("_")[1])
            ):
                iter_group = iterations_group[iter_name]

                # Build trace from attributes and datasets
                trace = {}

                # Load scalar attributes
                for attr_name in [
                    "iteration",
                    "timestamp",
                    "mutation_strategy",
                    "violation_found",
                    "coverage",
                    "coverage_delta",
                    "energy",
                    "seed_id",
                    "parent_id",
                    "depth",
                    "loss_value",
                ]:
                    if attr_name in iter_group.attrs:
                        trace[attr_name] = iter_group.attrs[attr_name]

                # Load tensor datasets
                for key in ["input_before", "input_after"]:
                    if key in iter_group:
                        trace[key] = torch.tensor(iter_group[key][:])

                # Load activations
                if "activations" in iter_group:
                    activations = {}
                    for layer_name in iter_group["activations"].keys():
                        activations[layer_name] = torch.tensor(
                            iter_group["activations"][layer_name][:]
                        )
                    trace["activations"] = activations

                # Load gradients
                if "gradients" in iter_group:
                    gradients = {}
                    for layer_name in iter_group["gradients"].keys():
                        gradients[layer_name] = torch.tensor(
                            iter_group["gradients"][layer_name][:]
                        )
                    trace["gradients"] = gradients

                self.traces.append(trace)


def create_reader(path: Path) -> TraceReader:
    """
    Factory to create appropriate trace reader.

    Args:
        path: Path to trace file

    Returns:
        TraceReader instance

    Raises:
        ValueError: If file format is not supported
    """
    if not path.exists():
        raise FileNotFoundError(f"Trace file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".json":
        return JSONTraceReader(path)
    elif suffix in [".h5", ".hdf5"]:
        return HDF5TraceReader(path)
    else:
        raise ValueError(f"Unsupported trace format: {suffix}. Use .json or .h5/.hdf5")


# ============================================================================
# Visualization Functions
# ============================================================================


def print_summary(reader: TraceReader):
    """Print trace summary statistics."""
    summary = reader.get_summary()

    print(f"\n{rule()}")
    print(f"TRACE SUMMARY: {reader.path.name}")
    print(f"{rule()}\n")

    print(f"📊 General Statistics:")
    print(f"   Total traces: {summary['total_traces']}")
    print(
        f"   Iterations: {summary['iterations_range'][0]} → {summary['iterations_range'][1]}"
    )
    print(f"   Violations found: {summary['violations_found']}")

    print(f"\n📈 Coverage:")
    print(f"   Min: {summary['coverage_min']:.2%}")
    print(f"   Max: {summary['coverage_max']:.2%}")
    print(f"   Final: {summary['coverage_final']:.2%}")

    print(f"\n🔄 Mutation Strategies:")
    strategies = summary["strategies"]
    total = sum(strategies.values())
    for strat, count in sorted(strategies.items(), key=lambda x: x[1], reverse=True):
        pct = 100 * count / total if total > 0 else 0
        print(f"   {strat:15s}: {count:5d} ({pct:5.1f}%)")

    print(f"\n{rule()}\n")


def print_trace_detail(trace: Dict[str, Any], idx: int):
    """Print detailed information about a single trace."""
    print(f"\n{rule()}")
    print(f"TRACE #{idx} (Iteration {trace.get('iteration', '?')})")
    print(f"{rule()}\n")

    # Basic info
    print(f"⏱️  Timestamp: {trace.get('timestamp', 'N/A')}")
    print(f"🔄 Strategy: {trace.get('mutation_strategy', 'unknown')}")
    print(
        f"📊 Coverage: {trace.get('coverage', 0.0):.2%} (Δ {trace.get('coverage_delta', 0.0):+.2%})"
    )
    print(f"⚡ Energy: {trace.get('energy', 0.0):.4f}")
    print(f"🆔 Seed ID: {trace.get('seed_id', 'unknown')}")

    if trace.get("parent_id"):
        print(f"👪 Parent: {trace['parent_id']} (depth: {trace.get('depth', '?')})")

    if trace.get("violation_found"):
        print(f"⚠️  VIOLATION FOUND!")

    # Input tensors
    if "input_before" in trace:
        tensor = trace["input_before"]
        print(f"\n📥 Input (before mutation):")
        print(f"   Shape: {tuple(tensor.shape)}")
        print(f"   Range: [{tensor.min():.4f}, {tensor.max():.4f}]")
        print(f"   Mean: {tensor.mean():.4f}, Std: {tensor.std():.4f}")

    if "input_after" in trace:
        tensor = trace["input_after"]
        print(f"\n📥 Input (after mutation):")
        print(f"   Shape: {tuple(tensor.shape)}")
        print(f"   Range: [{tensor.min():.4f}, {tensor.max():.4f}]")
        print(f"   Mean: {tensor.mean():.4f}, Std: {tensor.std():.4f}")

    # Activations
    if "activations" in trace:
        activations = trace["activations"]
        print(f"\n🧠 Layer Activations ({len(activations)} layers):")
        for layer_name, activation in activations.items():
            print(
                f"   {layer_name:20s}: shape={tuple(activation.shape)}, "
                f"range=[{activation.min():.4f}, {activation.max():.4f}]"
            )

    # Gradients
    if "gradients" in trace:
        gradients = trace["gradients"]
        print(f"\n∇ Gradients ({len(gradients)} layers):")
        for layer_name, gradient in gradients.items():
            grad_norm = gradient.norm().item()
            print(
                f"   {layer_name:20s}: shape={tuple(gradient.shape)}, "
                f"norm={grad_norm:.6f}"
            )

    if "loss_value" in trace:
        print(f"\n📉 Loss: {trace['loss_value']:.6f}")

    print(f"\n{rule()}\n")


# ============================================================================
# CLI Main Entry Point
# ============================================================================


def main():
    """
    CLI entry point for trace inspection and export.

    For visual analysis, use TraceAnalyzer in Jupyter notebooks instead.
    """
    parser = argparse.ArgumentParser(
        description="ACTFuzzer trace inspection and export tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Show summary statistics
  python -m act.pipeline.fuzzing.trace_reader traces.json --summary
  
  # List all traces
  python -m act.pipeline.fuzzing.trace_reader traces.json --list
  
  # List trace range
  python -m act.pipeline.fuzzing.trace_reader traces.json --list 10 20
  
  # Export specific trace
  python -m act.pipeline.fuzzing.trace_reader traces.json --export 42 --output trace_42.pt

Note: For visual analysis with charts and interactive widgets, use:
  from act.pipeline.fuzzing.trace_reader import TraceAnalyzer
  analyzer = TraceAnalyzer("traces.json")
  analyzer.plot_overview()  # In Jupyter notebook
        """,
    )

    parser.add_argument(
        "trace_file", type=str, help="Path to trace file (.json or .h5/.hdf5)"
    )
    parser.add_argument(
        "--summary", "-s", action="store_true", help="Show summary statistics"
    )
    parser.add_argument(
        "--list",
        "-l",
        nargs="*",
        metavar=("START", "END"),
        help="List traces (optionally specify range)",
    )
    parser.add_argument(
        "--export",
        "-e",
        type=int,
        metavar="INDEX",
        help="Export specific trace by index",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Output path for export (required with --export)",
    )
    parser.add_argument(
        "--show",
        type=int,
        metavar="INDEX",
        help="Show detailed information for specific trace",
    )

    args = parser.parse_args()

    # Load trace file
    try:
        trace_path = Path(args.trace_file)
        reader = create_reader(trace_path)
        print(f"✅ Loaded {len(reader)} traces from {trace_path.name}")
    except Exception as e:
        print(f"❌ Failed to load trace file: {e}")
        sys.exit(1)

    # Execute command
    try:
        if args.summary:
            print_summary(reader)

        elif args.list is not None:
            if len(args.list) == 0:
                start, end = 0, len(reader)
            elif len(args.list) == 1:
                start, end = int(args.list[0]), len(reader)
            else:
                start, end = int(args.list[0]), int(args.list[1])

            print(
                f"\n{'Idx':>5s} {'Iter':>6s} {'Strategy':>12s} {'Coverage':>10s} {'Violation':>10s}"
            )
            print(f"{rule(50, "-")}")
            for i in range(start, min(end, len(reader))):
                trace = reader[i]
                print(
                    f"{i:5d} {trace.get('iteration', 0):6d} "
                    f"{trace.get('mutation_strategy', 'unknown'):>12s} "
                    f"{trace.get('coverage', 0.0):9.2%} "
                    f"{'✓' if trace.get('violation_found') else '':<10s}"
                )

        elif args.show is not None:
            idx = args.show
            if 0 <= idx < len(reader):
                print_trace_detail(reader[idx], idx)
            else:
                print(f"❌ Index {idx} out of range [0, {len(reader) - 1}]")
                sys.exit(1)

        elif args.export is not None:
            if not args.output:
                print("❌ --output is required with --export")
                sys.exit(1)

            idx = args.export
            if 0 <= idx < len(reader):
                trace = reader[idx]
                output = Path(args.output)
                torch.save(trace, output)
                print(f"✅ Exported trace #{idx} to {os.path.relpath(output)}")
            else:
                print(f"❌ Index {idx} out of range [0, {len(reader) - 1}]")
                sys.exit(1)

        else:
            # No command specified - show help
            parser.print_help()
            print(f"\n💡 Tip: Use --summary to see trace statistics")
            print(f"💡 Tip: Use TraceAnalyzer in Jupyter for visual analysis")

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


# ============================================================================
# TraceAnalyzer: Jupyter-friendly trace analysis with visualizations
# ============================================================================


class TraceAnalyzer:
    """
    Jupyter-friendly trace analysis with visualizations.

    Designed for interactive exploration in Jupyter notebooks with:
    - Rich text summaries
    - Matplotlib visualizations
    - Interactive widgets (ipywidgets)
    - Pandas DataFrame export

    Example:
        >>> analyzer = TraceAnalyzer("traces.json")
        >>> analyzer.show_overview()
        >>> analyzer.plot_overview()
        >>> analyzer.interactive_explorer()
    """

    def __init__(self, trace_file: Union[str, Path]):
        """
        Initialize TraceAnalyzer.

        Args:
            trace_file: Path to trace file (JSON or HDF5)
        """
        self.trace_file = Path(trace_file)

        # Load traces using appropriate reader
        if self.trace_file.suffix == ".json":
            reader = JSONTraceReader(self.trace_file)
        elif self.trace_file.suffix in [".h5", ".hdf5"]:
            reader = HDF5TraceReader(self.trace_file)
        else:
            raise ValueError(f"Unsupported file format: {self.trace_file.suffix}")

        self.traces = reader.traces
        self._reader = reader

    def show_overview(self) -> None:
        """Print formatted summary statistics."""
        if not self.traces:
            print("❌ No traces loaded")
            return

        summary = self._reader.get_summary()

        print(f"✅ Loaded {len(self.traces)} traces from {self.trace_file.name}\n")

        print("📊 Summary Statistics")
        print("━" * 50)
        print(f"  Total Traces:        {summary['total_traces']}")
        print(
            f"  Iterations:          {summary['iterations_range'][0]} → {summary['iterations_range'][1]}"
        )
        print(f"  Violations Found:    {summary['violations_found']}")

        # Handle both old and new key names
        coverage_min = summary.get(
            "coverage_range",
            [summary.get("coverage_min", 0), summary.get("coverage_max", 0)],
        )[0]
        coverage_max = summary.get(
            "coverage_range",
            [summary.get("coverage_min", 0), summary.get("coverage_max", 0)],
        )[1]
        print(f"  Coverage Range:      {coverage_min:.1%} → {coverage_max:.1%}")
        print(f"  Final Coverage:      {summary['coverage_final']:.1%}")

        print(f"\n🔄 Mutation Strategies")
        print("━" * 50)

        # Handle both 'strategies' and 'strategy_distribution' keys
        strategies = summary.get("strategy_distribution", summary.get("strategies", {}))
        for strategy, count in strategies.items():
            pct = count / summary["total_traces"] * 100
            print(f"  {strategy:15s} : {count:3d} ({pct:.1f}%)")

    def show_trace_detail(self, trace_idx: int) -> None:
        """
        Display detailed information for a specific trace.

        Args:
            trace_idx: Index of trace to display
        """
        if trace_idx < 0 or trace_idx >= len(self.traces):
            print(
                f"❌ Invalid trace index: {trace_idx} (valid: 0-{len(self.traces) - 1})"
            )
            return

        trace = self.traces[trace_idx]

        print(f"\n{rule(60)}")
        print(f"Trace #{trace_idx} Details")
        print(f"{rule(60)}")
        print(f"Iteration:           {trace.get('iteration', 'N/A')}")
        print(f"Mutation Strategy:   {trace.get('mutation_strategy', 'N/A')}")
        print(f"Coverage:            {trace.get('coverage', 0):.2%}", end="")

        if "coverage_delta" in trace:
            delta = trace["coverage_delta"]
            print(f" ({delta:+.2%})", end="")
        print()

        if "energy" in trace:
            print(f"Energy:              {trace['energy']:.6f}")

        if "violation_found" in trace:
            if trace["violation_found"]:
                print(f"Violation:           ⚠️  FOUND!")
            else:
                print(f"Violation:           ✅ None")

        # Show data availability
        print(f"\n📊 Available Data:")
        print(f"  Input (before):      {'✓' if 'input_before' in trace else '✗'}")
        print(f"  Input (after):       {'✓' if 'input_after' in trace else '✗'}")
        print(f"  Activations:         {'✓' if 'activations' in trace else '✗'}")
        print(f"  Gradients:           {'✓' if 'gradients' in trace else '✗'}")

    def plot_overview(self, figsize=(14, 10)) -> None:
        """
        Plot 2×2 grid with key visualizations.

        Args:
            figsize: Figure size (width, height)
        """
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle(
            f"Fuzzing Trace Overview - {self.trace_file.name}",
            fontsize=16,
            fontweight="bold",
        )

        self.plot_coverage(ax=axes[0, 0])
        self.plot_strategies(ax=axes[0, 1])
        self._plot_strategy_effectiveness(ax=axes[1, 0])
        self._plot_violations_timeline(ax=axes[1, 1])

        plt.tight_layout()
        plt.show()

    def plot_coverage(self, ax=None) -> None:
        """
        Plot coverage over time.

        Args:
            ax: Matplotlib axes (creates new figure if None)
        """
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 5))

        iterations = [t.get("iteration", i) for i, t in enumerate(self.traces)]
        coverages = [t.get("coverage", 0) * 100 for t in self.traces]

        ax.plot(iterations, coverages, linewidth=2, marker="o", markersize=3, alpha=0.7)
        ax.set_xlabel("Iteration", fontsize=11)
        ax.set_ylabel("Coverage (%)", fontsize=11)
        ax.set_title("Coverage Over Time", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 105])

    def plot_strategies(self, ax=None) -> None:
        """
        Plot mutation strategy distribution as pie chart.

        Args:
            ax: Matplotlib axes (creates new figure if None)
        """
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))

        summary = self._reader.get_summary()
        # Handle both 'strategies' and 'strategy_distribution' keys
        strategies = summary.get("strategy_distribution", summary.get("strategies", {}))

        labels = list(strategies.keys())
        sizes = list(strategies.values())

        ax.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=90)
        ax.set_title("Mutation Strategy Distribution", fontsize=12, fontweight="bold")

    def _plot_strategy_effectiveness(self, ax=None) -> None:
        """Plot box plot of coverage gains per strategy."""
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 5))

        # Group traces by strategy
        strategy_data = {}
        for trace in self.traces:
            strategy = trace.get("mutation_strategy", "unknown")
            coverage_delta = trace.get("coverage_delta", 0) * 100

            if strategy not in strategy_data:
                strategy_data[strategy] = []
            strategy_data[strategy].append(coverage_delta)

        # Create box plot
        strategies = list(strategy_data.keys())
        data = [strategy_data[s] for s in strategies]

        bp = ax.boxplot(data, tick_labels=strategies, patch_artist=True)

        # Color boxes
        colors = plt.cm.Set3(range(len(strategies)))
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)

        ax.set_xlabel("Mutation Strategy", fontsize=11)
        ax.set_ylabel("Coverage Gain (%)", fontsize=11)
        ax.set_title("Strategy Effectiveness", fontsize=12, fontweight="bold")
        ax.grid(True, axis="y", alpha=0.3)
        ax.axhline(y=0, color="red", linestyle="--", alpha=0.5)

    def _plot_violations_timeline(self, ax=None) -> None:
        """Plot violations over time."""
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 5))

        iterations = [t.get("iteration", i) for i, t in enumerate(self.traces)]
        coverages = [t.get("coverage", 0) * 100 for t in self.traces]
        violations = [t.get("violation_found", False) for t in self.traces]

        # Plot all traces
        ax.scatter(iterations, coverages, c="blue", alpha=0.3, s=30, label="Normal")

        # Highlight violations
        viol_iters = [iterations[i] for i, v in enumerate(violations) if v]
        viol_covs = [coverages[i] for i, v in enumerate(violations) if v]

        if viol_iters:
            ax.scatter(
                viol_iters,
                viol_covs,
                c="red",
                s=100,
                marker="*",
                label="Violation",
                zorder=10,
            )

        ax.set_xlabel("Iteration", fontsize=11)
        ax.set_ylabel("Coverage (%)", fontsize=11)
        ax.set_title(
            f"Violations Timeline ({len(viol_iters)} found)",
            fontsize=12,
            fontweight="bold",
        )
        ax.legend()
        ax.grid(True, alpha=0.3)

    def plot_input_comparison(self, trace_idx: int, axes=None) -> None:
        """
        Plot before/after/diff heatmaps for input tensors.

        Args:
            trace_idx: Index of trace
            axes: List of 3 matplotlib axes (creates new figure if None)
        """
        import matplotlib.pyplot as plt
        import numpy as np

        if trace_idx < 0 or trace_idx >= len(self.traces):
            print(f"❌ Invalid trace index: {trace_idx}")
            return

        trace = self.traces[trace_idx]

        if "input_before" not in trace or "input_after" not in trace:
            print(
                f"❌ Trace #{trace_idx} missing input data (requires --trace-level 1+)"
            )
            return

        input_before = trace["input_before"]
        input_after = trace["input_after"]

        # Convert to numpy if needed
        if isinstance(input_before, torch.Tensor):
            input_before = input_before.cpu().numpy()
        if isinstance(input_after, torch.Tensor):
            input_after = input_after.cpu().numpy()

        # Compute difference
        diff = input_after - input_before

        # Handle different tensor shapes
        if input_before.ndim == 4:  # Batch dimension
            input_before = input_before[0]
            input_after = input_after[0]
            diff = diff[0]

        if input_before.ndim == 3:  # Channel dimension
            if input_before.shape[0] == 1:  # Grayscale
                input_before = input_before[0]
                input_after = input_after[0]
                diff = diff[0]
            elif input_before.shape[0] == 3:  # RGB
                # Take mean across channels for visualization
                input_before = input_before.mean(axis=0)
                input_after = input_after.mean(axis=0)
                diff = diff.mean(axis=0)

        # Create axes if not provided
        if axes is None:
            fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        # Plot heatmaps
        im1 = axes[0].imshow(input_before, cmap="viridis")
        axes[0].set_title("Input Before")
        axes[0].axis("off")
        plt.colorbar(im1, ax=axes[0])

        im2 = axes[1].imshow(input_after, cmap="viridis")
        axes[1].set_title("Input After")
        axes[1].axis("off")
        plt.colorbar(im2, ax=axes[1])

        # Use diverging colormap for difference
        vmax = np.abs(diff).max()
        im3 = axes[2].imshow(diff, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        axes[2].set_title("Difference")
        axes[2].axis("off")
        plt.colorbar(im3, ax=axes[2])

        if axes is None:
            plt.tight_layout()
            plt.show()

    def interactive_explorer(self) -> None:
        """
        Launch interactive widget-based trace explorer.

        Requires ipywidgets to be installed.
        """
        try:
            from ipywidgets import interact, Dropdown, Button, Output, HBox, VBox
            from IPython.display import display, clear_output
        except ImportError:
            print("❌ ipywidgets not installed. Install with: pip install ipywidgets")
            return

        import matplotlib.pyplot as plt

        # Create output widget
        out = Output()

        # Create dropdown for trace selection
        trace_options = [
            (
                f"Iteration {t.get('iteration', i)} - {t.get('mutation_strategy', 'unknown')}",
                i,
            )
            for i, t in enumerate(self.traces)
        ]

        dropdown = Dropdown(
            options=trace_options,
            description="Select Trace:",
            style={"description_width": "initial"},
        )

        # Export button
        export_btn = Button(description="Export This Trace", button_style="success")

        def on_trace_select(change):
            """Handle trace selection."""
            with out:
                clear_output(wait=True)
                trace_idx = change["new"]

                # Show details
                self.show_trace_detail(trace_idx)

                # Plot inputs if available
                trace = self.traces[trace_idx]
                if "input_before" in trace and "input_after" in trace:
                    print("\n")
                    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
                    self.plot_input_comparison(trace_idx, axes=axes)
                    plt.suptitle(
                        f"Trace #{trace_idx} - Input Visualization", fontsize=14
                    )
                    plt.tight_layout()
                    plt.show()

        def on_export_click(b):
            """Handle export button click."""
            trace_idx = dropdown.value
            output_path = f"trace_{trace_idx}.pt"

            trace = self.traces[trace_idx]
            torch.save(trace, output_path)

            with out:
                print(f"\n✅ Exported trace #{trace_idx} to {output_path}")

        # Attach handlers
        dropdown.observe(on_trace_select, names="value")
        export_btn.on_click(on_export_click)

        # Display widgets
        display(VBox([dropdown, export_btn, out]))

        # Trigger initial display
        on_trace_select({"new": dropdown.value})

    def to_dataframe(self):
        """
        Convert traces to pandas DataFrame.

        Returns:
            pd.DataFrame with trace data
        """
        try:
            import pandas as pd
        except ImportError:
            print("❌ pandas not installed. Install with: pip install pandas")
            return None

        # Extract key fields
        data = []
        for i, trace in enumerate(self.traces):
            row = {
                "trace_idx": i,
                "iteration": trace.get("iteration", i),
                "strategy": trace.get("mutation_strategy", "unknown"),
                "coverage": trace.get("coverage", 0),
                "coverage_delta": trace.get("coverage_delta", 0),
                "energy": trace.get("energy", 0),
                "violation": trace.get("violation_found", False),
            }
            data.append(row)

        return pd.DataFrame(data)


if __name__ == "__main__":
    main()
