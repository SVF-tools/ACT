"""
ACTFuzzer: Inference-based whitebox fuzzing for neural network verification.

Main fuzzer engine that orchestrates mutation, coverage tracking, and
property checking to find counterexamples.

Copyright (C) 2025 SVF-tools/ACT
License: AGPLv3+
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
import os
import time
import json
import yaml
import torch
import torch.nn as nn
from pathlib import Path

from act.front_end.specs import InputSpec, OutputSpec
from act.front_end.spec_creator_base import LabeledInputTensor
from act.front_end.verifiable_model import InputSpecLayer, OutputSpecLayer
from act.pipeline.fuzzing.mutations import MutationEngine
from act.pipeline.fuzzing.coverage import CoverageTracker
from act.pipeline.fuzzing.corpus import SeedCorpus, FuzzingSeed
from act.pipeline.fuzzing.checker import Counterexample, PropertyChecker
from act.util.path_config import get_pipeline_log_dir, get_project_root
from act.util.device_manager import get_default_device


@dataclass
class FuzzingConfig:
    """
    Fuzzing configuration (immutable).

    Attributes:
        max_iterations: Maximum fuzzing iterations
        timeout_seconds: Total time budget
        seed_selection_strategy: "energy" or "random"
        mutation_weights: Dict of strategy weights
        coverage_strategy: Coverage tracking strategy ("BestInputCov" or "GlobalCov")
        activation_threshold: Neuron activation threshold for coverage tracking
        perturb_mode: Perturbation size computation mode ("adaptive_scalar", "adaptive_perdim", "fixed")
        perturb_scale: Fraction of range per mutation perturbation (e.g., 0.1 = 10% = ~10 steps to traverse)
        save_counterexamples: Whether to save counterexamples incrementally
        output_dir: Output directory for results
        report_interval: Print progress every N iterations
        verbose: Logging verbosity (0=silent, 1=report violations in progress only, 2=print each violation immediately)
        trace_level: Execution tracing level (0=disabled, 1=default, 2=full, 3=debug)
        trace_sample_rate: Capture every Nth iteration (1=all iterations)
        trace_storage: Storage backend ("hdf5" or "json")
        trace_output: Trace output path (None=auto-generate)

    Device Management:
        Device is controlled by act.util.device_manager (the single source of truth for the entire ACT pipeline).

    Perturbation Size Configuration:
        NOTE: We use "perturb_size" (not "epsilon") to avoid confusion with InputSpec.eps (L∞ radius).
        - InputSpec.eps: Defines constraint boundaries (e.g., center ± eps for LINF_BALL)
        - Mutation perturb_size: Controls mutation perturbation magnitude (exploration granularity)

        perturb_mode determines how mutation perturbation sizes are computed:
        - "adaptive_scalar": Single perturb_size from mean(ub-lb) * perturb_scale (default, best for uniform ranges)
        - "adaptive_perdim": Per-dimension perturb_size from (ub-lb) * perturb_scale (best for non-uniform ranges)
        - "fixed": Legacy hardcoded values (0.01 for gradient/activation, 0.005 for boundary/random)

        coverage_strategy determines the coverage strategy to use:
        - "BestInputCov": Per-input coverage (best per-input coverage over time)
        - "GlobalCov": Global union coverage (monotonic union over all inputs)

        perturb_scale interpretation:
        - Fraction of feasible range each mutation perturbation covers
        - steps_to_traverse = 1 / perturb_scale
        - Example: perturb_scale=0.1 → 10% per perturbation → ~10 steps to traverse from lb to ub
    """

    # All configuration values are loaded from config.yaml via from_yaml().
    # from_yaml() is the single source of configuration truth.
    max_iterations: int
    timeout_seconds: float
    seed_selection_strategy: str
    mutation_weights: Dict[str, float]
    coverage_strategy: str
    activation_threshold: float
    perturb_mode: str
    perturb_scale: float
    save_counterexamples: bool
    output_dir: Path
    report_interval: int
    verbose: int

    # Tracing configuration
    trace_level: int
    trace_sample_rate: int
    trace_storage: str
    trace_output: Optional[Path]

    # Stop as soon as the first counterexample is found (for a fast pre-attack:
    # total_time then measures time-to-first-counterexample). Default off.
    stop_on_first_violation: bool = False

    def __post_init__(self):
        """Normalize output_dir to Path object."""
        if isinstance(self.output_dir, str):
            self.output_dir = Path(get_pipeline_log_dir()) / self.output_dir
        elif not isinstance(self.output_dir, Path):
            self.output_dir = Path(self.output_dir)

    @classmethod
    def from_yaml(
        cls, config_path: Optional[str | Path] = None, **overrides
    ) -> "FuzzingConfig":
        """
        Load FuzzingConfig from YAML file with optional overrides.

        Args:
            config_path: Path to config YAML file (default: act/pipeline/fuzzing/config.yaml)
            **overrides: Keyword arguments to override YAML values

        Returns:
            FuzzingConfig instance with merged configuration

        Example:
            >>> # Load defaults from YAML
            >>> config = FuzzingConfig.from_yaml()
            >>>
            >>> # Override specific values
            >>> config = FuzzingConfig.from_yaml(
            ...     timeout_seconds=60.0,
            ...     max_iterations=1000
            ... )
        """
        # Default config path
        if config_path is None:
            config_path = Path(get_project_root()) / "act/pipeline/fuzzing/config.yaml"
        else:
            config_path = Path(config_path)

        # Verify config file exists
        if not config_path.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {config_path}\n"
                f"Expected location: act/pipeline/fuzzing/config.yaml"
            )

        # Load YAML
        with open(config_path) as f:
            yaml_data = yaml.safe_load(f)
            yaml_config = yaml_data["fuzzing"]

        # Merge YAML config with overrides (overrides take precedence)
        merged_config = {**yaml_config, **overrides}

        # Convert output_dir string to Path if present
        if "output_dir" in merged_config and isinstance(
            merged_config["output_dir"], str
        ):
            merged_config["output_dir"] = (
                Path(get_pipeline_log_dir()) / merged_config["output_dir"]
            )

        # Create FuzzingConfig instance
        return cls(**merged_config)


@dataclass
class FuzzingReport:
    """
    Fuzzing results summary.

    Attributes:
        total_iterations: Number of iterations completed
        total_time: Time elapsed in seconds
        counterexamples: List of found counterexamples
        neuron_coverage: Final neuron coverage (0.0 to 1.0)
        total_mutations: Total mutations applied
        seeds_explored: Number of unique seeds explored
        num_of_never_activated_neurons: Number of neurons that were never activated across all iterations
        never_activated_neurons: Sample of never-activated neuron ids (layer_name, neuron_idx)
    """

    total_iterations: int
    total_time: float
    counterexamples: List[Counterexample]
    neuron_coverage: float
    total_mutations: int
    seeds_explored: int
    num_of_never_activated_neurons: int = 0
    never_activated_neurons: List[Tuple[str, int]] = field(default_factory=list)

    def save(self, output_dir: Path):
        """Save report and counterexamples to disk."""
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save summary as JSON
        summary = {
            "iterations": self.total_iterations,
            "time_seconds": self.total_time,
            "counterexamples_found": len(self.counterexamples),
            "neuron_coverage": self.neuron_coverage,
            "mutations": self.total_mutations,
            "seeds_explored": self.seeds_explored,
            "num_of_never_activated_neurons": self.num_of_never_activated_neurons,
            # JSON-friendly: list of [layer_name, neuron_idx]
            "never_activated_neurons": [
                [ln, int(i)] for (ln, i) in self.never_activated_neurons
            ],
        }

        with open(output_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        # Save counterexamples
        for i, ce in enumerate(self.counterexamples):
            ce.save(output_dir / f"counterexample_{i}.pt")

        print(f"✅ Report saved to {os.path.relpath(output_dir)}")


class ACTFuzzer:
    """
    Inference-based whitebox fuzzer for neural network verification.

    Features:
    - Gradient-guided mutations (FGSM-style)
    - Neuron coverage tracking (DeepXplore)
    - Energy-based seed scheduling (AFL)
    - OutputSpec violation detection
    - InputSpec constraint projection

    Workflow:
    1. Initialize with wrapped model and seeds
    2. Loop: Select seed → Mutate → Inference → Check violation → Update coverage
    3. Return report with counterexamples

    Example:
        >>> config = FuzzingConfig.from_yaml(max_iterations=5000)
        >>> fuzzer = ACTFuzzer(
        ...     wrapped_model=model,
        ...     initial_seeds=labeled_tensors,
        ...     config=config
        ... )
        >>> report = fuzzer.fuzz()
        >>> print(f"Found {len(report.counterexamples)} violations")
    """

    def __init__(
        self,
        wrapped_model: nn.Module,
        initial_seeds: List[LabeledInputTensor],
        config: Optional[FuzzingConfig] = None,
    ):
        """
        Initialize ACTFuzzer.

        Args:
            wrapped_model: VerifiableModel from model_synthesis.
                          Contains InputSpecLayer and OutputSpecLayer with batched specs
                          sized for N VNNLib instances.
            initial_seeds: List of LabeledInputTensor from spec creators
            config: Fuzzing configuration (uses from_yaml() defaults if None)

        Initialization Steps:
            1. Load config → from_yaml() or provided FuzzingConfig
            2. Get device from device_manager → get_default_device()
            3. Use wrapped model directly → self.model (no spec layer stripping)
            4. Extract specs → _extract_spec(InputSpecLayer), _extract_spec(OutputSpecLayer)
            5. Determine batch size → from InputSpec bounds shape[0] (model synthesis N)
            6. Initialize components → MutationEngine, CoverageTracker, PropertyChecker, SeedCorpus
            7. Setup tracer → ExecutionTracer (if trace_level > 0)
        """
        self.config = config or FuzzingConfig.from_yaml()
        self.device = get_default_device()

        self.model = wrapped_model.to(self.device)

        # Extract specs for MutationEngine (projection) and PropertyChecker (violation detection).
        self.input_spec = self._extract_spec(InputSpecLayer)
        self.output_spec = self._extract_spec(OutputSpecLayer)

        # Batch size is determined by model synthesis (number of VNNLib instances).
        self.batch_size = (
            self.input_spec.lb.shape[0]
            if self.input_spec and self.input_spec.lb is not None
            else len(initial_seeds)
        )

        # Initialize components
        self.mutation_engine = MutationEngine(
            model=self.model,
            input_spec=self.input_spec,
            weights=self.config.mutation_weights,
            perturb_mode=self.config.perturb_mode,
            perturb_scale=self.config.perturb_scale,
        )
        self.coverage_tracker = CoverageTracker(
            model=self.model,
            threshold=self.config.activation_threshold,
            strategy=self.config.coverage_strategy,
        )
        self.property_checker = PropertyChecker(self.output_spec)
        self.seed_corpus = SeedCorpus(
            initial_seeds=initial_seeds, strategy=self.config.seed_selection_strategy
        )

        # Initialize tracer (only if trace_level > 0)
        if self.config.trace_level > 0:
            from act.pipeline.fuzzing.tracer import ExecutionTracer

            # Auto-generate trace output path if not specified
            # Class-level counter for unique trace filenames across multiple ACTFuzzer instances.
            # When fuzzing multiple VNNLib instances (one ACTFuzzer per instance), each needs a
            # distinct trace file (traces_0.json, traces_1.json, ...) to avoid overwriting.
            if self.config.trace_output is not None:
                trace_output = self.config.trace_output
            else:
                if not hasattr(ACTFuzzer, "_trace_counter"):
                    ACTFuzzer._trace_counter = 0
                ext = self._get_trace_ext()
                trace_output = (
                    self.config.output_dir / f"traces_{ACTFuzzer._trace_counter}.{ext}"
                )
                ACTFuzzer._trace_counter += 1

            self.tracer = ExecutionTracer(
                level=self.config.trace_level,
                sample_rate=self.config.trace_sample_rate,
                storage_backend=self.config.trace_storage,
                output_path=trace_output,
            )

            print(
                f"📊 Tracing enabled: Level {self.config.trace_level}, "
                f"sampling every {self.config.trace_sample_rate} iteration(s)"
            )
            print(f"   Output: {os.path.relpath(trace_output)}")
        else:
            self.tracer = None  # No overhead when disabled

        # Statistics
        self.counterexamples: List[Counterexample] = []
        self.iterations = 0
        self.start_time = 0.0
        self.never_activated_neurons: List[Tuple[str, int]] = []
        self.last_report_ce_count = 0  # Track counterexamples count at last report

    def _get_trace_ext(self) -> str:
        """Get file extension for trace storage."""
        return {"hdf5": "h5", "json": "json"}[self.config.trace_storage]

    def _extract_spec(self, layer_type) -> Optional[InputSpec | OutputSpec]:
        """Extract spec from wrapper layer by type."""
        for layer in self.model.children():
            if isinstance(layer, layer_type):
                return layer.spec
        return None

    def fuzz(self) -> FuzzingReport:
        """
        Main fuzzing loop.

        Returns:
            FuzzingReport with counterexamples and statistics
        """
        print(f"{'=' * 80}")
        print(f"ACT: Abstract Constraint Transformer")
        print(f"Inference-based whitebox fuzzing for neural network verification")
        print(f"{'=' * 80}\n")

        batch_size = self.batch_size

        print(f"🚀 Starting ACTFuzzer with {len(self.seed_corpus)} seeds")
        print(f"   Device: {self.device}")
        print(f"   Batch size: {batch_size} (from model synthesis)")
        print(f"   Max iterations: {self.config.max_iterations}")
        print(f"   Timeout: {self.config.timeout_seconds}s\n")

        self.start_time = time.time()
        iteration = 0

        while iteration < self.config.max_iterations:
            if time.time() - self.start_time > self.config.timeout_seconds:
                print(f"⏱️  Timeout reached after {iteration} iterations")
                break

            # Always use full batch size to match VerifiableModel's spec layer dimensions.
            # The wrapped model's InputSpecLayer has bounds sized [N, ...] from model synthesis,
            # so every forward pass must use exactly N inputs.
            self._fuzz_iteration(iteration, batch_size)
            iteration += batch_size

            if self.config.stop_on_first_violation and self.counterexamples:
                print(f"✋ First counterexample at {time.time() - self.start_time:.3f}s; stopping")
                break

            if iteration > 0 and iteration % self.config.report_interval < batch_size:
                self._print_progress(iteration)

        return self._generate_report()

    def _fuzz_iteration(self, start_iteration: int, batch_size: int):
        """
        Run one batch-native fuzzing iteration over batch_size samples.

        All operations use FuzzingSeed batch tensors

        Args:
            start_iteration: Starting iteration number
            batch_size: Number of samples to process
        """
        # 1. select — returns FuzzingSeed batch (B=batch_size)
        seeds: FuzzingSeed = self.seed_corpus.select(batch_size)

        # 2. mutate — takes FuzzingSeed, returns Tensor[B, ...]
        inputs = self.mutation_engine.mutate(seeds)

        # 3. inference
        with torch.no_grad():
            output = self.model(inputs)
        outputs = output["output"] if isinstance(output, dict) else output

        # 4. violation check — returns (BoolTensor[B], List[Counterexample])
        violation_mask, counterexamples = self.property_checker.check(
            inputs=inputs,
            outputs=outputs,
            seeds=seeds,
        )

        # 5. coverage update — returns per-sample interestingness mask
        activations = self.mutation_engine.get_activation_map()
        global_delta, cov_interesting = self.coverage_tracker.update(
            inputs, activations
        )
        # Update secondary strategy for dual coverage reporting
        _other = "BestInputCov" if self.config.coverage_strategy == "GlobalCov" else "GlobalCov"
        self.coverage_tracker.update(inputs, activations, strategy=_other)

        # 6. energy computation (fully vectorized)
        interesting_mask = violation_mask | cov_interesting
        energies = cov_interesting.float() * 10.0 + violation_mask.float() * 100.0
        energies = torch.clamp(energies, min=0.1)

        # 7. counterexamples — already sparse list from checker
        for ce in counterexamples:
            self.counterexamples.append(ce)
            if self.config.verbose >= 2:
                print(f"🚨 Counterexample #{len(self.counterexamples)}: {ce.summary()}")
            if self.config.save_counterexamples:
                self.config.output_dir.mkdir(parents=True, exist_ok=True)
                ce.save(self.config.output_dir / f"ce_{len(self.counterexamples)}.pt")

        # 8. Corpus add — batch add with mask (no per-sample loop)
        child_seeds = FuzzingSeed(
            tensor=inputs,
            original_tensor=seeds.original_tensor,
            original_index=seeds.original_index,
            label=seeds.label,
            energy=energies,
            depth=seeds.depth + 1,
            parent_id=seeds.id,
        )
        self.seed_corpus.add(child_seeds, interesting_mask)

        # 9. Tracing ( per-sample for detail)
        if self.tracer:
            coverage = self.coverage_tracker.get_coverage()
            mutation_strategy = self.mutation_engine.last_strategy or "unknown"
            gradients = None
            loss_value = None
            if self.config.trace_level >= 3:
                gradients = self.mutation_engine.get_last_gradients()
                loss_value = self.mutation_engine.get_last_loss()

            for i in range(batch_size):
                iteration = start_iteration + i
                if self.tracer.should_trace(iteration):
                    # Build per-sample violation for tracer (None if no violation at this index)
                    violation_at_i = None
                    if violation_mask[i]:
                        # Find the counterexample for this index (sparse search)
                        seed_idx_val = int(seeds.original_index[i].item())
                        for ce in counterexamples:
                            if ce.seed_index == seed_idx_val:
                                violation_at_i = ce
                                break

                    self.tracer.record_iteration(
                        iteration=iteration,
                        timestamp=time.time(),
                        mutation_strategy=mutation_strategy,
                        violation=violation_at_i,
                        coverage=coverage,
                        coverage_delta=global_delta / batch_size,
                        energy=float(energies[i]),
                        seed_id=int(seeds.id[i].item()),
                        input_before=seeds.tensor[i : i + 1],
                        input_after=inputs[i : i + 1],
                        parent_id=int(seeds.parent_id[i].item()),
                        depth=int(seeds.depth[i].item()),
                        activations=activations,
                        gradients=gradients,
                        loss_value=loss_value,
                    )

        self.iterations = start_iteration + batch_size

    def _compute_energy(self, coverage_delta: float, found_violation: bool) -> float:
        """Compute seed energy (higher = more interesting)."""
        energy = coverage_delta * 10.0
        if found_violation:
            energy += 100.0  # Violations are very interesting
        return max(energy, 0.1)  # Minimum energy

    def _print_progress(self, iteration: int):
        """Print fuzzing progress with incremental counterexample count."""
        elapsed = time.time() - self.start_time
        iter_per_sec = iteration / elapsed if elapsed > 0 else 0
        glc = self.coverage_tracker.get_coverage(strategy="GlobalCov")
        bic = self.coverage_tracker.get_coverage(strategy="BestInputCov")

        # Calculate new counterexamples since last report
        ce_total = len(self.counterexamples)
        ce_new = ce_total - self.last_report_ce_count
        self.last_report_ce_count = ce_total

        samples_per_sec = iter_per_sec * self.batch_size
        print(
            f"📊 Iteration {iteration:6d} | "
            f"GlobalCov: {glc:6.2%} BestInputCov: {bic:6.2%} | "
            f"Seeds: {len(self.seed_corpus):4d} | "
            f"Violations: {ce_total:3d} (+{ce_new}) | "
            f"Speed: {iter_per_sec:5.1f} it/s ({samples_per_sec:.0f} samples/s)"
        )

    def _generate_report(self) -> FuzzingReport:
        """Generate final report."""
        total_time = time.time() - self.start_time

        # Neurons that were never activated across all iterations
        never_activated_neurons: List[Tuple[str, int]] = []
        never_activated_count = 0
        try:
            uncovered = self.coverage_tracker.get_uncovered_neurons()
            never_activated_count = len(uncovered)
            # Deterministic small sample for logs/report
            never_activated_neurons = sorted(list(uncovered))[:20]
        except Exception:
            never_activated_count = 0
            never_activated_neurons = []

        report = FuzzingReport(
            total_iterations=self.iterations,
            total_time=total_time,
            counterexamples=self.counterexamples,
            neuron_coverage=self.coverage_tracker.get_coverage(),
            total_mutations=self.mutation_engine.total_mutations,
            seeds_explored=len(self.seed_corpus),
            num_of_never_activated_neurons=never_activated_count,
            never_activated_neurons=never_activated_neurons,
        )

        # Print summary
        print(f"\n{'=' * 80}")
        print(f"🎉 ACTFuzzer completed in {total_time:.1f}s")
        print(f"   Iterations: {report.total_iterations}")
        print(f"   Counterexamples: {len(report.counterexamples)}")
        bic = self.coverage_tracker.get_coverage(strategy="BestInputCov")
        print(f"   GlobalCov: {report.neuron_coverage:.2%}  BestInputCov: {bic:.2%}")
        print(f"   Seeds explored: {report.seeds_explored}")
        print(f"   Never-activated neurons: {report.num_of_never_activated_neurons}")
        if report.never_activated_neurons:
            sample_str = ", ".join(
                [f"{ln}[{i}]" for (ln, i) in report.never_activated_neurons[:10]]
            )
            print(f"   Never-activated sample: {sample_str}")
        print(f"{'=' * 80}\n")

        if self.config.save_counterexamples and report.counterexamples:
            report.save(self.config.output_dir)

        # Close tracer if enabled
        if self.tracer:
            self.tracer.close()

        return report


def pgd_preattack(wrapped_model, seeds, budget, scale=0.5):
    """Anisotropic-PGD pre-attack (the FALSIFY path of the VNN-COMP runner);
    returns (counterexample_or_None, elapsed_s)."""
    from act.util.device_manager import get_default_device

    device = get_default_device()
    wrapped_model = wrapped_model.to(device)
    seeds = [
        type(s)(tensor=s.tensor.to(device), label=(s.label.to(device) if s.label is not None else None))
        for s in seeds
    ]
    cfg = FuzzingConfig.from_yaml(
        timeout_seconds=float(budget),
        max_iterations=10_000_000,
        mutation_weights={"gradient": 0.0, "pgd": 1.0, "activation": 0.0,
                          "boundary": 0.0, "random": 0.0},
        perturb_mode="adaptive_perdim",
        perturb_scale=scale,
        save_counterexamples=False,
        stop_on_first_violation=True,
        verbose=0,
    )
    report = ACTFuzzer(wrapped_model=wrapped_model, initial_seeds=seeds, config=cfg).fuzz()
    ce = report.counterexamples[0] if report.counterexamples else None
    return ce, report.total_time
