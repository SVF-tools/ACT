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
from act.pipeline.fuzzing.checker import PropertyChecker, Counterexample
from act.util.path_config import get_pipeline_log_dir, get_project_root


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
        device: Torch device ("cuda" or "cpu")
        save_counterexamples: Whether to save counterexamples incrementally
        output_dir: Output directory for results
        report_interval: Print progress every N iterations
        verbose: Logging verbosity (0=silent, 1=report violations in progress only, 2=print each violation immediately)
        trace_level: Execution tracing level (0=disabled, 1=default, 2=full, 3=debug)
        trace_sample_rate: Capture every Nth iteration (1=all iterations)
        trace_storage: Storage backend ("hdf5" or "json")
        trace_output: Trace output path (None=auto-generate)
    
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
    # NOTE: All configuration values are loaded from config.yaml via from_yaml() class method.
    # Direct instantiation without from_yaml() is not supported.
    max_iterations: int
    timeout_seconds: float
    seed_selection_strategy: str
    mutation_weights: Dict[str, float]
    coverage_strategy: str
    activation_threshold: float
    perturb_mode: str
    perturb_scale: float
    device: str
    save_counterexamples: bool
    output_dir: Path
    report_interval: int
    verbose: int
    trace_level: int
    trace_sample_rate: int
    trace_storage: str
    trace_output: Optional[Path] = None
    
    def __post_init__(self):
        """Normalize output_dir to Path object."""
        if isinstance(self.output_dir, str):
            self.output_dir = Path(get_pipeline_log_dir()) / self.output_dir
        elif not isinstance(self.output_dir, Path):
            self.output_dir = Path(self.output_dir)
    
    @classmethod
    def from_yaml(cls, config_path: Optional[str | Path] = None, **overrides) -> "FuzzingConfig":
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
            yaml_config = yaml_data['fuzzing']
        
        # Merge YAML config with overrides (overrides take precedence)
        merged_config = {**yaml_config, **overrides}
        
        # Convert output_dir string to Path if present
        if 'output_dir' in merged_config and isinstance(merged_config['output_dir'], str):
            merged_config['output_dir'] = Path(get_pipeline_log_dir()) / merged_config['output_dir']
        
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
    never_activated_neurons: List[Dict[str, Any]] = field(default_factory=list)
    
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
            "never_activated_neurons": [[ln, int(i)] for (ln, i) in self.never_activated_neurons]
        }
        
        with open(output_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        
        # Save counterexamples
        for i, ce in enumerate(self.counterexamples):
            ce.save(output_dir / f"counterexample_{i}.pt")
        
        print(f"✅ Report saved to {output_dir}")


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
        >>> fuzzer = ACTFuzzer(
        ...     wrapped_model=model,
        ...     initial_seeds=labeled_tensors,
        ...     config=FuzzingConfig(max_iterations=5000)
        ... )
        >>> report = fuzzer.fuzz()
        >>> print(f"Found {len(report.counterexamples)} violations")
    """
    
    def __init__(self,
                 wrapped_model: nn.Module,
                 initial_seeds: List[LabeledInputTensor],
                 config: Optional[FuzzingConfig] = None):
        """
        Initialize ACTFuzzer.
        
        Args:
            wrapped_model: VerifiableModel from model_synthesis
                          (contains InputSpecLayer and OutputSpecLayer)
            initial_seeds: List of LabeledInputTensor from spec creators
            config: Fuzzing configuration (uses defaults if None)
        """
        self.config = config or FuzzingConfig.from_yaml()
        self.model = wrapped_model.to(self.config.device)
        self.device = torch.device(self.config.device)
        
        # Extract specs from model
        self.input_spec = self._extract_spec(InputSpecLayer)
        self.output_spec = self._extract_spec(OutputSpecLayer)
        
        # Initialize components
        self.mutation_engine = MutationEngine(
            model=self.model,
            input_spec=self.input_spec,
            weights=self.config.mutation_weights,
            device=self.device,
            perturb_mode=self.config.perturb_mode,
            perturb_scale=self.config.perturb_scale
        )
        self.coverage_tracker = CoverageTracker(model=self.model,threshold=self.config.activation_threshold, strategy=self.config.coverage_strategy)
        self.property_checker = PropertyChecker(self.output_spec)
        self.seed_corpus = SeedCorpus(
            initial_seeds=initial_seeds,
            strategy=self.config.seed_selection_strategy
        )
        
        # Initialize tracer (only if trace_level > 0)
        if self.config.trace_level > 0:
            from act.pipeline.fuzzing.tracer import ExecutionTracer
            
            # Auto-generate trace output path if not specified
            trace_output = self.config.trace_output or (
                self.config.output_dir / f"traces.{self._get_trace_ext()}"
            )
            
            self.tracer = ExecutionTracer(
                level=self.config.trace_level,
                sample_rate=self.config.trace_sample_rate,
                storage_backend=self.config.trace_storage,
                output_path=trace_output
            )
            
            print(f"📊 Tracing enabled: Level {self.config.trace_level}, "
                  f"sampling every {self.config.trace_sample_rate} iteration(s)")
            print(f"   Output: {trace_output}")
        else:
            self.tracer = None  # No overhead when disabled
        
        # Statistics
        self.counterexamples: List[Counterexample] = []
        self.iterations = 0
        self.start_time = 0.0
        self.never_activated_neurons: List[Dict[str, Any]] = []
        self.last_report_ce_count = 0  # Track counterexamples count at last report
    
    def _get_trace_ext(self) -> str:
        """Get file extension for trace storage."""
        return {"hdf5": "h5", "json": "json"}[self.config.trace_storage]
    
    def _extract_spec(self, layer_type) -> Optional[InputSpec | OutputSpec]:
        """Extract spec from wrapper layer."""
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
        print(f"{'='*80}")
        print(f"ACT: Abstract Constraint Transformer")
        print(f"Inference-based whitebox fuzzing for neural network verification")
        print(f"{'='*80}\n")
        
        print(f"🚀 Starting ACTFuzzer with {len(self.seed_corpus)} seeds")
        print(f"   Device: {self.device}")
        print(f"   Max iterations: {self.config.max_iterations}")
        print(f"   Timeout: {self.config.timeout_seconds}s\n")
        
        self.start_time = time.time()
        
        for iteration in range(self.config.max_iterations):
            # Check timeout
            if time.time() - self.start_time > self.config.timeout_seconds:
                print(f"⏱️  Timeout reached after {iteration} iterations")
                break
            
            # Fuzzing iteration
            self._fuzz_iteration(iteration)
            
            # Periodic reporting
            if iteration > 0 and iteration % self.config.report_interval == 0:
                self._print_progress(iteration)
        
        return self._generate_report()
    
    def _fuzz_iteration(self, iteration: int):
        """Single fuzzing iteration with optional tracing."""
        # 1. Select seed
        seed = self.seed_corpus.select()
        
        # 2. Get seed tensor (already has batch dimension)
        seed_tensor = seed.tensor  # Already (1, C, H, W)
        
        # 3. Mutate with feedback (pass labeled_tensor)
        candidate = self.mutation_engine.mutate(seed)
        mutation_strategy = self.mutation_engine.last_strategy
        
        # 4. Run inference
        with torch.no_grad():
            output_dict = self.model(candidate)
        
        # Handle VerifiableModel output (dict) or plain tensor
        if isinstance(output_dict, dict):
            output = output_dict['output']
        else:
            output = output_dict
        
        # 5. Check violation
        violation = self.property_checker.check(
            input_tensor=candidate,
            output=output,
            label=seed.label,
            seed_tensor=seed.tensor
        )
        
        # 6. Update coverage
        activations = self.mutation_engine.get_activation_map()

        
        coverage_delta = self.coverage_tracker.update(candidate, activations)
        coverage = self.coverage_tracker.get_coverage()
        
        # 7. Compute energy
        if violation or coverage_delta > 0:
            energy = self._compute_energy(coverage_delta, violation is not None)
        else:
            energy = 0.0
        
        # 8. UNIFIED TRACING HOOK (all levels)
        if self.tracer and self.tracer.should_trace(iteration):
            # Collect gradients only if Level 3
            gradients = None
            loss_value = None
            if self.config.trace_level >= 3:
                gradients = self.mutation_engine.get_last_gradients()
                loss_value = self.mutation_engine.get_last_loss()
            
            # Single tracing call - tracer handles level-specific storage
            self.tracer.record_iteration(
                iteration=iteration,
                timestamp=time.time(),
                mutation_strategy=mutation_strategy,
                violation=violation,
                coverage=coverage,
                coverage_delta=coverage_delta,
                energy=energy,
                seed_id=seed.id,
                # Level 1+ data
                input_before=seed_tensor,
                input_after=candidate,
                parent_id=seed.parent_id,
                depth=seed.depth,
                # Level 2+ data
                activations=activations,
                # Level 3+ data
                gradients=gradients,
                loss_value=loss_value
            )
        
        # 9. Handle results
        if violation:
            self.counterexamples.append(violation)
            # Only print individual counterexamples in debug mode (verbose >= 2)
            # Regular reporting happens every report_interval in _print_progress
            if self.config.verbose >= 2:
                print(f"🚨 Counterexample #{len(self.counterexamples)}: {violation.summary()}")
            
            if self.config.save_counterexamples:
                self.config.output_dir.mkdir(parents=True, exist_ok=True)
                violation.save(self.config.output_dir / f"ce_{len(self.counterexamples)}.pt")
        
        # 10. Add to corpus if interesting
        if violation or coverage_delta > 0:
            new_seed = FuzzingSeed(
                tensor=candidate.cpu(),
                label=seed.label,
                energy=energy,
                depth=seed.depth + 1,
                parent_id=seed.id
            )
            self.seed_corpus.add(new_seed)
        
        self.iterations = iteration + 1
    
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
        coverage = self.coverage_tracker.get_coverage()
        
        # Calculate new counterexamples since last report
        ce_total = len(self.counterexamples)
        ce_new = ce_total - self.last_report_ce_count
        self.last_report_ce_count = ce_total
        
        print(f"📊 Iteration {iteration:6d} | "
              f"Coverage: {coverage:6.2%} | "
              f"Seeds: {len(self.seed_corpus):4d} | "
              f"Violations: {ce_total:3d} (+{ce_new}) | "
              f"Speed: {iter_per_sec:5.1f} it/s")
    
    def _generate_report(self) -> FuzzingReport:
        """Generate final report."""
        total_time = time.time() - self.start_time

        # Neurons that were never activated across all iterations (per coverage tracker definition)
        never_activated_neurons: List[Tuple[str, int]] = []
        never_activated_count = 0
        try:
            uncovered = self.coverage_tracker.get_uncovered_neurons()
            never_activated_count = len(uncovered)
            # Deterministic small sample for logs/report
            never_activated_neurons = sorted(list(uncovered))[:20]
        except Exception:
            # Fallback if tracker doesn't support uncovered queries
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
        print(f"\n{'='*80}")
        print(f"🎉 ACTFuzzer completed in {total_time:.1f}s")
        print(f"   Iterations: {report.total_iterations}")
        print(f"   Counterexamples: {len(report.counterexamples)}")
        print(f"   Coverage: {report.neuron_coverage:.2%}")
        print(f"   Seeds explored: {report.seeds_explored}")
        print(f"   Never-activated neurons: {report.num_of_never_activated_neurons}")
        if report.never_activated_neurons:
            sample_str = ", ".join([f"{ln}[{i}]" for (ln, i) in report.never_activated_neurons[:10]])
            print(f"   Never-activated sample: {sample_str}")
        print(f"{'='*80}\n")
        
        if self.config.save_counterexamples and report.counterexamples:
            report.save(self.config.output_dir)
        
        # Close tracer if enabled
        if self.tracer:
            self.tracer.close()
        
        return report