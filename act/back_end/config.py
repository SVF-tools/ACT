# ===- act/back_end/config.py - Backend Configuration ---------------------====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
# ===---------------------------------------------------------------------====#

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Final, List, Optional, Union

import yaml

_DEFAULT_YAML = Path(__file__).parent / "config.yaml"

_VALID_SOLVERS = {"auto", "gurobi", "torchlp", "dual"}
_VALID_DEVICES = {"cpu", "cuda", "gpu"}
_VALID_DTYPES = {"float32", "float64"}
_VALID_REGISTRY_MODES = {"intersection", "union"}
_VALID_COVERAGE_MODES = {"basic", "full"}
VALID_SOLVER_TIERS: Final[tuple[str, ...]] = ("lp", "dual", "dual_alpha", "dual_alpha_eta")
VALID_BERT_METHODS: Final[tuple[str, ...]] = (
    "planar",
    "rule",
    "alpha",
    "ibp",
    "discrete",
)


@dataclass(frozen=True)
class BertMethodSelection:
    """Resolved attention-relaxation BERT verification method."""

    method: str
    internal_method: str
    baf: bool
    alpha_mode: str
    solver_tier: str
    use_bab: bool = True


_BERT_METHOD_SELECTIONS: Final[dict[str, BertMethodSelection]] = {
    "planar": BertMethodSelection("planar", "planar", True, "fixed", "dual"),
    "rule": BertMethodSelection("rule", "rule", True, "rule", "dual"),
    "alpha": BertMethodSelection("alpha", "alpha", True, "optimized", "dual_alpha"),
    "ibp": BertMethodSelection("ibp", "ibp", False, "none", "dual"),
    "discrete": BertMethodSelection("discrete", "discrete", False, "none", "dual"),
}

BERT_METHOD_TIERS: Final[dict[str, str]] = {
    key: value.solver_tier for key, value in _BERT_METHOD_SELECTIONS.items()
}


def normalize_bert_method(method: str) -> str:
    """Normalize a public BERT method name."""
    key = method.strip().lower().replace("-", "_")
    if key not in _BERT_METHOD_SELECTIONS:
        valid = ", ".join(name.replace("_", "-") for name in VALID_BERT_METHODS)
        raise ValueError(f"Invalid bert method {method!r}; expected one of: {valid}")
    return key


def select_bert_method(method: str) -> BertMethodSelection:
    """Resolve a user-facing SST/Yelp method into ACT back-end settings."""
    return _BERT_METHOD_SELECTIONS[normalize_bert_method(method)]


# ---------------------------------------------------------------------------
# BaBConfig — Branch-and-Bound algorithm parameters
# ---------------------------------------------------------------------------


@dataclass
class BaBConfig:
    """Configuration for Branch-and-Bound verification algorithm.

    Construction::

        BaBConfig()                     # programmatic defaults
        BaBConfig.from_yaml()           # load from act/back_end/config.yaml
        BaBConfig.from_yaml(path, **kw) # custom YAML + overrides
    """

    max_depth: int = 20
    max_nodes: int = 2000
    frontier_cap: int = 0
    input_split_fanout: int = 2

    branching_method: str = "random"
    bounding_method: str = "random"
    bounding_order: str = "depth_lb"
    bounding_depth_weight: float = 0.5
    bounding_bound_weight: float = 0.5
    sa_cooling_rate: float = 0.99

    # Dual-tier solver knobs — support solver_tier="dual_alpha_eta" with
    # Iterative slope + Lagrange-multiplier optimization for the dual backward pass.
    solver_tier: str = "lp"
    f"""Solver tier for BaB bound computation. Valid: {VALID_SOLVER_TIERS}."""

    dual_n_iters: int = 50
    """Number of Adam iterations for α/η optimization (only used in ``dual_alpha`` / ``dual_alpha_eta`` tiers)."""

    lr_alpha: float = 0.1
    """Adam learning rate for α (slope) variables."""

    lr_beta: float = 0.1
    """Adam learning rate for η (split-constraint KKT multipliers). 0.1 default; tune per network."""

    lr_decay: float = 0.98
    """Multiplicative learning-rate decay applied each Adam iteration."""

    incremental_start_enabled: bool = True
    """Reuse α/η tensors from the parent subproblem as the initial point for child optimization."""

    per_class_alpha: bool = True
    """Allocate separate α tensors per output class (tighter bounds) rather than sharing one α."""

    provenance_enabled: bool = False
    """Track logical BaB node ids and parent ids in TopKBounding."""

    eta_only_children: bool = False
    """Freeze alpha in child subproblems (depth > 0): children inherit the
    parent's optimized alpha and refine only the split multipliers (eta).
    Cuts the per-node Adam graph and, combined with reuse_root_bounds,
    removes the per-iteration forward pass entirely."""

    presplit_levels: int = 0
    """Pre-split the root's top-k scored unstable neurons into all 2^k sign
    combinations before the main loop (LEAPS-style leap: descendants are
    materialized directly, intermediate tree levels are never bounded). The
    combinations exactly partition the root region, so soundness is
    unaffected. Requires a dual tier with neuron branching state."""

    intermediate_refine: str = "none"
    """Backward refinement of intermediate pre-activation bounds at the root:
    'none' (off), 'auto' (refine activation layers whose mean width exceeds
    intermediate_refine_ratio x the median - targets wide fan-in
    concretization loss), 'all' (every unstable activation layer)."""

    intermediate_refine_ratio: float = 10.0
    """Width-blowup threshold multiplier for intermediate_refine='auto'."""

    reuse_root_bounds: bool = False
    """Reuse the root box's forward bounds for every descendant (dual tiers).

    Sound by monotonicity: a child box is contained in the root box, so the
    root's per-layer bounds remain valid over-approximations. Children only
    override the INPUT/INPUT_SPEC bounds with their own sub-box; intermediate
    ReLU relaxations stay at root tightness, with branching gains recovered by
    the input-term concretization and the eta split multipliers. Eliminates
    the per-node forward pass (the dominant time and memory cost)."""

    per_subproblem_refine: str = "none"
    """Per-subproblem sparse backward refinement of intermediate bounds in the
    BaB loop (requires reuse_root_bounds): 'none' (off), 'tail' (last two
    unstable activation layers), 'all' (every unstable activation layer). For
    each child batch, the split-hardened bounds are re-tightened by a K-lane
    backward pass over the unstable-neuron union only (stable phases are
    exact, so refining them gains nothing), so splits propagate relationally
    downstream instead of only through the interval refresh."""

    per_subproblem_refine_iters: int = 0
    """Adam iterations for per-subproblem refine rows (0 = single fixed-slope
    backward, cheapest)."""

    per_subproblem_refine_rows_cap: int = 64
    """Max refined neurons per layer per batch (top-cap by interval width);
    bounds the K x 2*cap backward cost."""

    auto_batch_safety: float = 0.55
    """Fraction of GPU memory the auto batch sizer (max_batch_size='auto') may
    target; lowered on a shared GPU. The sizer also never exceeds 90% of the
    currently-reclaimable memory (free + this process's reserved cache)."""

    auto_batch_cap: int = 2048
    """Hard upper bound on the auto-sized batch (also the CPU fallback)."""

    auto_batch_floor: int = 8
    """Lower bound on the auto-sized batch."""

    multi_split_levels: int = 1
    """Simultaneous neuron splits per branching step (gain branching only).
    Each lane splits its top-k scored neurons jointly into all 2^k sign
    combinations. Joint splits are super-additive: the bound gain of
    constraining k neurons together exceeds the sum of the k individual
    split gains, because the split multipliers are optimized jointly
    against all constraints.     1 = single-split behavior."""

    llm_probe_enabled: bool = False
    llm_probe_backend: str = "mock"
    llm_probe_model: str = ""
    llm_probe_base_url: str = ""
    llm_probe_api_key_env: str = ""
    llm_probe_temperature: float = 0.0
    llm_probe_timeout: float = 30.0
    llm_probe_max_candidates: int = 8
    llm_probe_max_candidates_total: int = 1024
    llm_probe_neuron_topk: int = 512
    llm_probe_cadence: int = 1
    llm_probe_history: int = 8
    llm_probe_max_failures: int = 3
    llm_probe_decisions: str = "split,frontier,refine"
    """Comma-separated decision types the LLM may steer: 'split' (joint neuron
    split depth), 'frontier' (wave width), 'refine' (per-subproblem refinement),
    'neuron' (joint neuron-group selection), 'input_split' (which input
    dimension to bisect and its fanout, input-domain-splitting BaB only)."""
    llm_probe_log: bool = False

    verbose: bool = False

    method: Optional[str] = None
    baf: bool = True
    alpha_mode: str = "fixed"
    p: float = 2.0
    perturbed_words: int = 1
    eps: float = 1e-5
    max_eps: float = 0.01
    num_verify_iters: int = 5
    k: int = 1
    alpha_opt_steps: int = 1000

    def __post_init__(self) -> None:
        if self.solver_tier not in VALID_SOLVER_TIERS:
            raise ValueError(
                f"Invalid solver_tier {self.solver_tier!r}; expected {VALID_SOLVER_TIERS}"
            )
        if self.method is not None:
            selection = select_bert_method(self.method)
            self.method = selection.method
            self.baf = selection.baf
            self.alpha_mode = selection.alpha_mode
            if self.solver_tier == "lp":
                self.solver_tier = selection.solver_tier
        if self.perturbed_words not in (1, 2):
            raise ValueError("perturbed_words must be 1 or 2")
        if self.num_verify_iters < 0:
            raise ValueError("num_verify_iters must be non-negative")
        if self.max_eps < 0 or self.eps < 0:
            raise ValueError("eps and max_eps must be non-negative")

    @classmethod
    def from_yaml(
        cls,
        config_path: Optional[Union[str, Path]] = None,
        **overrides,
    ) -> BaBConfig:
        """Load BaB settings from YAML with optional keyword overrides.

        Reads from ``backend.bab`` in the unified backend config, falling
        back to a top-level ``bab`` key for standalone BaB YAML files.
        """
        path = Path(config_path) if config_path else _DEFAULT_YAML

        if not path.exists():
            raise FileNotFoundError(
                f"Backend config not found: {path}\nExpected: act/back_end/config.yaml"
            )

        with open(path) as f:
            yaml_data = yaml.safe_load(f) or {}

        # Support both nested (backend.bab) and flat (bab) YAML layouts.
        backend_section = yaml_data.get("backend", {})
        yaml_config: dict[str, Any] = backend_section.get("bab", yaml_data.get("bab", {}))

        valid_keys = {fld.name for fld in fields(cls)}
        merged = {k: v for k, v in yaml_config.items() if k in valid_keys}
        merged.update({k: v for k, v in overrides.items() if k in valid_keys})

        return cls(**merged)

    def to_yaml(self, path: Union[str, Path]) -> Path:
        """Write BaB settings to a standalone YAML file (top-level ``bab`` key)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            yaml.dump(
                {"bab": asdict(self)}, f, default_flow_style=False, sort_keys=False
            )

        return path


# ---------------------------------------------------------------------------
# GenerationConfig — network generation (net_factory) parameters
# ---------------------------------------------------------------------------

_DEFAULT_GEN_CONFIG = str(
    Path(__file__).parent / "examples" / "config_gen_act_net.yaml"
)


@dataclass
class GenerationConfig:
    """Configuration for network generation via ``NetFactory``.

    Controls the simple knobs (how many, where, seed, TF filtering).
    The architecture sampling DSL lives in a separate file referenced
    by ``gen_config_path``.
    """

    gen_config_path: str = _DEFAULT_GEN_CONFIG
    output_dir: str = "act/back_end/examples/nets"
    num_instances: int = 15
    base_seed: int = 42
    name_prefix: str = "cfg_seed"
    tf_targets: Optional[List[str]] = None
    registry_mode: str = "intersection"
    coverage_mode: str = "basic"
    coverage_max_attempts: int = 1000
    coverage_report: bool = True
    write_manifest: bool = True

    def __post_init__(self) -> None:
        if self.registry_mode not in _VALID_REGISTRY_MODES:
            raise ValueError(
                f"Invalid registry_mode {self.registry_mode!r}; "
                f"expected one of {_VALID_REGISTRY_MODES}"
            )
        if self.coverage_mode not in _VALID_COVERAGE_MODES:
            raise ValueError(
                f"Invalid coverage_mode {self.coverage_mode!r}; "
                f"expected one of {_VALID_COVERAGE_MODES}"
            )


# ---------------------------------------------------------------------------
# BackendConfig — unified back-end configuration
# ---------------------------------------------------------------------------


@dataclass
class BackendConfig:
    """Unified configuration for the ACT back-end.

    Covers runtime selectors (solver / device / dtype), verification timeout,
    and nested BaB settings.  The canonical source is ``act/back_end/config.yaml``;
    CLI flags and environment variables override it at load time.

    Construction::

        BackendConfig()                     # programmatic defaults
        BackendConfig.from_yaml()           # load from default YAML
        BackendConfig.from_yaml(path, **kw) # custom YAML + overrides
    """

    solver: str = "auto"
    device: str = "cpu"
    dtype: str = "float64"
    verbose: bool = False
    timeout: float = 300.0

    bab_enabled: bool = False
    bab: BaBConfig = field(default_factory=BaBConfig)

    # -- batched-API knobs (C11) --------------------------------------------
    lp_enabled: bool = True
    """Enable the LP-batched tier (tier 2) in the 3-tier cascade.

    Set to False to skip verify_lp_batched and fall through directly to BaB.
    Must be False when solver='gurobi' (Gurobi solve_batch is N=1 only;
    see commit af797ff / C6).
    """

    bab_max_batch_size: int = 8
    """Maximum K for BaB sub-problem batching (tier 3).

    BaB dispatches up to K sub-problems per solve_batch call.  Set to 1 to
    disable batching inside BaB (equivalent to the legacy sequential loop).
    Must be 1 when solver='gurobi' (same N=1 restriction as lp_enabled).
    """

    generation: GenerationConfig = field(default_factory=GenerationConfig)

    method: Optional[str] = None
    p: float = 2.0
    perturbed_words: int = 1
    eps: float = 1e-5
    max_eps: float = 0.01
    num_verify_iters: int = 5
    k: int = 1
    alpha_opt_steps: int = 1000

    # -- validation ---------------------------------------------------------

    def __post_init__(self) -> None:
        if self.solver not in _VALID_SOLVERS:
            raise ValueError(
                f"Invalid solver {self.solver!r}; expected one of {_VALID_SOLVERS}"
            )
        if self.device not in _VALID_DEVICES:
            raise ValueError(
                f"Invalid device {self.device!r}; expected one of {_VALID_DEVICES}"
            )
        if self.dtype not in _VALID_DTYPES:
            raise ValueError(
                f"Invalid dtype {self.dtype!r}; expected one of {_VALID_DTYPES}"
            )
        if self.method is not None:
            selection = select_bert_method(self.method)
            self.method = selection.method
            self.bab.method = selection.method
            self.bab.baf = selection.baf
            self.bab.alpha_mode = selection.alpha_mode
            self.bab.solver_tier = selection.solver_tier
            self.bab.p = float(self.p)
            self.bab.perturbed_words = int(self.perturbed_words)
            self.bab.eps = float(self.eps)
            self.bab.max_eps = float(self.max_eps)
            self.bab.num_verify_iters = int(self.num_verify_iters)
            self.bab.k = int(self.k)
            self.bab.alpha_opt_steps = int(self.alpha_opt_steps)
        # Gurobi solve_batch is restricted to N=1 (commit af797ff / C6).
        # Fail loud at config-load time rather than at the first batched call.
        if self.solver == "gurobi":
            if self.lp_enabled:
                raise ValueError(
                    "BackendConfig: solver='gurobi' is incompatible with "
                    "lp_enabled=True.  GurobiSolver.solve_batch raises for N>1 "
                    "(Gurobi does not expose a truly parallel multi-LP API for "
                    "varying constraint matrices; see commit af797ff).  "
                    "Either set lp_enabled=False or switch to solver='torchlp'."
                )
            if self.bab_max_batch_size > 1:
                raise ValueError(
                    f"BackendConfig: solver='gurobi' is incompatible with "
                    f"bab_max_batch_size={self.bab_max_batch_size} > 1.  "
                    f"GurobiSolver.solve_batch raises for N>1.  "
                    f"Either set bab_max_batch_size=1 or switch to solver='torchlp'."
                )

    # -- YAML I/O -----------------------------------------------------------

    @classmethod
    def from_yaml(
        cls,
        config_path: Optional[Union[str, Path]] = None,
        **overrides,
    ) -> BackendConfig:
        """Load config from YAML with optional keyword overrides.

        YAML layout::

            backend:
              solver: "torchlp"
              ...
              bab:
                enabled: true
                ...
              generation:
                num_instances: 15
                ...

        Override naming:
          - ``bab_<field>`` → ``BaBConfig.<field>``
          - ``gen_<field>`` → ``GenerationConfig.<field>``
          - ``bab_enabled`` → top-level ``bab_enabled``
        """
        path = Path(config_path) if config_path else _DEFAULT_YAML
        if not path.exists():
            raise FileNotFoundError(f"Backend config not found: {path}")

        with open(path) as f:
            raw = yaml.safe_load(f) or {}

        backend_raw: dict[str, Any] = raw.get("backend", {})
        bab_raw: dict[str, Any] = backend_raw.pop("bab", {})
        gen_raw: dict[str, Any] = backend_raw.pop("generation", {})

        # Extract "enabled" from bab section → top-level bab_enabled
        bab_enabled = bab_raw.pop("enabled", None)

        # Route prefixed overrides to the right sub-config
        bab_fields = {fld.name for fld in fields(BaBConfig)}
        gen_fields = {fld.name for fld in fields(GenerationConfig)}
        bab_overrides: dict[str, Any] = {}
        gen_overrides: dict[str, Any] = {}
        top_overrides: dict[str, Any] = {}
        for k, v in overrides.items():
            if k.startswith("bab_") and k[4:] in bab_fields:
                bab_overrides[k[4:]] = v
            elif k.startswith("gen_") and k[4:] in gen_fields:
                gen_overrides[k[4:]] = v
            else:
                top_overrides[k] = v

        # Build BaBConfig
        bab_merged = {k: v for k, v in bab_raw.items() if k in bab_fields}
        bab_merged.update(bab_overrides)
        bab_config = BaBConfig(**bab_merged)

        # Build GenerationConfig
        gen_merged = {k: v for k, v in gen_raw.items() if k in gen_fields}
        gen_merged.update(gen_overrides)
        gen_config = GenerationConfig(**gen_merged)

        # Build top-level config
        top_fields = {fld.name for fld in fields(cls)} - {"bab", "generation"}
        top_merged: dict[str, Any] = {}
        for k, v in backend_raw.items():
            if k in top_fields:
                top_merged[k] = v

        if bab_enabled is not None:
            top_merged["bab_enabled"] = bab_enabled

        top_merged.update({k: v for k, v in top_overrides.items() if k in top_fields})

        return cls(bab=bab_config, generation=gen_config, **top_merged)

    def to_yaml(self, path: Union[str, Path]) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        d = asdict(self)
        bab_d = d.pop("bab")
        gen_d = d.pop("generation")
        bab_enabled = d.pop("bab_enabled")
        bab_d["enabled"] = bab_enabled

        with open(path, "w") as f:
            yaml.dump(
                {"backend": {**d, "bab": bab_d, "generation": gen_d}},
                f,
                default_flow_style=False,
                sort_keys=False,
            )
        return path


if __name__ == "__main__":
    import sys

    passed = 0
    failed = 0

    def _check(label: str, fn) -> None:  # pragma: no cover
        global passed, failed
        try:
            fn()
            print(f"  PASS  {label}")
            passed += 1
        except Exception as exc:
            print(f"  FAIL  {label}: {exc}")
            failed += 1

    print("BackendConfig.__post_init__ rejection tests")

    def _t1():  # pragma: no cover
        try:
            BackendConfig(solver="gurobi", lp_enabled=True)
            raise AssertionError("expected ValueError not raised")
        except ValueError as e:
            assert "lp_enabled" in str(e), f"wrong message: {e}"

    def _t2():  # pragma: no cover
        try:
            BackendConfig(solver="gurobi", lp_enabled=False, bab_max_batch_size=2)
            raise AssertionError("expected ValueError not raised")
        except ValueError as e:
            assert "bab_max_batch_size" in str(e), f"wrong message: {e}"

    def _t3():  # pragma: no cover
        cfg = BackendConfig(solver="gurobi", lp_enabled=False, bab_max_batch_size=1)
        assert cfg.solver == "gurobi"
        assert not cfg.lp_enabled
        assert cfg.bab_max_batch_size == 1

    def _t4():  # pragma: no cover
        cfg = BackendConfig()
        assert cfg.lp_enabled is True
        assert cfg.bab_max_batch_size == 8

    _check("gurobi + lp_enabled=True raises ValueError", _t1)
    _check("gurobi + bab_max_batch_size=2 raises ValueError", _t2)
    _check("gurobi + lp_enabled=False + bab_max_batch_size=1 succeeds", _t3)
    _check("default config has lp_enabled=True, bab_max_batch_size=8", _t4)

    print(f"\n{passed}/{passed + failed} passed")
    sys.exit(0 if failed == 0 else 1)


def build_vnncomp_bab_config(
    config_label: str,
    *,
    llm_backend: str = "mock",
    llm_decisions: str = "split,frontier,refine,input_split",
    llm_timeout: float = 30.0,
    llm_model: str = "",
    llm_cadence: int = 1,
    llm_neuron_topk: int = 0,
    llm_log: bool = False,
    multi_split_levels: int = 4,
    max_depth: int = 1_000_000,
    max_nodes: int = 1_000_000_000,
    solver_tier: str = "dual_alpha_eta",
    dual_n_iters: int = 100,
) -> BaBConfig:
    """BaBConfig for real VNNLIB instances (the VNN-COMP runner profile):
    ``fsb``/``babsr`` keep single-neuron splits, ``gain``/``gain+llm`` use joint-split
    depth, and only ``gain+llm`` enables the LLM probe."""
    branching_method = config_label if config_label in ("fsb", "babsr") else "gain"
    common: dict[str, Any] = dict(
        solver_tier=solver_tier,
        branching_method=branching_method,
        bounding_method="topk",
        bounding_order="depth_lb",
        frontier_cap=25000,
        max_depth=max_depth,
        max_nodes=max_nodes,
        dual_n_iters=dual_n_iters,
        lr_alpha=0.25,
        lr_beta=0.1,
        lr_decay=0.98,
        incremental_start_enabled=True,
        per_class_alpha=True,
        reuse_root_bounds=True,
        intermediate_refine="all",
        presplit_levels=0,
        eta_only_children=False,
        multi_split_levels=1 if branching_method != "gain" else max(1, int(multi_split_levels)),
    )
    if config_label != "gain+llm":
        return BaBConfig(**common)
    cfg = BaBConfig(
        llm_probe_enabled=True,
        llm_probe_backend=llm_backend,
        llm_probe_decisions=llm_decisions,
        llm_probe_timeout=llm_timeout,
        llm_probe_cadence=llm_cadence,
        llm_probe_neuron_topk=llm_neuron_topk,
        llm_probe_log=llm_log,
        **common,
    )
    if llm_model:
        cfg.llm_probe_model = llm_model
    return cfg
