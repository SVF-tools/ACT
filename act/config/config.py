from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field, fields
from importlib import import_module
from pathlib import Path
from typing import Any, Final, List, Optional, Union

import yaml

_BACKEND_YAML = Path(__file__).parent / "backend.yaml"
_NETGEN_YAML = Path(__file__).parent / "gen_act_net.yaml"
_PIPELINE_YAML = Path(__file__).parent / "pipeline.yaml"
_FRONTEND_YAML = Path(__file__).parent / "frontend.yaml"

_VALID_SOLVERS = {"auto", "gurobi", "torchlp", "dual", "hybridz"}
_VALID_DEVICES = {"cpu", "cuda", "gpu"}
_VALID_DTYPES = {"float32", "float64"}
_VALID_REGISTRY_MODES = {"intersection", "union"}
_VALID_COVERAGE_MODES = {"basic", "full"}
VALID_SOLVER_TIERS: Final[tuple[str, ...]] = ("lp", "dual", "dual_alpha", "dual_alpha_eta")
# BaB subproblem-pool selection strategies (``--bab-bounding``).
#
#   value                | pool class          | order function                       | --bab-top-k | reference
#   ---------------------|---------------------|--------------------------------------|-------------|----------
#   depth_bound_blend    | TopKBounding        | DepthLowerBoundOrder (depth+urgency) | honoured    | default
#   greedy               | TopKBounding        | GreedyOrder (best-first on |lb|)     | honoured    | Oliva-Greedy, ECOOP 2025
#   annealed             | TopKBounding        | SAOrder (Gumbel, cooling_rate**step) | honoured    | Oliva-SA, ECOOP 2025
#   diverse_split_signs  | DiverseTopKBounding | any order + split-sign repulsion     | honoured    | PR #106
#   random               | RandomBounding      | none (uniform sampling)              | ValueError  | baseline
#   mcts                 | MCTSBounding        | DepthLowerBoundOrder (pinned)        | ValueError  | docs/design/mcts_bab.md
#
# The first four share the TopKBounding pool and therefore honour ``top_k``;
# ``random`` and ``mcts`` do not rank by an order function, so passing
# ``top_k`` with them is a configuration error rather than a silent no-op.
VALID_BOUNDINGS: Final[tuple[str, ...]] = (
    "depth_bound_blend",
    "greedy",
    "annealed",
    "diverse_split_signs",
    "random",
    "mcts",
)
TOP_K_BOUNDINGS: Final[tuple[str, ...]] = VALID_BOUNDINGS[:4]
TOP_K_INCOMPATIBLE_BOUNDINGS: Final[tuple[str, ...]] = VALID_BOUNDINGS[4:]
VALID_BERT_METHODS: Final[tuple[str, ...]] = (
    "planar",
    "rule",
    "alpha",
    "ibp",
    "discrete",
)


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path) as handle:
        return yaml.safe_load(handle) or {}


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
        BaBConfig.from_yaml()           # load from act/config/backend.yaml
        BaBConfig.from_yaml(path, **kw) # custom YAML + overrides
    """

    max_depth: int = 20
    max_nodes: int = 2000
    frontier_cap: int = 0
    top_k: int = 0
    """Cap on how many pooled subproblems a single BaB wave may pop, independently
    of the batch size. 0 = unbounded. Honoured by the TopKBounding family
    (depth_bound_blend / greedy / annealed / diverse_split_signs); raises
    ValueError with random / mcts, which do not rank the pool by an order
    function. Unlike frontier_cap this never discards work: the remainder stays
    pooled, so a smaller top_k only re-sorts priorities more often."""

    input_split_fanout: int = 2

    branching_method: str = "random"
    bounding: str = "random"
    bounding_depth_weight: float = field(default=0.5, metadata={"in_yaml": False})
    bounding_bound_weight: float = field(default=0.5, metadata={"in_yaml": False})
    sa_cooling_rate: float = 0.99
    mcts_exploration: float = 1.0
    mcts_lambda: float = 0.5
    mcts_virtual_loss: float = 1.0

    # Dual-tier solver knobs — support solver_tier="dual_alpha_eta" with
    # Iterative slope + Lagrange-multiplier optimization for the dual backward pass.
    solver_tier: str = "lp"
    f"""Solver tier for BaB bound computation. Valid: {VALID_SOLVER_TIERS}."""

    provenance_enabled: bool = False
    """Track logical BaB node ids and parent ids in TopKBounding."""

    eta_only_children: bool = field(default=False, metadata={"in_yaml": False})
    """Freeze alpha in child subproblems (depth > 0): children inherit the
    parent's optimized alpha and refine only the split multipliers (eta).
    Cuts the per-node Adam graph and, combined with reuse_root_bounds,
    removes the per-iteration forward pass entirely."""

    presplit_levels: int = field(default=0, metadata={"in_yaml": False})
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

    intermediate_refine_ratio: float = field(default=10.0, metadata={"in_yaml": False})
    """Width-blowup threshold multiplier for intermediate_refine='auto'."""

    reuse_root_bounds: bool = False
    """Reuse the root box's forward bounds for every descendant (dual tiers).

    Sound by monotonicity: a child box is contained in the root box, so the
    root's per-layer bounds remain valid over-approximations. Children only
    override the INPUT/INPUT_SPEC bounds with their own sub-box; intermediate
    ReLU relaxations stay at root tightness, with branching gains recovered by
    the input-term concretization and the eta split multipliers. Eliminates
    the per-node forward pass (the dominant time and memory cost)."""

    per_subproblem_refine: str = field(default="none", metadata={"in_yaml": False})
    """Per-subproblem sparse backward refinement of intermediate bounds in the
    BaB loop (requires reuse_root_bounds): 'none' (off), 'tail' (last two
    unstable activation layers), 'all' (every unstable activation layer). For
    each child batch, the split-hardened bounds are re-tightened by a K-lane
    backward pass over the unstable-neuron union only (stable phases are
    exact, so refining them gains nothing), so splits propagate relationally
    downstream instead of only through the interval refresh."""

    per_subproblem_refine_iters: int = field(default=0, metadata={"in_yaml": False})
    """Adam iterations for per-subproblem refine rows (0 = single fixed-slope
    backward, cheapest)."""

    per_subproblem_refine_rows_cap: int = field(default=64, metadata={"in_yaml": False})
    """Max refined neurons per layer per batch (top-cap by interval width);
    bounds the K x 2*cap backward cost."""

    auto_batch_safety: float = field(default=0.55, metadata={"in_yaml": False})
    """Fraction of GPU memory the auto batch sizer (max_batch_size='auto') may
    target; lowered on a shared GPU. The sizer also never exceeds 90% of the
    currently-reclaimable memory (free + this process's reserved cache)."""

    auto_batch_cap: int = field(default=2048, metadata={"in_yaml": False})
    """Hard upper bound on the auto-sized batch (also the CPU fallback)."""

    auto_batch_floor: int = field(default=8, metadata={"in_yaml": False})
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
    llm_probe_model: str = field(default="", metadata={"in_yaml": False})
    llm_probe_base_url: str = field(default="", metadata={"in_yaml": False})
    llm_probe_api_key_env: str = field(default="", metadata={"in_yaml": False})
    llm_probe_temperature: float = field(default=0.0, metadata={"in_yaml": False})
    llm_probe_timeout: float = field(default=30.0, metadata={"in_yaml": False})
    llm_probe_max_candidates: int = field(default=8, metadata={"in_yaml": False})
    llm_probe_max_candidates_total: int = field(default=1024, metadata={"in_yaml": False})
    llm_probe_neuron_topk: int = field(default=512, metadata={"in_yaml": False})
    llm_probe_cadence: int = 1
    llm_probe_history: int = field(default=8, metadata={"in_yaml": False})
    llm_probe_max_failures: int = field(default=3, metadata={"in_yaml": False})
    llm_probe_decisions: str = "split,frontier,refine"
    """Comma-separated decision types the LLM may steer: 'split' (joint neuron
    split depth), 'frontier' (wave width), 'refine' (per-subproblem refinement),
    'neuron' (joint neuron-group selection), 'input_split' (which input
    dimension to bisect and its fanout, input-domain-splitting BaB only)."""
    llm_probe_log: bool = False

    verbose: bool = field(default=False, metadata={"in_yaml": False})

    method: Optional[str] = None
    baf: bool = field(default=True, metadata={"in_yaml": False})
    alpha_mode: str = field(default="fixed", metadata={"in_yaml": False})
    p: float = 2.0
    perturbed_words: int = field(default=1, metadata={"in_yaml": False})
    eps: float = 1e-5
    max_eps: float = 0.01
    num_verify_iters: int = field(default=5, metadata={"in_yaml": False})
    k: int = 1
    alpha_opt_steps: int = field(default=1000, metadata={"in_yaml": False})

    def __post_init__(self) -> None:
        if self.solver_tier not in VALID_SOLVER_TIERS:
            raise ValueError(
                f"Invalid solver_tier {self.solver_tier!r}; expected {VALID_SOLVER_TIERS}"
            )
        if self.bounding not in VALID_BOUNDINGS:
            raise ValueError(
                f"Invalid bounding {self.bounding!r}; expected {VALID_BOUNDINGS}"
            )
        if self.top_k < 0:
            raise ValueError(f"top_k must be non-negative, got {self.top_k}")
        if self.top_k > 0 and self.bounding in TOP_K_INCOMPATIBLE_BOUNDINGS:
            raise ValueError(
                f"top_k={self.top_k} is not supported by bounding={self.bounding!r}; "
                f"it applies only to the order-ranked pools {TOP_K_BOUNDINGS}"
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
        path = Path(config_path) if config_path else _BACKEND_YAML

        if not path.exists():
            raise FileNotFoundError(
                f"Backend config not found: {path}\nExpected: act/config/backend.yaml"
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

@dataclass
class GenerationConfig:
    """Configuration for network generation via ``NetFactory``.

    Controls network generation knobs and the architecture-sampling DSL loaded
    from ``act/config/gen_act_net.yaml``.
    """

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

    net_factory: dict[str, Any] = field(default_factory=dict)

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

@dataclass
class HybridZConfig:
    timeout: Optional[float] = None
    tolerance: float = 1e-7
    max_input_dim: int = 1024
    sigmoid_segments: int = 2
    fuse_sigmoid_affine: bool = False


@dataclass
class GurobiConfig:
    time_limit: Optional[float] = None
    mip_gap: float = 1e-4
    threads: int = 0
    output_flag: int = 0


@dataclass
class DualConfig:
    n_iters: int = 50
    """Number of Adam iterations for α/η optimization in BaB dual tiers."""

    lr_alpha: float = 0.1
    """Adam learning rate for α (slope) variables."""

    lr_beta: float = field(default=0.1, metadata={"in_yaml": False})
    """Adam learning rate for η (split-constraint KKT multipliers)."""

    lr_decay: float = field(default=0.98, metadata={"in_yaml": False})
    """Multiplicative learning-rate decay applied each Adam iteration."""

    per_class_alpha: bool = True
    """Allocate separate α tensors per output class rather than sharing one α."""

    incremental_start_enabled: bool = True
    """Reuse α/η tensors from the parent subproblem as the child initialization."""


@dataclass
class TorchLPConfig:
    rho_eq: float = 10.0
    rho_ineq: float = 10.0
    max_iter: int = 2000
    tol_feas: float = 1e-4
    lr: float = 1e-2
    beta1: float = 0.9
    beta2: float = 0.999
    weight_decay: float = 0.0
    large_n_threshold: int = 20000
    large_n_max_iter: int = 800
    large_n_tol: float = 1e-3
    stagnation_patience: int = 300
    stagnation_tol: float = 1e-5
    feas_check_stride: int = 5


# ---------------------------------------------------------------------------
# BackendConfig — unified back-end configuration
# ---------------------------------------------------------------------------


@dataclass
class BackendConfig:
    """Unified configuration for the ACT back-end.

    Covers runtime selectors (solver / device / dtype), verification timeout,
    and nested BaB settings.  The canonical source is ``act/config/backend.yaml``;
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
    hybridz: HybridZConfig = field(default_factory=HybridZConfig)
    gurobi: GurobiConfig = field(default_factory=GurobiConfig)
    torchlp: TorchLPConfig = field(default_factory=TorchLPConfig)
    dual: DualConfig = field(default_factory=DualConfig)

    method: Optional[str] = field(default=None, metadata={"in_yaml": False})
    p: float = field(default=2.0, metadata={"in_yaml": False})
    perturbed_words: int = field(default=1, metadata={"in_yaml": False})
    eps: float = field(default=1e-5, metadata={"in_yaml": False})
    max_eps: float = field(default=0.01, metadata={"in_yaml": False})
    num_verify_iters: int = field(default=5, metadata={"in_yaml": False})
    k: int = field(default=1, metadata={"in_yaml": False})
    alpha_opt_steps: int = field(default=1000, metadata={"in_yaml": False})

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
            generation settings are loaded from act/config/gen_act_net.yaml

        Override naming:
          - ``bab_<field>`` → ``BaBConfig.<field>``
          - ``gen_<field>`` → ``GenerationConfig.<field>``
          - ``hybridz_<field>`` → ``HybridZConfig.<field>``
          - ``gurobi_<field>`` → ``GurobiConfig.<field>``
          - ``torchlp_<field>`` → ``TorchLPConfig.<field>``
          - ``dual_<field>`` → ``DualConfig.<field>``
          - ``bab_enabled`` → top-level ``bab_enabled``
        """
        path = Path(config_path) if config_path else _BACKEND_YAML
        if not path.exists():
            raise FileNotFoundError(f"Backend config not found: {path}")

        raw = _load_yaml(path)

        backend_raw: dict[str, Any] = raw.get("backend", {})
        bab_raw: dict[str, Any] = backend_raw.pop("bab", {})
        gen_raw: dict[str, Any] = _load_yaml(_NETGEN_YAML) if _NETGEN_YAML.exists() else {}
        hz_raw: dict[str, Any] = backend_raw.pop("hybridz", {})
        gurobi_raw: dict[str, Any] = backend_raw.pop("gurobi", {})
        torchlp_raw: dict[str, Any] = backend_raw.pop("torchlp", {})
        dual_raw: dict[str, Any] = backend_raw.pop("dual", {})

        # Extract "enabled" from bab section → top-level bab_enabled
        bab_enabled = bab_raw.pop("enabled", None)

        # Route prefixed overrides to the right sub-config
        bab_fields = {fld.name for fld in fields(BaBConfig)}
        gen_fields = {fld.name for fld in fields(GenerationConfig)}
        hz_fields = {fld.name for fld in fields(HybridZConfig)}
        gurobi_fields = {fld.name for fld in fields(GurobiConfig)}
        torchlp_fields = {fld.name for fld in fields(TorchLPConfig)}
        dual_fields = {fld.name for fld in fields(DualConfig)}
        bab_overrides: dict[str, Any] = {}
        gen_overrides: dict[str, Any] = {}
        hz_overrides: dict[str, Any] = {}
        gurobi_overrides: dict[str, Any] = {}
        torchlp_overrides: dict[str, Any] = {}
        dual_overrides: dict[str, Any] = {}
        top_overrides: dict[str, Any] = {}
        for k, v in overrides.items():
            if k.startswith("bab_") and k[4:] in bab_fields:
                bab_overrides[k[4:]] = v
            elif k.startswith("gen_") and k[4:] in gen_fields:
                gen_overrides[k[4:]] = v
            elif k.startswith("hybridz_") and k[8:] in hz_fields:
                hz_overrides[k[8:]] = v
            elif k.startswith("gurobi_") and k[7:] in gurobi_fields:
                gurobi_overrides[k[7:]] = v
            elif k.startswith("torchlp_") and k[8:] in torchlp_fields:
                torchlp_overrides[k[8:]] = v
            elif k.startswith("dual_") and k[5:] in dual_fields:
                dual_overrides[k[5:]] = v
            else:
                top_overrides[k] = v

        # Build BaBConfig
        bab_in_yaml = {
            fld.name for fld in fields(BaBConfig) if fld.metadata.get("in_yaml", True)
        }
        bab_merged = {
            k: v for k, v in bab_raw.items() if k in bab_fields and k in bab_in_yaml
        }
        bab_merged.update(bab_overrides)
        bab_config = BaBConfig(**bab_merged)

        # Build GenerationConfig
        gen_merged = {k: v for k, v in gen_raw.items() if k in gen_fields}
        gen_merged.update(gen_overrides)
        gen_config = GenerationConfig(**gen_merged)

        hz_merged = {k: v for k, v in hz_raw.items() if k in hz_fields}
        hz_merged.update(hz_overrides)
        hz_config = HybridZConfig(**hz_merged)

        gurobi_config = GurobiConfig(
            **{k: v for k, v in gurobi_raw.items() if k in gurobi_fields} | gurobi_overrides
        )

        torchlp_merged = {k: v for k, v in torchlp_raw.items() if k in torchlp_fields}
        torchlp_merged.update(torchlp_overrides)
        torchlp_config = TorchLPConfig(**torchlp_merged)

        dual_in_yaml = {
            fld.name for fld in fields(DualConfig) if fld.metadata.get("in_yaml", True)
        }
        dual_merged = {
            k: v for k, v in dual_raw.items() if k in dual_fields and k in dual_in_yaml
        }
        dual_merged.update(dual_overrides)
        dual_config = DualConfig(**dual_merged)

        # Build top-level config
        top_fields = {fld.name for fld in fields(cls)} - {
            "bab",
            "generation",
            "hybridz",
            "gurobi",
            "torchlp",
            "dual",
        }
        top_merged: dict[str, Any] = {}
        for k, v in backend_raw.items():
            if k in top_fields:
                top_merged[k] = v

        if bab_enabled is not None:
            top_merged["bab_enabled"] = bab_enabled

        top_merged.update({k: v for k, v in top_overrides.items() if k in top_fields})

        return cls(
            bab=bab_config,
            generation=gen_config,
            hybridz=hz_config,
            gurobi=gurobi_config,
            torchlp=torchlp_config,
            dual=dual_config,
            **top_merged,
        )

    def to_yaml(self, path: Union[str, Path]) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        d = asdict(self)
        bab_d = d.pop("bab")
        d.pop("generation")
        hz_d = d.pop("hybridz")
        gurobi_d = d.pop("gurobi")
        torchlp_d = d.pop("torchlp")
        dual_d = d.pop("dual")
        bab_enabled = d.pop("bab_enabled")
        bab_d["enabled"] = bab_enabled

        with open(path, "w") as f:
            yaml.dump(
                {
                    "backend": {
                        **d,
                        "bab": bab_d,
                        "hybridz": hz_d,
                        "gurobi": gurobi_d,
                        "torchlp": torchlp_d,
                        "dual": dual_d,
                    }
                },
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
) -> tuple[BaBConfig, DualConfig]:
    """BaBConfig for real VNNLIB instances (the VNN-COMP runner profile):
    ``fsb``/``babsr`` keep single-neuron splits, ``gain``/``gain+llm`` use joint-split
    depth, and only ``gain+llm`` enables the LLM probe."""
    branching_method = config_label if config_label in ("fsb", "babsr") else "gain"
    common: dict[str, Any] = dict(
        solver_tier=solver_tier,
        branching_method=branching_method,
        bounding="depth_bound_blend",
        frontier_cap=25000,
        max_depth=max_depth,
        max_nodes=max_nodes,
        reuse_root_bounds=True,
        intermediate_refine="all",
        presplit_levels=0,
        eta_only_children=False,
        multi_split_levels=max(1, int(multi_split_levels)),
    )
    dual_cfg = DualConfig(
        n_iters=dual_n_iters,
        lr_alpha=0.25,
        lr_beta=0.1,
        lr_decay=0.98,
        incremental_start_enabled=True,
        per_class_alpha=True,
    )
    if config_label != "gain+llm":
        return BaBConfig(**common), dual_cfg
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
    return cfg, dual_cfg


# ---------------------------------------------------------------------------
# Pipeline configuration
# ---------------------------------------------------------------------------


FuzzingConfig = Any


@dataclass
class ValidationConfig:
    solvers: list[str]
    tf_modes: list[str]
    samples: int
    per_neuron_topk: int
    bounds_tolerance: str
    batch_sizes: Optional[list[Optional[int]]]


@dataclass
class PipelineConfig:
    fuzzing: FuzzingConfig
    bab: BaBConfig
    dual: DualConfig
    validation: ValidationConfig

    @classmethod
    def from_yaml(
        cls,
        config_path: Optional[str | Path] = None,
        **overrides: Any,
    ) -> "PipelineConfig":
        path = Path(config_path) if config_path else _PIPELINE_YAML
        if not path.exists():
            raise FileNotFoundError(
                f"Pipeline config not found: {path}\nExpected: act/config/pipeline.yaml"
            )

        FuzzingConfig = import_module("act.pipeline.fuzzing.actfuzzer").FuzzingConfig

        with open(path) as f:
            yaml_data = yaml.safe_load(f) or {}

        fuzz_overrides = _strip_prefixed_overrides(overrides, "fuzz_")
        bab_overrides = _strip_prefixed_overrides(overrides, "bab_")
        dual_overrides = _strip_prefixed_overrides(overrides, "dual_")
        val_overrides = _strip_prefixed_overrides(overrides, "val_")

        fuzzing = FuzzingConfig.from_mapping(
            yaml_data.get("fuzzing") or {}, **fuzz_overrides
        )
        verification_data = yaml_data.get("verification") or {}
        bab_data = verification_data.get("bab") or {}
        dual_data = verification_data.get("dual") or {}
        validation_data = yaml_data.get("validation") or {}

        bab = BaBConfig(**_merge_dataclass_fields(BaBConfig, bab_data, bab_overrides))
        dual = DualConfig(**_merge_dataclass_fields(DualConfig, dual_data, dual_overrides))
        validation = ValidationConfig(
            **_merge_dataclass_fields(ValidationConfig, validation_data, val_overrides)
        )
        return cls(fuzzing=fuzzing, bab=bab, dual=dual, validation=validation)


def _strip_prefixed_overrides(overrides: dict[str, Any], prefix: str) -> dict[str, Any]:
    return {
        key[len(prefix) :]: value
        for key, value in overrides.items()
        if key.startswith(prefix) and value is not None
    }


def _merge_dataclass_fields(
    dataclass_type: type,
    yaml_values: dict[str, Any],
    overrides: dict[str, Any],
) -> dict[str, Any]:
    valid_keys = {field.name for field in fields(dataclass_type)}
    merged = {key: value for key, value in yaml_values.items() if key in valid_keys}
    merged.update({key: value for key, value in overrides.items() if key in valid_keys})
    return merged


def read_fuzzing_section(config_path: Optional[str | Path] = None) -> dict[str, Any]:
    """Read the ``fuzzing`` section of the pipeline YAML.

    config.py is the single reader of the config YAML files; FuzzingConfig (in
    act.pipeline.fuzzing.actfuzzer) routes its YAML access through here.
    """
    path = Path(config_path) if config_path else _PIPELINE_YAML
    if not path.exists():
        raise FileNotFoundError(
            f"Pipeline config not found: {path}\nExpected: act/config/pipeline.yaml"
        )
    with open(path) as f:
        yaml_data = yaml.safe_load(f) or {}
    return yaml_data.get("fuzzing") or {}


# ---------------------------------------------------------------------------
# Front-end configuration loading
# ---------------------------------------------------------------------------


@dataclass
class FrontEndConfig:
    specs: dict[str, dict[str, Any]] = field(default_factory=dict)
    text_verification: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(
        cls,
        config_path: Optional[Union[str, Path]] = None,
        **overrides: Any,
    ) -> "FrontEndConfig":
        path = Path(config_path) if config_path else _FRONTEND_YAML
        if not path.exists():
            raise FileNotFoundError(f"Front-end config not found: {path}")

        with open(path) as f:
            raw = yaml.safe_load(f) or {}

        specs = deepcopy(raw.get("specs", {}))
        text_verification = deepcopy(raw.get("text_verification", {}))
        text_verification.update(
            {k: v for k, v in overrides.items() if k in text_verification and v is not None}
        )
        return cls(specs=specs, text_verification=text_verification)

    def spec_config(self, name: Optional[str]) -> dict[str, Any]:
        key = name or "default"
        if key not in self.specs:
            raise KeyError(f"Unknown front-end spec config: {key}")
        return deepcopy(self.specs[key])
