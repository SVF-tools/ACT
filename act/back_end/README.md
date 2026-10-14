
# ACT Back-End — Torch‑Native DNN Verification (Bounds + MILP/LP + BaB)

A concise, modular verification toolkit that does **all analysis in PyTorch** (for ergonomics and optional GPU),
and converts to **NumPy only at the solver boundary**. It supports MLPs and key Transformer components, with
worklist-based bounds propagation, constraint extraction, MILP/LP export, and Branch‑and‑Bound (BaB) with
counterexample validation.

---

## Why this design?

- **Torch inside**: easy interop with models, layers, and CE validation; one tensor world.
- **Single export boundary**: solver backends remain NumPy‑facing; plug in Gurobi/HiGHS/CBC, etc.
- **DAG‑aware worklist**: efficient, incremental propagation with caching and change detection.
- **Layer coverage**: MLP + less‑common activations + Transformer blocks with sound relaxations.
- **Refinement loop**: BaB + CE validation to distinguish true vs. false counterexamples.

---

## Core Data Structures

```python
@dataclass(eq=True, frozen=True)
class Bounds:      # Box bounds for a contiguous vector of variables
    lb: torch.Tensor
    ub: torch.Tensor

@dataclass(eq=False)
class Con:         # Canonical constraint template (solver-agnostic)
    kind: str                  # 'EQ' | 'INEQ' | 'BIN'
    var_ids: Tuple[int, ...]   # variable ids referenced by this constraint
    meta: Dict[str, Any]       # parameters (Torch tensors allowed) + {'tag': str}
    # Optional numeric payloads (unused in-core; only for compatibility)
    A: Any=None; b: Any=None; C: Any=None; d: Any=None
    def signature(self) -> Tuple[str, Tuple[int, ...], str]: ...

@dataclass
class ConSet:       # Replace-by-signature semantics
    S: Dict[Tuple[str, Tuple[int, ...], str], Con]
    def replace(self, c: Con): ...
    def add_box(self, layer_id: int, var_ids: List[int], B: Bounds): ...

@dataclass
class Fact:         # Dataflow fact per layer: (bounds, constraints)
    bounds: Bounds
    cons: ConSet

@dataclass
class Layer:
    id: int
    kind: str       # e.g. 'DENSE', 'RELU', 'SOFTMAX', ...
    params: Dict[str, Any]
    in_vars: List[int]
    out_vars: List[int]
    cache: Dict[str, Any]      # prev_lb/prev_ub/masks (for change detection)

@dataclass
class Net:          # Topo-ordered DAG
    layers: List[Layer]
    preds: Dict[int, List[int]]
    succs: Dict[int, List[int]]
    by_id: Dict[int, Layer] = field(init=False)
```

- **`meta['tag']`** drives export logic (e.g. `relu:{id}`, `dense:{id}`, `softmax:simplex:{id}`, …).
- Constraints are **templates** kept compact during analysis, then materialized at export time.

---

## Device & Dtype Policy

- Analysis uses **Torch** everywhere with `DEFAULT_DEVICE` (CPU/GPU) and `DEFAULT_DTYPE` (e.g. `float32`).
- Helpers:
  - `as_t(x, device, dtype)` converts inputs and parameters consistently.
  - `@torch.no_grad()` guards analysis & CE validation paths.
- Export converts tensors to **NumPy float64** in one place: `exporter.to_numpy(x)`.

---

## Supported Layers

### Layer Kinds
`INPUT, INPUT_SPEC, ASSERT` (Wrapper & Specs)

### MLP Basics
`DENSE, BIAS, SCALE, RELU, LRELU, ABS, CLIP, MUL, ADD, CONCAT, BN`

### Less‑Common MLP‑ish
`SIGMOID, TANH, SOFTPLUS, SILU, MAX, MIN, SQUARE, POWER`

### Transformer Components
`EMBEDDING, POSENC, LAYERNORM, GELU, ATT_SCORES, SOFTMAX, ATT_MIX, MHA_SPLIT, MHA_JOIN, MASK_ADD`

Each layer has a **transfer function** that:
1) computes new **interval bounds** (sound over-approximation);
2) adds a minimal **template constraint** (with `meta['tag']`) for later solver export.

---

## Bounds Propagation (Worklist, DAG‑aware)

- Start from the **entry layer**: seed bounds with the input spec (box / ℓ∞ ball).
- Use a **worklist** (queue) of layer ids.
- For a layer `L`:
  1. **Join** predecessors’ `after` bounds into `before[L]`.
  2. Run `dispatch_tf(L, before, after, net)` ⇒ `out_fact`.
  3. If changed (bounds or masks), update caches, merge constraints, push successors.
- Termination when the worklist empties (monotone joins + finite precision).

**Performance**: vectorized Torch ops, caching, and dedup constraints (replace-by-signature).

---

## Constraint Export (Solver‑Agnostic)

- `export_to_solver(globalC, solver)` performs the **only** Torch→NumPy conversion.
- Merges all `box:*` constraints into global variable bounds.
- Materializes per‑layer templates:
  - `dense:` → linear equalities (y = Wx + b)
  - `relu:` → phase splits (on/off) and convex relaxation for ambiguous
  - `mcc:`  → McCormick envelopes for bilinear terms (e.g., `MUL`, attention mixes)
  - `softmax:simplex:` → probability simplex per row (≥0, sum=1)
  - etc.
- Backends implement `Solver` (see next section).

---

## Solver Interface & Gurobi Backend

```python
class Solver:
    def begin(...); def add_vars(n); def set_bounds(idxs, lb, ub)
    def add_lin_eq(...); def add_lin_le(...); def add_lin_ge(...)
    def add_sum_eq(...); def add_ge_zero(...); def add_sos2(...)
    def set_objective_linear(..., sense="min"); def optimize(tlim=None)
    def status() -> str; def has_solution() -> bool; def get_values(ids) -> np.ndarray
    @property def n(self) -> int
```

- **Backends** only see NumPy arrays.
- Provided example: **`GurobiSolver`** (others can implement the same API).

---

## Input / Output Specs

**InputSpec**: `BOX`, `LINF_BALL`, or extra **linear polytope** constraints (`A x ≤ b`).  
**OutputSpec** (negated for searching counterexamples):
- `LINEAR_LE`: find `c^T y ≥ d + ε`
- `TOP1_ROBUST`: find a class with `y_j ≥ y_true`
- `MARGIN_ROBUST`: find `y_j − y_true ≥ δ`

`verify_once(...)`, `verify_lp_batched(...)`:
1. Run `analyse()` to collect bounds + constraint templates.
2. Export to solver, add input spec and **negated** output spec.
3. Optimize (optionally maximizing violation) →
   - **INFEASIBLE** ⇒ `CERTIFIED`
   - **FEASIBLE** ⇒ return **counterexample** (input/output witness).

`verify_bab(...)`, `verify_bab_batched(...)`:
- Priority queue by box “width” score (sum of side lengths).
- For each node:
  1. `verify_once` on the sub‑box.
  2. If **INFEASIBLE** ⇒ pruned (certified).
  3. If **CE found** ⇒ **validate** with Torch model forward:
     - If **violates** spec → **True CE** (stop).
     - If **does not** violate → **False CE** (likely relaxation gap).
  4. **Branch** on the widest dimension, enqueue children.
- Terminates with **CERTIFIED** if no true CE is found within limits.

Dual solver (`--solver dual`) supports `DualSolver.evaluate_spec()` for efficient dual-based bounds.

This forms a **refinement loop**: false CEs drive further splitting → tighter local bounds.

---

## Extending the System

- **New layers**: add transfer in `act/back_end/*_tf/` + export case in `cons_exportor.py`.
- **New solvers**: subclass `Solver` in `solver/` and implement the abstract methods.
- **Advanced relaxations**: you can store extra parameters in `Con.meta` (Torch tensors),
  then convert in the exporter when materializing constraints.

---

## Performance Tips

- Set a **global device** (CPU/GPU) and dtype up front.
- Use `@torch.no_grad()` in analysis & CE validation.
- Batch BaB nodes for large networks (optional extension).
- Keep constraints minimal; rely on `box:*` and a few tight relaxations.
- Watch BLAS threading: Torch vs. solver (set `OMP_NUM_THREADS`, `MKL_NUM_THREADS` if needed).

---

## Minimal Usage Sketch

```python
import torch, numpy as np
from act.back_end.core import Layer, Net, Bounds, as_t
from act.front_end.specs import InputSpec, OutputSpec, InKind, OutKind
from act.back_end.bab.bab import verify_bab
from act.back_end.solver.solver_gurobi import GurobiSolver

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
torch.set_default_dtype(torch.float32)

# Build a tiny MLP: x -> Dense -> ReLU -> y
# ... (rest of code)
```

---

## File Layout

```
act/back_end/
  core.py
  utils.py
  analyze.py
  verifier.py
  layer_schema.py
  bab/
    bab.py          # BaB engine wiring and orchestration
    climb.py
    violation.py
    node.py         # Subproblem batches and all child construction
    branching/
      branching.py  # Split decisions and branching heuristics
      multi_split.py # Multi-neuron selection and gain testing
  solver/
    solver_base.py
    solver_gurobi.py
    solver_torchlp.py
    solver_dual.py  # DualSolver.solve_spec_batch() handles K-lane dual solves
    solver_hz.py
  interval_tf/
    interval_tf.py
    tf_mlp.py
    tf_cnn.py
    ...
  hybridz_tf/
  dual_tf/
```

---

## LLM-Probe BaB Controller (experimental)

`verify_bab_batched` can consult an LLM as a **closed-loop search controller**. It is disabled by
default and, when disabled, the verifier is byte-identical to baseline.

Each BaB wave the controller observes the frontier (pool size, budget, depth/bound spread, recent
progress) and proposes, depending on `llm_probe_decisions`: how many sub-problems to advance
(`frontier` → `k_requested`), how deep to split (`split` → `split_k`), how much per-sub-problem
refinement to spend (`refine`), and — under `solver_tier` `dual_alpha`/`dual_alpha_eta` — **which
neurons to split jointly** (`neuron`). For neuron selection the controller is fed the **full** unstable
candidate set for each dequeued domain; the LLM returns a neuron *group* per lane and the verifier
forms the joint **2^k sign-combination cube** and solves it in one batched pass (super-additive joint
splitting). It records each wave outcome to condition the next decision.

**Soundness boundary:** the LLM only schedules work; every proposal is clipped to a sound legal range
(neuron groups are filtered to currently-unstable, not-already-split neurons, and the 2^k cube exactly
partitions the region for any choice), and bounding/certification/falsification stay entirely in the
verifier. Invalid/unavailable output (parse error, timeout, illegal value, or a candidate set above
`llm_probe_max_candidates_total`) falls back to FSB/BaBSR/gain; a circuit breaker disables the
controller after repeated failures.

**Providers** are swappable by config with no verifier code change. `mock` is offline/deterministic.
The recommended online path is **OpenRouter** — one OpenAI-compatible gateway that reaches OpenAI,
Claude, GLM, and MiniMax via the **model string** (`anthropic/claude-sonnet-4`, `openai/gpt-4o`,
`z-ai/glm-4.6`, `minimax/minimax-m1`). Direct presets `openai|glm|minimax` are also available. All use
stdlib HTTP (no extra dependencies) and a tolerant JSON parser, so models without strict json-mode
still work. Each preset supplies a default `base_url` and API-key env var (overridable):

| backend | base_url default | key env default |
|---|---|---|
| `openrouter` | `https://openrouter.ai/api/v1` | `OPENROUTER_API_KEY` |
| `openai` | `https://api.openai.com/v1` | `OPENAI_API_KEY` |
| `glm` | `https://open.bigmodel.cn/api/paas/v4` | `ZHIPUAI_API_KEY` |
| `minimax` | `https://api.minimaxi.com/v1` | `MINIMAX_API_KEY` |

Config (`BaBConfig`) / CLI flags:

| Config field | CLI flag | Default |
|---|---|---|
| `llm_probe_enabled` | `--bab-llm-probe-enabled` | `False` |
| `llm_probe_backend` | `--bab-llm-probe-backend {mock,openrouter,openai,glm,minimax}` | `mock` |
| `llm_probe_model` | `--bab-llm-probe-model` | `""` |
| `llm_probe_base_url` | `--bab-llm-probe-base-url` | `""` |
| `llm_probe_cadence` | `--bab-llm-probe-cadence` | `1` |
| `llm_probe_decisions` (e.g. `split,frontier,refine,neuron`) / `llm_probe_max_candidates_total` / `llm_probe_api_key_env` / `_temperature` / `_max_candidates` / `_history` / `_max_failures` / `_log` | `--bab-llm-probe-<field>` (e.g. `--bab-llm-probe-decisions`); also YAML | see `act/config/config.py` |

Implementation lives in `act/pipeline/verification/llm_probe.py`. That file has two independent
sections: the legacy TinyLlama input/output probe (Section A) and this BaB controller (Section B);
they share no state.

## License & Notes

- This blueprint focuses on clarity, extensibility, and solver portability.
- Replace/extend relaxations as needed for tighter bounds (CROWN, triangle relaxations, etc.).
- Backends other than Gurobi are straightforward once the `Solver` interface is implemented.
