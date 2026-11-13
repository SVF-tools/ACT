# Contributing to ACT

ACT (Abstract Constraint Transformer) is a PyTorch-native neural-network verification framework. Thanks for considering a contribution — this guide explains how to get set up, what kinds of changes are welcome, and what we expect in a pull request.

## What we welcome

The changes that get merged most easily:

- **Bug fixes** — especially soundness bugs (the verifier must never certify an unsafe property).
- **New layer support** — adding a `LayerKind` and its transfer functions.
- **New transfer functions / precision** — interval, hybrid-zonotope, or dual bounds for existing layers.
- **New solvers** — additional bounding/optimization backends behind the `Solver` interface.
- **Benchmark / dataset support** — new VNN-COMP categories or TorchVision dataset–model pairs.
- **Performance** — faster bounds propagation, batching, or BaB exploration (with before/after evidence).
- **Documentation** — READMEs, docstrings, examples.

For anything larger or architectural (new verification paradigm, major API change), open an issue to discuss first — see [Feature requests](#feature-requests).

## Before you start

- **Open an issue for non-trivial changes.** A short issue describing the bug or feature helps us avoid duplicate work and agree on scope before you invest time. Reference it with `Fixes #123` / `Closes #123` in your PR.
- **Check it doesn't already exist.** ACT has three TF families, several solvers, and many layer kinds — search the codebase before adding something new.
- **Trivial fixes** (typos, doc tweaks, obvious one-liners) don't need an issue.

## Development setup

**Requirements:** Conda (Miniconda), Python 3.12. Gurobi is optional (only for MILP/exact bounds).

```bash
# 1. Clone repository
git clone --recursive https://github.com/SVF-tools/ACT.git
cd ACT

# 2a. Primary core environment
conda env create -f environment.yml
conda activate act-py312

# 2b. (Optional) Run setup script (same act-py312 env, plus Gurobi)
bash modules/setup/setup.sh

# 3. (Optional) Gurobi license for MILP / exact HybridZ bounds
cp /path/to/gurobi.lic ./modules/gurobi/gurobi.lic
```

Verify the install:

```bash
python -m act.pipeline --help
```

## Project layout

ACT is a three-tier architecture:

- `act/front_end/` — data/model/spec loading and synthesis (`specs.py`, spec creators, `torchvision_loader/`, `vnnlib_loader/`).
- `act/back_end/` — the verification core: `core.py`, `verifier.py`, `layer_schema.py`, the `bab/` package, the transfer-function families (`interval_tf/`, `hybridz_tf/`, `dual_tf/`), and `solver/`.
- `act/pipeline/` — orchestration, Torch↔ACT conversion (`verification/`), validation, and fuzzing (`fuzzing/`).
- `act/util/` — shared device/path/config utilities.

## Running ACT

All native modules share `--device {cpu,cuda,gpu}` and `--dtype {float32,float64}`:

```bash
python -m act.back_end  --generate                          # build example nets from YAML
python -m act.back_end  --verify --network <name>           # verify a single ACT net (optionally --bab)
python -m act.pipeline  --verify vnnlib                      # end-to-end VNN-COMP verification
python -m act.pipeline  --fuzz --category acasxu_2023        # whitebox fuzzing
python -m act.front_end --list                               # list datasets/benchmarks
```

## Where to add things

| Contribution | Start here |
|---|---|
| New layer type | add to `LayerKind` in `act/back_end/layer_schema.py`, then wire transfer functions |
| Transfer function | `act/back_end/{interval_tf,hybridz_tf,dual_tf}/` (register the kind in the dispatcher + add the `tf_*` impl) |
| New solver | `act/back_end/solver/` implementing the `solver_base.py` interface |
| Benchmark / dataset | extend `BaseSpecCreator` (`act/front_end/spec_creator_base.py`) + register in `creator_registry.py` |
| Example network | add to the examples YAML in `act/back_end/examples/`, then `python -m act.back_end --generate` |

## Soundness first

ACT is a verifier — **soundness is non-negotiable**. A change is unsound if the verifier reports `CERTIFIED` when a concrete counterexample exists, or if abstract bounds fail to over-approximate concrete activations.

If you touch transfer functions, solvers, or BaB, validate before and after:

```bash
# Unified two-level validation (Level 1 counterexample + Level 2 per-neuron bounds)
python -m act.pipeline --verify netfactory --validate-soundness --device cpu --dtype float64
python -m act.pipeline --verify netfactory --validate-soundness --tf-modes interval hybridz dual
```

When a layer/solver can't yet handle an input soundly, **fail loud** (raise) rather than returning a possibly-unsound result.

## Testing & how to verify your change

CI runs across both precisions and all tiers. The relevant workflows:

- `act-backend-float64.yml` / `act-backend-float32.yml` — backend + verifier validation
- `act-bab.yml` — branch-and-bound (branching + bounding strategies)
- `act-frontend.yml` — loaders / spec creation
- `act-pipeline-verify.yml` — end-to-end VNNLIB + TorchVision
- `act-pipeline-fuzz.yml` — fuzzing

Reproduce the relevant ones locally (the `python -m act.*` commands above) before opening a PR. In your PR, state **what you ran and what the result was**.

For VNN-COMP changes, update the [operator, model, and property support
tables](vnncomp/README.md) in the same PR. This applies to ONNX conversion, layer
semantics, transfer functions, specification parsing/synthesis, counterexample
handling, and the competition runner. Record the tested scope and restrictions;
do not equate loading or catalogue membership with verification support.

## Pull request expectations

- **Branch and target `main`.** ACT integrates into a single branch — there's no separate `dev` branch. Work on a short-lived topic branch (e.g. `fix/...`, `feat/...`) and open your PR against `main`.
- **Keep PRs small and focused.** One concern per PR.
- **Explain the problem and your fix** in your own words — what changed and why.
- **Show how you verified it.** For logic changes: what you ran and how a reviewer can reproduce it. For numerical/soundness changes: include the `--validate-soundness` outcome.
- **No AI-generated walls of text.** Short, specific descriptions. If you can't explain it briefly, the PR is probably too large.
- **Don't suppress problems.** No bare `except:`, no silently swallowing errors, no deleting failing checks to go green.

### PR titles — Conventional Commits

`<type>(<scope>): <description>`, where `type` ∈ `feat | fix | docs | chore | refactor | test | perf | ci` and `scope` is the area touched (`back_end`, `front_end`, `pipeline`, `bab`, `dual`, `vnnlib`, `torch2act`, …).

```
fix(dual): correct ReLU backward LB on negative multipliers
feat(bab): add N-ary input splitting
perf(interval): vectorize per-sample CE collection
docs: refresh back_end README for dual solver
```

## Coding style

Follow the standards in [`.github/copilot-instructions.md`](./.github/copilot-instructions.md). In short:

- **Type hints + docstrings** on functions/classes; PEP 8 / PEP 257.
- **`dataclass`** for simple containers; ABCs for extensible interfaces.
- **`logging`, not `print`**, for diagnostics; clear, actionable error messages.
- **Precise types** — avoid `Any` and type-ignore escape hatches.
- **Device/dtype** via `act/util/device_manager.py`; **never** hardcode `torch.device("cuda")` or a dtype.
- **Paths** via `act/util/path_config.py`; never hardcode paths.
- **Remove legacy code and backward-compat shims** rather than layering around them — prefer clean over compatible.
- Prefer comprehensions, f-strings, and context managers where they read clearly.

## Feature requests

For net-new functionality, start with a design conversation: open an issue describing the problem, your proposed approach, and why it belongs in ACT. Please wait for maintainer agreement before opening a large feature PR.

## License

ACT is licensed under the **GNU Affero General Public License v3.0 or later (AGPLv3+)**. By contributing, you agree that your contributions are licensed under the same terms. Add the standard ACT license header to new source files.
