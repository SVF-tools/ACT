# VNN-COMP support

This directory contains ACT's competition entry points:
[install_tool.sh](install_tool.sh), [prepare_instance.sh](prepare_instance.sh),
[run_instance.sh](run_instance.sh), and [act_run_instance.py](act_run_instance.py).

The tables below record the current implementation and its evidence. **Loading a
model, bounding its operators, proving its property, and emitting a competition
witness are different capabilities.** An implemented operator does not imply
that every model containing it is supported or that verification will finish
within a competition timeout.

Last reviewed: **2026-11-16**, against `b0afdc1` (PR #139 and PR #140 merged)
plus the dynamic-batch conversion change for issue #141 and branch
`feat/vnncomp2026-instance-queries` (2026-11-14): every query of a property in
one model, integer-cast input cases, interval `ArgMax` and act2torch lane
alignment. Later changes must update this record in the same PR; see
[Maintenance](#maintenance).

## Operator support

These are ONNX-to-ACT conversion paths, including ONNX operators converted to
native PyTorch modules. "Implemented" means that a lowering exists under the
listed conditions, not unrestricted ONNX opset or abstract-domain coverage.

A dynamic leading (batch) input dimension is fixed to 1 before simplification
only when no node reads a tensor shape (`Shape`/`Size`). Otherwise it stays
symbolic, onnxruntime's symbolic shape inference annotates the graph, and the
converted PyTorch module accepts any batch size.

Implementation owners:

- [ONNX module handlers](../act/pipeline/verification/utils.py):
  `ONNX_HANDLERS` and `_convert_Onnx*`.
- [Native module and FX lowering](../act/pipeline/verification/torch2act.py).
- [ACT layer definitions](../act/back_end/layer_schema.py) and
  [concrete reconstruction](../act/pipeline/verification/act2torch.py).
- Abstract domains: [interval](../act/back_end/interval_tf/interval_tf.py),
  [HybridZ](../act/back_end/hybridz_tf/hybridz_tf.py), and
  [dual](../act/back_end/dual_tf/dual_tf.py).

| ONNX operator / family | Conversion status | Current scope and restrictions |
| --- | --- | --- |
| `Gemm`, constant-weight `MatMul` | Implemented | Native linear layers or `DENSE`; the dedicated `MatMul` path accepts 1-D and 2-D constant weights. Batched matrix multiplication preserves the last-dimension contract. |
| Variable-variable `MatMul` | Implemented | Lowered to bilinear `MATMUL`; input shapes must be resolved and compatible. Bounds depend on the selected domain. |
| `Add`, `Sub`, `Mul`, `Div`, `Neg` | Implemented | Shape-aware elementwise lowering; constants can become `BIAS`/`SCALE`. A zero-crossing divisor can prevent useful finite bounds. |
| `Conv` | Implemented | Native `Conv1d`, `Conv2d`, and `Conv3d` paths with resolved channel-first shapes and parameters. |
| `ConvTranspose` | Implemented, 2-D | `ConvTranspose2d`; channel counts, kernel size, and output padding are retained through the ACT round trip. Other ranks are not covered by this path. The dual domain has no `CONVTRANSPOSE2D` transfer function (cgan2026 dual runs stop with an unknown-kind error). |
| `Relu`, `LeakyRelu`, `PRelu`, `Sigmoid`, `Tanh` | Implemented | Recognized native activation modules; sigmoid/tanh also have ONNX-function dispatch. Exported graphs must lower to recognized modules/functions. |
| `Abs`, `Sign`, `Erf`, `Sqrt`, `Sin`, `Cos` | Implemented | Dedicated handlers or the explicit `OnnxFunction` allowlist. Check domain restrictions and numerical behavior for the actual input range. |
| `Floor`, `Ceil`, `Round` | Implemented | `OnnxRound` lowers to `ROUND(mode)`. Concrete execution and interval/HybridZ/dual use the same step function; `round` uses ties-to-even. |
| `Pow` | Restricted | The ONNX handler requires one static scalar integer exponent of at least 2 and lowers it through `SQUARE`/`MUL`. This differs from polynomial expressions in a VNNLIB property. |
| `Min`, `Max` | Restricted | Elementwise two-input lowering; this is not support for arbitrary reduction operators or all variadic forms. |
| `MaxPool`, `AveragePool`, `GlobalAveragePool` | Implemented | Native pooling modules; global average pooling lowers to spatial means. Parameters and shapes must be representable by the selected backend. |
| `ReduceMean`, `ReduceSum` | Implemented, static axes | Mean and sum have explicit lowerings. The generic static-reduction handler rejects other reductions such as `ReduceMax`, `ReduceMin`, and `ReduceL2`. |
| `BatchNormalization`, `Softmax` | Implemented paths | Native batch-normalization and softmax modules are recognized. This does not establish coverage for every exporter representation or every Transformer graph. |
| `Reshape`, `Flatten`, `Transpose` | Implemented, resolved shapes | Reshape resolves ONNX `0`/`-1` dimensions and checks element counts. Fixed model dimensions are kept separate from verifier lanes. Transpose permits valid per-model permutations. |
| `Squeeze`, `Unsqueeze` | Restricted | Resolved axes/shape transformations; squeeze preserves the live verifier-lane axis. Arbitrary input-dependent shape changes are not supported. |
| `Concat`, `Expand` | Implemented, resolved shapes | Concat inputs and expand targets must be resolvable; constants can be materialized. Backend handling of the requested layout must also be available. HybridZ concatenates lane by lane (2026-11-14: before, two or more lanes stopped with "HZ and interval bounds have a non-numerical conflict" on ml4acopf and ACC). |
| `Slice` | Restricted | Starts, ends, axes, and steps must resolve during conversion and pass the supported slice-shape checks. Not a general dynamic-index operator. |
| `Split` | Restricted | The `OnnxSplit13` path requires explicit, positive, resolvable split sizes covering the axis. Equal splitting without a sizes input is rejected by this handler. |
| `Gather` | Restricted | Integer indices must be constant-resolvable, in range, and associated with a valid axis. Negative indices are normalized. |
| `ScatterND` | Partial | Fixed-shape lowering exists. Interval bounds conservatively enclose data and updates; this does not provide input-dependent tensor shapes or universal solver support. |
| `ArgMax`, `ArgMin` | Partial | Lowered to `ARG_EXTREMUM`. Interval (and HybridZ, which delegates to it) bounds the index by the first and last element that can still be the extremum, so a point box without ties gives the exact index. Other domains are not established. |
| `Equal`, `Less`, `Greater`, `LessOrEqual`, `GreaterOrEqual`, `Where` | Implemented paths | Comparison and conditional-selection layers over compatible resolved shapes. Domain-specific precision and availability still apply. |
| `QuantizeLinear`, `DequantizeLinear` | Restricted | Real-valued QDQ representation with supported scale/zero-point broadcasting. Dequantization folds constants, recognizes matching QDQ pairs, or emits scale/bias. Not a claim of arbitrary integer-kernel equivalence. |
| `Constant`, `Shape` | Implemented | Constants retain integer shape metadata; `Shape` is evaluated from the verified lane's shape at conversion time. Nodes computed only from such values (e.g. `Gather`/`Unsqueeze`/`Concat` building a `Reshape` target from a symbolic batch dimension) are folded during conversion and emit no layer. |
| `Dropout` | Inference only | Evaluation-mode passthrough. Training-time stochastic dropout is not the verified behavior. |
| `Cast` | Partial | Inside the network the lowering is a value passthrough; it does not model float-to-integer truncation of intermediate values. Input coordinates read only through `Gather` + `Cast` to an integer type are enumerated instead (integer-cast inputs, property table). |
| Value-dependent `Range`/shape/index construction, `If`/`Loop`/`Scan`, unregistered ONNX modules | Not covered | No general lowering is claimed. Exception: a subgraph computed only from integer-cast input coordinates is evaluated per case and cut off, provided the tensors it hands to the rest of the network have static shapes and the input's element type. |

Native PyTorch modules such as recurrent or attention layers have additional ACT
paths. Their presence in the layer registry alone is not evidence that an ONNX
export using them passes the complete conversion and verification pipeline.

## Model and benchmark evidence

"Checked" below means the named instances and execution paths were exercised.
`UNKNOWN` is a normal verifier result, not an unsupported model; it is also not
a proof. No row claims every instance in a benchmark is solved.

Two evidence scopes must remain distinct:

- **Branch checks:** after rebasing PR #139 onto `65952d7`, native category runs
  used CPU/float64, interval/torchlp, `--validate-soundness`, and
  `--max-instances 5`. The six named category runs below passed.
- **Deployment checks:** the 2026-11-13 workbench ran the combined integration
  engine with PR #139 changes and pre-existing integration fixes. These results
  are useful instance evidence, but do not certify the standalone branch, every
  backend, or the competition shell runner.

| Model / benchmark | Status | Evidence and remaining limits |
| --- | --- | --- |
| ACAS Xu MLPs (`acasxu_2023`) | Checked | Branch category soundness run passed. Per-property verdict and completion still depend on the query and budget. |
| Distribution-shift models (`dist_shift_2023`) | Checked | Branch category soundness run passed; do not extrapolate it to all model/property combinations. |
| SafeNLP embedding MLPs (`safenlp_2024`) | Checked | Branch category soundness run passed. The verified model consumes a 30-D embedding, not raw text or a Transformer encoder. Text provenance is a separate GUI capability. |
| Conditional GANs (`cgan2026`) | Deployment checked | One instance for each of 8 ONNX models loaded, ran interval verification (`UNKNOWN`), and completed fuzzing. Four models produced counterexamples. Conversion fixes include ConvTranspose, Resize, and shape operations. Each property has 2 queries (one per output side); 2026-11-14 branch category run passed with both as lanes of one model. |
| Power-grid networks (`ml4acopf_2024`) | Deployment checked | One instance for each of 9 ONNX models loaded and completed interval/fuzz runs. Residual variants require rounding support. The 2026-11-13 deployment `CERTIFIED` on three 300-bus instances covered only the first of their 276 queries; shape grouping now stacks all queries (1 to 600 per property) in one model. The category soundness run fails identically on PR #139 and this branch: the bounds validator reports `Hookable count mismatch` for these graphs. |
| Coupled monotonic ACAS Xu (`monotonic_acasxu_2026`) | API/category checked | Branch category soundness and direct build/interval/fuzz checks passed. Example combined model: 6 inputs, 10 outputs; shared coordinates plus a retained `LIN_POLY` relation. The competition shell entry still accepts one ONNX path. |
| Isomorphic ACAS Xu (`isomorphic_acasxu_2026`) | API/category checked | Branch category soundness and direct build/interval/fuzz checks passed. Example combined model: 5 inputs, 10 outputs. Each property has 10 queries, stacked in one model by shape grouping. Witness export depends on the caller; see the property table. |
| Nonlinear ACC (`adaptive_cruise_control_non_linear_2026`) | Checked | Branch category soundness run passed. Polynomial terms become extra model outputs. Properties have 38 or 62 queries with 2/4/5/7 output rows, padded into one model (2026-11-14). Interval with the concrete center check: instances 1 and 17 certify all 38 queries (VNN-COMP 2026: unsat); instance 2 (sat) and instance 32 (unsat) stay `UNKNOWN`. The 2026-11-13 deployment verdicts covered query 1 only. |
| Vision Transformers (`vit_2023`) | Checked | 2026-11-16 on `b0afdc1` plus the issue #141 change, CPU/float64. `pgd_2_3_16` and `ibp_3_3_8` keep their symbolic batch: the converted modules match ONNX Runtime at batch 1 and 3 (max abs difference ≤ 1.5e-6), and `--fuzz --max-instances 4` fuzzes four instances in one batch. `--verify vnnlib --max-instances 1` passed `--validate-soundness` for interval/torchlp, HybridZ TF/torchlp, the HybridZ solver, and dual; Level-2 bound alignment is skipped for this transformer lowering, as before. The ACT net keeps the Q/K/V projections as three `DENSE` layers (previously one `DENSE` and three `SLICE`s); interval output bounds and verdicts equal those of the batch-1 conversion. |
| Patch-based YOLO (`cctsdb_yolo_2023`) | Checked | The input is rank-1 `[12296]`: an image, the patch position `X[12288]`, `X[12289]` (read only through `Cast` to int64) and the reference box. The 63 x 63 positions give 190 distinct patch masks, one lane each of the static remainder model (input row: 12296 declared values, then 12288 mask values). 2026-11-14, CPU: patch-1 `idx_00559_0` certifies all 190 lanes (unsat); patch-3 `idx_05840_0` falsifies lane 88 (sat, position 1/25), and onnxruntime on the original ONNX gives `Y[0] = 0 <= 0.5` at the written witness inside the box. Both match VNN-COMP 2026 (alpha-beta-CROWN, PyRAT, NNV); the competition runner returns the same. Category run with `--validate-soundness`: counterexample validation passes; bounds validation errors (`Hookable count mismatch`) because the remainder runs one lane per call. |
| Other registered categories, CNN/ViT/recurrent exports | Not established by this record | [The category catalogue](../act/front_end/vnnlib_loader/category_mapping.py) controls discovery, not verified coverage. Add a dated model/property result here before claiming support. |

The deployment investigation also compared 17 cgan/ml4acopf converted models
with ONNX Runtime at 16 input-box points per model and rechecked 8 generated
counterexamples on their original ONNX models. Sampling of certified instances
found no violation; sampling is supplementary evidence, not a certification
proof or exhaustive operator validation.

For the issue #141 change, the 71 local VNN-COMP models with a dynamic batch
dimension and a listed instance (14 benchmarks) were rebuilt from their first
instance on `b0afdc1` and on the branch. Only the two `vit_2023` models changed
their ACT layers; interval verdicts and output bounds were unchanged for all 71.
`smart_turn_multimodal_2026`, the other instance-backed model that reads tensor
shapes, now also keeps its batch symbolic; its ACT net matched ONNX Runtime at
the instance centre.

## Property and entry-point support

Owners: [parser](../act/front_end/vnnlib_loader/vnnlib_parser.py),
[loaders](../act/front_end/vnnlib_loader/data_model_loader.py),
[spec creation](../act/front_end/vnnlib_loader/create_specs.py),
[model synthesis](../act/front_end/model_synthesis.py), and
[counterexample validation](../act/util/counterexample.py).

| Format / property / path | Status | Current scope and limits |
| --- | --- | --- |
| VNNLIB 2.0 tensor declarations and bracket indexing | Implemented | Declared tensor coordinates are mapped to flat ACT variables with shape checks. |
| Legacy VNNLIB 1.0 flat declarations | Unsupported | Rejected with `UnsupportedSpecError`; the shell protocol's `v1` argument is not a VNNLIB-version selector. |
| Box input bounds and linear output comparisons | Implemented | BOX inputs and `UNSAFE_LINEAR` output rows; recognizable classification properties can be promoted to `TOP1_ROBUST`. |
| Conjunction, disjunction, equality, `!=` / `distinct` | Implemented for supported expressions | Assertions are combined into queries; equality expands to two inequalities and disequality to strict alternatives. Unencodable branches raise rather than being silently dropped. |
| Linear relations between inputs | Implemented, closed constraints | Relations remain `LIN_POLY` rows with an enclosing BOX. Synthesis, fuzz mutation projection, and concrete counterexample checks retain them. |
| Strict inequalities | Partial | Output rows retain strictness. Input-box/polytope construction uses closed bounds and does not retain an exact open-domain flag; do not claim exact strict-input witness semantics. |
| Polynomial and mixed input/output assertions | Implemented for expandable polynomials | Sums/products, constant division, and nonnegative integer powers are expanded; terms other than a single output become extra model outputs. Variable division and other unencodable expressions are rejected. |
| Several `declare-network` blocks | API/category support | `create_specs_from_paths` accepts one ONNX path per network in declaration order. Each network must declare one input tensor. Top-level equal input coordinates share positions; other input relations remain constraints. |
| Several queries in one property | Implemented | `synthesize_models_and_seeds_from_specs(cd_group="shape")` stacks every BOX query of a property in one model, one lane per query; an `UNSAFE_LINEAR` lane with fewer rows repeats its own first row, which leaves its unsafe set unchanged. `LIN_POLY` and fixed-shape inputs keep one lane per model. The property is unsat only when every lane is certified. The competition runner aggregates its models; `verify_bab_batched` takes one root lane, so a multi-lane model gets `verify_once` with the concrete center check on every lane. |
| Integer-cast inputs | Implemented for BOX queries | Input coordinates read only through `Gather` + `Cast` to an integer type take `trunc(lb) .. trunc(ub)` (at most 65536 combinations per query). Each combination fixes them to a representative; tensors computed only from them become fixed extra inputs of the remainder model, and combinations with equal values share one lane. A lane's leading values are an input of the original ONNX model. The remainder module runs one lane per call in eager mode; traced conversion is unchanged. |
| Competition shell invocation | Single ONNX path | `run_instance.sh v1 BENCH ONNX VNNLIB RESULTS TIMEOUT` forwards one path. Python multi-network loading is not yet a multi-network shell protocol. |
| VNN-COMP `sat` assignment output | Single-network/single-input header path | The writer emits shaped input/output assignments, the declared input prefix of a longer integer-case row, and omits lifted terms from declared outputs. `extract_vnnlib_2_io_decls` returns no header for multiple networks/inputs; multi-network witness export is not established. |
| Empty or unsupported queries | Restricted | Empty input boxes can be discarded by the parser; a runner with no synthesized model raises. Unsupported conversion/specification errors are not certification results. |

## Reproducing checks

Run from the ACT repository root in the documented environment:

```bash
python -m act.back_end --generate --device cpu --dtype float32
python -m act.pipeline --verify act2torch --device cpu --dtype float32
python -m act.pipeline --verify netfactory --validate-soundness \
  --device cpu --dtype float32 --tf-modes interval hybridz --solvers torchlp
python -m act.pipeline --verify netfactory --validate-soundness \
  --device cpu --dtype float32 --solvers dual
python -m act.pipeline --verify vnnlib --category monotonic_acasxu_2026 \
  --max-instances 5 --validate-soundness --device cpu --dtype float64 \
  --tf-modes interval --solvers torchlp
```

Repeat the category command with the other checked categories and installed
benchmark data (`cctsdb_yolo_2023` with `--max-instances 3`). Integer-cast
inputs through the competition runner:

```bash
python vnncomp/act_run_instance.py <cctsdb>/onnx/patch-3.onnx \
  <cctsdb>/vnnlib/spec_onnx_patch-3_idx_05840_0.vnnlib out.txt 120 --device cpu
```

[NetFactory](../act/back_end/net_factory.py) contains the
`layer_testing_*` examples, including `layer_testing_round`; these exercise
conversion and abstract-domain behavior without requiring competition downloads.
For an actual competition result, separately exercise
`python vnncomp/act_run_instance.py --help` and the chosen model/property through
the runner, including witness validation when the result is `sat`.

## Maintenance

This is the maintained support record, not a one-time PR report.

1. Any change to ONNX lowering, ACT layer semantics, transfer functions,
   VNNLIB parsing/synthesis, counterexample handling, or the competition runner
   must update the affected table rows in the **same PR**.
2. State the supported parameter/shape range, input format, caller, backend,
   and known restrictions. Keep conversion, verification, and witness export
   separate; list unsupported cases explicitly.
3. Record the date and tested revision, model/property scope, command, device,
   precision, and result. Mark untested paths as untested. Catalogue membership,
   a handler registration, or a successful sample is not whole-benchmark coverage.
4. Add or extend the owning native checks when behavior changes. Keep runtime
   checks in their existing modules; do not make prose-string tests for this table.
5. When upstream fixes remove a limitation, update its row and evidence rather
   than leaving contradictory status notes. Preserve evidence scope when a test
   used an integration engine containing additional patches.
