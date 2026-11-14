#===- act/front_end/vnnlib_loader/vnnlib_parser.py - VNNLIB Parser ----====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
#
# Purpose:
#   Parse VNNLIB SMT-LIB format files to extract input tensors and constraints.
#   Converts VNNLIB specifications to InputSpec and OutputSpec objects.
#
#===---------------------------------------------------------------------===#

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Tuple, Optional, Any, TYPE_CHECKING
import logging
import math
import torch
import re

from act.front_end.specs import InputSpec, OutputSpec, InKind, OutKind

if TYPE_CHECKING:
    from act.front_end.spec_creator_base import LabeledInputTensor

logger = logging.getLogger(__name__)


class VNNLibParseError(Exception):
    """Exception raised when VNNLIB parsing fails."""
    pass


class UnsupportedSpecError(ValueError):
    """Exception raised for soundly unsupported VNNLIB features."""
    pass


# -------------------------------------------------------------------------
# Public API
# -------------------------------------------------------------------------


def parse_vnnlib_to_tensors(
    vnnlib_path: Path,
    input_shape: Optional[Tuple[int, ...]] = None
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Parse a VNNLIB file to extract input tensor and metadata.
    
    The input tensor represents the center of the constrained input region.
    For box constraints with bounds [lb, ub], the center is (lb + ub) / 2.
    
    Args:
        vnnlib_path: Path to .vnnlib file
        input_shape: Expected input shape INCLUDING batch dimension (e.g., (1, 3, 32, 32))
                    If None, will be inferred or use flat shape
        
    Returns:
        Tuple of (input_tensor, metadata_dict) where:
        - input_tensor: torch.Tensor with batch dimension (e.g., shape (1, 3, 32, 32))
        - metadata_dict: Contains 'input_bounds', 'num_outputs', 'property_type'
        
    Raises:
        VNNLibParseError: If parsing fails
    """
    if not vnnlib_path.exists():
        raise VNNLibParseError(f"VNNLIB file not found: {vnnlib_path}")

    try:
        with open(vnnlib_path, 'r') as f:
            content = f.read()
        
        # VNNLIB 2.0 (declare-network) uses bracket vars X[i,..]/Y[j,..]; rewrite
        # them to flat X_n/Y_n so the shared bound-extraction logic below applies.
        if "(vnnlib-version" in content or "(declare-network" in content:
            if len(_network_blocks(content)) >= 2:
                content, layout = _multinet_rewrite(content)
                num_inputs, num_outputs = layout.num_inputs, layout.num_outputs
            else:
                content, num_inputs, num_outputs = _single_network_rewrite(content)
        else:
            raise UnsupportedSpecError(
                f"{vnnlib_path.name}: VNNLIB 1.0 flat format is no longer supported "
                f"(ACT is VNNLIB 2.0-only); provide a 2.0 file declaring "
                f"(vnnlib-version)/(declare-network)."
            )

        # Extract input bounds from top-level simple X-bound asserts only.
        # Constraints inside (or ...) branches must not be intersected here —
        # that would produce empty boxes for disjunctive-input properties
        # (e.g. ACAS Xu prop_5..10).
        conjuncts = _assert_conjuncts(_parse_all_forms(content))
        simple_bodies = [b for b in conjuncts if _is_simple_x_bound(b)]
        input_bounds = _extract_input_bounds(simple_bodies, num_inputs)
        # Relations between inputs (e.g. across the networks of a multi-network
        # file) also bound inputs that have no simple bound of their own.
        input_bounds = _tighten_bounds(
            input_bounds, _top_level_input_rows(conjuncts, num_inputs, num_outputs)
        )
        
        # Create input tensor from bounds center
        input_values = []
        for i in range(num_inputs):
            if i in input_bounds:
                lb, ub = input_bounds[i]
                center = (lb + ub) / 2.0
            else:
                # Default to 0 if no constraint
                center = 0.0
            input_values.append(center)
        
        input_tensor = torch.tensor(input_values)
        
        # Reshape if shape is provided; the shape already includes a leading batch dimension.
        if input_shape is not None:
            expected_numel = 1
            for dim in input_shape:
                expected_numel *= dim
            if input_tensor.numel() != expected_numel:
                raise VNNLibParseError(
                    f"Input size mismatch: got {input_tensor.numel()} "
                    f"values but expected {expected_numel} from shape {input_shape}"
                )
            # Reshape directly - input_shape already includes batch dimension
            input_tensor = input_tensor.view(*input_shape)
        
        # Infer property type
        property_type = _infer_property_type(content, num_outputs)
        
        metadata = {
            'input_bounds': input_bounds,
            'num_inputs': num_inputs,
            'num_outputs': num_outputs,
            'property_type': property_type,
            'vnnlib_path': str(vnnlib_path)
        }
        
        logger.info(
            f"Parsed VNNLIB: {num_inputs} inputs, {num_outputs} outputs, "
            f"type={property_type}"
        )
        
        return input_tensor, metadata

    except (VNNLibParseError, UnsupportedSpecError):
        raise
    except Exception as e:
        raise VNNLibParseError(f"Failed to parse {vnnlib_path}: {str(e)}")


def parse_vnnlib_queries(
    vnnlib_path: Path,
    labeled_tensor: Optional['LabeledInputTensor'] = None
) -> List[Tuple[InputSpec, OutputSpec]]:
    """
    Parse a VNNLIB 2.0 file into a list of verification queries.

    ACT is VNNLIB 2.0-only: the file must declare ``(vnnlib-version ...)`` /
    ``(declare-network ...)``. Legacy flat 1.0 files (bare ``declare-const X_0``)
    are rejected. Parsing is delegated to :func:`parse_vnnlib_2_0`, which
    ravel-rewrites bracket vars ``X[i,..]``/``Y[j,..]`` to flat ``X_n``/``Y_n``
    before the shared query-assembly core.

    Semantics:
      - Multiple top-level ``(assert ...)`` forms are conjunctive (implicit AND).
      - ``(or ...)`` inside an assert expands to multiple queries
        (Cartesian product across all asserts).
      - Inequalities linear in X alone are folded into the input BOX / LIN_POLY.
      - The others become rows of an UNSAFE_LINEAR OutputSpec over the outputs,
        followed by the products and inputs of :func:`lifted_output_terms`
        (the model must append those terms to its outputs).
      - When ``labeled_tensor.label`` is provided and queries match the
        classification pattern (one row Y_true - Y_j <= 0 for each j != true), the
        result collapses to a single TOP1_ROBUST OutputSpec.

    Raises:
        UnsupportedSpecError: If the file is legacy VNNLIB 1.0 flat format.
        VNNLibParseError: If the file is missing or unparseable.
    """
    if not vnnlib_path.exists():
        raise VNNLibParseError(f"VNNLIB file not found: {vnnlib_path}")
    try:
        with open(vnnlib_path, 'r') as f:
            content = f.read()
    except Exception as e:
        raise VNNLibParseError(f"Failed to read {vnnlib_path}: {e}") from e

    if "(vnnlib-version" in content or "(declare-network" in content:
        return parse_vnnlib_2_0(vnnlib_path, labeled_tensor=labeled_tensor)

    raise UnsupportedSpecError(
        f"{vnnlib_path.name}: VNNLIB 1.0 flat format is no longer supported "
        f"(ACT is VNNLIB 2.0-only); provide a 2.0 file declaring "
        f"(vnnlib-version)/(declare-network)."
    )


def extract_label_from_vnnlib(vnnlib_path: Path) -> Optional[int]:
    """
    Extract ground truth label from VNNLIB file comment.
    
    Many VNNLIB files (e.g., CIFAR-100) include ground truth labels in comments:
    ; CIFAR100 property with label: 14.
    
    Args:
        vnnlib_path: Path to .vnnlib file
        
    Returns:
        Ground truth label as integer, or None if not found
        
    Example:
        >>> label = extract_label_from_vnnlib(Path("spec.vnnlib"))
        >>> print(label)
        14
    """
    with open(vnnlib_path, 'r') as f:
        # Read first few lines (label is typically in first comment)
        for _ in range(5):
            line = f.readline()
            if not line:
                break
            
            # Match patterns like: ; CIFAR100 property with label: 14.
            match = re.search(r'label:\s*(\d+)', line, re.IGNORECASE)
            if match:
                return int(match.group(1))
    
    return None


# -------------------------------------------------------------------------
# Module-level regex patterns and type aliases
# -------------------------------------------------------------------------


_X_RE = re.compile(r"X_(\d+)")
_Y_RE = re.compile(r"Y_(\d+)")
_Ineq = Tuple[List[float], List[float], float, bool]
_Query = List[_Ineq]
# A monomial is the sorted tuple of its variables, e.g. ("X_1", "X_1"); () is the constant term.
_Monomial = Tuple[str, ...]
_Poly = Dict[_Monomial, float]
_COMPARISONS = ("<", ">", "<=", ">=")


# -------------------------------------------------------------------------
# VNNLIB 2.0 parsing and legacy-token rewrite
# -------------------------------------------------------------------------


def parse_vnnlib_2_0(
    vnnlib_path: Path,
    labeled_tensor: Optional['LabeledInputTensor'] = None,
) -> List[Tuple[InputSpec, OutputSpec]]:
    """Parse VNNLIB 2.0 by ravel-rewriting bracket variables to legacy names."""
    if not vnnlib_path.exists():
        raise VNNLibParseError(f"VNNLIB file not found: {vnnlib_path}")
    try:
        with open(vnnlib_path, 'r') as f:
            content = f.read()
    except Exception as e:
        raise VNNLibParseError(f"Failed to read {vnnlib_path}: {e}") from e

    if len(_network_blocks(content)) >= 2:
        return _parse_vnnlib_2_0_multinet(content, labeled_tensor)

    inputs = _extract_all_vnnlib_2_decls(content, "input")
    rewritten, num_inputs, num_outputs = _single_network_rewrite(content)
    default_shape = inputs[0][1] if len(inputs) == 1 else (1, num_inputs)
    tensor_shape = tuple(labeled_tensor.tensor.shape) if labeled_tensor is not None else tuple(default_shape)
    tensor_numel = _numel(tensor_shape)
    if tensor_numel != num_inputs:
        raise VNNLibParseError(
            f"VNNLIB 2.0 declared inputs {[shape for _, shape in inputs]} have {num_inputs} elements, "
            f"but model/sample input shape {tensor_shape} has {tensor_numel}"
        )
    true_label = labeled_tensor.label if labeled_tensor is not None else None
    return _queries_from_rewritten(
        rewritten, num_inputs, num_outputs, tensor_shape, true_label, vnnlib_path.name
    )


def _queries_from_rewritten(
    rewritten: str,
    num_inputs: int,
    num_outputs: int,
    tensor_shape: Tuple[int, ...],
    true_label,
    name: str,
) -> List[Tuple[InputSpec, OutputSpec]]:
    """Shared core: turn a flat-name-rewritten 2.0 body into (InputSpec, OutputSpec)
    queries. Used by both single-network and isomorphic dual-network 2.0 parsing."""
    try:
        forms = _parse_all_forms(rewritten)
    except VNNLibParseError:
        raise
    except Exception as e:
        raise VNNLibParseError(f"S-expression parse failed: {e}") from e

    assert_bodies = _assert_conjuncts(forms)
    simple_assert_bodies = [f for f in assert_bodies if _is_simple_x_bound(f)]
    complex_assert_bodies = [f for f in assert_bodies if not _is_simple_x_bound(f)]
    bounds_dict = _extract_input_bounds(simple_assert_bodies, num_inputs)
    base_in_spec = _build_input_spec(num_inputs, tensor_shape, bounds_dict, [])

    if not complex_assert_bodies:
        out_spec = _build_trivial_output_spec(num_outputs, true_label)
        logger.info(f"Parsed {name}: 1 query(ies) [vnnlib 2.0 input-only]")
        return [(base_in_spec, out_spec)]

    # Products of variables, and inputs next to outputs, become extra model
    # outputs (lifted_output_terms); their constraints are linear rows over them.
    lifted = {
        term: num_outputs + k
        for k, term in enumerate(_lifted_terms(complex_assert_bodies, num_inputs, num_outputs))
    }
    complex_queries = _combine_conjunctive_queries(
        [_process_body(body, num_inputs, num_outputs, lifted) for body in complex_assert_bodies]
    )
    in_specs: List[InputSpec] = []
    per_query_y_ineqs: List[_Query] = []
    for q in complex_queries:
        x_ineqs: _Query = []
        y_ineqs: _Query = []
        skip = False
        for xc, yc, d, strict in q:
            if any(v != 0 for v in yc):
                y_ineqs.append((xc, yc, d, strict))
            elif any(v != 0 for v in xc):
                x_ineqs.append((xc, yc, d, strict))
            elif d < 0 or (strict and d == 0):
                logger.debug(f"Infeasible constant constraint: 0 <= {d}")
                skip = True
                break
        if skip:
            continue
        in_spec = _build_input_spec(num_inputs, tensor_shape, bounds_dict, x_ineqs) if x_ineqs else base_in_spec
        lb, ub = in_spec.lb, in_spec.ub
        if lb is not None and ub is not None and bool((lb > ub).any()):
            logger.debug("Query with an empty input box dropped: no input satisfies its input rows")
            continue
        in_specs.append(in_spec)
        per_query_y_ineqs.append(y_ineqs)

    out_specs = _build_output_specs(per_query_y_ineqs, num_outputs + len(lifted), true_label)
    results: List[Tuple[InputSpec, OutputSpec]] = list(zip(in_specs, out_specs))

    # Classification promotion reads rows over the network outputs alone.
    if true_label is not None and not lifted:
        promoted = _try_promote_to_top1(results, num_outputs, true_label)
        if promoted is not None:
            results = [promoted]

    if len(results) > 1 and not lifted:
        promoted = _try_promote_to_top1_unlabeled(results, num_outputs)
        if promoted is not None:
            results = [promoted]

    logger.info(f"Parsed {name}: {len(results)} query(ies) [vnnlib 2.0]")
    return results


def _extract_all_vnnlib_2_decls(content: str, io_kind: str) -> List[Tuple[str, Tuple[int, ...]]]:
    """All declare-input/output entries (multi-network files have one per network)."""
    pattern = re.compile(
        rf"\(\s*declare-{io_kind}\s+([A-Za-z_]\w*)\s+\S+\s+\[([^\]]+)\]\s*\)",
        re.MULTILINE,
    )
    decls: List[Tuple[str, Tuple[int, ...]]] = []
    for m in pattern.finditer(content):
        dims = tuple(int(p.strip()) for p in m.group(2).split(",") if p.strip())
        if not dims or any(d <= 0 for d in dims):
            raise VNNLibParseError(f"Invalid VNNLIB 2.0 declare-{io_kind} shape: {m.group(2)}")
        decls.append((m.group(1), dims))
    if not decls:
        raise VNNLibParseError(f"VNNLIB 2.0 missing declare-{io_kind}")
    return decls


@dataclass(frozen=True)
class MultiNetworkLayout:
    """How the networks of a multi-network VNN-LIB 2.0 file read one combined input.

    The combined input holds every distinct input scalar once: scalars tied by a
    top-level equality between two inputs (e.g. ``(== X_f[i] X_g[i])``) share a
    position. ``input_index[k][i]`` is the position of the i-th row-major scalar
    of network ``k``'s input. Outputs concatenate in declaration order,
    ``[Y_f ; Y_g ; ...]``.
    """

    names: Tuple[str, ...]
    input_shapes: Tuple[Tuple[int, ...], ...]
    input_index: Tuple[Tuple[int, ...], ...]
    num_inputs: int
    num_outputs: int

    @property
    def shared_input(self) -> bool:
        """Whether every network reads the whole combined input as is (e.g. isomorphic-to)."""
        whole = tuple(range(self.num_inputs))
        return all(index == whole for index in self.input_index) and len(set(self.input_shapes)) == 1

    @property
    def input_shape(self) -> Tuple[int, ...]:
        """Shape of the combined input: a shared input keeps the networks' own shape."""
        return self.input_shapes[0] if self.shared_input else (1, self.num_inputs)


def multi_network_layout(vnnlib_path: Path) -> Optional[MultiNetworkLayout]:
    """Layout of a multi-network VNN-LIB 2.0 file, or None when it declares one network."""
    content = Path(vnnlib_path).read_text()
    if len(_network_blocks(content)) < 2:
        return None
    return _multinet_rewrite(content)[1]


@dataclass(frozen=True)
class LiftedTerms:
    """Terms ACT appends to the model outputs to check a property, in column order.

    A constraint with a product of variables (e.g. ``(>= (* X[0] 200) (* X[1] X[1]))``),
    or one relating inputs to outputs, is checked as a linear row over the
    network outputs followed by these terms. Each term is a product of factors
    ``("X", i)`` / ``("Y", j)`` over the flat (combined) input of ``num_inputs``
    scalars and output of ``num_outputs`` scalars; an input of such a constraint
    is a one-factor term. ``terms`` is empty when every constraint is linear in
    the outputs, or in the inputs alone.
    """

    terms: Tuple[Tuple[Tuple[str, int], ...], ...]
    num_inputs: int
    num_outputs: int


def lifted_output_terms(vnnlib_path: Path) -> LiftedTerms:
    """The :class:`LiftedTerms` of a VNN-LIB 2.0 property."""
    content = Path(vnnlib_path).read_text()
    if len(_network_blocks(content)) >= 2:
        flat, layout = _multinet_rewrite(content)
        num_inputs, num_outputs = layout.num_inputs, layout.num_outputs
    else:
        flat, num_inputs, num_outputs = _single_network_rewrite(content)
    bodies = [b for b in _assert_conjuncts(_parse_all_forms(flat)) if not _is_simple_x_bound(b)]
    terms = tuple(
        tuple((token[0], int(token[2:])) for token in term)
        for term in _lifted_terms(bodies, num_inputs, num_outputs)
    )
    return LiftedTerms(terms, num_inputs, num_outputs)


_DECLARE_NETWORK_RE = re.compile(r"\(\s*declare-network\s+([A-Za-z_]\w*)")


def _network_blocks(content: str) -> List[Tuple[str, str]]:
    """``(name, text)`` of every ``(declare-network ...)`` block, in declaration order."""
    text = re.sub(r";[^\n]*", "", content)
    blocks: List[Tuple[str, str]] = []
    for match in _DECLARE_NETWORK_RE.finditer(text):
        depth, end = 0, match.start()
        for end in range(match.start(), len(text)):
            depth += {"(": 1, ")": -1}.get(text[end], 0)
            if depth == 0:
                break
        blocks.append((match.group(1), text[match.start():end + 1]))
    return blocks


def _multinet_rewrite(content: str) -> Tuple[str, MultiNetworkLayout]:
    """Flatten a multi-network file to ``X_n``/``Y_n`` over the combined input and output.

    Each network declares one input and its outputs. Inputs are numbered network
    by network, then scalars tied by a top-level equality between two inputs
    merge into one position; the tie then reads ``X_k == X_k`` and drops out as
    a trivial row. Any other relation between inputs (e.g. ``X_f[0] >= X_g[0]``
    in a monotonicity property) stays a constraint on the combined input.
    """
    name_map: Dict[str, Tuple[str, Tuple[int, ...], int]] = {}
    names: List[str] = []
    shapes: List[Tuple[int, ...]] = []
    spans: List[range] = []
    num_x = num_y = 0
    for name, block in _network_blocks(content):
        inputs = _extract_all_vnnlib_2_decls(block, "input")
        if len(inputs) != 1:
            raise UnsupportedSpecError(
                f"network {name} declares {len(inputs)} inputs; multi-network files need one input per network"
            )
        in_name, in_shape = inputs[0]
        name_map[in_name] = ("X", in_shape, num_x)
        names.append(name)
        shapes.append(in_shape)
        spans.append(range(num_x, num_x + _numel(in_shape)))
        num_x += _numel(in_shape)
        for out_name, out_shape in _extract_all_vnnlib_2_decls(block, "output"):
            name_map[out_name] = ("Y", out_shape, num_y)
            num_y += _numel(out_shape)
    flat = _rewrite_bracket_vars(content, name_map)

    parent = list(range(num_x))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    # (i, j) means X_i <= X_j holds in every query; both directions tie them.
    below = set()
    for body in _assert_conjuncts(_parse_all_forms(flat)):
        if isinstance(body, list) and len(body) == 3 and body[0] in ("<=", ">="):
            lhs, rhs = (_X_RE.fullmatch(t) if isinstance(t, str) else None for t in body[1:])
            if lhs and rhs:
                i, j = int(lhs.group(1)), int(rhs.group(1))
                below.add((i, j) if body[0] == "<=" else (j, i))
    for i, j in below:
        if (j, i) in below:
            parent[find(i)] = find(j)
    position: Dict[int, int] = {}
    index = [position.setdefault(find(i), len(position)) for i in range(num_x)]
    rewritten = re.sub(r"\bX_(\d+)\b", lambda m: f"X_{index[int(m.group(1))]}", flat)
    layout = MultiNetworkLayout(
        names=tuple(names),
        input_shapes=tuple(shapes),
        input_index=tuple(tuple(index[i] for i in span) for span in spans),
        num_inputs=len(position),
        num_outputs=num_y,
    )
    return rewritten, layout


def _single_network_rewrite(content: str) -> Tuple[str, int, int]:
    """Flatten a one-network file to ``X_n``/``Y_n``; returns (rewritten, num_inputs, num_outputs).

    Several ``declare-input`` tensors are concatenated row-major in declaration
    order, matching the packed input ``onnx_converter.pack_onnx_inputs`` builds
    from the ONNX graph inputs (VNNLIB 2.0 declares inputs in graph order).
    Several outputs concatenate the same way.
    """
    name_map: Dict[str, Tuple[str, Tuple[int, ...], int]] = {}
    totals: Dict[str, int] = {}
    for io_kind, prefix in (("input", "X"), ("output", "Y")):
        base = 0
        for name, shape in _extract_all_vnnlib_2_decls(content, io_kind):
            name_map[name] = (prefix, shape, base)
            base += _numel(shape)
        totals[prefix] = base
    return _rewrite_bracket_vars(content, name_map), totals["X"], totals["Y"]


def _rewrite_bracket_vars(content: str, name_map: Dict[str, Tuple[str, Tuple[int, ...], int]]) -> str:
    """Rewrite ``name[i,..]`` to ``<prefix>_<base + ravel(i,..)>`` for every name in ``name_map``."""
    var_re = re.compile(r"\b([A-Za-z_]\w*)\s*\[([^\]]+)\]")

    def repl(m: "re.Match[str]") -> str:
        info = name_map.get(m.group(1))
        if info is None:
            return m.group(0)
        prefix, shape, base = info
        idx = tuple(int(p.strip()) for p in m.group(2).split(",") if p.strip())
        return f"{prefix}_{base + _ravel_c_order(idx, shape)}"

    return var_re.sub(repl, content)


def _parse_vnnlib_2_0_multinet(
    content: str,
    labeled_tensor: Optional['LabeledInputTensor'] = None,
) -> List[Tuple[InputSpec, OutputSpec]]:
    """Several networks (equal-to, isomorphic-to, ...): queries over the combined
    input and the concatenated output ``[Y_f ; Y_g ; ...]`` of the combined model."""
    rewritten, layout = _multinet_rewrite(content)
    tensor_shape = tuple(labeled_tensor.tensor.shape) if labeled_tensor is not None else layout.input_shape
    if _numel(tensor_shape) != layout.num_inputs:
        raise VNNLibParseError(
            f"multi-network combined input has {layout.num_inputs} elements, "
            f"but the sample shape {tensor_shape} has {_numel(tensor_shape)}"
        )
    return _queries_from_rewritten(
        rewritten, layout.num_inputs, layout.num_outputs, tensor_shape, None, "multi-network"
    )


def _extract_vnnlib_2_decl(content: str, io_kind: str) -> Tuple[str, str, Tuple[int, ...]]:
    """Return ``(var_name, dtype, shape)``; dtype feeds the 2.0 witness header."""
    pattern = re.compile(
        rf"\(\s*declare-{io_kind}\s+([A-Za-z_]\w*)\s+(\S+)\s+\[([^\]]+)\]\s*\)",
        re.MULTILINE,
    )
    match = pattern.search(content)
    if match is None:
        raise VNNLibParseError(f"VNNLIB 2.0 missing declare-{io_kind}")
    dims = tuple(int(part.strip()) for part in match.group(3).split(",") if part.strip())
    if not dims or any(dim <= 0 for dim in dims):
        raise VNNLibParseError(f"Invalid VNNLIB 2.0 declare-{io_kind} shape: {match.group(3)}")
    return match.group(1), match.group(2), dims


def _numel(shape: Tuple[int, ...]) -> int:
    total = 1
    for dim in shape:
        total *= int(dim)
    return total


def _ravel_c_order(indices: Tuple[int, ...], shape: Tuple[int, ...]) -> int:
    if len(indices) != len(shape):
        raise VNNLibParseError(f"Index rank {indices} does not match declared shape {shape}")
    flat = 0
    for idx, dim in zip(indices, shape):
        if idx < 0 or idx >= dim:
            raise VNNLibParseError(f"Index {indices} out of bounds for declared shape {shape}")
        flat = flat * dim + idx
    return flat


def _assert_conjuncts(forms: List[Any]) -> List[Any]:
    """Normalized bodies of the top-level asserts, with top-level ``and`` split.

    Asserts are conjunctive, so ``(assert (and a b))`` equals ``(assert a)
    (assert b)``. Splitting lets per-variable box bounds written as
    ``(and (>= X_i lo) (<= X_i hi))`` take the fast simple-bound path and
    set the input center, instead of entering query expansion as one
    complex assertion each.
    """
    bodies: List[Any] = []
    pending = [
        _normalize_vnnlib_2_body(f[1]) for f in reversed(forms)
        if isinstance(f, list) and len(f) >= 2 and f[0] == "assert"
    ]
    while pending:
        body = pending.pop()
        if isinstance(body, list) and body and body[0] == "and":
            pending.extend(reversed(body[1:]))
        else:
            bodies.append(body)
    return bodies


def _normalize_vnnlib_2_body(body: Any) -> Any:
    if not isinstance(body, list) or not body:
        return body
    op = body[0]
    children = [_normalize_vnnlib_2_body(child) for child in body[1:]]
    if op in ("=", "==") and len(children) == 2:
        return ["and", ["<=", children[0], children[1]], [">=", children[0], children[1]]]
    if op in ("!=", "distinct") and len(children) == 2:
        return ["or", ["<", children[0], children[1]], [">", children[0], children[1]]]
    return [op] + children


# -------------------------------------------------------------------------
# Shared bound/property extractors (operate on flat X_n/Y_n post-rewrite)
# -------------------------------------------------------------------------


def _extract_input_bounds(
    simple_bodies: List[Any],
    num_inputs: int,
) -> Dict[int, Tuple[float, float]]:
    """Extract per-variable [lb, ub] from top-level simple X-bound asserts.

    ``simple_bodies`` is the list of assertion bodies that already passed
    ``_is_simple_x_bound``. Constraints inside ``(or ...)`` branches must be
    excluded by the caller — intersecting them here would produce empty boxes
    for disjunctive-input properties (ACAS Xu prop_5..10).
    """
    bounds = {i: [float('-inf'), float('inf')] for i in range(num_inputs)}
    for body in simple_bodies:
        if not (isinstance(body, list) and len(body) == 3):
            continue
        op, left, right = body
        if op not in ("<", ">", "<=", ">="):
            continue
        x_is_left = isinstance(left, str) and bool(_X_RE.fullmatch(left))
        x_is_right = isinstance(right, str) and bool(_X_RE.fullmatch(right))
        if x_is_left == x_is_right:
            continue
        x_tok = left if x_is_left else right
        lit_tok = right if x_is_left else left
        val = float(lit_tok)  # _is_simple_x_bound admitted only numeric literals
        x_match = _X_RE.fullmatch(x_tok)
        if x_match is None:
            continue
        idx = int(x_match.group(1))
        assert idx < num_inputs, f"X_{idx} is bounded but the property declares {num_inputs} inputs"
        # (<= X val)  -> X <= val   (upper)
        # (>= X val)  -> X >= val   (lower)
        # (<= val X)  -> val <= X   -> X >= val (lower)
        # (>= val X)  -> val >= X   -> X <= val (upper)
        tighten_upper = (op in ("<", "<=") and x_is_left) or (op in (">", ">=") and not x_is_left)
        if tighten_upper:
            bounds[idx][1] = min(bounds[idx][1], val)
        else:
            bounds[idx][0] = max(bounds[idx][0], val)
    return {
        i: (lb, ub)
        for i, (lb, ub) in bounds.items()
        if lb != float('-inf') or ub != float('inf')
    }


def _infer_property_type(content: str, num_outputs: int) -> str:
    """
    Infer the property type from VNNLIB content.
    
    Returns:
        One of: 'classification', 'safety', 'unknown'
    """
    content_lower = content.lower()
    
    # Classification properties often involve comparisons between outputs
    if 'y_' in content_lower and num_outputs > 1:
        # Check for patterns like Y_i - Y_j > 0 (classification margin)
        if re.search(r'y_\d+\s*[-]\s*y_\d+', content_lower):
            return 'classification'
    
    # Safety properties typically have output range constraints
    if num_outputs == 1 or 'range' in content_lower:
        return 'safety'
    
    return 'unknown'


# -------------------------------------------------------------------------
# Helper: label coercion
# -------------------------------------------------------------------------


def _coerce_label_to_tensor(true_label: Any) -> torch.Tensor:
    """Coerce int/tensor label to 1-D int64 tensor (defensive)."""
    if isinstance(true_label, torch.Tensor):
        return true_label.clone()
    return torch.tensor([int(true_label)], dtype=torch.int64)


# -------------------------------------------------------------------------
# Step 1: S-expression tokenization and AST parsing
# -------------------------------------------------------------------------


def _tokenize_sexpr(text: str) -> List[str]:
    lines = []
    for raw in text.split("\n"):
        idx = raw.find(";")
        if idx >= 0:
            raw = raw[:idx]
        lines.append(raw)
    text = " ".join(lines)
    text = text.replace("(", " ( ").replace(")", " ) ")
    return [str(tok) for tok in text.split()]


def _parse_sexpr(tokens: List[str], pos: int) -> Tuple[Any, int]:
    if pos >= len(tokens):
        raise VNNLibParseError("Unexpected EOF in S-expression")
    tok = tokens[pos]
    if tok == "(":
        result = []
        pos += 1
        while pos < len(tokens) and tokens[pos] != ")":
            item, pos = _parse_sexpr(tokens, pos)
            result.append(item)
        if pos >= len(tokens):
            raise VNNLibParseError("Unbalanced '('")
        return result, pos + 1
    if tok == ")":
        raise VNNLibParseError(f"Unexpected ')' at pos {pos}")
    return tok, pos + 1


def _parse_all_forms(text: str) -> List[Any]:
    tokens = _tokenize_sexpr(text)
    out = []
    pos = 0
    while pos < len(tokens):
        form, pos = _parse_sexpr(tokens, pos)
        out.append(form)
    return out


# -------------------------------------------------------------------------
# Step 2: Linear algebra on S-expressions
# -------------------------------------------------------------------------


def _poly_add(*polys: _Poly) -> _Poly:
    out: _Poly = {}
    for poly in polys:
        for mono, coef in poly.items():
            out[mono] = out.get(mono, 0.0) + coef
    return {mono: coef for mono, coef in out.items() if coef != 0.0}


def _poly_scale(poly: _Poly, factor: float) -> _Poly:
    return {mono: coef * factor for mono, coef in poly.items() if coef * factor != 0.0}


def _poly_mul(a: _Poly, b: _Poly) -> _Poly:
    out: _Poly = {}
    for ma, ca in a.items():
        for mb, cb in b.items():
            mono = tuple(sorted(ma + mb))
            out[mono] = out.get(mono, 0.0) + ca * cb
    return {mono: coef for mono, coef in out.items() if coef != 0.0}


def _parse_poly(expr: Any, num_inputs: int, num_outputs: int) -> Optional[_Poly]:
    """Expand an arithmetic expression into a polynomial over ``X_i`` / ``Y_j``.

    Numbers, variables, n-ary ``+`` and ``*``, unary and n-ary ``-``, division
    by a constant and non-negative integer powers expand exactly; anything else
    (e.g. division by a variable) gives None.
    """
    if isinstance(expr, str):
        match = _X_RE.fullmatch(expr) or _Y_RE.fullmatch(expr)
        if match:
            bound = num_inputs if expr.startswith("X") else num_outputs
            return {(expr,): 1.0} if int(match.group(1)) < bound else None
        try:
            return {(): float(expr)}
        except ValueError:
            return None
    if not (isinstance(expr, list) and len(expr) >= 2):
        return None
    op = expr[0]
    parsed = [_parse_poly(arg, num_inputs, num_outputs) for arg in expr[1:]]
    args: List[_Poly] = [arg for arg in parsed if arg is not None]
    if len(args) != len(parsed):
        return None
    if op == "+":
        return _poly_add(*args)
    if op == "-":
        if len(args) == 1:
            return _poly_scale(args[0], -1.0)
        return _poly_add(args[0], *(_poly_scale(arg, -1.0) for arg in args[1:]))
    if op == "*":
        product: _Poly = {(): 1.0}
        for arg in args:
            product = _poly_mul(product, arg)
        return product
    if op == "/" and len(args) >= 2 and all(set(arg) <= {()} for arg in args[1:]):
        denominator = math.prod(arg.get((), 0.0) for arg in args[1:])
        return _poly_scale(args[0], 1.0 / denominator) if denominator != 0.0 else None
    if op in ("^", "pow") and len(args) == 2 and set(args[1]) <= {()}:
        exponent = args[1].get((), 0.0)
        if exponent >= 0 and float(exponent).is_integer():
            power: _Poly = {(): 1.0}
            for _ in range(int(exponent)):
                power = _poly_mul(power, args[0])
            return power
    return None


def _comparison_poly(lhs: Any, rhs: Any, num_inputs: int, num_outputs: int) -> Optional[_Poly]:
    """``lhs - rhs`` as a polynomial, or None when either side does not expand."""
    left = _parse_poly(lhs, num_inputs, num_outputs)
    right = _parse_poly(rhs, num_inputs, num_outputs)
    return None if left is None or right is None else _poly_add(left, _poly_scale(right, -1.0))


def _is_input_poly(poly: _Poly) -> bool:
    """Whether a constraint polynomial is linear in the inputs alone."""
    return all(not mono or (len(mono) == 1 and mono[0].startswith("X")) for mono in poly)


def _is_output_term(mono: _Monomial) -> bool:
    return len(mono) == 1 and mono[0].startswith("Y")


def _parse_inequality(
    op: str, lhs: Any, rhs: Any, num_inputs: int, num_outputs: int,
    lifted: Optional[Dict[_Monomial, int]] = None,
) -> Optional[_Ineq]:
    """``op(lhs, rhs)`` as a row ``xc·x + yc·y <= d`` (strict for ``<`` and ``>``).

    A comparison linear in the inputs alone keeps its input coefficients ``xc``.
    Any other is a row over the outputs: ``yc`` has a column per output and one
    per lifted term (``lifted`` maps a term to its column), and ``xc`` is zero.
    None when the comparison does not expand or needs a term that is not lifted.
    """
    poly = _comparison_poly(lhs, rhs, num_inputs, num_outputs)
    if poly is None or op not in _COMPARISONS:
        return None
    lifted = lifted or {}
    xc = [0.0] * num_inputs
    yc = [0.0] * (num_outputs + len(lifted))
    input_row = _is_input_poly(poly)
    for mono, coef in poly.items():
        if not mono:
            continue
        if input_row:
            xc[int(mono[0][2:])] += coef
        elif _is_output_term(mono):
            yc[int(mono[0][2:])] += coef
        elif mono in lifted:
            yc[lifted[mono]] += coef
        else:
            return None
    d = -poly.get((), 0.0)
    if op in ("<", "<="):
        return xc, yc, d, op == "<"
    return [-v for v in xc], [-v for v in yc], -d, op == ">"


# -------------------------------------------------------------------------
# Step 3: AND/OR composition and query assembly
# -------------------------------------------------------------------------


def _is_simple_x_bound(body: Any) -> bool:
    """Fast check: is body a simple single-variable X bound (op X_i val) / (op val X_i)?

    These asserts are fully handled by the regex-based :func:`_extract_input_bounds`
    and do NOT need S-expression processing; skipping them avoids quadratic blow-up
    from the Cartesian product when many X-bound asserts coexist with a large
    disjunctive output (common in CIFAR-100-style classification files).
    """
    if not (isinstance(body, list) and len(body) == 3):
        return False
    op = body[0]
    if op not in ("<", ">", "<=", ">="):
        return False
    left, right = body[1], body[2]
    if not (isinstance(left, str) and isinstance(right, str)):
        return False
    left_is_x = bool(_X_RE.fullmatch(left))
    right_is_x = bool(_X_RE.fullmatch(right))
    if left_is_x == right_is_x:
        return False
    other = right if left_is_x else left
    try:
        float(other)
        return True
    except (ValueError, TypeError):
        return False


def _process_body(
    body: Any, num_inputs: int, num_outputs: int,
    lifted: Optional[Dict[_Monomial, int]] = None,
) -> List[_Query]:
    """DNF of an assertion body: one query (a conjunction of rows) per disjunct.

    Raises UnsupportedSpecError on anything it cannot encode exactly: dropping
    a conjunct or a disjunct would change the property.
    """
    if isinstance(body, list) and body and body[0] in ("and", "or"):
        subs = [_process_body(child, num_inputs, num_outputs, lifted) for child in body[1:]]
        if body[0] == "or":
            return [query for sub in subs for query in sub]
        return _combine_conjunctive_queries(subs)
    if isinstance(body, list) and len(body) == 3 and body[0] in _COMPARISONS:
        ineq = _parse_inequality(body[0], body[1], body[2], num_inputs, num_outputs, lifted)
        if ineq is not None:
            return [[ineq]]
    raise UnsupportedSpecError(f"unsupported VNNLIB 2.0 assertion: {str(body)[:300]}")


def _combine_conjunctive_queries(qs: List[List[_Query]]) -> List[_Query]:
    combined = [[]]
    for q_list in qs:
        new_combined = []
        for base in combined:
            for q in q_list:
                new_combined.append(base + q)
        combined = new_combined
    return combined


# -------------------------------------------------------------------------
# Step 4 (+ Step 5 TOP1 promotion): Spec builders
# -------------------------------------------------------------------------


def _build_input_spec(
    num_inputs: int,
    input_shape: Optional[Tuple[int, ...]],
    bounds_dict: Dict[int, Tuple[float, float]],
    extra_x_ineqs: _Query,
) -> InputSpec:
    """Build an InputSpec from regex-extracted base bounds + the query's own X rows.

    ``bounds_dict`` is the output of :func:`_extract_input_bounds` — fast regex
    per-variable (lb, ub) pairs. ``extra_x_ineqs`` holds the X-only inequalities
    of one query (from complex asserts, e.g. disjunctive X ranges).
    Single-variable rows refine the BOX. Rows relating several inputs (e.g.
    ``X_f[0] >= X_g[0]`` across the networks of a multi-network file) are no box:
    they tighten it and stay a LIN_POLY polytope ``A x <= b`` over that
    enclosing box.
    """
    lb_vals = [bounds_dict.get(i, (float('-inf'), float('inf')))[0] for i in range(num_inputs)]
    ub_vals = [bounds_dict.get(i, (float('-inf'), float('inf')))[1] for i in range(num_inputs)]
    lb_tensor = torch.tensor(lb_vals)
    ub_tensor = torch.tensor(ub_vals)
    for xc, _yc, d, _strict in extra_x_ineqs:
        nonzero = [(i, v) for i, v in enumerate(xc) if v != 0]
        if len(nonzero) == 1:
            idx, coef = nonzero[0]
            bound = d / coef
            if coef > 0 and bound < ub_tensor[idx].item():
                ub_tensor[idx] = bound
            elif coef < 0 and bound > lb_tensor[idx].item():
                lb_tensor[idx] = bound
    rows = [(xc, d) for xc, _yc, d, _strict in extra_x_ineqs if sum(v != 0 for v in xc) > 1]
    if rows:
        box = _tighten_bounds(
            {i: (float(lb_tensor[i]), float(ub_tensor[i])) for i in range(num_inputs)}, rows
        )
        lb_tensor = torch.tensor([box[i][0] for i in range(num_inputs)])
        ub_tensor = torch.tensor([box[i][1] for i in range(num_inputs)])
    if input_shape is not None:
        lb_tensor = lb_tensor.view(*input_shape)
        ub_tensor = ub_tensor.view(*input_shape)
    if not rows:
        return InputSpec(kind=InKind.BOX, lb=lb_tensor, ub=ub_tensor)
    return InputSpec(
        kind=InKind.LIN_POLY, lb=lb_tensor, ub=ub_tensor,
        A=torch.tensor([xc for xc, _d in rows]), b=torch.tensor([d for _xc, d in rows]),
    )


def _comparisons(body: Any) -> Iterator[List[Any]]:
    """The comparisons of an and/or assertion body, depth first."""
    if isinstance(body, list) and body and body[0] in ("and", "or"):
        for child in body[1:]:
            yield from _comparisons(child)
    elif isinstance(body, list) and len(body) == 3 and body[0] in _COMPARISONS:
        yield body
    else:
        raise UnsupportedSpecError(f"unsupported VNNLIB 2.0 assertion: {str(body)[:300]}")


def _input_only(body: Any, num_inputs: int, num_outputs: int) -> bool:
    """Whether every comparison of an assertion body is linear in the inputs alone."""
    for _op, lhs, rhs in _comparisons(body):
        poly = _comparison_poly(lhs, rhs, num_inputs, num_outputs)
        if poly is None or not _is_input_poly(poly):
            return False
    return True


def _lifted_terms(bodies: List[Any], num_inputs: int, num_outputs: int) -> List[_Monomial]:
    """Terms of the constraints checked on the outputs, other than a single output.

    A comparison linear in the inputs alone constrains the input domain. Any
    other comparison is checked on the model outputs, so each of its terms that
    is not a single ``Y_j`` (an input, or a product of variables) becomes an
    extra model output, in order of first appearance.
    """
    terms: Dict[_Monomial, None] = {}
    for body in bodies:
        for _op, lhs, rhs in _comparisons(body):
            poly = _comparison_poly(lhs, rhs, num_inputs, num_outputs)
            if poly is None or _is_input_poly(poly):
                continue
            for mono in poly:
                if mono and not _is_output_term(mono):
                    terms.setdefault(mono)
    return list(terms)


def _top_level_input_rows(
    conjuncts: List[Any], num_inputs: int, num_outputs: int,
) -> List[Tuple[List[float], float]]:
    """``a·x <= d`` rows of the top-level asserts that relate several inputs.

    Only conjunctive (single-query) asserts linear in the inputs alone qualify:
    they hold in every query, so they may tighten the input box shared by all
    queries.
    """
    rows: List[Tuple[List[float], float]] = []
    for body in conjuncts:
        if _is_simple_x_bound(body) or not _input_only(body, num_inputs, num_outputs):
            continue
        queries = _process_body(body, num_inputs, num_outputs)
        if len(queries) == 1:
            rows.extend((xc, d) for xc, _yc, d, _strict in queries[0] if sum(v != 0 for v in xc) > 1)
    return rows


def _tighten_bounds(
    bounds: Dict[int, Tuple[float, float]], rows: List[Tuple[List[float], float]],
) -> Dict[int, Tuple[float, float]]:
    """Tighten per-variable bounds with linear rows ``a·x <= d`` (interval propagation).

    A row bounds each of its variables by ``d`` minus the least value its other
    terms take over the current bounds. Only points violating a row are cut, so
    the box keeps enclosing the feasible set; a few sweeps reach the fixpoint for
    the short relational rows VNN-LIB files carry.
    """
    if not rows:
        return bounds
    lo = {i: float(b[0]) for i, b in bounds.items()}
    hi = {i: float(b[1]) for i, b in bounds.items()}
    for _sweep in range(8):
        changed = False
        for coeffs, d in rows:
            terms = [(j, a) for j, a in enumerate(coeffs) if a != 0]
            least = {j: min(a * lo[j], a * hi[j]) for j, a in terms}
            for j, a in terms:
                rest = sum(v for k, v in least.items() if k != j)
                if not math.isfinite(rest):
                    continue
                bound = (d - rest) / a
                if a > 0 and bound < hi[j]:
                    hi[j], changed = bound, True
                elif a < 0 and bound > lo[j]:
                    lo[j], changed = bound, True
        if not changed:
            break
    return {i: (lo[i], hi[i]) for i in bounds}


def _build_trivial_output_spec(num_outputs: int, true_label) -> OutputSpec:
    """OutputSpec for a query that carries no output inequality: TOP1 against the
    known label, else an unconstrained RANGE."""
    if true_label is not None:
        y_true = _coerce_label_to_tensor(true_label)
        return OutputSpec(kind=OutKind.TOP1_ROBUST, y_true=y_true)
    return OutputSpec(
        kind=OutKind.RANGE,
        lb=torch.tensor([float('-inf')] * num_outputs),
        ub=torch.tensor([float('inf')] * num_outputs),
    )


def _build_output_specs(
    per_query_ineqs: List[_Query],
    num_outputs: int,
    true_label,
) -> List[OutputSpec]:
    """Build one OutputSpec per query, materialising EVERY constraint row of the
    file in a single tensor construction.

    Each query then takes a contiguous row slice of that matrix, so row order,
    values, dtype and device are exactly those of a per-query construction. The
    batching matters because the default device is CUDA: one construction per
    query is one host-to-device transfer per query, and a CIFAR-100 file holds
    99 single-row queries — the dominant cost of VNNLIB parsing.
    """
    rows_c: List[List[float]] = []
    rows_d: List[float] = []
    rows_strict: List[bool] = []
    spans: List[Tuple[int, int]] = []
    for ineqs in per_query_ineqs:
        start = len(rows_c)
        for _xc, yc, d, strict in ineqs:
            rows_c.append(list(yc))
            rows_d.append(float(d))
            rows_strict.append(strict)
        spans.append((start, len(rows_c)))

    if not rows_c:
        return [_build_trivial_output_spec(num_outputs, true_label) for _ in spans]

    c_all = torch.tensor(rows_c)
    d_all = torch.tensor(rows_d)
    strict_all = torch.tensor(rows_strict, dtype=torch.bool) if any(rows_strict) else None
    return [
        _build_trivial_output_spec(num_outputs, true_label) if start == stop
        else OutputSpec(
            kind=OutKind.UNSAFE_LINEAR,
            c=c_all[start:stop],
            d=d_all[start:stop],
            strict=strict_all[start:stop] if strict_all is not None else None,
        )
        for start, stop in spans
    ]


def _try_promote_to_top1(
    queries: List[Tuple[InputSpec, OutputSpec]],
    num_outputs: int,
    true_label,
) -> Optional[Tuple[InputSpec, OutputSpec]]:
    """Collapse N UNSAFE_LINEAR queries into a single TOP1_ROBUST when possible.

    Structural requirements:
      - All queries share identical input BOX bounds.
      - Every query has exactly ONE output inequality of the form
        ``Y_true - Y_other <= 0`` (coefficients +1 on the true class, -1 on
        the other class, RHS = 0), i.e. unsafe iff Y_other >= Y_true.
      - The set of ``other`` indices covers every class except ``true_label``.

    Orientation (CRITICAL): VNNLIB uses BOTH ``(>= Y_j Y_true)`` and
    ``(<= Y_true Y_j)``; ``_parse_inequality`` canonicalises both to the row
    ``Y_true - Y_j <= 0``, which is accepted. A row with the true class at -1
    (``Y_j - Y_true <= 0``) encodes "Y_true is minimal" (an argmin property)
    and is NOT a top-1 robustness property, so it stays UNSAFE_LINEAR.
    """
    if not queries:
        return None
    t_idx = (int(true_label.item()) if isinstance(true_label, torch.Tensor)
             else int(true_label))
    first_in = queries[0][0]
    if first_in.lb is None or first_in.ub is None:
        return None
    for in_spec, _ in queries[1:]:
        if in_spec.lb is None or in_spec.ub is None:
            return None
        if not torch.equal(first_in.lb, in_spec.lb) or not torch.equal(first_in.ub, in_spec.ub):
            return None
    expected = {j for j in range(num_outputs) if j != t_idx}
    covered = set()
    for _, out_spec in queries:
        if out_spec.kind != OutKind.UNSAFE_LINEAR:
            return None
        if out_spec.strict is not None and bool(out_spec.strict.any()):
            return None
        c_mat = out_spec.c
        d_vec = out_spec.d
        if c_mat is None or d_vec is None:
            return None
        if c_mat.dim() == 1:
            c_mat = c_mat.unsqueeze(0)
        if c_mat.shape[0] != 1 or d_vec.reshape(-1).shape[0] != 1:
            return None
        row = c_mat[0].tolist()
        d_val = float(d_vec.reshape(-1)[0].item())
        if abs(d_val) > 1e-6:
            return None
        nz = [(i, v) for i, v in enumerate(row) if abs(v) > 1e-9]
        if len(nz) != 2:
            return None
        pos = [i for i, v in nz if v > 0]
        neg = [i for i, v in nz if v < 0]
        if len(pos) != 1 or len(neg) != 1:
            return None
        val_pos = [v for _, v in nz if v > 0][0]
        val_neg = [v for _, v in nz if v < 0][0]
        if abs(val_pos - 1.0) > 1e-6 or abs(val_neg + 1.0) > 1e-6:
            return None
        if pos[0] != t_idx:
            return None
        covered.add(neg[0])
    if covered != expected:
        return None
    y_true = _coerce_label_to_tensor(true_label)
    return queries[0][0], OutputSpec(kind=OutKind.TOP1_ROBUST, y_true=y_true)


def _try_promote_to_top1_unlabeled(
    queries: List[Tuple[InputSpec, OutputSpec]],
    num_outputs: int,
) -> Optional[Tuple[InputSpec, OutputSpec]]:
    """Label-agnostic TOP1 recognition for 2.0 files that omit the label comment.

    Soundness invariant: the shared class ``t`` must be one of the two nonzero
    indices of the first canonicalised row ``e_t - e_j``, so only those two are
    trialled; the full structural + full-coverage check is delegated to
    :func:`_try_promote_to_top1`, which rejects any non-top-1 OR (mixed labels,
    extra conjuncts, nonzero RHS) — so no unsound collapse is possible.
    """
    if len(queries) < 2:
        return None
    first_out = queries[0][1]
    if first_out.kind != OutKind.UNSAFE_LINEAR or first_out.c is None:
        return None
    c0 = first_out.c
    if c0.dim() == 1:
        c0 = c0.unsqueeze(0)
    if c0.shape[0] != 1:
        return None
    candidates = [i for i, v in enumerate(c0[0].tolist()) if abs(v) > 1e-9]
    if len(candidates) != 2:
        return None
    for cand in candidates:
        promoted = _try_promote_to_top1(queries, num_outputs, cand)
        if promoted is not None:
            return promoted
    return None


def extract_vnnlib_2_io_decls(vnnlib_path):
    """``((in_name, in_dtype, in_shape), (out_name, out_dtype, out_shape))`` for a
    single-network VNNLIB 2.0 file, else None (legacy 1.0, or multi-network): the
    caller then has no single-tensor witness header and degrades ``sat`` to
    ``unknown``."""
    try:
        with open(vnnlib_path, "r") as f:
            content = f.read()
    except OSError:
        return None
    if "(vnnlib-version" not in content and "(declare-network" not in content:
        return None
    if len(re.findall(r"\(\s*declare-network\b", content)) >= 2:
        return None
    if len(re.findall(r"\(\s*declare-input\b", content)) >= 2:
        return None
    try:
        return (_extract_vnnlib_2_decl(content, "input"),
                _extract_vnnlib_2_decl(content, "output"))
    except VNNLibParseError:
        return None


def write_vnncomp_result(out_path, token: str, *, x=None, y=None,
                         in_decl=None, out_decl=None) -> None:
    """Emit the VNN-COMP result token; for ``sat`` append the counterexample as a
    VNNLIB 2.0 command-line assignment (VNNLIB-Standard §5.3): per variable a
    header ``<name> <dtype> [d0,d1,..]`` then its values one-per-line in
    row-major (C) order, input decl then output decl. This is what the 2026
    checker (``counterexamples_v2.py::parse_text_assignment``) parses — not the
    legacy flat ``((X_0 v)...)`` pairs. ``in_decl``/``out_decl`` are
    ``(name, dtype, shape)`` triples; if either is None the token is written
    witness-less so the caller can degrade to ``unknown``."""
    def _assignment(name, dtype, shape, values):
        dims = ",".join(str(int(d)) for d in shape)
        return [f"{name} {dtype} [{dims}]"] + [f"{v:.16g}" for v in values]

    with open(out_path, "w") as f:
        if (token != "sat" or x is None or y is None
                or in_decl is None or out_decl is None):
            f.write(token + "\n")
            return
        xf = x.detach().cpu().flatten().tolist()
        # The declared outputs come first; lifted spec terms (lifted_output_terms) follow.
        yf = y.detach().cpu().flatten().tolist()[:math.prod(int(d) for d in out_decl[2])]
        lines = ["sat"]
        lines += _assignment(in_decl[0], in_decl[1], in_decl[2], xf)
        lines += _assignment(out_decl[0], out_decl[1], out_decl[2], yf)
        f.write("\n".join(lines) + "\n")
