#===- act/back_end/tf_capabilities.py - TF Layer Capabilities Registry ---====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
#
# Purpose:
#   TF Capabilities Registry. Collects supported layer types from each
#   Transfer Function implementation (IntervalTF, HybridzTF, DualTF) and
#   provides utilities to compute layer intersections/unions for NetFactory.
#
#   Design Principles:
#   1. Lazy import - Import TF modules only when called to avoid circular deps
#   2. Cache results - Use lru_cache to avoid repeated registry parsing
#   3. Fallback mechanism - Use predefined fallbacks if dynamic loading fails
#   4. Normalization - All layer names are uppercase for consistency
#
#===---------------------------------------------------------------------===#

from typing import Dict, FrozenSet, List, Optional
import functools
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# Special Layers (not in all TF registries)
# ============================================================================

# Layers that may not be in registry but should be considered supported
# Most layers (INPUT, INPUT_SPEC, ASSERT, FLATTEN, etc.) are already in registries
_IMPLICIT_LAYERS = frozenset({
    'OUTPUT_SPEC',  # Output specification (not always in registry)
    'DROPOUT',      # Dropout (identity at inference, not always in registry)
    'IDENTITY',     # Explicit identity (not always in registry)
})


# ============================================================================
# Core Functions
# ============================================================================

@functools.lru_cache(maxsize=1)
def get_tf_capabilities() -> Dict[str, FrozenSet[str]]:
    """
    Collect supported layers from each TF's registry dynamically.

    Reads directly from:
    - IntervalTF._LAYER_REGISTRY
    - HybridzTF._LAYER_REGISTRY
    - DualTF._BACKWARD_REGISTRY

    Returns:
        Dict mapping TF name to frozenset of supported layer kinds (uppercase).

    Note:
        Results are cached. Call get_tf_capabilities.cache_clear() to refresh.
    """
    result = {}

    # ---- Interval TF ----
    try:
        from act.back_end.interval_tf import IntervalTF
        registry = getattr(IntervalTF, '_LAYER_REGISTRY', {})
        layers = set(k.upper() for k in registry.keys())
        layers.update(_IMPLICIT_LAYERS)  # Add implicit layers not in registry
        result["interval"] = frozenset(layers)
        logger.debug(f"IntervalTF: loaded {len(layers)} layers from registry")
    except (ImportError, AttributeError) as e:
        logger.error(f"Failed to load IntervalTF registry: {e}")
        raise RuntimeError(f"Cannot load IntervalTF._LAYER_REGISTRY: {e}")

    # ---- HybridZ TF ----
    try:
        from act.back_end.hybridz_tf import HybridzTF
        registry = getattr(HybridzTF, '_LAYER_REGISTRY', {})
        layers = set(k.upper() for k in registry.keys())
        layers.update(_IMPLICIT_LAYERS)
        result["hybridz"] = frozenset(layers)
        logger.debug(f"HybridzTF: loaded {len(layers)} layers from registry")
    except (ImportError, AttributeError) as e:
        logger.error(f"Failed to load HybridzTF registry: {e}")
        raise RuntimeError(f"Cannot load HybridzTF._LAYER_REGISTRY: {e}")

    # ---- Dual TF ----
    try:
        from act.back_end.dual_tf import DualTF
        # Dual uses _BACKWARD_REGISTRY
        registry = getattr(DualTF, '_BACKWARD_REGISTRY', {})
        layers = set(k.upper() for k in registry.keys())
        layers.update(_IMPLICIT_LAYERS)
        result["dual"] = frozenset(layers)
        logger.debug(f"DualTF: loaded {len(layers)} layers from registry")
    except (ImportError, AttributeError) as e:
        logger.error(f"Failed to load DualTF registry: {e}")
        raise RuntimeError(f"Cannot load DualTF._BACKWARD_REGISTRY: {e}")

    return result


def get_allowed_layers(
    tf_targets: Optional[List[str]] = None,
    mode: str = "intersection"
) -> FrozenSet[str]:
    """
    Compute allowed layer set based on target TFs and combination mode.

    Args:
        tf_targets: Target TF list.
            - None: All TFs ["interval", "hybridz", "dual"]
            - ["interval"]: Only interval (returns all interval layers)
            - ["interval", "hybridz"]: Intersection or union of both

        mode: Combination mode for multiple TFs.
            - "intersection": Layers supported by ALL targets (default, safest)
            - "union": Layers supported by ANY target (broader coverage)

    Returns:
        FrozenSet of allowed layer kinds (uppercase).

    Raises:
        ValueError: Unknown TF name, unknown mode, or empty result.

    Examples:
        >>> # All TFs common layers (for CI testing)
        >>> layers = get_allowed_layers(["interval", "hybridz", "dual"])

        >>> # Only interval layers (for interval-specific testing)
        >>> layers = get_allowed_layers(["interval"])

        >>> # interval + hybridz intersection
        >>> layers = get_allowed_layers(["interval", "hybridz"], "intersection")
    """
    if tf_targets is None:
        tf_targets = ["interval", "hybridz", "dual"]

    # Normalize input
    tf_targets = [t.lower().strip() for t in tf_targets]

    # Get capabilities
    capabilities = get_tf_capabilities()

    # Validate TF names
    unknown = set(tf_targets) - set(capabilities.keys())
    if unknown:
        raise ValueError(
            f"Unknown TF targets: {unknown}. "
            f"Available: {list(capabilities.keys())}"
        )

    # Collect target sets
    target_sets = [capabilities[tf] for tf in tf_targets]

    # Compute result
    if len(target_sets) == 1:
        # Single TF: return its full layer set
        result = target_sets[0]
    elif mode == "intersection":
        # Intersection: layers supported by ALL
        result = target_sets[0]
        for s in target_sets[1:]:
            result = result & s
    elif mode == "union":
        # Union: layers supported by ANY
        result = frozenset().union(*target_sets)
    else:
        raise ValueError(
            f"Unknown mode: '{mode}'. Expected 'intersection' or 'union'."
        )

    # Validate non-empty
    if not result:
        raise ValueError(
            f"Empty layer set for tf_targets={tf_targets}, mode={mode}. "
            f"Check TF registries or try mode='union'."
        )

    logger.info(
        f"Allowed layers: {len(result)} for tf_targets={tf_targets}, mode={mode}"
    )

    return result


# ============================================================================
# Utility Functions
# ============================================================================

def get_tf_specific_layers(tf_name: str) -> FrozenSet[str]:
    """
    Get layers unique to a specific TF (not supported by others).

    Args:
        tf_name: TF name (case-insensitive).

    Returns:
        Layers only supported by this TF.
    """
    caps = get_tf_capabilities()
    tf_name = tf_name.lower()

    if tf_name not in caps:
        raise ValueError(f"Unknown TF: {tf_name}")

    this_tf_layers = caps[tf_name]
    other_tf_layers = set()
    for name, layers in caps.items():
        if name != tf_name:
            other_tf_layers.update(layers)

    return this_tf_layers - other_tf_layers


def get_common_layers() -> FrozenSet[str]:
    """Get layers supported by all TFs (intersection)."""
    return get_allowed_layers(None, "intersection")


def get_all_known_layers() -> FrozenSet[str]:
    """Get layers supported by any TF (union)."""
    return get_allowed_layers(None, "union")


# ============================================================================
# Diagnostics and Reporting
# ============================================================================

def print_capabilities_report():
    """Print detailed TF layer support report."""
    caps = get_tf_capabilities()

    print("=" * 70)
    print("TF Layer Capabilities Report")
    print("=" * 70)

    # Layer categories for organized display
    CATEGORIES = {
        "Basic": ['INPUT', 'INPUT_SPEC', 'OUTPUT_SPEC', 'ASSERT'],
        "Dense": ['DENSE', 'BIAS', 'SCALE', 'BN'],
        "Activation": ['RELU', 'LRELU', 'TANH', 'SIGMOID', 'RELU6', 'HARDTANH',
                       'HARDSIGMOID', 'HARDSWISH', 'SILU', 'SOFTPLUS', 'MISH',
                       'SOFTSIGN', 'ABS', 'CLIP', 'SQUARE', 'POWER'],
        "Conv": ['CONV1D', 'CONV2D', 'CONV3D', 'CONVTRANSPOSE2D'],
        "Pool": ['MAXPOOL1D', 'MAXPOOL2D', 'MAXPOOL3D',
                 'AVGPOOL1D', 'AVGPOOL2D', 'AVGPOOL3D'],
        "MultiInput": ['ADD', 'SUB', 'MUL', 'DIV', 'POW', 'MAX', 'MIN',
                       'MATMUL', 'CONCAT'],
        "Reshape": ['FLATTEN', 'RESHAPE', 'TRANSPOSE', 'SQUEEZE', 'UNSQUEEZE',
                    'DROPOUT', 'IDENTITY'],
        "Other": ['PAD', 'UPSAMPLE', 'SLICE', 'GATHER', 'TILE', 'EXPAND',
                  'INDEX_SELECT', 'LSTM', 'GRU', 'RNN', 'EMBEDDING'],
    }

    # 1. Per-TF layers
    for tf_name in sorted(caps.keys()):
        layers = caps[tf_name]
        print(f"\n{tf_name.upper()} ({len(layers)} layers):")

        for cat_name, cat_layers in CATEGORIES.items():
            supported = [l for l in cat_layers if l in layers]
            if supported:
                print(f"    {cat_name}: {', '.join(supported)}")

        # Uncategorized layers
        all_categorized = set()
        for cat_layers in CATEGORIES.values():
            all_categorized.update(cat_layers)
        uncategorized = layers - all_categorized
        if uncategorized:
            print(f"    Other: {', '.join(sorted(uncategorized))}")

    # 2. Intersection and Union
    print(f"\n{'=' * 70}")

    intersection = get_allowed_layers(list(caps.keys()), "intersection")
    union = get_allowed_layers(list(caps.keys()), "union")

    print(f"\nINTERSECTION - All TFs support ({len(intersection)} layers):")
    print(f"    {sorted(intersection)}")

    print(f"\nUNION - Any TF supports ({len(union)} layers):")
    print(f"    {sorted(union)}")

    # 3. TF-specific layers
    print(f"\n{'=' * 70}")
    print("TF-SPECIFIC LAYERS (only supported by one TF):")

    for tf_name in sorted(caps.keys()):
        specific = get_tf_specific_layers(tf_name)
        if specific:
            print(f"    {tf_name.upper()} only: {sorted(specific)}")
        else:
            print(f"    {tf_name.upper()} only: (none)")

    # 4. Support matrix
    print(f"\n{'=' * 70}")
    print("LAYER SUPPORT MATRIX:")
    print(f"    {'Layer':<20} {'interval':>10} {'hybridz':>10} {'dual':>10}")
    print(f"    {'-'*20} {'-'*10} {'-'*10} {'-'*10}")

    for layer in sorted(union):
        row = f"    {layer:<20}"
        for tf_name in ['interval', 'hybridz', 'dual']:
            supported = 'Y' if layer in caps.get(tf_name, set()) else '-'
            row += f" {supported:>10}"
        print(row)


# ============================================================================
# Convenience Functions
# ============================================================================

def for_interval() -> FrozenSet[str]:
    """Get all layers supported by IntervalTF."""
    return get_allowed_layers(["interval"])


def for_hybridz() -> FrozenSet[str]:
    """Get all layers supported by HybridzTF."""
    return get_allowed_layers(["hybridz"])


def for_dual() -> FrozenSet[str]:
    """Get all layers supported by DualTF."""
    return get_allowed_layers(["dual"])


def for_all_tfs() -> FrozenSet[str]:
    """Get layers supported by all TFs (intersection)."""
    return get_allowed_layers(["interval", "hybridz", "dual"], "intersection")
