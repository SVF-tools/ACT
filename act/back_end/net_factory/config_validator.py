#===- act/back_end/net_factory/config_validator.py - Config Validation ====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
#
# Purpose:
#   Strict validation of YAML configuration to catch typos and unsupported
#   configuration items. Implements centralized assertions to ensure all
#   config items are properly handled before generation begins.
#
#===---------------------------------------------------------------------===#

from __future__ import annotations

from typing import Any, Dict, List, Set


# Allowed rule keywords in sampling rules
ALLOWED_RULE_KEYS: Set[str] = {
    "choice",
    "range",
    "weighted",
    "repeat",
    "probability",
    "const",
}

# Supported family types (builder implementations)
SUPPORTED_FAMILY_TYPES: Set[str] = {"mlp", "cnn2d"}

# Supported input spec kinds
SUPPORTED_INPUT_KINDS: Set[str] = {"BOX", "LINF_BALL"}

# Supported output spec kinds
SUPPORTED_OUTPUT_KINDS: Set[str] = {"TOP1_ROBUST", "MARGIN_ROBUST", "LINEAR_LE", "RANGE"}

# Required parameters for MLP families
# NOTE: 'depth' removed - MLP builder doesn't use it, uses len(hidden_sizes) instead
MLP_REQUIRED_PARAMS: Set[str] = {
    "type",
    "input_shape",
    "hidden_sizes",
    "activation",
    "variant",
    "num_blocks",
    "block_width",
    "post_block_activation",
    "num_residual_blocks",
    "residual_width",
    "use_bias",
    "num_classes",
}

# Required parameters for CNN2D families
CNN2D_REQUIRED_PARAMS: Set[str] = {
    "type",
    "input_shape",
    "variant",
    "num_blocks",
    "conv_channels",
    "kernel_sizes",
    "strides",
    "paddings",
    "use_maxpool",
    "maxpool_kernel",
    "maxpool_stride",
    "fc_hidden",
    "stages",
    "blocks_per_stage",
    "base_channels",
    "channel_mult",
    "downsample",
    "double_conv_p",
    "head_pool_to_1x1",
    "activation",
    "use_bias",
    "num_classes",
}


class ConfigValidationError(ValueError):
    """Raised when configuration validation fails."""
    pass


def _validate_rule_dict(rule: Any, path: str) -> None:
    """
    Validate that a rule dictionary only contains allowed rule keywords.

    STRICT MODE: If a dict contains any key that looks like a rule keyword
    (even a typo), it must ONLY contain valid rule keywords.

    Args:
        rule: Rule definition to validate
        path: Configuration path for error messages

    Raises:
        ConfigValidationError: If unknown rule keys are found
    """
    if not isinstance(rule, dict):
        return

    # Check if this is a rule dict or a nested config dict
    # A dict is considered a "rule dict" if:
    # 1. It contains at least one valid rule key, OR
    # 2. It contains only lowercase keys (likely intended as rules)
    has_valid_rule_key = any(k in ALLOWED_RULE_KEYS for k in rule.keys())
    has_lowercase_keys = all(isinstance(k, str) and k.islower() for k in rule.keys())

    is_rule = has_valid_rule_key or (has_lowercase_keys and len(rule) <= 3)

    if is_rule:
        # Validate rule keys - ALL keys must be valid if this is a rule dict
        unknown_keys = set(rule.keys()) - ALLOWED_RULE_KEYS
        if unknown_keys:
            raise ConfigValidationError(
                f"Unknown rule keys at '{path}': {unknown_keys}. "
                f"Allowed rule keys: {ALLOWED_RULE_KEYS}. "
                f"Did you mean one of: {', '.join(ALLOWED_RULE_KEYS)}?"
            )

        # Recursively validate nested rules (e.g., in 'repeat')
        if "repeat" in rule:
            repeat_rule = rule["repeat"]
            if isinstance(repeat_rule, dict):
                if "count" in repeat_rule:
                    _validate_rule_dict(repeat_rule["count"], f"{path}.repeat.count")
                if "value" in repeat_rule:
                    _validate_rule_dict(repeat_rule["value"], f"{path}.repeat.value")
    else:
        # Recursively validate nested dicts
        for key, value in rule.items():
            _validate_rule_dict(value, f"{path}.{key}")


def _validate_family_params(
    family_name: str,
    family_config: Dict[str, Any],
    required_params: Set[str]
) -> None:
    """
    Validate that a family configuration contains all required parameters.

    STRICT MODE: Rejects both missing AND unknown parameters to ensure
    all configuration items are properly handled by the builder.

    Args:
        family_name: Name of the family
        family_config: Family configuration dict
        required_params: Set of required parameter names

    Raises:
        ConfigValidationError: If required parameters are missing OR unknown parameters present
    """
    actual_params = set(family_config.keys())
    missing_params = required_params - actual_params
    unknown_params = actual_params - required_params

    errors = []

    if missing_params:
        errors.append(
            f"Family '{family_name}' is missing required parameters: {missing_params}"
        )

    if unknown_params:
        errors.append(
            f"Family '{family_name}' has unknown parameters: {unknown_params}. "
            f"All parameters must be explicitly handled by the builder."
        )

    if errors:
        raise ConfigValidationError("\n".join(errors))

    # Validate each parameter's rule definition
    for param_name, param_value in family_config.items():
        if param_name == "type":
            continue  # Type is not a rule
        _validate_rule_dict(param_value, f"families.{family_name}.{param_name}")


def validate_config(config: Dict[str, Any]) -> None:
    """
    Validate entire configuration with strict assertions.

    This function performs comprehensive validation to catch:
    - Typos in rule keywords
    - Unknown family types
    - Missing required family parameters
    - Unsupported spec kinds
    - Mismatches between family_selection and families

    Args:
        config: Loaded YAML configuration dictionary

    Raises:
        ConfigValidationError: If any validation check fails
    """
    errors: List[str] = []

    # =========================================================================
    # 1. Validate Common Section
    # =========================================================================
    if "common" not in config:
        errors.append("Missing required section: 'common'")
    else:
        common = config["common"]
        required_common_keys = {
            "num_instances", "base_seed", "name_prefix", "output_dir",
            "write_manifest", "manifest_path", "dtype"
        }
        actual_common_keys = set(common.keys())
        missing_common = required_common_keys - actual_common_keys
        if missing_common:
            errors.append(f"Missing required common parameters: {missing_common}")

    # =========================================================================
    # 2. Validate Input Spec
    # =========================================================================
    if "input_spec" not in config:
        errors.append("Missing required section: 'input_spec'")
    else:
        input_spec = config["input_spec"]

        # Validate kind weights
        if "kind" not in input_spec:
            errors.append("input_spec missing required field: 'kind'")
        elif "weighted" in input_spec["kind"]:
            weighted = input_spec["kind"]["weighted"]
            for kind in weighted.keys():
                if kind not in SUPPORTED_INPUT_KINDS:
                    errors.append(
                        f"Unsupported input_spec kind '{kind}'. "
                        f"Supported: {SUPPORTED_INPUT_KINDS}"
                    )

        # Validate rule syntax in input_spec
        try:
            _validate_rule_dict(input_spec, "input_spec")
        except ConfigValidationError as e:
            errors.append(str(e))

    # =========================================================================
    # 3. Validate Output Spec
    # =========================================================================
    if "output_spec" not in config:
        errors.append("Missing required section: 'output_spec'")
    else:
        output_spec = config["output_spec"]

        # Validate kind weights
        if "kind" not in output_spec:
            errors.append("output_spec missing required field: 'kind'")
        elif "weighted" in output_spec["kind"]:
            weighted = output_spec["kind"]["weighted"]
            for kind in weighted.keys():
                if kind not in SUPPORTED_OUTPUT_KINDS:
                    errors.append(
                        f"Unsupported output_spec kind '{kind}'. "
                        f"Supported: {SUPPORTED_OUTPUT_KINDS}"
                    )

        # Validate rule syntax in output_spec
        try:
            _validate_rule_dict(output_spec, "output_spec")
        except ConfigValidationError as e:
            errors.append(str(e))

    # =========================================================================
    # 4. Validate Family Selection
    # =========================================================================
    if "family_selection" not in config:
        errors.append("Missing required section: 'family_selection'")
    else:
        family_selection = config["family_selection"]
        if "weighted" not in family_selection:
            errors.append("family_selection must have 'weighted' strategy")
        else:
            selected_families = set(family_selection["weighted"].keys())

            # Check that all selected families exist in families section
            if "families" in config:
                defined_families = set(config["families"].keys())
                undefined_families = selected_families - defined_families
                if undefined_families:
                    errors.append(
                        f"family_selection references undefined families: {undefined_families}"
                    )

                # Warn about unused families (not critical)
                unused_families = defined_families - selected_families
                if unused_families:
                    print(f"Warning: Defined families not in selection: {unused_families}")

    # =========================================================================
    # 5. Validate Families
    # =========================================================================
    if "families" not in config:
        errors.append("Missing required section: 'families'")
    else:
        families = config["families"]

        for family_name, family_config in families.items():
            # Check that family has 'type' field
            if "type" not in family_config:
                errors.append(f"Family '{family_name}' missing required field: 'type'")
                continue

            family_type = family_config["type"]

            # Validate family type
            if family_type not in SUPPORTED_FAMILY_TYPES:
                errors.append(
                    f"Family '{family_name}' has unsupported type '{family_type}'. "
                    f"Supported types: {SUPPORTED_FAMILY_TYPES}"
                )
                continue

            # Validate family parameters based on type
            if family_type == "mlp":
                try:
                    _validate_family_params(family_name, family_config, MLP_REQUIRED_PARAMS)
                except ConfigValidationError as e:
                    errors.append(str(e))

            elif family_type == "cnn2d":
                try:
                    _validate_family_params(family_name, family_config, CNN2D_REQUIRED_PARAMS)
                except ConfigValidationError as e:
                    errors.append(str(e))

    # =========================================================================
    # Raise All Errors
    # =========================================================================
    if errors:
        error_message = "Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
        raise ConfigValidationError(error_message)

    print("✅ Configuration validation passed")


def get_validation_summary(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate a validation summary for the configuration.

    Returns a dictionary with statistics about the configuration:
    - Number of families by type
    - Family selection weights
    - Supported spec kinds

    Args:
        config: Loaded YAML configuration dictionary

    Returns:
        Dictionary with validation summary
    """
    summary = {
        "total_families": 0,
        "families_by_type": {},
        "family_selection_weights": {},
        "input_spec_kinds": [],
        "output_spec_kinds": [],
    }

    if "families" in config:
        families = config["families"]
        summary["total_families"] = len(families)

        for family_name, family_config in families.items():
            family_type = family_config.get("type", "unknown")
            if family_type not in summary["families_by_type"]:
                summary["families_by_type"][family_type] = []
            summary["families_by_type"][family_type].append(family_name)

    if "family_selection" in config and "weighted" in config["family_selection"]:
        summary["family_selection_weights"] = config["family_selection"]["weighted"]

    if "input_spec" in config and "kind" in config["input_spec"]:
        if "weighted" in config["input_spec"]["kind"]:
            summary["input_spec_kinds"] = list(config["input_spec"]["kind"]["weighted"].keys())

    if "output_spec" in config and "kind" in config["output_spec"]:
        if "weighted" in config["output_spec"]["kind"]:
            summary["output_spec_kinds"] = list(config["output_spec"]["kind"]["weighted"].keys())

    return summary
