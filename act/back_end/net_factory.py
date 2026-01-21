#===- act/back_end/net_factory.py - YAML-Driven ACT Net Factory ----------====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
#
# Purpose:
#   Generator-driven ACT Net generator. Samples MLP/CNN2D
#   configs, fills INPUT_SPEC/ASSERT params, builds layer vars/weights, and
#   writes Net JSONs + optional manifest for downstream verification.
#   Runtime uses JSON nets under act/back_end/examples/nets; YAML is config-only.
#   CLI entry: python -m act.back_end --generate (see act/back_end/cli.py).
#   INPUT.meta.dtype is set from device_manager default (get_default_dtype()).
#
#===---------------------------------------------------------------------===#
# ASSERT semantics are documented in act/back_end/README.md.
#===---------------------------------------------------------------------===#
#
# This module has been refactored into a package structure:
#   - net_factory/factory.py: Main NetFactory class
#   - net_factory/sampler.py: Configuration sampling logic
#   - net_factory/layer_builder.py: Layer construction functions
#   - net_factory/utils.py: Utility functions
#
# For backward compatibility, NetFactory is re-exported from this module.
#===---------------------------------------------------------------------===#

from __future__ import annotations

from act.back_end.net_factory import NetFactory

__all__ = ["NetFactory"]


if __name__ == "__main__":
    factory = NetFactory()
    factory.generate()
