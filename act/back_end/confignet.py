#!/usr/bin/env python3
#===- act/back_end/confignet.py - ConfigNet Driver + CLI --------------====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
#
# Purpose:
#   ConfigNet entrypoints:
#   - schema + sampler (imported from confignet_spec)
#   - examples_config materialization + CLI
#
#===---------------------------------------------------------------------===#

from __future__ import annotations

import argparse
import logging
import secrets
from pathlib import Path
from typing import Dict, List, Optional

from act.back_end.confignet_spec import (
    ConfigNetConfig,
    InstanceSpec,
    ModelFamily,
    MLPConfig,
    CNN2DConfig,
    InputSpecConfig,
    OutputSpecConfig,
    derive_seed,
    sample_instances,
)
from act.back_end.confignet_io import (
    instance_to_examples_entry,
    write_confignet_entries_to_examples_config,
    DEFAULT_EXAMPLES_CONFIG,
    CONFIGNET_PREFIX,
)

logger = logging.getLogger(__name__)


# --------------------------
# Confignet → examples_config.yaml materialization
# --------------------------


def materialize_to_examples_config(
    *,
    instances: List[InstanceSpec],
    config_path: str = DEFAULT_EXAMPLES_CONFIG,
    dtype: str = "torch.float64",
    prefix: str = CONFIGNET_PREFIX,
) -> Dict[str, str]:
    """
    Convert Confignet instances into examples_config.yaml entries.

    Returns:
        Mapping of instance_id -> generated network name.
    """
    entries: Dict[str, Dict[str, object]] = {}
    name_map: Dict[str, str] = {}

    def _confignet_name(inst: InstanceSpec) -> str:
        base_seed = inst.meta.get("base_seed")
        idx = inst.meta.get("idx")
        if base_seed is not None and idx is not None:
            return f"{prefix}{int(base_seed)}_idx{int(idx):05d}"
        parts = str(inst.instance_id).split("_")
        if len(parts) >= 3 and parts[1].isdigit():
            base_seed = parts[1]
            idx = parts[2]
            return f"{prefix}{base_seed}_idx{idx}"
        return f"{prefix}{inst.instance_id}"

    for inst in instances:
        name = _confignet_name(inst)
        entries[name] = instance_to_examples_entry(inst, dtype=dtype)
        name_map[inst.instance_id] = name

    write_confignet_entries_to_examples_config(entries=entries, config_path=config_path, prefix=prefix)
    return name_map


# --------------------------
# CLI
# --------------------------


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO)
    p = argparse.ArgumentParser("ConfigNet CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_gen = sub.add_parser("generate", help="sample + write into examples_config.yaml")
    p_gen.add_argument("--num", type=int, default=5)
    p_gen.add_argument("--dtype", type=str, default="torch.float64", choices=["torch.float32", "torch.float64"])
    p_gen.add_argument("--examples-config", type=str, default=DEFAULT_EXAMPLES_CONFIG)
    p_gen.add_argument("--base-seed", type=int, default=None)
    p_gen.add_argument("--prefix", type=str, default=CONFIGNET_PREFIX)

    args = p.parse_args(argv)

    if args.cmd == "generate":
        base_seed = int(secrets.randbits(32)) if args.base_seed is None else int(args.base_seed)
        cfg = ConfigNetConfig(num_instances=args.num, base_seed=base_seed)
        instances = sample_instances(cfg)
        examples_path = Path(str(args.examples_config))
        if not examples_path.exists():
            examples_path.parent.mkdir(parents=True, exist_ok=True)
            examples_path.write_text("networks: {}\n", encoding="utf-8")
        materialize_to_examples_config(
            instances=instances,
            config_path=str(args.examples_config),
            dtype=str(args.dtype),
            prefix=str(args.prefix),
        )
        return 0

    return 1


__all__ = [
    "ConfigNetConfig",
    "InstanceSpec",
    "ModelFamily",
    "MLPConfig",
    "CNN2DConfig",
    "InputSpecConfig",
    "OutputSpecConfig",
    "derive_seed",
    "sample_instances",
    "materialize_to_examples_config",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
