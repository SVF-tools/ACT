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
from typing import List, Optional, Union

import yaml

_DEFAULT_YAML = Path(__file__).parent / "config.yaml"

_VALID_SOLVERS = {"auto", "gurobi", "torch"}
_VALID_DEVICES = {"cpu", "cuda", "gpu"}
_VALID_DTYPES = {"float32", "float64"}
_VALID_REGISTRY_MODES = {"intersection", "union"}
_VALID_COVERAGE_MODES = {"basic", "full"}


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

    branching_method: str = "random"
    bounding_method: str = "random"

    verbose: bool = False

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
        yaml_config: dict = backend_section.get("bab", yaml_data.get("bab", {}))

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

    generation: GenerationConfig = field(default_factory=GenerationConfig)

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
              solver: "torch"
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

        backend_raw: dict = raw.get("backend", {})
        bab_raw: dict = backend_raw.pop("bab", {})
        gen_raw: dict = backend_raw.pop("generation", {})

        # Extract "enabled" from bab section → top-level bab_enabled
        bab_enabled = bab_raw.pop("enabled", None)

        # Route prefixed overrides to the right sub-config
        bab_fields = {fld.name for fld in fields(BaBConfig)}
        gen_fields = {fld.name for fld in fields(GenerationConfig)}
        bab_overrides: dict = {}
        gen_overrides: dict = {}
        top_overrides: dict = {}
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
        top_merged: dict = {}
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
