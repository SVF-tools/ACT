"""Consistency check: CLI options <-> YAML keys <-> config.py dataclass fields.

Run in CI (``.github/workflows/act.config.yml``) or locally
(``python -m act.config.check_parity``). Exits non-zero on any drift, so an
option added to one surface can never be silently missed by the others.

The config.py dataclasses are the single source of truth. One documented
asymmetry is allowed:

* pipeline ``verification.bab`` is a SPARSE override of ``backend.yaml``
  (its YAML lists only non-default keys), so there the YAML keys must be a
  subset of the CLI surface rather than equal to it.
"""
from __future__ import annotations

import sys
from dataclasses import fields
from pathlib import Path
from typing import Any

import yaml


def _field_names(dataclass_type) -> set[str]:
    return {field.name for field in fields(dataclass_type)}


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path) as handle:
        return yaml.safe_load(handle) or {}


def _fmt(names: set[str]) -> str:
    return "{" + ", ".join(sorted(names)) + "}" if names else "{}"


class ParityReport:
    """Collects per-check results and prints a readable pass/fail line for each."""

    def __init__(self) -> None:
        self.failures: list[str] = []

    def require(self, label: str, passed: bool, detail: str = "") -> None:
        line = f"  [{'OK  ' if passed else 'FAIL'}] {label}"
        if detail and not passed:
            line += f"  ->  {detail}"
        print(line)
        if not passed:
            self.failures.append(label)


def check_backend(report: ParityReport) -> None:
    from act.config.config import _BACKEND_YAML, _NETGEN_YAML, BaBConfig
    from act.config.backend_cli import (
        _BACKEND_OVERRIDE_SPEC,
        _BACKEND_SUBCONFIG_PREFIX,
        _backend_override_keys_from_dataclasses,
    )

    # Single source of truth: the CLI/override key set is DERIVED from the config
    # dataclasses (auto-includes any new field or sub-config). No hand-coded list.
    dataclass_fields = _backend_override_keys_from_dataclasses()
    cli_options = {key for key, *_ in _BACKEND_OVERRIDE_SPEC}

    yaml_keys: set[str] = set()
    backend_document = _load_yaml(_BACKEND_YAML)
    backend_yaml = backend_document.get("backend", {})
    for key, value in backend_yaml.items():
        prefix = _BACKEND_SUBCONFIG_PREFIX.get(key)
        if prefix is None:
            yaml_keys.add(key)
            continue
        for sub_key in (value or {}):
            # backend.bab.enabled is surfaced as the top-level bab_enabled flag.
            if key == "bab" and sub_key == "enabled":
                yaml_keys.add("bab_enabled")
            else:
                yaml_keys.add(f"{prefix}{sub_key}")

    if _NETGEN_YAML.exists():
        for key in _load_yaml(_NETGEN_YAML):
            yaml_keys.add(f"gen_{key}")

    print("[backend]")
    report.require("CLI options are backed by a dataclass field",
                   cli_options <= dataclass_fields, _fmt(cli_options - dataclass_fields))
    report.require("every dataclass field is CLI-exposed",
                   dataclass_fields <= cli_options, _fmt(dataclass_fields - cli_options))
    report.require("YAML keys are backed by a dataclass field",
                   yaml_keys <= dataclass_fields, _fmt(yaml_keys - dataclass_fields))
    report.require("CLI and YAML are 1-to-1",
                   cli_options == yaml_keys,
                   f"cli-only={_fmt(cli_options - yaml_keys)} yaml-only={_fmt(yaml_keys - cli_options)}")

    print("[backend.bab_presets]")
    bab_yaml_keys = set((backend_yaml.get("bab") or {}).keys()) - {"enabled"}
    for preset_name, preset in (backend_document.get("bab_presets") or {}).items():
        preset_keys = set(preset or {})
        report.require(
            f"preset '{preset_name}' only uses backend.bab keys",
            preset_keys <= bab_yaml_keys,
            _fmt(preset_keys - bab_yaml_keys),
        )
        try:
            BaBConfig(**(preset or {}))
        except (TypeError, ValueError) as exc:
            report.require(
                f"preset '{preset_name}' constructs a valid BaBConfig",
                False,
                str(exc),
            )
        else:
            report.require(
                f"preset '{preset_name}' constructs a valid BaBConfig",
                True,
            )


def check_pipeline(report: ParityReport) -> None:
    from act.config.config import _PIPELINE_YAML, BaBConfig, DualConfig, ValidationConfig
    from act.pipeline.fuzzing.actfuzzer import FuzzingConfig
    from act.config.pipeline_cli import (
        _FUZZ_OVERRIDE_SPEC, _PIPELINE_BAB_OVERRIDE_FIELDS, _PIPELINE_DUAL_OVERRIDE_FIELDS, _PIPELINE_VAL_ATTR_MAP,
    )
    pipeline_yaml = _load_yaml(_PIPELINE_YAML)

    fuzz_fields = _field_names(FuzzingConfig)
    fuzz_cli = {key for key, *_ in _FUZZ_OVERRIDE_SPEC}
    fuzz_yaml = set((pipeline_yaml.get("fuzzing") or {}).keys())
    print("[pipeline.fuzzing]")
    report.require("CLI options are backed by a dataclass field",
                   fuzz_cli <= fuzz_fields, _fmt(fuzz_cli - fuzz_fields))
    report.require("YAML keys are backed by a dataclass field",
                   fuzz_yaml <= fuzz_fields, _fmt(fuzz_yaml - fuzz_fields))
    report.require("CLI and YAML are 1-to-1",
                   fuzz_cli == fuzz_yaml,
                   f"cli-only={_fmt(fuzz_cli - fuzz_yaml)} yaml-only={_fmt(fuzz_yaml - fuzz_cli)}")

    val_fields = _field_names(ValidationConfig)
    val_cli = set(_PIPELINE_VAL_ATTR_MAP)
    val_yaml = set((pipeline_yaml.get("validation") or {}).keys())
    print("[pipeline.validation]")
    report.require("CLI options are backed by a dataclass field",
                   val_cli <= val_fields, _fmt(val_cli - val_fields))
    report.require("YAML keys are backed by a dataclass field",
                   val_yaml <= val_fields, _fmt(val_yaml - val_fields))
    report.require("CLI and YAML are 1-to-1",
                   val_cli == val_yaml,
                   f"cli-only={_fmt(val_cli - val_yaml)} yaml-only={_fmt(val_yaml - val_cli)}")

    bab_fields = _field_names(BaBConfig)
    bab_cli = set(_PIPELINE_BAB_OVERRIDE_FIELDS)
    verification_yaml = pipeline_yaml.get("verification") or {}
    bab_yaml = set((verification_yaml.get("bab") or {}).keys())
    print("[pipeline.verification.bab]  (sparse override of backend.yaml)")
    report.require("CLI options are backed by a dataclass field",
                   bab_cli <= bab_fields, _fmt(bab_cli - bab_fields))
    report.require("YAML keys are backed by a dataclass field",
                   bab_yaml <= bab_fields, _fmt(bab_yaml - bab_fields))
    report.require("YAML keys are a subset of the CLI surface",
                   bab_yaml <= bab_cli, _fmt(bab_yaml - bab_cli))

    dual_fields = _field_names(DualConfig)
    dual_cli = set(_PIPELINE_DUAL_OVERRIDE_FIELDS)
    dual_yaml = set((verification_yaml.get("dual") or {}).keys())
    print("[pipeline.verification.dual]  (sparse override of backend.yaml)")
    report.require("CLI options are backed by a dataclass field",
                   dual_cli <= dual_fields, _fmt(dual_cli - dual_fields))
    report.require("YAML keys are backed by a dataclass field",
                   dual_yaml <= dual_fields, _fmt(dual_yaml - dual_fields))
    report.require("YAML keys are a subset of the CLI surface",
                   dual_yaml <= dual_cli, _fmt(dual_yaml - dual_cli))


def check_frontend(report: ParityReport) -> None:
    from act.config.config import _FRONTEND_YAML
    from act.config.frontend_cli import (
        _FRONTEND_SPEC_OVERRIDE_KEYS, _FRONTEND_TEXTVERIFY_OVERRIDE_KEYS,
    )
    frontend_yaml = _load_yaml(_FRONTEND_YAML)

    text_cli = set(_FRONTEND_TEXTVERIFY_OVERRIDE_KEYS)
    text_yaml = set((frontend_yaml.get("text_verification") or {}).keys())
    print("[frontend.text_verification]")
    report.require("CLI and YAML are 1-to-1",
                   text_cli == text_yaml,
                   f"cli-only={_fmt(text_cli - text_yaml)} yaml-only={_fmt(text_yaml - text_cli)}")

    # specs are per-benchmark preset data, not scalar knobs; the CLI exposes a few
    # override knobs that must exist in every preset for the override to apply.
    spec_knobs = set(_FRONTEND_SPEC_OVERRIDE_KEYS)
    print("[frontend.specs]  (CLI knobs override preset values)")
    for preset_name, preset in (frontend_yaml.get("specs") or {}).items():
        missing = spec_knobs - set((preset or {}).keys())
        report.require(f"preset '{preset_name}' exposes every CLI override knob",
                       not missing, _fmt(missing))


def main() -> int:
    report = ParityReport()
    check_backend(report)
    check_pipeline(report)
    check_frontend(report)
    print()
    if report.failures:
        print(f"CONFIG PARITY: {len(report.failures)} violation(s) - CLI/YAML/dataclass out of sync:")
        for failure in report.failures:
            print(f"  - {failure}")
        print("\nAdd the option in all three places: the CLI (act/config/*_cli.py), the "
              "YAML (act/config/*_config.yaml), and the dataclass (act/config/config.py).")
        return 1
    print("CONFIG PARITY: OK - CLI options, YAML keys, and dataclass fields are in sync.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
