#===- act/util/format_utils.py - Shared formatting/size helpers --------====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
#
# Purpose:
#   Small shared helpers for human-readable size formatting, directory-size
#   reporting, and separator lines. Consolidates logic previously duplicated
#   across data loaders and CLIs.
#
#===---------------------------------------------------------------------===#

from __future__ import annotations
from pathlib import Path
from typing import Union

__all__ = ["format_bytes", "dir_size", "rule"]

_UNITS = ("B", "KB", "MB", "GB", "TB", "PB")


def format_bytes(size_bytes: Union[int, float], precision: int = 1) -> str:
    """Human-readable size string; ``precision`` sets decimal places."""
    size = float(size_bytes)
    for unit in _UNITS:
        if size < 1024.0:
            return f"{size:.{precision}f} {unit}"
        size /= 1024.0
    return f"{size:.{precision}f} EB"


def dir_size(path: Union[str, Path]) -> int:
    """Return total size in bytes of all files under ``path`` (0 on error)."""
    total = 0
    try:
        for item in Path(path).rglob("*"):
            if item.is_file():
                total += item.stat().st_size
    except Exception:
        pass
    return total


def rule(width: int = 80, char: str = "=") -> str:
    """Horizontal separator line (``char * width``); the shared banner idiom."""
    return char * width
