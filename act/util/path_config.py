#===- util.path_config.py ----ACT Path Configuration ---------------------#
#
#                 ACT: Abstract Constraints Transformer
#
# Copyright (C) <2025->  ACT Team
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# Purpose:
#   Python path configuration utilities for the Abstract Constraints Transformer
#   (ACT), ensuring proper module imports and path resolution across the
#   verification framework components.
#
#===----------------------------------------------------------------------===#

import os
import sys
from typing import Any, Optional, Tuple


def setup_act_paths() -> str:
    """Set up ACT project paths for proper module imports."""
    current_file = os.path.abspath(__file__)
    # From: /path/to/act/util/path_config.py
    # Need to go up 2 levels: util -> act
    act_root = os.path.dirname(os.path.dirname(current_file))
    
    # Add both act_root and its parent (project_root) to sys.path
    # This ensures proper module imports from wrapper_exts and other modules
    if act_root not in sys.path:
        sys.path.insert(0, act_root)
    
    project_root = os.path.dirname(act_root)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    return act_root


def get_project_root() -> str:
    """Get the project root directory (parent of act/)."""
    current_file = os.path.abspath(__file__)
    # Go up 3 levels: util -> act -> project_root
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
    return project_root


# Set up paths
act_root = setup_act_paths()
project_root = get_project_root()


def ensure_gurobi_license() -> Optional[str]:
    """Ensure GRB_LICENSE_FILE is set to a valid Gurobi license file."""
    existing_license = os.environ.get('GRB_LICENSE_FILE')
    if existing_license:
        license_path = os.path.abspath(existing_license)
        print(f"[ACT] Using existing Gurobi license: {license_path}")
        return license_path

    if 'ACTHOME' in os.environ:
        acthome = os.environ['ACTHOME']
        print(f"[ACT] Using ACTHOME environment variable: {acthome}")
        license_path = os.path.abspath(os.path.join(acthome, 'modules', 'gurobi', 'gurobi.lic'))
        if os.path.exists(license_path):
            os.environ['GRB_LICENSE_FILE'] = license_path
            print(f"[ACT] Gurobi license found and set: {license_path}")
            return license_path
        else:
            print(f"[WARN] Gurobi license not found at: {license_path}")
            print(f"[INFO] Please ensure gurobi.lic is placed in: {os.path.dirname(license_path)}")

    print(f"[ACT] Auto-detecting project root from path_config")
    license_path = os.path.abspath(os.path.join(project_root, 'modules', 'gurobi', 'gurobi.lic'))
    if os.path.exists(license_path):
        os.environ['GRB_LICENSE_FILE'] = license_path
        print(f"[ACT] Gurobi license found and set: {license_path}")
        return license_path

    print(f"[WARN] Gurobi license not found at: {license_path}")
    print(f"[INFO] Please ensure gurobi.lic is placed in: {os.path.dirname(license_path)}")
    return None


def import_gurobi(ensure_license: bool = False) -> Tuple[bool, Optional[Any], Optional[Any]]:
    """Attempt to import Gurobi and return availability with module handles."""
    if ensure_license:
        ensure_gurobi_license()

    try:
        import gurobipy as gp  # type: ignore[import-not-found]
        from gurobipy import GRB  # type: ignore[import-not-found]
        return True, gp, GRB
    except ImportError:
        print("Warning: Gurobi not available. Some verification methods may use alternative solvers.")
        return False, None, None


def get_data_root() -> str:
    """Get the data directory path."""
    return os.path.join(get_project_root(), 'data')


def get_torchvision_data_root() -> str:
    """Get the torchvision dataset downloads directory path."""
    return os.path.join(get_project_root(), 'data', 'torchvision')


def get_config_root() -> str:
    """Get the configs directory path."""
    # After reorganizing repository, configs live under modules/configs
    return os.path.join(get_project_root(), 'modules', 'configs')


def get_modules_root() -> str:
    """Get the modules directory path."""
    return os.path.join(get_project_root(), 'modules')


def get_eran_tf_verify_path() -> str:
    """Get the ERAN tf_verify directory path."""
    return os.path.join(get_modules_root(), 'eran', 'tf_verify')


def get_abcrown_path() -> str:
    """Get the abCrown module directory path."""
    return os.path.join(get_modules_root(), 'abcrown')


def get_pipeline_log_dir() -> str:
    """Get the pipeline log directory path.
    
    Returns:
        Absolute path to act/pipeline/log/ directory
    """
    return os.path.join(act_root, 'pipeline', 'log')


def get_path_relative_to_project(relative_path: str) -> str:
    """Get absolute path for a path relative to project root.
    
    Args:
        relative_path: Path relative to project root (e.g., 'data', 'configs/eran_defaults.ini')
    
    Returns:
        Absolute path
    """
    return os.path.abspath(os.path.join(get_project_root(), relative_path))


def configure_torch_print(linewidth: int = 500,
                          threshold: int = 10000,
                          sci_mode: bool = False,
                          precision: int = 4) -> None:
    """Configure default Torch print options for consistent tensor logging."""
    import torch

    torch.set_printoptions(
        linewidth=linewidth,
        threshold=threshold,
        sci_mode=sci_mode,
        precision=precision
    )


def get_vnnlib_data_root() -> str:
    """Get the VNNLIB benchmark data directory path, creating if needed."""
    from pathlib import Path
    vnnlib_root = Path(get_project_root()) / 'data' / 'vnnlib'
    vnnlib_root.mkdir(parents=True, exist_ok=True)
    return str(vnnlib_root)


def get_spec_config_root() -> str:
    """Get spec configuration directory (configs/specs/), creating if needed."""
    from pathlib import Path
    config_root = Path(get_project_root()) / 'configs' / 'specs'
    config_root.mkdir(parents=True, exist_ok=True)
    return str(config_root)


def get_default_spec_config_path() -> str:
    """Get path to default spec configuration."""
    from pathlib import Path
    return str(Path(get_spec_config_root()) / 'default_spec_config.yaml')


def get_spec_config_path(name: str) -> str:
    """
    Resolve named spec config to full path.
    
    Args:
        name: Config name (with or without .yaml extension)
    
    Returns:
        Full path to spec config file
    
    Raises:
        FileNotFoundError: If config file doesn't exist
    """
    from pathlib import Path
    if not name.endswith('.yaml'):
        name = f"{name}.yaml"
    path = Path(get_spec_config_root()) / name
    if not path.exists():
        raise FileNotFoundError(f"Spec config '{name}' not found in {get_spec_config_root()}")
    return str(path)


def list_spec_configs() -> list:
    """List all available spec configuration files."""
    from pathlib import Path
    config_root = Path(get_spec_config_root())
    return sorted([f.stem for f in config_root.glob('*.yaml')])