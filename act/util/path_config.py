# pyright: reportMissingImports=false, reportMissingTypeArgument=false
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


def setup_act_paths() -> str:
    """Set up ACT project paths for proper module imports."""
    current_file = os.path.abspath(__file__)
    # From: /path/to/act/util/path_config.py
    # Need to go up 2 levels: util -> act
    act_root = os.path.dirname(os.path.dirname(current_file))
    
    # Add both act_root and its parent (project_root) to sys.path
    # This ensures proper module imports from other modules
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


def get_data_root() -> str:
    """Get the data directory path."""
    return os.path.join(get_project_root(), 'data')


def get_bert_artifact_root() -> str:
    """Get the BERT artifact directory, retaining the on-disk buffet layout."""
    return os.path.join(get_data_root(), 'buffet')


def get_torchvision_data_root() -> str:
    """Get the torchvision dataset downloads directory path."""
    return os.path.join(get_project_root(), 'data', 'torchvision')


def get_pipeline_log_dir() -> str:
    """Get the pipeline log directory path, creating it because git keeps none of its files.
    
    Returns:
        Absolute path to act/pipeline/log/ directory
    """
    log_dir = os.path.join(act_root, 'pipeline', 'log')
    os.makedirs(log_dir, exist_ok=True)
    return log_dir


def get_vnnlib_data_root() -> str:
    """Get the VNNLIB benchmark data directory path, creating if needed."""
    from pathlib import Path
    vnnlib_root = Path(get_project_root()) / 'data' / 'vnnlib'
    vnnlib_root.mkdir(parents=True, exist_ok=True)
    return str(vnnlib_root)


# ============================================================================
# NetFactory Paths
# ============================================================================

def get_examples_nets_dir() -> str:
    """Get directory containing pre-generated Net JSON files."""
    from pathlib import Path
    d = Path(get_project_root()) / 'act' / 'back_end' / 'examples' / 'nets'
    d.mkdir(parents=True, exist_ok=True)
    return str(d)
