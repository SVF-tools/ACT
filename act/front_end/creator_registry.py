#!/usr/bin/env python3
"""
Creator Registry for ACT Specification Creators.

Provides factory pattern for managing multiple spec creators (TorchVision, VNNLIB, bert)
with automatic detection and routing.

Copyright (C) 2025 SVF-tools/ACT
License: AGPLv3+
"""

from functools import cache
from importlib import import_module
import logging
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

from act.front_end.spec_creator_base import BaseSpecCreator

logger = logging.getLogger(__name__)


class _Creator(NamedTuple):
    """Lazy import paths of one creator and how error messages name it."""

    creator_path: str
    finder_path: str
    not_found: str
    label: str


_CREATORS: Dict[str, _Creator] = {
    'torchvision': _Creator(
        'act.front_end.torchvision_loader.create_specs:TorchVisionSpecCreator',
        'act.front_end.torchvision_loader.data_model_mapping:find_dataset_name',
        "Dataset '{}' not found in TorchVision creator.",
        'TorchVision',
    ),
    'vnnlib': _Creator(
        'act.front_end.vnnlib_loader.create_specs:VNNLibSpecCreator',
        'act.front_end.vnnlib_loader.category_mapping:find_category_name',
        "Category '{}' not found in VNNLIB creator.",
        'VNNLIB',
    ),
    'bert': _Creator(
        'act.front_end.bert_loader.create_specs:BertSpecCreator',
        'act.front_end.bert_loader.data_model_loader:find_bert_dataset_name',
        "Dataset '{}' not found in bert creator.",
        'BERT',
    ),
}


def _load(path: str) -> Any:
    """Import the attribute named by a 'module:attribute' path."""
    module, attribute = path.split(':')
    return getattr(import_module(module), attribute)


@cache
def get_creator(name: str) -> BaseSpecCreator:
    """
    Get the shared spec creator instance for a name.
    
    Args:
        name: Creator name ('torchvision', 'vnnlib' or 'bert')
        
    Returns:
        Creator instance
        
    Raises:
        ValueError: If creator not found
    """
    if name not in _CREATORS:
        raise ValueError(
            f"Unknown creator '{name}'. "
            f"Available creators: {list_creators()}"
        )
    return _load(_CREATORS[name].creator_path)()


def list_creators() -> List[str]:
    """List all available creator names."""
    return list(_CREATORS)


def detect_creator(name: str, explicit_creator: Optional[str] = None) -> Tuple[str, str]:
    """
    Auto-detect which creator to use based on name.
    
    Tries to match the name against every creator's datasets or categories.
    If explicit_creator is provided, skips detection and validates the name
    against that creator only.
    
    Args:
        name: Dataset or category name (case-insensitive)
        explicit_creator: Optional creator name to force
        
    Returns:
        Tuple of (creator_name, normalized_name)
        - creator_name: 'torchvision', 'vnnlib' or 'bert'
        - normalized_name: Properly cased name for the creator
        
    Raises:
        ValueError: If name not found, ambiguous, or invalid for explicit creator
        
    Examples:
        >>> detect_creator('MNIST')
        ('torchvision', 'MNIST')
        
        >>> detect_creator('acasxu_2023')
        ('vnnlib', 'acasxu_2023')
        
        >>> detect_creator('mnist', 'vnnlib')
        ValueError: 'mnist' not found in VNNLIB categories
    """
    if explicit_creator:
        if explicit_creator not in _CREATORS:
            raise ValueError(
                f"Unknown creator '{explicit_creator}'. "
                f"Available: {list_creators()}"
            )
        try:
            return (explicit_creator, _load(_CREATORS[explicit_creator].finder_path)(name))
        except ValueError as e:
            raise ValueError(f"{_CREATORS[explicit_creator].not_found.format(name)}\n{str(e)}")
    
    matches: Dict[str, str] = {}
    for creator, entry in _CREATORS.items():
        try:
            matches[creator] = _load(entry.finder_path)(name)
        except ValueError as e:
            # Intentional: auto-detection probe; absence is reported through the matches below.
            logger.debug("suppressed: %s", e)
    
    if len(matches) > 1:
        rows = "".join(
            f"  {entry.label}: {matches.get(creator)}\n" for creator, entry in _CREATORS.items()
        )
        raise ValueError(
            f"Ambiguous name '{name}' matches multiple creators:\n"
            f"{rows}"
            f"Use --creator to specify explicitly:\n"
            f"  --creator torchvision  OR  --creator vnnlib  OR  --creator bert"
        )
    if not matches:
        raise ValueError(
            f"Dataset/category '{name}' not found in any creator.\n"
            f"Use --list to see all available options."
        )
    return next(iter(matches.items()))
