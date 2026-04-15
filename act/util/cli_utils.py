#!/usr/bin/env python3
"""
Shared CLI utilities for ACT.

Provides common argument parsing and device initialization logic to ensure
consistency across all ACT CLIs:
- act.front_end (unified front-end CLI)
- act.front_end.torchvision_loader (TorchVision-specific)
- act.front_end.vnnlib_loader (VNNLIB-specific)
- act.pipeline (fuzzing and testing)
- act.back_end (verification core)

Copyright (C) 2025 SVF-tools/ACT
License: AGPLv3+
"""

import argparse
import torch


def add_device_args(parser: argparse.ArgumentParser) -> None:
    """
    Add standard device and dtype arguments to an ArgumentParser.
    
    This ensures consistent device/dtype handling across all ACT CLIs.
    All CLIs should use this function to add device arguments, ensuring:
    - Consistent default values (cuda if available, else cpu)
    - Consistent argument names (--device, --dtype)
    - Consistent choices (cpu/cuda/gpu, float32/float64)
    
    Args:
        parser: ArgumentParser to add arguments to
        
    Example:
        >>> parser = argparse.ArgumentParser()
        >>> add_device_args(parser)
        >>> args = parser.parse_args(['--device', 'cpu', '--dtype', 'float32'])
        >>> initialize_from_args(args)
    """
    parser.add_argument(
        "--device",
        type=str,
        default='cuda' if torch.cuda.is_available() else 'mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else 'cpu',
        choices=['cpu', 'cuda', 'gpu', 'mps'],
        help="Device to use for computation (default: best available)"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default='float64',
        choices=['float32', 'float64'],
        help="Default dtype for tensors (default: float64)"
    )


def initialize_from_args(args: argparse.Namespace) -> None:
    """
    Initialize device manager from parsed CLI arguments.
    
    This should be called immediately after parser.parse_args() in all
    front-end CLIs to ensure device manager is configured before any
    operations that might need it.
    
    Args:
        args: Parsed arguments containing 'device' and 'dtype' attributes
    """
    from act.util.device_manager import initialize_device
    initialize_device(device=args.device, dtype=args.dtype)
