# device_manager.py
# Simplified device/dtype management using PyTorch global defaults.
# Now with explicit initialization API (no argparse dependency).

import torch
from typing import Tuple, Optional

# Global initialization state
_INITIALIZED = False


def initialize_device(device: str = 'cuda', dtype: str = 'float64') -> None:
    """
    Explicitly initialize device and dtype settings.
    
    This should be called ONCE at the entry point of your application
    (e.g., in CLI main() after parsing arguments).
    
    Args:
        device: Computation device - 'cpu', 'cuda', or 'gpu' (gpu aliased to cuda)
        dtype: PyTorch data type - 'float32' or 'float64'
    
    Examples:
        # In CLI after parsing args:
        from act.util.device_manager import initialize_device
        initialize_device(device=args.device, dtype=args.dtype)
        
        # For testing with specific settings:
        initialize_device(device='cpu', dtype='float32')
    """
    global _INITIALIZED
    
    try:
        # Handle gpu/cuda aliasing
        if device == 'gpu':
            device = 'cuda'
            print(f"🔄 Device alias: 'gpu' → 'cuda'")
        
        # Determine target device
        if device == 'cpu':
            target_device = torch.device("cpu")
        elif device == 'cuda':
            if torch.cuda.is_available():
                target_device = torch.device("cuda:0")
            else:
                target_device = torch.device("cpu")
                print(f"⚠️ CUDA not available, falling back to CPU")
        elif device == 'mps':
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                target_device = torch.device("mps")
            else:
                target_device = torch.device("cpu")
                print(f"⚠️ MPS not available, falling back to CPU")
        else:
            # Unknown device, default to CPU
            target_device = torch.device("cpu")
            print(f"⚠️ Unknown device '{device}', using CPU")
        
        # Determine target dtype
        if dtype == 'float32':
            target_dtype = torch.float32
        elif dtype == 'float64':
            target_dtype = torch.float64
        else:
            # Unknown dtype, default to float64
            target_dtype = torch.float64
            print(f"⚠️ Unknown dtype '{dtype}', using float64")
        
        # Set PyTorch global defaults
        torch.set_default_dtype(target_dtype)
        if hasattr(torch, 'set_default_device'):
            torch.set_default_device(target_device)
        
        print(f"✅ Device Manager Initialized: device={target_device}, dtype={target_dtype}")
        _INITIALIZED = True
        
    except Exception as e:
        print(f"❌ Device initialization failed: {e}")
        print(f"   Falling back to CPU + float64")
        torch.set_default_dtype(torch.float64)
        _INITIALIZED = True


def get_default_device() -> torch.device:
    """
    Get current PyTorch default device.
    
    Auto-initializes with sensible defaults if not yet initialized
    (CUDA if available, else CPU).
    """
    _ensure_initialized()
    
    if hasattr(torch, 'get_default_device'):
        try:
            return torch.get_default_device()
        except:
            return torch.device("cpu")
    else:
        # For older PyTorch versions, check where a test tensor is created
        test_tensor = torch.zeros(1)
        device = test_tensor.device
        del test_tensor
        return device


def get_default_dtype() -> torch.dtype:
    """
    Get current PyTorch default dtype.
    
    Auto-initializes with sensible defaults if not yet initialized (float64).
    """
    _ensure_initialized()
    return torch.get_default_dtype()


def get_current_settings() -> Tuple[torch.device, torch.dtype]:
    """
    Get current PyTorch default device and dtype settings.
    
    Auto-initializes with sensible defaults if not yet initialized.
    
    Returns:
        Tuple of (device, dtype)
    """
    _ensure_initialized()
    return get_default_device(), get_default_dtype()


def _ensure_initialized():
    """
    Lazy initialization with sensible defaults if not explicitly initialized.
    
    This is called automatically by get_default_device() and get_default_dtype()
    to ensure the device manager is always ready to use.
    
    Default behavior:
    - Device: CUDA if available, else CPU
    - Dtype: float64
    """
    global _INITIALIZED
    
    if not _INITIALIZED:
        # Auto-detect best device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        dtype = 'float64'
        
        # Initialize with defaults (no print statements for lazy init)
        try:
            target_device = torch.device("cuda:0") if device == 'cuda' else torch.device("cpu")
            target_dtype = torch.float64
            
            torch.set_default_dtype(target_dtype)
            if hasattr(torch, 'set_default_device'):
                torch.set_default_device(target_device)
            
            _INITIALIZED = True
        except Exception:
            # Silent fallback to CPU + float64
            torch.set_default_dtype(torch.float64)
            _INITIALIZED = True
