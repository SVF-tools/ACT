# device_manager.py
# Simplified device/dtype management using PyTorch global defaults.
# Now with explicit initialization API (no argparse dependency).

import threading
from typing import Optional, Tuple

import torch

if hasattr(torch.sparse, "check_sparse_tensor_invariants"):
    torch.sparse.check_sparse_tensor_invariants.disable()

# Global initialization state
_INITIALIZED = False
# torch.set_default_device is thread-local: keep the process default and apply it once per thread.
_DEFAULT_DEVICE: Optional[torch.device] = None
_THREAD_STATE = threading.local()


def _apply_precision_policy() -> str:
    """Keep float32 matmul and convolution at full float32 precision.

    TF32 truncates the float32 mantissa to 10 bits, which silently makes the
    analysed network differ from the one on disk. Deliberately not
    configurable: it is a correctness invariant, not a tuning knob.

    Returns:
        Description of the resulting precision state, for the startup banner.

    Raises:
        RuntimeError: If either backend still reports TF32 as enabled, which
            means this torch build no longer honours these setters.
    """
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    if torch.backends.cudnn.allow_tf32 or torch.backends.cuda.matmul.allow_tf32:
        raise RuntimeError(
            "Refusing to run with TF32 enabled: float32 convolutions would use "
            "a 10-bit mantissa and the analysed network would differ from the "
            "file on disk. Setters did not take effect "
            f"(cudnn={torch.backends.cudnn.allow_tf32}, "
            f"matmul={torch.backends.cuda.matmul.allow_tf32}); this torch "
            f"version ({torch.__version__}) may have changed their semantics."
        )
    return "tf32=off"


def initialize_device(device: str, dtype: str) -> None:
    """
    Explicitly initialize device and dtype settings.
    
    This should be called ONCE at the entry point of your application
    (e.g., in CLI main() after parsing arguments).
    
    Args:
        device: Computation device - 'cpu', 'cuda', or 'gpu' (gpu aliased to cuda)
        dtype: PyTorch data type - 'float32' or 'float64'. Required, so that
            each tier's config YAML stays the single authority for its own
            value: the pipeline tier runs float32 and the back_end tier
            float64, and no shared default can silently answer for either.
    
    Examples:
        # In CLI after parsing args:
        from act.util.device_manager import initialize_device
        initialize_device(device=args.device, dtype=args.dtype)
        
        # For testing with specific settings:
        initialize_device(device='cpu', dtype='float32')
    """
    global _INITIALIZED, _DEFAULT_DEVICE
    
    # Handle gpu/cuda aliasing
    if device == 'gpu':
        device = 'cuda'
        print(f"🔄 Device alias: 'gpu' → 'cuda'")
        
    # Determine target device. CLI defaults already select an available device,
    # so an unavailable explicit request is a configuration error.
    if device == 'cpu':
        target_device = torch.device("cpu")
    elif device == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested but CUDA is not available")
        target_device = torch.device("cuda:0")
    elif device == 'mps':
        if not hasattr(torch.backends, 'mps') or not torch.backends.mps.is_available():
            raise RuntimeError("MPS device requested but MPS is not available")
        target_device = torch.device("mps")
    else:
        raise ValueError(
            f"Unknown device {device!r}; expected one of 'cpu', 'cuda', 'gpu', or 'mps'"
        )
        
    # Determine target dtype.
    if dtype == 'float32':
        target_dtype = torch.float32
    elif dtype == 'float64':
        target_dtype = torch.float64
    else:
        raise ValueError(
            f"Unknown dtype {dtype!r}; expected 'float32' or 'float64'"
        )
        
    # Set PyTorch global defaults
    torch.set_default_dtype(target_dtype)
    if hasattr(torch, 'set_default_device'):
        torch.set_default_device(target_device)
        _DEFAULT_DEVICE = target_device
        _THREAD_STATE.applied = True

    precision = _apply_precision_policy()

    print(
        f"✅ Device Manager Initialized: device={target_device}, "
        f"dtype={target_dtype}, {precision}"
    )
    _INITIALIZED = True


def get_default_device() -> torch.device:
    """
    Get current PyTorch default device.
    
    Auto-initializes with sensible defaults if not yet initialized
    (CUDA if available, else CPU).
    """
    _ensure_initialized()
    
    if hasattr(torch, 'get_default_device'):
        return torch.get_default_device()
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
    global _INITIALIZED, _DEFAULT_DEVICE
    
    if not _INITIALIZED:
        # Auto-detect best device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Initialize with defaults (no print statements for lazy init).
        target_device = torch.device("cuda:0") if device == 'cuda' else torch.device("cpu")
        target_dtype = torch.float64

        torch.set_default_dtype(target_dtype)
        if hasattr(torch, 'set_default_device'):
            torch.set_default_device(target_device)
            _DEFAULT_DEVICE = target_device
            _THREAD_STATE.applied = True

        _apply_precision_policy()
        _INITIALIZED = True

    # A thread that has not seen the process default device yet (e.g. a server
    # worker thread) gets it now; later per-thread changes are left alone.
    if _DEFAULT_DEVICE is not None and not getattr(_THREAD_STATE, "applied", False):
        torch.set_default_device(_DEFAULT_DEVICE)
    _THREAD_STATE.applied = True
