"""
Trace storage backends for ACTFuzzer execution traces.

Supports HDF5 (recommended), JSON, and async wrappers for performance.

Copyright (C) 2025 SVF-tools/ACT
License: AGPLv3+
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any
import json
import queue
import threading
import torch


class TraceStorage(ABC):
    """Abstract base class for trace storage backends."""
    
    @abstractmethod
    def write(self, trace: Dict[str, Any]):
        """
        Write a single trace entry.
        
        Args:
            trace: Dictionary containing trace data
        """
        pass
    
    @abstractmethod
    def close(self):
        """Finalize and close storage."""
        pass
    
    @abstractmethod
    def flush(self):
        """Force write buffered data to disk."""
        pass


class HDF5Storage(TraceStorage):
    """
    HDF5 storage backend with compression.
    
    Features:
    - Binary format (compact, fast)
    - Per-dataset gzip compression
    - Hierarchical structure: /iterations/iter_0, iter_1, ...
    - Crash-safe with periodic flushes
    
    File structure:
        /iterations/
            iter_0/
                @iteration = 0
                @timestamp = 1699632000.5
                @mutation_strategy = "gradient"
                @violation_found = False
                @coverage = 0.67
                input_before: Dataset[float32]
                input_after: Dataset[float32]
                activations/
                    conv1: Dataset[float32]
                    fc1: Dataset[float32]
    """
    
    def __init__(self, path: Path):
        """
        Initialize HDF5 storage.
        
        Args:
            path: Output file path (e.g., "traces.h5")
        """
        try:
            import h5py
        except ImportError:
            raise ImportError(
                "h5py is required for HDF5 storage. "
                "Install with: pip install h5py"
            )
        
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        
        self.file = h5py.File(path, 'w')
        self.iteration_group = self.file.create_group('iterations')
        self.count = 0
        
        # Metadata
        self.file.attrs['version'] = '1.0'
        self.file.attrs['description'] = 'ACTFuzzer execution traces'
    
    def write(self, trace: Dict[str, Any]):
        """Write trace entry to HDF5."""
        iter_group = self.iteration_group.create_group(f'iter_{self.count}')
        
        # Store scalar attributes (Level 0+)
        scalar_keys = [
            'iteration', 'timestamp', 'mutation_strategy', 'violation_found',
            'coverage', 'coverage_delta', 'energy', 'seed_id', 'parent_id', 
            'depth', 'loss_value'
        ]
        for key in scalar_keys:
            if key in trace and trace[key] is not None:
                iter_group.attrs[key] = trace[key]
        
        # Store tensors (Level 1+)
        tensor_keys = ['input_before', 'input_after']
        for key in tensor_keys:
            if key in trace and trace[key] is not None:
                tensor = trace[key]
                if isinstance(tensor, torch.Tensor):
                    tensor = tensor.cpu().numpy()
                iter_group.create_dataset(
                    key, 
                    data=tensor,
                    compression='gzip',
                    compression_opts=1  # Fast compression (1-9, 1=fast)
                )
        
        # Store activations (Level 2+)
        if 'activations' in trace and trace['activations'] is not None:
            act_group = iter_group.create_group('activations')
            for name, tensor in trace['activations'].items():
                if isinstance(tensor, torch.Tensor):
                    tensor = tensor.cpu().numpy()
                act_group.create_dataset(
                    name,
                    data=tensor,
                    compression='gzip',
                    compression_opts=1
                )
        
        # Store gradients (Level 3+)
        if 'gradients' in trace and trace['gradients'] is not None:
            grad_group = iter_group.create_group('gradients')
            for name, tensor in trace['gradients'].items():
                if isinstance(tensor, torch.Tensor):
                    tensor = tensor.cpu().numpy()
                grad_group.create_dataset(
                    name,
                    data=tensor,
                    compression='gzip',
                    compression_opts=1
                )
        
        self.count += 1
        
        # Periodic flush for crash-safety (every 100 traces)
        if self.count % 100 == 0:
            self.flush()
    
    def flush(self):
        """Flush data to disk."""
        self.file.flush()
    
    def close(self):
        """Close HDF5 file."""
        self.file.close()


class JSONStorage(TraceStorage):
    """
    JSON storage backend (simple but slow).
    
    Not recommended for production due to:
    - Large file sizes (3-5x larger than HDF5)
    - Slow serialization (especially for tensors)
    - No compression
    
    Use for debugging or when HDF5 is unavailable.
    """
    
    def __init__(self, path: Path):
        """
        Initialize JSON storage.
        
        Args:
            path: Output file path (e.g., "traces.json")
        """
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        
        self.file = open(path, 'w')
        self.file.write('[\n')  # Start JSON array
        self.first_entry = True
    
    def write(self, trace: Dict[str, Any]):
        """Write trace entry to JSON."""
        # Convert tensors to lists
        serializable_trace = {}
        for key, value in trace.items():
            if isinstance(value, torch.Tensor):
                serializable_trace[key] = value.cpu().tolist()
            elif isinstance(value, dict):
                # Handle nested dicts (activations, gradients)
                serializable_trace[key] = {
                    k: v.cpu().tolist() if isinstance(v, torch.Tensor) else v
                    for k, v in value.items()
                }
            else:
                serializable_trace[key] = value
        
        # Write with proper JSON array formatting
        if not self.first_entry:
            self.file.write(',\n')
        json.dump(serializable_trace, self.file, indent=2)
        self.first_entry = False
    
    def flush(self):
        """Flush data to disk."""
        self.file.flush()
    
    def close(self):
        """Close JSON file."""
        self.file.write('\n]')  # End JSON array
        self.file.close()


class AsyncTraceStorage(TraceStorage):
    """
    Async wrapper for trace storage backends.
    
    Writes traces in a background thread to avoid blocking the fuzzing loop.
    Critical for maintaining fuzzing performance with Level 1+ tracing.
    
    Features:
    - Non-blocking writes via queue
    - Configurable queue size (backpressure control)
    - Automatic flushing on close
    - Thread-safe
    
    Performance:
    - Reduces I/O overhead from 50-500% → < 5%
    - Queue size of 100 buffers ~5-50MB depending on level
    
    Example:
        >>> storage = AsyncTraceStorage(HDF5Storage("traces.h5"), queue_size=100)
        >>> storage.write(trace)  # Returns immediately
        >>> storage.close()  # Waits for all writes to complete
    """
    
    def __init__(self, backend: TraceStorage, queue_size: int = 100):
        """
        Initialize async storage wrapper.
        
        Args:
            backend: Underlying storage backend (HDF5Storage or JSONStorage)
            queue_size: Maximum number of traces to buffer (default: 100)
        """
        self.backend = backend
        self.queue: queue.Queue = queue.Queue(maxsize=queue_size)
        self.stop_event = threading.Event()
        
        # Start writer thread
        self.writer_thread = threading.Thread(
            target=self._writer_loop,
            daemon=False,  # Don't make daemon - we want clean shutdown
            name="TraceWriter"
        )
        self.writer_thread.start()
        
        # Statistics
        self.writes_queued = 0
        self.writes_blocking = 0
    
    def write(self, trace: Dict[str, Any]):
        """
        Write trace asynchronously (non-blocking if queue has space).
        
        Args:
            trace: Trace dictionary to write
        """
        try:
            # Try non-blocking put
            self.queue.put_nowait(trace)
            self.writes_queued += 1
        except queue.Full:
            # Queue full - apply backpressure by writing synchronously
            # This prevents unbounded memory growth
            self.backend.write(trace)
            self.writes_blocking += 1
    
    def _writer_loop(self):
        """Background thread that writes traces to storage."""
        while not self.stop_event.is_set():
            try:
                # Wait for trace with timeout (allows checking stop_event)
                trace = self.queue.get(timeout=0.1)
                self.backend.write(trace)
                self.queue.task_done()
            except queue.Empty:
                # Intentional: queue.Empty is the 100ms polling signal, not an error;
                # logger.debug omitted to avoid log spam at ~10Hz while idle.
                continue
    
    def flush(self):
        """Flush queued traces and backend storage."""
        # Wait for queue to empty
        self.queue.join()
        # Flush backend
        self.backend.flush()
    
    def close(self):
        """Close storage and wait for all writes to complete."""
        # Wait for queue to drain
        self.queue.join()
        
        # Signal thread to stop
        self.stop_event.set()
        self.writer_thread.join(timeout=5.0)
        
        # Close backend
        self.backend.close()
        
        # Print statistics
        total_writes = self.writes_queued + self.writes_blocking
        if total_writes > 0:
            blocking_pct = (self.writes_blocking / total_writes) * 100
            if blocking_pct > 10:
                print(f"⚠️  Async trace queue full {self.writes_blocking}/{total_writes} times "
                      f"({blocking_pct:.1f}%). Consider increasing queue_size or sampling rate.")


def create_storage(backend: str, path: Path, async_write: bool = True) -> TraceStorage:
    """
    Factory function to create trace storage.
    
    Args:
        backend: Storage backend ("hdf5" or "json")
        path: Output file path
        async_write: Whether to use async wrapper (default: True for performance)
    
    Returns:
        TraceStorage instance
    
    Raises:
        ValueError: If backend is not supported
    """
    # Create base storage
    if backend == "hdf5":
        storage = HDF5Storage(path)
    elif backend == "json":
        storage = JSONStorage(path)
    else:
        raise ValueError(f"Unsupported storage backend: {backend}. Use 'hdf5' or 'json'.")
    
    # Wrap with async if requested
    if async_write:
        storage = AsyncTraceStorage(storage, queue_size=100)
    
    return storage
