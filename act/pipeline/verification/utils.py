#===- act/pipeline/utils.py - Pipeline Testing Utilities ---------------====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
#
# Purpose:
#   Shared utilities for ACT pipeline testing framework. Provides common
#   utilities for parallel execution, performance profiling, logging,
#   and other shared functionality across the pipeline testing system.
#
#===---------------------------------------------------------------------===#

import time
import psutil
import logging
import functools
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Callable, Any, Dict, Optional, Tuple
from dataclasses import dataclass
from contextlib import contextmanager
import torch

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Shape utilities
# -----------------------------------------------------------------------------

def _prod(shape_tail: Tuple[int, ...]) -> int:
    """Compute product of shape dimensions."""
    result = 1
    for s in shape_tail:
        result *= int(s)
    return result


def _normalize_tuple(val: Any, default: Tuple[int, int] = (1, 1)) -> Tuple[int, int]:
    """Normalize int, list, or tuple to 2-tuple for kernel_size/stride/padding.

    Lists are accepted because onnx2torch constructs nn.MaxPool2d/AvgPool2d with
    list-typed kernel_size / stride / padding (PyTorch's own constructors
    typically store them as tuples).
    """
    if isinstance(val, (list, tuple)):
        return tuple(val)
    return (val, val) if val is not None else default


def _assert_dag(preds: Dict[int, List[int]], succs: Dict[int, List[int]], n_layers: int) -> None:
    """Kahn's algorithm cycle check. Raises ValueError listing the cycle nodes."""
    if n_layers == 0:
        return
    in_degree = {i: len(preds.get(i, [])) for i in range(n_layers)}
    queue = [i for i in range(n_layers) if in_degree[i] == 0]
    visited = 0
    while queue:
        node = queue.pop(0)
        visited += 1
        for succ in succs.get(node, []):
            in_degree[succ] -= 1
            if in_degree[succ] == 0:
                queue.append(succ)
    if visited != n_layers:
        cycle_nodes = [i for i in range(n_layers) if in_degree[i] > 0]
        raise ValueError(f"Layer graph contains a cycle! Nodes: {cycle_nodes}")


def _normalize_axes(axes: Any, rank: int) -> List[int]:
    """Sort + dedupe + normalise possibly-negative axes against ``rank``."""
    return sorted({(a + rank) if a < 0 else a for a in (int(x) for x in axes)})


def _reduce_output_shape(input_shape: Tuple[int, ...], norm_axes: List[int],
                         keepdims: bool) -> Tuple[int, ...]:
    """Output shape of a reduce-along-axes op. ``norm_axes`` must be already-normalised."""
    if keepdims:
        return tuple(1 if i in norm_axes else int(d) for i, d in enumerate(input_shape))
    return tuple(int(d) for i, d in enumerate(input_shape) if i not in norm_axes) or (1,)


def _compute_slice_output_shape(
    input_shape: Tuple[int, ...],
    starts: List[int], ends: List[int],
    axes: List[int], steps: List[int],
) -> Tuple[Tuple[int, ...], List[int], List[int], List[int]]:
    """ONNX Slice output shape with negative-index + clamp + step semantics.

    Returns ``(output_shape, n_starts, n_ends, n_axes)`` with all indices
    normalised against ``input_shape``. Raises on zero step.
    """
    rank = len(input_shape)
    n_starts: List[int] = []
    n_ends: List[int] = []
    n_axes: List[int] = []
    output_shape = list(input_shape)
    for s, e, ax, st in zip(starts, ends, axes, steps):
        ax = int(ax) + rank if int(ax) < 0 else int(ax)
        dim = int(input_shape[ax])
        st = int(st)
        if st == 0:
            raise ValueError("OnnxSlice: zero step")
        s = int(s) + dim if int(s) < 0 else int(s)
        e = int(e) + dim if int(e) < 0 else int(e)
        if st > 0:
            s, e = min(max(s, 0), dim), min(max(e, 0), dim)
        else:
            s, e = min(max(s, -1), dim - 1), min(max(e, -1), dim - 1)
        output_shape[ax] = max(0, len(range(s, e, st)))
        n_starts.append(s); n_ends.append(e); n_axes.append(ax)
    return tuple(output_shape), n_starts, n_ends, n_axes


def _broadcast_const_to_size(const: torch.Tensor, size: int, dtype: torch.dtype) -> torch.Tensor:
    """Broadcast a constant tensor to a flat variable count.

    Handles scalars (numel==1), exact-size vectors, and integer-multiple
    repetitions (shape (C,) → flat (C*spatial,)). Empty constants fall back
    to zeros — only sound for ADD/SUB; MUL/DIV callers must reject empty.
    """
    flat = const.reshape(-1)
    if flat.numel() == 0:
        return torch.zeros(size, dtype=dtype)
    if flat.numel() == 1:
        return flat.expand(size).clone().to(dtype)
    if flat.numel() == size:
        return flat.clone().to(dtype)
    if size % flat.numel() == 0:
        return flat.repeat(size // flat.numel()).to(dtype)
    if flat.numel() % size == 0:
        return flat[:size].clone().to(dtype)
    raise ValueError(
        f"Cannot broadcast constant of shape {tuple(const.shape)} to flat size {size}"
    )


@dataclass
class PerformanceMetrics:  # pragma: no cover
    """Performance metrics for validation operations."""
    execution_time: float
    peak_memory_mb: float
    cpu_usage_percent: float
    gpu_memory_mb: Optional[float] = None


@dataclass
class ParallelResult:  # pragma: no cover
    """Result from parallel execution."""
    results: List[Any]
    failed_tasks: List[Tuple[int, Exception]]
    total_time: float
    metrics: PerformanceMetrics


class PerformanceProfiler:  # pragma: no cover
    """Performance profiling utilities for validation operations."""
    
    def __init__(self):
        self.start_time: Optional[float] = None
        self.start_memory: Optional[float] = None
        self.peak_memory: float = 0
        self.cpu_usage_samples: List[float] = []
        self._monitoring_thread: Optional[threading.Thread] = None
        self._stop_monitoring = threading.Event()
    
    def start(self) -> None:
        """Start performance monitoring."""
        self.start_time = time.time()
        self.start_memory = self._get_memory_usage()
        self.peak_memory = self.start_memory
        self.cpu_usage_samples = []
        self._stop_monitoring.clear()
        
        # Start monitoring thread
        self._monitoring_thread = threading.Thread(target=self._monitor_resources)
        self._monitoring_thread.daemon = True
        self._monitoring_thread.start()
        
        logger.debug("Performance profiling started")
    
    def stop(self) -> PerformanceMetrics:
        """Stop monitoring and return performance metrics."""
        if self.start_time is None:
            raise RuntimeError("Profiler not started")
        
        # Stop monitoring thread
        self._stop_monitoring.set()
        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=1.0)
        
        execution_time = time.time() - self.start_time
        avg_cpu_usage = sum(self.cpu_usage_samples) / len(self.cpu_usage_samples) if self.cpu_usage_samples else 0
        
        metrics = PerformanceMetrics(
            execution_time=execution_time,
            peak_memory_mb=self.peak_memory,
            cpu_usage_percent=avg_cpu_usage,
            gpu_memory_mb=self._get_gpu_memory() if torch.cuda.is_available() else None
        )
        
        logger.debug(f"Performance metrics: {metrics}")
        return metrics
    
    def _monitor_resources(self) -> None:
        """Monitor resource usage in background thread."""
        while not self._stop_monitoring.wait(0.1):  # Sample every 100ms
            try:
                # Monitor memory
                current_memory = self._get_memory_usage()
                self.peak_memory = max(self.peak_memory, current_memory)
                
                # Monitor CPU
                cpu_usage = psutil.cpu_percent(interval=None)
                self.cpu_usage_samples.append(cpu_usage)
                
            except Exception as e:
                logger.warning(f"Resource monitoring error: {e}")
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    
    def _get_gpu_memory(self) -> Optional[float]:
        """Get GPU memory usage in MB."""
        if not torch.cuda.is_available():
            return None
        try:
            return torch.cuda.memory_allocated() / 1024 / 1024
        except Exception:
            return None


@contextmanager
def profile_performance():  # pragma: no cover
    """Context manager for performance profiling."""
    profiler = PerformanceProfiler()
    profiler.start()
    try:
        yield profiler
    finally:
        metrics = profiler.stop()
        yield metrics


class ParallelExecutor:  # pragma: no cover
    """Utilities for parallel execution of validation tasks."""
    
    def __init__(self, max_workers: Optional[int] = None, timeout: Optional[float] = None):
        """
        Initialize parallel executor.
        
        Args:
            max_workers: Maximum number of worker threads
            timeout: Timeout for individual tasks in seconds
        """
        self.max_workers = max_workers or min(32, (psutil.cpu_count() or 1) + 4)
        self.timeout = timeout
    
    def execute_parallel(self, 
                        tasks: List[Callable],
                        task_args: Optional[List[Tuple]] = None,
                        task_kwargs: Optional[List[Dict]] = None) -> ParallelResult:
        """
        Execute tasks in parallel.
        
        Args:
            tasks: List of callable tasks to execute
            task_args: List of argument tuples for each task
            task_kwargs: List of keyword argument dicts for each task
            
        Returns:
            ParallelResult with results and performance metrics
        """
        if task_args is None:
            task_args = [() for _ in tasks]
        if task_kwargs is None:
            task_kwargs = [{} for _ in tasks]
        
        if len(tasks) != len(task_args) or len(tasks) != len(task_kwargs):
            raise ValueError("tasks, task_args, and task_kwargs must have same length")
        
        results = []
        failed_tasks = []
        
        with profile_performance() as profiler:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all tasks
                future_to_index = {
                    executor.submit(task, *args, **kwargs): i
                    for i, (task, args, kwargs) in enumerate(zip(tasks, task_args, task_kwargs))
                }
                
                # Collect results
                for future in as_completed(future_to_index, timeout=self.timeout):
                    task_index = future_to_index[future]
                    try:
                        result = future.result()
                        results.append((task_index, result))
                    except Exception as e:
                        failed_tasks.append((task_index, e))
                        logger.error(f"Task {task_index} failed: {e}")
        
        # Sort results by original task index
        results.sort(key=lambda x: x[0])
        sorted_results = [result for _, result in results]
        
        metrics = profiler.stop()
        
        return ParallelResult(
            results=sorted_results,
            failed_tasks=failed_tasks,
            total_time=metrics.execution_time,
            metrics=metrics
        )
    
    def map_parallel(self, func: Callable, items: List[Any]) -> ParallelResult:
        """
        Apply function to list of items in parallel.
        
        Args:
            func: Function to apply to each item
            items: List of items to process
            
        Returns:
            ParallelResult with mapped results
        """
        tasks = [func for _ in items]
        task_args = [(item,) for item in items]
        
        return self.execute_parallel(tasks, task_args)


def print_memory_usage(prefix: str = "") -> None:  # pragma: no cover
    """Print current memory usage information."""
    process = psutil.Process()
    memory_info = process.memory_info()
    
    memory_mb = memory_info.rss / 1024 / 1024
    cpu_percent = process.cpu_percent()
    
    gpu_info = ""
    if torch.cuda.is_available():
        gpu_memory_mb = torch.cuda.memory_allocated() / 1024 / 1024
        gpu_max_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
        gpu_info = f", GPU: {gpu_memory_mb:.1f}MB (max: {gpu_max_mb:.1f}MB)"
    
    logger.info(f"{prefix}Memory: {memory_mb:.1f}MB, CPU: {cpu_percent:.1f}%{gpu_info}")


def clear_torch_cache() -> None:  # pragma: no cover
    """Clear PyTorch GPU cache if available."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.debug("Cleared PyTorch GPU cache")


def setup_logging(level: str = "INFO", format_str: Optional[str] = None) -> None:  # pragma: no cover
    """
    Setup logging configuration for the pipeline.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        format_str: Custom format string for log messages
    """
    if format_str is None:
        format_str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Ensure log file goes to pipeline/log directory
    import os
    from pathlib import Path
    pipeline_dir = Path(__file__).parent
    log_dir = pipeline_dir / "log"
    log_dir.mkdir(exist_ok=True)
    log_file_path = log_dir / "pipeline_tests.log"
    
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=format_str,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file_path)
        ]
    )
    
    # Reduce noise from some libraries
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)


def retry_on_failure(max_retries: int = 3, delay: float = 1.0, backoff: float = 2.0):  # pragma: no cover
    """
    Decorator to retry function on failure.
    
    Args:
        max_retries: Maximum number of retry attempts
        delay: Initial delay between retries in seconds
        backoff: Backoff multiplier for delay
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            current_delay = delay
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries:
                        logger.warning(f"Attempt {attempt + 1} failed for {func.__name__}: {e}")
                        time.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        logger.error(f"All {max_retries + 1} attempts failed for {func.__name__}")
            
            raise last_exception
        
        return wrapper
    return decorator


def timeout_handler(timeout_seconds: float):  # pragma: no cover
    """
    Decorator to add timeout to function execution.
    
    Args:
        timeout_seconds: Timeout in seconds
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            import signal
            
            def timeout_signal_handler(signum, frame):
                raise TimeoutError(f"Function {func.__name__} timed out after {timeout_seconds} seconds")
            
            # Set up signal handler
            old_handler = signal.signal(signal.SIGALRM, timeout_signal_handler)
            signal.alarm(int(timeout_seconds))
            
            try:
                result = func(*args, **kwargs)
            finally:
                # Restore old signal handler
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
            
            return result
        
        return wrapper
    return decorator


class ProgressTracker:  # pragma: no cover
    """Track progress of long-running operations."""
    
    def __init__(self, total_items: int, description: str = "Processing"):
        self.total_items = total_items
        self.description = description
        self.completed_items = 0
        self.start_time = time.time()
    
    def update(self, completed: int = 1) -> None:
        """Update progress by specified number of completed items."""
        self.completed_items += completed
        self._print_progress()
    
    def _print_progress(self) -> None:
        """Print current progress."""
        if self.total_items == 0:
            return
        
        percentage = (self.completed_items / self.total_items) * 100
        elapsed_time = time.time() - self.start_time
        
        if self.completed_items > 0:
            eta = (elapsed_time / self.completed_items) * (self.total_items - self.completed_items)
            eta_str = f", ETA: {eta:.1f}s"
        else:
            eta_str = ""
        
        logger.info(f"{self.description}: {self.completed_items}/{self.total_items} "
                   f"({percentage:.1f}%) - {elapsed_time:.1f}s elapsed{eta_str}")
    
    def finish(self) -> None:
        """Mark progress as complete."""
        self.completed_items = self.total_items
        elapsed_time = time.time() - self.start_time
        logger.info(f"{self.description} completed in {elapsed_time:.1f}s")


# -----------------------------------------------------------------------------
# ONNX -> ACT layer handlers (bound onto _LayerGraphBuilder via setattr in
# torch2act.py). Kept here only to keep that file manageable; ``self`` is always
# a _LayerGraphBuilder instance and these touch its private API directly.
# -----------------------------------------------------------------------------

import torch.fx as fx
import torch.nn as nn
from act.back_end.layer_schema import LayerKind

def _convert_OnnxNeg(self, mod: nn.Module, node: fx.Node) -> None:
    """OnnxNeg: y = -x. Emitted as SCALE with a = -1."""
    if not self._get_predecessor_state(node):
        raise ValueError(f"OnnxNeg: missing predecessor for {node.name}")
    size = len(self.prev_out)
    out_vars = self._same_size_forward()
    layer_id = self._add_layer(
        LayerKind.SCALE.value,
        {"a": torch.full((size,), -1.0, dtype=self.dtype),
         "input_shape": self.shape, "output_shape": self.shape},
        self.prev_out, out_vars,
    )
    self.prev_out = out_vars
    self._register_node(node.name, layer_id)

def _convert_OnnxTranspose(self, mod: nn.Module, node: fx.Node) -> None:
    """OnnxTranspose: y = x.permute(perm)."""
    if not self._get_predecessor_state(node):
        raise ValueError(f"OnnxTranspose: missing predecessor for {node.name}")
    perm = tuple(int(p) for p in (getattr(mod, 'perm', None) or list(range(len(self.shape)))[::-1]))
    if len(perm) != len(self.shape):
        raise ValueError(f"OnnxTranspose: perm rank {len(perm)} != input rank {len(self.shape)}")
    output_shape = tuple(self.shape[p] for p in perm)
    input_shape_t = tuple(self.shape)
    out_vars = self._same_size_forward()
    layer_id = self._add_layer(
        LayerKind.TRANSPOSE.value, {"perm": perm, "input_shape": input_shape_t},
        self.prev_out, out_vars,
    )
    self.prev_out = out_vars
    self.shape = output_shape
    self._register_node(node.name, layer_id)

def _convert_OnnxReshape(self, mod: nn.Module, node: fx.Node) -> None:
    """OnnxReshape with ONNX 0/-1 dim semantics (0 = keep input dim, -1 = infer).

    Target shape resolution falls back through three tiers:
    direct get_attr → upstream layer's stored value → constant subgraph
    evaluation (handles e.g. shape derived via Concat-of-Shape ops).
    """
    if not self._get_predecessor_state(node):
        raise ValueError(f"OnnxReshape: missing predecessor for {node.name}")
    args = [a for a in node.args if isinstance(a, fx.Node)]
    shape_tensor = self._resolve_constant_tensor(args[1].name) if len(args) >= 2 else None
    if shape_tensor is None and len(args) >= 2:
        shape_tensor = self._evaluate_constant_subgraph(args[1].name)
    if shape_tensor is None:
        raise ValueError(f"OnnxReshape: cannot resolve target shape at {node.name}")
    raw = [int(x) for x in shape_tensor.flatten().tolist()]
    resolved = [int(self.shape[i]) if d == 0 else d for i, d in enumerate(raw)]
    if -1 in resolved:
        known = _prod(tuple(d for d in resolved if d != -1)) or 1
        resolved[resolved.index(-1)] = _prod(self.shape) // known
    output_shape = tuple(resolved)
    out_vars = self._same_size_forward()
    layer_id = self._add_layer(
        LayerKind.RESHAPE.value, {"target_shape": output_shape}, self.prev_out, out_vars,
    )
    self.prev_out = out_vars
    self.shape = output_shape
    self._register_node(node.name, layer_id)

def _convert_OnnxConstant(self, mod: nn.Module, node: fx.Node) -> None:
    """Emit ACT CONSTANT layer for an onnx2torch OnnxConstant module.

    OnnxConstant wraps a fixed tensor that the model's forward returns at this
    position.  Materialising it as a CONSTANT layer lets registered-var
    consumers (Reshape's shape arg, Slice bounds, Pow exponent, MatMul second
    operand, etc.) resolve it via ``node_outputs[node.name]``.  Lazy consumers
    that walk the FX graph through ``_evaluate_constant_subgraph`` continue
    to work because ``mod.forward()`` is unchanged.

    Integer dtypes are preserved (shape constants must stay int); only
    floating-point values get cast to ``self.dtype``.
    """
    val = getattr(mod, "value", None)
    if val is None:
        val = next(iter(mod.buffers()), None)
    if val is None:
        raise NotImplementedError(
            f"OnnxConstant at {node.name} has no .value attribute or buffer"
        )
    if not isinstance(val, torch.Tensor):
        val = torch.tensor(val)
    val = val.detach().clone()
    if val.is_floating_point():
        val = val.to(self.dtype)
    flat = val.reshape(-1)
    shape = tuple(int(d) for d in val.shape) or (1,)
    out_vars = self._alloc_ids(int(flat.numel()) or 1)
    layer_id = self._add_layer(
        LayerKind.CONSTANT.value,
        {"value": flat, "input_shape": shape, "output_shape": shape},
        [], out_vars,
    )
    self.node_outputs[node.name] = out_vars
    self.node_shapes[node.name] = shape
    self.node_to_layer_id[node.name] = layer_id


def _convert_OnnxConcat(self, mod: nn.Module, node: fx.Node) -> None:
    """OnnxConcat: y = cat(*input_tensors, axis).

    Constant-initializer args (e.g. ViT's [CLS] token) are materialised
    on-demand via ``_ensure_constant_vars`` before the CONCAT layer is
    emitted, so each input has registered vars in ``node_outputs``.
    """
    axis = int(getattr(mod, 'axis', 0))
    args = [a for a in node.args if isinstance(a, fx.Node)]
    if not args:
        raise ValueError(f"OnnxConcat: no inputs at {node.name}")
    all_vars: List[int] = []
    shapes: List[Tuple[int, ...]] = []
    for arg in args:
        if arg.name not in self.node_outputs and not self._ensure_constant_vars(arg.name):
            raise ValueError(f"OnnxConcat at {node.name}: input '{arg.name}' is neither a registered variable nor a resolvable constant")
        all_vars.extend(self.node_outputs[arg.name])
        shapes.append(self.node_shapes[arg.name])
    norm_axis = axis if axis >= 0 else axis + len(shapes[0])
    out_shape = list(shapes[0])
    out_shape[norm_axis] = sum(int(s[norm_axis]) for s in shapes)
    output_shape = tuple(out_shape)
    out_vars = self._alloc_ids(len(all_vars))
    layer_id = self._add_layer(
        LayerKind.CONCAT.value, {"concat_dim": axis}, all_vars, out_vars,
    )
    self.prev_out = out_vars
    self.shape = output_shape
    self._register_node(node.name, layer_id)

def _convert_OnnxReduceStaticAxes(self, mod: nn.Module, node: fx.Node) -> None:
    """OnnxReduceStaticAxes (ReduceMean / ReduceMax / ReduceMin / ReduceSum etc.).

    Currently only ReduceMean is mapped to LayerKind.MEAN. Other reductions
    (Max / Min / L2-norm) need their own LayerKind mapping (future enhancement).
    """
    if not self._get_predecessor_state(node):
        raise ValueError(f"OnnxReduceStaticAxes: missing predecessor for {node.name}")
    op_func = getattr(mod, 'math_op_function', None)
    op_name = getattr(op_func, '__name__', '').lower() if op_func is not None else ''
    if 'mean' not in op_name:
        raise NotImplementedError(
            f"OnnxReduceStaticAxes at {node.name}: only ReduceMean supported (got '{op_name}'; future enhancement)"
        )
    # OnnxReduceStaticAxes uses public ``axes`` / ``keepdims`` (different from
    # OnnxReduceSumStaticAxes which uses private ``_axes`` / ``_keepdims``).
    axes_attr = getattr(mod, 'axes', None) or list(range(len(self.shape)))
    keepdims = bool(getattr(mod, 'keepdims', True))
    norm_axes = _normalize_axes(axes_attr, len(self.shape))
    output_shape = _reduce_output_shape(self.shape, norm_axes, keepdims)
    out_vars = self._alloc_ids(_prod(output_shape) or 1)
    layer_id = self._add_layer(
        LayerKind.MEAN.value,
        {"dim": list(norm_axes), "keepdim": int(keepdims),
         "input_shape": self.shape, "output_shape": output_shape},
        self.prev_out, out_vars,
    )
    self.prev_out = out_vars
    self.shape = output_shape
    self._register_node(node.name, layer_id)

def _convert_OnnxReduceSumStaticAxes(self, mod: nn.Module, node: fx.Node) -> None:
    """OnnxReduceSumStaticAxes: y = sum(x, axes, keepdim)."""
    if not self._get_predecessor_state(node):
        raise ValueError(f"OnnxReduceSumStaticAxes: missing predecessor for {node.name}")
    axes = getattr(mod, '_axes', None) or list(range(len(self.shape)))
    keepdims = bool(int(getattr(mod, '_keepdims', 1)))
    norm_axes = _normalize_axes(axes, len(self.shape))
    output_shape = _reduce_output_shape(self.shape, norm_axes, keepdims)
    out_vars = self._alloc_ids(_prod(output_shape) or 1)
    layer_id = self._add_layer(
        LayerKind.REDUCE_SUM.value,
        {"axes": list(norm_axes), "keepdims": int(keepdims),
         "input_shape": self.shape, "output_shape": output_shape},
        self.prev_out, out_vars,
    )
    self.prev_out = out_vars
    self.shape = output_shape
    self._register_node(node.name, layer_id)

def _convert_OnnxGather(self, mod: nn.Module, node: fx.Node) -> None:
    """OnnxGather: numpy.take(x, indices, axis=_axis)."""
    if not self._get_predecessor_state(node):
        raise ValueError(f"OnnxGather: missing predecessor for {node.name}")
    axis = int(getattr(mod, '_axis', 0))
    args = [a for a in node.args if isinstance(a, fx.Node)]
    idx = self._resolve_constant_tensor(args[1].name) if len(args) >= 2 else None
    if idx is None:
        raise ValueError(f"OnnxGather: cannot resolve indices at {node.name}")
    indices = idx.detach().clone().to(torch.int64)
    norm_axis = axis if axis >= 0 else axis + len(self.shape)
    if indices.dim() == 0:
        output_shape = tuple(self.shape[:norm_axis] + self.shape[norm_axis + 1:]) or (1,)
    else:
        output_shape = (*self.shape[:norm_axis], *indices.shape, *self.shape[norm_axis + 1:])
    out_vars = self._alloc_ids(_prod(output_shape) or 1)
    layer_id = self._add_layer(
        LayerKind.GATHER.value,
        {"indices": indices, "axis": axis,
         "input_shape": self.shape, "output_shape": output_shape},
        self.prev_out, out_vars,
    )
    self.prev_out = out_vars
    self.shape = output_shape
    self._register_node(node.name, layer_id)

def _convert_OnnxMatMul(self, mod: nn.Module, node: fx.Node) -> None:
    """OnnxMatMul: dispatches three cases by operand kind.

    - var × const-2D weight  → DENSE (Linear-equivalent, W.T as weight)
    - var × const-1D weight  → SCALE + REDUCE_SUM (matrix-vector dot)
    - var × var              → MATMUL layer (bilinear; TF support deferred)
    """
    args = [a for a in node.args if isinstance(a, fx.Node)]
    x_node, w_node = args[0], args[1]
    x_var = x_node.name in self.node_outputs
    w_var = w_node.name in self.node_outputs

    if x_var and w_var:
        xv = self.node_outputs[x_node.name]
        yv = self.node_outputs[w_node.name]
        xs = self.node_shapes[x_node.name]
        ys = self.node_shapes[w_node.name]
        try:
            output_shape = tuple(int(d) for d in torch.matmul(
                torch.zeros(xs, dtype=self.dtype),
                torch.zeros(ys, dtype=self.dtype),
            ).shape)
        except RuntimeError as e:
            raise ValueError(
                f"OnnxMatMul at {node.name}: incompatible var-var shapes {xs} @ {ys} ({e})"
            )
        out_vars = self._alloc_ids(_prod(output_shape) or 1)
        layer_id = self._add_layer(
            LayerKind.MATMUL.value,
            {"x_vars": xv, "y_vars": yv,
             "x_shape": xs, "y_shape": ys,
             "input_shape": xs, "output_shape": output_shape},
            xv + yv, out_vars,
        )
        self.prev_out = out_vars
        self.shape = output_shape
        self._register_node(node.name, layer_id)
        return

    if not x_var:
        # const × var: materialise the constant via _ensure_constant_vars
        # then re-enter the var × var branch above. Note matmul is
        # non-commutative — we keep operand order intact.
        if not self._ensure_constant_vars(x_node.name):
            raise ValueError(
                f"OnnxMatMul at {node.name}: cannot resolve constant first operand"
            )
        xv = self.node_outputs[x_node.name]
        yv = self.node_outputs[w_node.name]
        xs = self.node_shapes[x_node.name]
        ys = self.node_shapes[w_node.name]
        try:
            output_shape = tuple(int(d) for d in torch.matmul(
                torch.zeros(xs, dtype=self.dtype),
                torch.zeros(ys, dtype=self.dtype),
            ).shape)
        except RuntimeError as e:
            raise ValueError(
                f"OnnxMatMul at {node.name}: incompatible const-var shapes {xs} @ {ys} ({e})"
            )
        out_vars = self._alloc_ids(_prod(output_shape) or 1)
        layer_id = self._add_layer(
            LayerKind.MATMUL.value,
            {"x_vars": xv, "y_vars": yv,
             "x_shape": xs, "y_shape": ys,
             "input_shape": xs, "output_shape": output_shape},
            xv + yv, out_vars,
        )
        self.prev_out = out_vars
        self.shape = output_shape
        self._register_node(node.name, layer_id)
        return
    W = self._resolve_constant_tensor(w_node.name)
    if W is None:
        raise ValueError(f"OnnxMatMul at {node.name}: cannot resolve constant weight")
    self.prev_out = self.node_outputs[x_node.name].copy()
    self.shape = self.node_shapes[x_node.name]

    if W.dim() == 1:
        # PyTorch matrix-vector: (..., K) @ (K,) -> (...) — sum-product along last dim.
        # Realised as element-wise SCALE (broadcast W over var's leading dims) then REDUCE_SUM
        # along the last axis with keepdims=0.
        K = int(W.shape[0])
        if not self.shape or int(self.shape[-1]) != K:
            raise ValueError(
                f"OnnxMatMul at {node.name}: var last dim {self.shape[-1] if self.shape else None} != W len {K}"
            )
        scale_a = W.expand(*self.shape).contiguous().to(self.dtype).flatten()
        scale_out = self._same_size_forward()
        self._add_layer(
            LayerKind.SCALE.value,
            {"a": scale_a, "input_shape": self.shape, "output_shape": self.shape},
            self.prev_out, scale_out,
        )
        self.prev_out = scale_out
        output_shape = tuple(self.shape[:-1]) or (1,)
        out_vars = self._alloc_ids(_prod(output_shape) or 1)
        layer_id = self._add_layer(
            LayerKind.REDUCE_SUM.value,
            {"axes": [len(self.shape) - 1], "keepdims": 0,
             "input_shape": self.shape, "output_shape": output_shape},
            self.prev_out, out_vars,
        )
        self.prev_out = out_vars
        self.shape = output_shape
        self._register_node(node.name, layer_id)
        return

    if W.dim() != 2:
        raise NotImplementedError(
            f"OnnxMatMul at {node.name}: only 1D / 2D constant weights supported (got {tuple(W.shape)})"
        )
    in_features, out_features = int(W.shape[0]), int(W.shape[1])
    # Batched matmul: var shape (..., M, K) @ const W (K, N) -> (..., M, N).
    # The right consistency check is on the LAST dim of the var shape, not
    # the flattened length (which counts batch * M * K, not just K).
    if not self.shape or int(self.shape[-1]) != in_features:
        raise ValueError(
            f"OnnxMatMul at {node.name}: var last dim "
            f"{self.shape[-1] if self.shape else None} != weight in_features {in_features}"
        )
    output_shape = tuple(self.shape[:-1]) + (out_features,)
    out_vars = self._alloc_ids(_prod(output_shape) or out_features)
    # DENSE concretizes against the FLAT var vector, so a batched
    # (..., M, K) @ (K, N) that shares W across M tokens must become a
    # block-diagonal weight (token_count copies of W.T) — exact, not relaxed.
    weight_t = W.t().contiguous().detach().clone().to(self.dtype)
    n_in_vars = len(self.prev_out)
    if n_in_vars % in_features != 0:
        raise ValueError(
            f"OnnxMatMul at {node.name}: flat input var count {n_in_vars} "
            f"is not a multiple of weight in_features {in_features}"
        )
    token_count = n_in_vars // in_features
    weight_mat = torch.block_diag(*([weight_t] * token_count)) if token_count > 1 else weight_t
    layer_id = self._add_layer(
        LayerKind.DENSE.value,
        {"weight": weight_mat,
         "in_features": int(weight_mat.shape[1]), "out_features": int(weight_mat.shape[0]),
         "input_shape": self.shape, "output_shape": output_shape},
        self.prev_out, out_vars,
    )
    self.prev_out = out_vars
    self.shape = output_shape
    self._register_node(node.name, layer_id)

def _convert_OnnxShape(self, mod: nn.Module, node: fx.Node) -> None:
    """OnnxShape: pure compile-time. Stash value in side map; emit no layer.

    A runtime CONSTANT layer was unsafe (zero-indegree → never seeded into
    ``analyze()``'s worklist; DualTF treats unknown layers as identity).
    Leaving ``node_outputs`` empty also keeps ``_build_preds_succs`` and
    ``_get_predecessor_state`` from mistaking Shape for a runtime tensor.
    """
    args = [a for a in node.args if isinstance(a, fx.Node)]
    if not args:
        raise ValueError(f"OnnxShape at {node.name}: no inputs")
    src = args[0]
    if src.name in self.node_shapes:
        src_shape = self.node_shapes[src.name]
    elif src.op == 'placeholder':
        src_shape = self.input_shape
    else:
        raise ValueError(f"OnnxShape at {node.name}: cannot resolve input shape for '{src.name}'")
    start = int(getattr(mod, 'start', 0) or 0)
    end_attr = getattr(mod, 'end', None)
    end = int(end_attr) if end_attr is not None else len(src_shape)
    if start < 0:
        start += len(src_shape)
    if end < 0:
        end += len(src_shape)
    start = max(0, min(start, len(src_shape)))
    end = max(start, min(end, len(src_shape)))
    self._compile_time_values[node.name] = torch.tensor(
        src_shape[start:end], dtype=self._ONNX_SHAPE_DTYPE,
    )

def _convert_OnnxSlice(self, mod: nn.Module, node: fx.Node) -> None:
    """OnnxSlice: y = x[starts:ends:steps along axes].

    The input tensor can be a constant initializer (e.g. YOLO's anchor
    constants); materialise it via ``_ensure_constant_vars`` before
    running the slice arithmetic.
    """
    args = [a for a in node.args if isinstance(a, fx.Node)]
    if len(args) < 3:
        raise ValueError(f"OnnxSlice at {node.name}: need at least 3 args")
    if args[0].name in self.node_outputs:
        self.prev_out = self.node_outputs[args[0].name].copy()
        self.shape = self.node_shapes[args[0].name]
    elif self._ensure_constant_vars(args[0].name):
        self.prev_out = self.node_outputs[args[0].name].copy()
        self.shape = self.node_shapes[args[0].name]
    else:
        raise ValueError(f"OnnxSlice: missing predecessor for {node.name}")
    starts = self._resolve_slice_input_to_int_list(args[1].name)
    ends = self._resolve_slice_input_to_int_list(args[2].name)
    if starts is None or ends is None:
        raise ValueError(f"OnnxSlice at {node.name}: cannot resolve starts/ends")
    axes = (self._resolve_slice_input_to_int_list(args[3].name)
            if len(args) > 3 else None) or list(range(len(starts)))
    steps = (self._resolve_slice_input_to_int_list(args[4].name)
             if len(args) > 4 else None) or [1] * len(starts)
    try:
        out_shape, n_starts, n_ends, n_axes = _compute_slice_output_shape(
            self.shape, starts, ends, axes, steps,
        )
    except ValueError as e:
        raise ValueError(f"OnnxSlice at {node.name}: {e}")
    out_vars = self._alloc_ids(_prod(out_shape) or 1)
    layer_id = self._add_layer(
        LayerKind.SLICE.value,
        {"starts": n_starts, "ends": n_ends, "axes": n_axes,
         "input_shape": self.shape, "output_shape": out_shape},
        self.prev_out, out_vars,
    )
    self.prev_out = out_vars
    self.shape = out_shape
    self._register_node(node.name, layer_id)

def _convert_OnnxPow(self, mod: nn.Module, node: fx.Node) -> None:
    """OnnxPow with constant integer exponent.

    Currently supports exponent==2 (squaring) by emitting MUL(var, var).
    Higher integer exponents would chain MULs; non-integer exponents need
    a real POW transfer-function and are deferred (future enhancement).
    """
    if not self._get_predecessor_state(node):
        raise ValueError(f"OnnxPow: missing predecessor for {node.name}")
    args = [a for a in node.args if isinstance(a, fx.Node)]
    if len(args) < 2:
        raise ValueError(f"OnnxPow at {node.name}: expected 2 args")
    exp_t = self._resolve_constant_tensor(args[1].name)
    if exp_t is None:
        raise NotImplementedError(f"OnnxPow at {node.name}: dynamic exponent (future enhancement)")
    exp_val = float(exp_t.flatten().tolist()[0])
    if abs(exp_val - 2.0) > 1e-9:
        raise NotImplementedError(
            f"OnnxPow at {node.name}: only exponent==2 supported (got {exp_val}; future enhancement)"
        )
    var_vars = self.node_outputs[args[0].name]
    out_vars = self._alloc_ids(len(var_vars))
    layer_id = self._add_layer(
        LayerKind.MUL.value,
        {"x_vars": var_vars, "y_vars": var_vars,
         "input_shape": self.shape, "output_shape": self.shape},
        var_vars + var_vars, out_vars,
    )
    self.prev_out = out_vars
    self._register_node(node.name, layer_id)

def _convert_OnnxSplit13(self, mod: nn.Module, node: fx.Node) -> None:
    """OnnxSplit13: split input along an axis into chunks of given sizes.

    Decomposes into N SLICE layers (one per output chunk). Each downstream
    ``getitem(split, i)`` fx node is pre-registered to point at the i-th
    SLICE's outputs; ``_process_getitem_operation`` honours the pre-registered
    state via its early-return guard.
    """
    if not self._get_predecessor_state(node):
        raise ValueError(f"OnnxSplit13: missing predecessor for {node.name}")
    args = [a for a in node.args if isinstance(a, fx.Node)]
    if len(args) < 2:
        raise NotImplementedError(
            f"OnnxSplit13 at {node.name}: equal-axis split (no sizes input) not supported"
        )
    split_t = self._resolve_constant_tensor(args[1].name)
    if split_t is None:
        raise ValueError(f"OnnxSplit13 at {node.name}: cannot resolve split sizes")
    split_sizes = [int(x) for x in split_t.flatten().tolist()]
    axis_attr = getattr(mod, '_axis', None)
    if axis_attr is None:
        axis_attr = getattr(mod, 'axis', 0)
    rank = len(self.shape)
    norm_axis = int(axis_attr) + rank if int(axis_attr) < 0 else int(axis_attr)

    getitem_children: Dict[int, fx.Node] = {}
    if self.fx_graph is not None:
        for n in self.fx_graph.nodes:
            if n.op == 'call_function' and 'getitem' in str(n.target).lower() and n.args:
                if isinstance(n.args[0], fx.Node) and n.args[0].name == node.name and len(n.args) > 1:
                    idx_arg = n.args[1]
                    if isinstance(idx_arg, int):
                        getitem_children[idx_arg] = n

    var_vars = list(self.prev_out)
    var_shape = self.shape
    # All chunk SLICEs read the split's data input; pin their predecessor to
    # that producing layer so graph-edge resolution can't mis-wire siblings to
    # the split node (which maps only to the last chunk).
    input_layer_id = self.node_to_layer_id.get(args[0].name)
    last_chunk_vars: List[int] = var_vars
    last_chunk_shape: Tuple[int, ...] = var_shape
    last_layer_id = -1
    offset = 0
    for i, size in enumerate(split_sizes):
        chunk_shape = list(var_shape)
        chunk_shape[norm_axis] = size
        chunk_shape_t = tuple(chunk_shape)
        chunk_vars = self._alloc_ids(_prod(chunk_shape_t) or 1)
        layer_id = self._add_layer(
            LayerKind.SLICE.value,
            {"starts": [offset], "ends": [offset + size], "axes": [norm_axis],
             "input_shape": var_shape, "output_shape": chunk_shape_t},
            var_vars, chunk_vars,
        )
        if input_layer_id is not None and input_layer_id >= 0:
            self._fx_pred_override[layer_id] = [input_layer_id]
        if i in getitem_children:
            git_node = getitem_children[i]
            self.node_outputs[git_node.name] = chunk_vars
            self.node_shapes[git_node.name] = chunk_shape_t
            self.node_to_layer_id[git_node.name] = layer_id
        last_chunk_vars = chunk_vars
        last_chunk_shape = chunk_shape_t
        last_layer_id = layer_id
        offset += size

    # The Split node itself isn't directly consumed (only via getitem), but
    # downstream code that propagates from it should see at least *some*
    # valid state -- use the last chunk as the canonical output.
    self.prev_out = last_chunk_vars
    self.shape = last_chunk_shape
    self._register_node(node.name, last_layer_id)

def _convert_OnnxResize(self, mod: nn.Module, node: fx.Node) -> None:
    """OnnxResize: spatial up/downsampling (nearest / linear / bilinear etc.).

    ONNX Resize takes (input, roi, scales, sizes) where any of roi/scales/sizes
    may be empty. We resolve a float scales tensor (preferred) or fall back to
    an int sizes tensor, then compute the output shape and emit UPSAMPLE.
    """
    if not self._get_predecessor_state(node):
        raise ValueError(f"OnnxResize: missing predecessor for {node.name}")
    args = [a for a in node.args if isinstance(a, fx.Node)]
    scales_t: Optional[torch.Tensor] = None
    sizes_t: Optional[torch.Tensor] = None
    for a in args[1:]:
        t = self._resolve_constant_tensor(a.name)
        if t is None or t.numel() == 0:
            continue
        if t.dtype.is_floating_point and scales_t is None:
            scales_t = t
        elif not t.dtype.is_floating_point and sizes_t is None:
            sizes_t = t
    if scales_t is not None and scales_t.numel() == len(self.shape):
        output_shape = tuple(int(round(s * sc))
                             for s, sc in zip(self.shape, scales_t.tolist()))
        scale_factor = tuple(float(x) for x in scales_t.tolist())
        size_param = None
    elif sizes_t is not None and sizes_t.numel() == len(self.shape):
        output_shape = tuple(int(x) for x in sizes_t.tolist())
        scale_factor = None
        size_param = tuple(int(x) for x in sizes_t.tolist())
    else:
        raise ValueError(f"OnnxResize at {node.name}: cannot resolve scales or sizes")
    params: Dict[str, Any] = {
        "mode": str(getattr(mod, 'onnx_mode', 'nearest')),
        "input_shape": self.shape,
        "output_shape": output_shape,
    }
    if getattr(mod, 'align_corners', None) is not None:
        params["align_corners"] = bool(mod.align_corners)
    if scale_factor is not None:
        params["scale_factor"] = scale_factor
    if size_param is not None:
        params["size"] = size_param
    out_vars = self._alloc_ids(_prod(output_shape) or 1)
    layer_id = self._add_layer(
        LayerKind.UPSAMPLE.value, params, self.prev_out, out_vars,
    )
    self.prev_out = out_vars
    self.shape = output_shape
    self._register_node(node.name, layer_id)

def _convert_OnnxBinaryMathOperation(self, mod: nn.Module, node: fx.Node) -> None:
    """Add/Sub/Mul/Div: var-var → ADD/SUB/MUL/DIV; var-const → BIAS/SCALE (or SCALE+BIAS)."""
    op_raw = getattr(getattr(mod, 'math_op_function', None), '__name__', '').lower()
    op = {'add': 'add', 'sub': 'sub', 'mul': 'mul',
          '_onnx_div': 'div', 'div': 'div'}.get(op_raw)
    if op is None:
        raise NotImplementedError(f"OnnxBinaryMathOperation: unrecognised op '{op_raw}' at {node.name}")
    args = [a for a in node.args if isinstance(a, fx.Node)]
    x, y = args[0], args[1]
    x_var = x.name in self.node_outputs
    y_var = y.name in self.node_outputs

    if x_var and y_var:
        xv, yv = self.node_outputs[x.name], self.node_outputs[y.name]
        xs, ys = self.node_shapes[x.name], self.node_shapes[y.name]
        if len(xv) != len(yv):
            raise NotImplementedError(
                f"Var-var '{op}' size mismatch ({len(xv)} vs {len(yv)}) at {node.name}"
            )
        kind = {'add': LayerKind.ADD, 'sub': LayerKind.SUB,
                'mul': LayerKind.MUL, 'div': LayerKind.DIV}[op]
        out_shape = xs if _prod(xs) >= _prod(ys) else ys
        out_vars = self._alloc_ids(len(xv))
        layer_id = self._add_layer(
            kind.value,
            {"x_vars": xv, "y_vars": yv,
             "input_shape": xs, "output_shape": out_shape},
            xv + yv, out_vars,
        )
        self.prev_out = out_vars
        self.shape = out_shape
        self._register_node(node.name, layer_id)
        return

    if x_var:
        var_node, const_node, var_first = x, y, True
    else:
        var_node, const_node, var_first = y, x, False
    const = self._resolve_constant_tensor(const_node.name)
    if const is None:
        raise ValueError(f"OnnxBinaryMathOperation: cannot resolve constant at {node.name}")
    self.prev_out = self.node_outputs[var_node.name].copy()
    self.shape = self.node_shapes[var_node.name]

    # PyTorch broadcasting may yield an output shape *larger* than either
    # operand (outer-product case, e.g. (1,226,1) op (54,) -> (1,226,54)).
    # Detect this and prepend an EXPAND layer that replicates the variable
    # to the broadcast shape before applying BIAS/SCALE; the constant gets
    # pre-broadcasted offline since it's known at conversion time.
    try:
        broadcast_shape = tuple(int(d) for d in torch.broadcast_shapes(self.shape, tuple(const.shape)))
    except RuntimeError as e:
        raise ValueError(
            f"OnnxBinaryMathOperation at {node.name}: shapes {self.shape} and "
            f"{tuple(const.shape)} are not broadcast-compatible ({e})"
        )
    if broadcast_shape != self.shape:
        expanded_size = _prod(broadcast_shape) or 1
        expanded_vars = self._alloc_ids(expanded_size)
        self._add_layer(
            LayerKind.EXPAND.value,
            {"shape": broadcast_shape},
            self.prev_out, expanded_vars,
        )
        self.prev_out = expanded_vars
        self.shape = broadcast_shape

    size = len(self.prev_out)
    const_b = const.expand(*broadcast_shape).contiguous() if tuple(const.shape) != broadcast_shape else const
    c = _broadcast_const_to_size(const_b, size, self.dtype)

    def emit(kind: LayerKind, key: str, t: torch.Tensor, register: bool) -> None:
        out = self._same_size_forward()
        lid = self._add_layer(
            kind.value,
            {key: t, "input_shape": self.shape, "output_shape": self.shape},
            self.prev_out, out,
        )
        self.prev_out = out
        if register:
            self._register_node(node.name, lid)

    if op == 'add':
        emit(LayerKind.BIAS, "c", c, register=True)
    elif op == 'sub':
        if var_first:
            emit(LayerKind.BIAS, "c", (-c).contiguous(), register=True)
        else:
            emit(LayerKind.SCALE, "a", torch.full((size,), -1.0, dtype=self.dtype), register=False)
            emit(LayerKind.BIAS, "c", c.contiguous(), register=True)
    elif op == 'mul':
        emit(LayerKind.SCALE, "a", c, register=True)
    else:  # 'div'
        if not var_first:
            raise NotImplementedError(f"const/var Div at {node.name} (future enhancement)")
        emit(LayerKind.SCALE, "a", (1.0 / c).to(self.dtype), register=True)

def _convert_OnnxExpand(self, mod: nn.Module, node: fx.Node) -> None:
    """OnnxExpand: y = x.expand(shape) — broadcast to the given shape.

    The first arg can be a constant initializer (cctsdb_yolo); fall back
    to ``_ensure_constant_vars`` before reading state.
    """
    args = [a for a in node.args if isinstance(a, fx.Node)]
    if len(args) < 2:
        raise ValueError(f"OnnxExpand at {node.name}: expected 2 args")
    if args[0].name in self.node_outputs:
        self.prev_out = self.node_outputs[args[0].name].copy()
        self.shape = self.node_shapes[args[0].name]
    elif self._ensure_constant_vars(args[0].name):
        self.prev_out = self.node_outputs[args[0].name].copy()
        self.shape = self.node_shapes[args[0].name]
    else:
        raise ValueError(f"OnnxExpand: missing predecessor for {node.name}")
    shape_t = self._resolve_constant_tensor(args[1].name)
    if shape_t is None:
        shape_t = self._evaluate_constant_subgraph(args[1].name)
    if shape_t is None:
        raise ValueError(f"OnnxExpand at {node.name}: cannot resolve target shape")
    target_shape = tuple(int(x) for x in shape_t.flatten().tolist())
    try:
        broadcast_shape = tuple(int(d) for d in torch.broadcast_shapes(self.shape, target_shape))
    except RuntimeError as e:
        raise ValueError(
            f"OnnxExpand at {node.name}: cannot broadcast {self.shape} → {target_shape} ({e})"
        )
    out_vars = self._alloc_ids(_prod(broadcast_shape) or 1)
    layer_id = self._add_layer(
        LayerKind.EXPAND.value,
        {"shape": broadcast_shape,
         "input_shape": self.shape, "output_shape": broadcast_shape},
        self.prev_out, out_vars,
    )
    self.prev_out = out_vars
    self.shape = broadcast_shape
    self._register_node(node.name, layer_id)

def _convert_OnnxFlatten(self, mod: nn.Module, node: fx.Node) -> None:
    """OnnxFlatten: flatten dims [axis:] into one trailing dim, keep [:axis] intact."""
    if not self._get_predecessor_state(node):
        raise ValueError(f"OnnxFlatten: missing predecessor for {node.name}")
    axis = int(getattr(mod, '_axis', getattr(mod, 'axis', 1)))
    if axis < 0:
        axis += len(self.shape)
    a = _prod(self.shape[:axis]) or 1
    b = _prod(self.shape[axis:]) or 1
    output_shape = (a, b)
    out_vars = self._same_size_forward()
    layer_id = self._add_layer(
        LayerKind.FLATTEN.value,
        {"start_dim": axis, "end_dim": -1,
         "input_shape": self.shape, "output_shape": output_shape},
        self.prev_out, out_vars,
    )
    self.prev_out = out_vars
    self.shape = output_shape
    self._register_node(node.name, layer_id)

def _convert_OnnxMinMax(self, mod: nn.Module, node: fx.Node) -> None:
    """OnnxMinMax: element-wise Min / Max of two operands -> LayerKind.MIN / MAX."""
    op_func = getattr(mod, '_operator', None)
    op_name = getattr(op_func, '__name__', '').lower() if op_func is not None else ''
    kind = LayerKind.MIN if 'min' in op_name else LayerKind.MAX
    args = [a for a in node.args if isinstance(a, fx.Node)]
    if len(args) < 2:
        raise ValueError(f"OnnxMinMax at {node.name}: expected 2 args")
    for n in args[:2]:
        if n.name not in self.node_outputs and not self._ensure_constant_vars(n.name):
            raise ValueError(f"OnnxMinMax at {node.name}: '{n.name}' not registered")
    xv, yv = self.node_outputs[args[0].name], self.node_outputs[args[1].name]
    xs, ys = self.node_shapes[args[0].name], self.node_shapes[args[1].name]
    try:
        output_shape = tuple(int(d) for d in torch.broadcast_shapes(xs, ys))
    except RuntimeError:
        output_shape = xs if _prod(xs) >= _prod(ys) else ys
    out_vars = self._alloc_ids(_prod(output_shape) or 1)
    layer_id = self._add_layer(
        kind.value,
        {"x_vars": xv, "y_vars": yv,
         "input_shape": xs, "output_shape": output_shape},
        xv + yv, out_vars,
    )
    self.prev_out = out_vars
    self.shape = output_shape
    self._register_node(node.name, layer_id)

def _convert_OnnxScatterND(self, mod: nn.Module, node: fx.Node) -> None:
    """OnnxScatterND: y = data; y[indices] = updates (writes ``updates`` into ``data``).

    For static-shape conversion the output shape equals the data input's
    shape; we only emit a SCATTER_ND layer recording the three input
    var-streams. Soundness depends on the verifier's TF (deferred).
    """
    args = [a for a in node.args if isinstance(a, fx.Node)]
    if len(args) < 3:
        raise ValueError(f"OnnxScatterND at {node.name}: expected 3 args")
    for n in args[:3]:
        if n.name not in self.node_outputs and not self._ensure_constant_vars(n.name):
            raise ValueError(f"OnnxScatterND at {node.name}: '{n.name}' not registered")
    data_vars = self.node_outputs[args[0].name]
    idx_vars = self.node_outputs[args[1].name]
    upd_vars = self.node_outputs[args[2].name]
    data_shape = self.node_shapes[args[0].name]
    out_vars = self._alloc_ids(_prod(data_shape) or 1)
    layer_id = self._add_layer(
        LayerKind.SCATTER_ND.value,
        {"data_vars": data_vars, "indices_vars": idx_vars, "updates_vars": upd_vars,
         "input_shape": data_shape, "output_shape": data_shape},
        data_vars + idx_vars + upd_vars, out_vars,
    )
    self.prev_out = out_vars
    self.shape = data_shape
    self._register_node(node.name, layer_id)

def _convert_OnnxSqueezeDynamicAxes(self, mod: nn.Module, node: fx.Node) -> None:
    """OnnxSqueezeDynamicAxes: drop size-1 dims at axes given by the second arg."""
    if not self._get_predecessor_state(node):
        raise ValueError(f"OnnxSqueezeDynamicAxes: missing predecessor for {node.name}")
    args = [a for a in node.args if isinstance(a, fx.Node)]
    axes_t = self._resolve_constant_tensor(args[1].name) if len(args) >= 2 else None
    if axes_t is None and len(args) >= 2:
        axes_t = self._evaluate_constant_subgraph(args[1].name)
    rank = len(self.shape)
    if axes_t is not None:
        axes = sorted({(int(a) + rank) if int(a) < 0 else int(a) for a in axes_t.flatten().tolist()})
    else:
        axes = [i for i, d in enumerate(self.shape) if int(d) == 1]
    output_shape = tuple(int(d) for i, d in enumerate(self.shape) if i not in axes) or (1,)
    out_vars = self._same_size_forward()
    layer_id = self._add_layer(
        LayerKind.SQUEEZE.value,
        {"dims": list(axes)},
        self.prev_out, out_vars,
    )
    self.prev_out = out_vars
    self.shape = output_shape
    self._register_node(node.name, layer_id)

def _convert_OnnxWhere(self, mod: nn.Module, node: fx.Node) -> None:
    """OnnxWhere: y = where(cond, x, y_else) — pointwise conditional select."""
    args = [a for a in node.args if isinstance(a, fx.Node)]
    if len(args) < 3:
        raise ValueError(f"OnnxWhere at {node.name}: expected 3 args")
    for n in args[:3]:
        if n.name not in self.node_outputs and not self._ensure_constant_vars(n.name):
            raise ValueError(f"OnnxWhere at {node.name}: '{n.name}' not registered")
    cv = self.node_outputs[args[0].name]
    xv = self.node_outputs[args[1].name]
    yv = self.node_outputs[args[2].name]
    try:
        output_shape = tuple(int(d) for d in torch.broadcast_shapes(
            self.node_shapes[args[0].name],
            self.node_shapes[args[1].name],
            self.node_shapes[args[2].name],
        ))
    except RuntimeError:
        output_shape = self.node_shapes[args[1].name]
    out_vars = self._alloc_ids(_prod(output_shape) or 1)
    layer_id = self._add_layer(
        LayerKind.WHERE.value,
        {"cond_vars": cv, "x_vars": xv, "y_vars": yv,
         "input_shape": self.node_shapes[args[1].name], "output_shape": output_shape},
        cv + xv + yv, out_vars,
    )
    self.prev_out = out_vars
    self.shape = output_shape
    self._register_node(node.name, layer_id)

def _convert_unary_same_shape(self, node: fx.Node, kind: LayerKind, op_name: str) -> None:
    if not self._get_predecessor_state(node):
        raise ValueError(f"{op_name}: missing predecessor for {node.name}")
    out_vars = self._same_size_forward()
    layer_id = self._add_layer(
        kind.value,
        {"input_shape": self.shape, "output_shape": self.shape},
        self.prev_out, out_vars,
    )
    self.prev_out = out_vars
    self._register_node(node.name, layer_id)

def _same_tensor_value(a: torch.Tensor, b: torch.Tensor) -> bool:
    return tuple(a.shape) == tuple(b.shape) and torch.equal(a.detach().cpu(), b.detach().cpu())


def _flat_broadcast_param(param: torch.Tensor, shape: Tuple[int, ...], axis: Optional[int], dtype: torch.dtype) -> torch.Tensor:
    param = param.detach().clone().to(dtype=dtype)
    n = _prod(shape) or 1
    if param.numel() == 1:
        return param.reshape(1).expand(n).contiguous()
    if param.numel() == n:
        return param.reshape(-1).contiguous()
    if axis is not None and param.dim() == 1 and shape:
        ax = axis + len(shape) if axis < 0 else axis
        view_shape = [1] * len(shape)
        view_shape[ax] = int(param.numel())
        return param.reshape(view_shape).expand(shape).contiguous().reshape(-1)
    raise NotImplementedError(f"quant param with shape {tuple(param.shape)} cannot broadcast to {shape}")


def _convert_OnnxQuantizeLinear(self, mod: nn.Module, node: fx.Node) -> None:
    """OnnxQuantizeLinear as real-valued QDQ map with guarded LP envelope."""
    if not self._get_predecessor_state(node):
        raise ValueError(f"OnnxQuantizeLinear: missing predecessor for {node.name}")
    out_vars = self._same_size_forward()
    scale = _flat_broadcast_param(getattr(mod, "scale"), self.shape, getattr(mod, "axis", None), self.dtype)
    zero_point = _flat_broadcast_param(getattr(mod, "zero_point"), self.shape, getattr(mod, "axis", None), self.dtype)
    layer_id = self._add_layer(
        LayerKind.QUANTIZE.value,
        {"scale": scale, "zero_point": zero_point,
         "qmin": int(getattr(mod, "qmin")), "qmax": int(getattr(mod, "qmax")),
         "axis": getattr(mod, "axis", None), "dtype": getattr(mod, "dtype_name", ""),
         "input_shape": self.shape, "output_shape": self.shape},
        self.prev_out, out_vars,
    )
    self.prev_out = out_vars
    self._register_node(node.name, layer_id)


def _convert_OnnxDequantizeLinear(self, mod: nn.Module, node: fx.Node) -> None:
    """OnnxDequantizeLinear: constant-fold, QDQ passthrough, or exact SCALE+BIAS."""
    args = [a for a in node.args if isinstance(a, fx.Node)]
    if not args:
        raise ValueError(f"OnnxDequantizeLinear at {node.name}: no data input")
    q_node = args[0]
    q_const = self._resolve_constant_tensor(q_node.name)
    if q_const is not None:
        with torch.no_grad():
            value = mod(q_const).detach().clone().to(dtype=self.dtype)
        flat = value.reshape(-1)
        out_vars = self._alloc_ids(int(flat.numel()) or 1)
        shape = tuple(int(d) for d in value.shape) or (1,)
        layer_id = self._add_layer(
            LayerKind.CONSTANT.value,
            {"value": flat, "input_shape": shape, "output_shape": shape},
            [], out_vars,
        )
        self.prev_out = out_vars
        self.shape = shape
        self._register_node(node.name, layer_id)
        return

    pred_mod = self.modules.get(str(q_node.target)) if q_node.op == "call_module" else None
    if pred_mod is not None and type(pred_mod).__name__ == "OnnxQuantizeLinear":
        if _same_tensor_value(getattr(pred_mod, "scale"), getattr(mod, "scale")) and _same_tensor_value(getattr(pred_mod, "zero_point"), getattr(mod, "zero_point")):
            if not self._propagate_node_state(node.name, q_node.name):
                raise ValueError(f"OnnxDequantizeLinear: missing Q predecessor state for {node.name}")
            return

    if not self._get_predecessor_state(node):
        raise ValueError(f"OnnxDequantizeLinear: missing predecessor for {node.name}")
    scale = _flat_broadcast_param(getattr(mod, "scale"), self.shape, getattr(mod, "axis", None), self.dtype)
    zp = _flat_broadcast_param(getattr(mod, "zero_point"), self.shape, getattr(mod, "axis", None), self.dtype)
    scaled_vars = self._same_size_forward()
    self._add_layer(
        LayerKind.SCALE.value,
        {"a": scale, "input_shape": self.shape, "output_shape": self.shape},
        self.prev_out, scaled_vars,
    )
    bias_vars = self._alloc_ids(len(scaled_vars))
    layer_id = self._add_layer(
        LayerKind.BIAS.value,
        {"c": -scale * zp, "input_shape": self.shape, "output_shape": self.shape},
        scaled_vars, bias_vars,
    )
    self.prev_out = bias_vars
    self._register_node(node.name, layer_id)

def _convert_OnnxErf(self, mod: nn.Module, node: fx.Node) -> None:
    """OnnxErf: y = erf(x), dedicated onnx2torch module route."""
    _convert_unary_same_shape(self, node, LayerKind.ERF, "OnnxErf")

def _convert_OnnxSqrt(self, mod: nn.Module, node: fx.Node) -> None:
    """OnnxSqrt: y = sqrt(x), dedicated onnx2torch module route."""
    _convert_unary_same_shape(self, node, LayerKind.SQRT, "OnnxSqrt")

def _convert_OnnxSin(self, mod: nn.Module, node: fx.Node) -> None:
    """OnnxSin: y = sin(x), dedicated onnx2torch module route."""
    _convert_unary_same_shape(self, node, LayerKind.SIN, "OnnxSin")

def _convert_OnnxCos(self, mod: nn.Module, node: fx.Node) -> None:
    """OnnxCos: y = cos(x), dedicated onnx2torch module route."""
    _convert_unary_same_shape(self, node, LayerKind.COS, "OnnxCos")

def _convert_OnnxFunction(self, mod: nn.Module, node: fx.Node) -> None:
    """OnnxFunction: dispatch by inner-function name for supported unary ops."""
    func_name = getattr(getattr(mod, 'function', None), '__name__', '').lower()
    kind = {'sign': LayerKind.SIGN, 'abs': LayerKind.ABS,
            'tanh': LayerKind.TANH, 'sigmoid': LayerKind.SIGMOID,
            'erf': LayerKind.ERF, 'sqrt': LayerKind.SQRT,
            'sin': LayerKind.SIN, 'cos': LayerKind.COS}.get(func_name)
    if kind is None:
        raise NotImplementedError(f"OnnxFunction({func_name}) at {node.name} (future enhancement)")
    if not self._get_predecessor_state(node):
        raise ValueError(f"OnnxFunction: missing predecessor for {node.name}")
    out_vars = self._same_size_forward()
    layer_id = self._add_layer(
        kind.value,
        {"input_shape": self.shape, "output_shape": self.shape},
        self.prev_out, out_vars,
    )
    self.prev_out = out_vars
    self._register_node(node.name, layer_id)

def _convert_OnnxCast(self, mod: nn.Module, node: fx.Node) -> None:
    """OnnxCast: dtype-only conversion; ACT tracks values, not dtype, so passthrough."""
    args = [a for a in node.args if isinstance(a, fx.Node)]
    if not args:
        raise ValueError(f"OnnxCast at {node.name}: no inputs")
    self._propagate_node_state(node.name, args[0].name)

def _convert_OnnxArgExtremum(self, mod: nn.Module, node: fx.Node) -> None:
    """OnnxArgExtremum: argmax / argmin along an axis."""
    if not self._get_predecessor_state(node):
        raise ValueError(f"OnnxArgExtremum: missing predecessor for {node.name}")
    op_func = getattr(mod, 'extremum_function', None)
    op_name = getattr(op_func, '__name__', '').lower() if op_func is not None else ''
    op = 'argmax' if 'max' in op_name else 'argmin'
    axis = int(getattr(mod, 'axis', 0))
    keepdims = bool(getattr(mod, 'keepdims', True))
    if axis < 0:
        axis += len(self.shape)
    if keepdims:
        output_shape = tuple(1 if i == axis else int(d) for i, d in enumerate(self.shape))
    else:
        output_shape = tuple(int(d) for i, d in enumerate(self.shape) if i != axis) or (1,)
    out_vars = self._alloc_ids(_prod(output_shape) or 1)
    layer_id = self._add_layer(
        LayerKind.ARG_EXTREMUM.value,
        {"op": op, "axis": axis, "keepdims": int(keepdims),
         "input_shape": self.shape, "output_shape": output_shape},
        self.prev_out, out_vars,
    )
    self.prev_out = out_vars
    self.shape = output_shape
    self._register_node(node.name, layer_id)

def _convert_OnnxCompare(self, mod: nn.Module, node: fx.Node) -> None:
    """OnnxCompare: element-wise comparison (eq/ne/lt/le/gt/ge) producing bool vars."""
    op_func = getattr(mod, 'compare_function', None)
    op_raw = getattr(op_func, '__name__', '').lower() if op_func is not None else ''
    op_map = {'equal': 'eq', 'eq': 'eq', 'less': 'lt', 'lt': 'lt',
              'greater': 'gt', 'gt': 'gt', 'less_equal': 'le', 'le': 'le',
              'greater_equal': 'ge', 'ge': 'ge', 'not_equal': 'ne', 'ne': 'ne'}
    op = op_map.get(op_raw)
    if op is None:
        raise NotImplementedError(f"OnnxCompare at {node.name}: unrecognised op '{op_raw}'")
    args = [a for a in node.args if isinstance(a, fx.Node)]
    x_node, y_node = args[0], args[1]
    for n in (x_node, y_node):
        if n.name not in self.node_outputs and not self._ensure_constant_vars(n.name):
            raise ValueError(f"OnnxCompare at {node.name}: '{n.name}' not registered")
    xv, yv = self.node_outputs[x_node.name], self.node_outputs[y_node.name]
    xs, ys = self.node_shapes[x_node.name], self.node_shapes[y_node.name]
    try:
        output_shape = tuple(int(d) for d in torch.broadcast_shapes(xs, ys))
    except RuntimeError:
        output_shape = xs if _prod(xs) >= _prod(ys) else ys
    out_vars = self._alloc_ids(_prod(output_shape) or 1)
    layer_id = self._add_layer(
        LayerKind.COMPARE.value,
        {"op": op, "x_vars": xv, "y_vars": yv,
         "input_shape": xs, "output_shape": output_shape},
        xv + yv, out_vars,
    )
    self.prev_out = out_vars
    self.shape = output_shape
    self._register_node(node.name, layer_id)

def _convert_OnnxDropoutDynamic(self, mod: nn.Module, node: fx.Node) -> None:
    """OnnxDropoutDynamic in eval mode: identity passthrough (no layer emitted)."""
    args = [a for a in node.args if isinstance(a, fx.Node)]
    if not args:
        raise ValueError(f"OnnxDropoutDynamic at {node.name}: no inputs")
    self._propagate_node_state(node.name, args[0].name)

def _convert_OnnxUnsqueezeStaticAxes(self, mod: nn.Module, node: fx.Node) -> None:
    """OnnxUnsqueezeStaticAxes: insert size-1 dims at static ``_axes``."""
    if not self._get_predecessor_state(node):
        raise ValueError(f"OnnxUnsqueezeStaticAxes: missing predecessor for {node.name}")
    axes = list(getattr(mod, '_axes', None) or [])
    if not axes:
        raise ValueError(f"OnnxUnsqueezeStaticAxes at {node.name}: missing _axes")
    rank_after = len(self.shape) + len(axes)
    norm_axes = sorted({a + rank_after if a < 0 else a for a in (int(x) for x in axes)})
    output_shape = list(self.shape)
    for ax in norm_axes:
        output_shape.insert(ax, 1)
    output_shape = tuple(output_shape)
    out_vars = self._same_size_forward()
    layer_id = self._add_layer(
        LayerKind.UNSQUEEZE.value,
        {"dims": list(norm_axes)},
        self.prev_out, out_vars,
    )
    self.prev_out = out_vars
    self.shape = output_shape
    self._register_node(node.name, layer_id)

ONNX_HANDLERS = {
    'OnnxArgExtremum': _convert_OnnxArgExtremum,
    'OnnxBinaryMathOperation': _convert_OnnxBinaryMathOperation,
    'OnnxCast': _convert_OnnxCast,
    'OnnxCompare': _convert_OnnxCompare,
    'OnnxConcat': _convert_OnnxConcat,
    'OnnxConstant': _convert_OnnxConstant,
    'OnnxDropoutDynamic': _convert_OnnxDropoutDynamic,
    'OnnxExpand': _convert_OnnxExpand,
    'OnnxFlatten': _convert_OnnxFlatten,
    'OnnxQuantizeLinear': _convert_OnnxQuantizeLinear,
    'OnnxDequantizeLinear': _convert_OnnxDequantizeLinear,
    'OnnxErf': _convert_OnnxErf,
    'OnnxSqrt': _convert_OnnxSqrt,
    'OnnxSin': _convert_OnnxSin,
    'OnnxCos': _convert_OnnxCos,
    'OnnxFunction': _convert_OnnxFunction,
    'OnnxGather': _convert_OnnxGather,
    'OnnxMatMul': _convert_OnnxMatMul,
    'OnnxMinMax': _convert_OnnxMinMax,
    'OnnxNeg': _convert_OnnxNeg,
    'OnnxPow': _convert_OnnxPow,
    'OnnxReduceStaticAxes': _convert_OnnxReduceStaticAxes,
    'OnnxReduceSumStaticAxes': _convert_OnnxReduceSumStaticAxes,
    'OnnxReshape': _convert_OnnxReshape,
    'OnnxResize': _convert_OnnxResize,
    'OnnxScatterND': _convert_OnnxScatterND,
    'OnnxShape': _convert_OnnxShape,
    'OnnxSlice': _convert_OnnxSlice,
    'OnnxSplit13': _convert_OnnxSplit13,
    'OnnxSqueezeDynamicAxes': _convert_OnnxSqueezeDynamicAxes,
    'OnnxTranspose': _convert_OnnxTranspose,
    'OnnxUnsqueezeStaticAxes': _convert_OnnxUnsqueezeStaticAxes,
    'OnnxWhere': _convert_OnnxWhere,
}
