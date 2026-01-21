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
    """Normalize int or tuple to 2-tuple for kernel_size/stride/padding."""
    if isinstance(val, tuple):
        return val
    return (val, val) if val is not None else default


@dataclass
class PerformanceMetrics:
    """Performance metrics for validation operations."""
    execution_time: float
    peak_memory_mb: float
    cpu_usage_percent: float
    gpu_memory_mb: Optional[float] = None


@dataclass
class ParallelResult:
    """Result from parallel execution."""
    results: List[Any]
    failed_tasks: List[Tuple[int, Exception]]
    total_time: float
    metrics: PerformanceMetrics


class PerformanceProfiler:
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
def profile_performance():
    """Context manager for performance profiling."""
    profiler = PerformanceProfiler()
    profiler.start()
    try:
        yield profiler
    finally:
        metrics = profiler.stop()
        yield metrics


class ParallelExecutor:
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


def print_memory_usage(prefix: str = "") -> None:
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


def clear_torch_cache() -> None:
    """Clear PyTorch GPU cache if available."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.debug("Cleared PyTorch GPU cache")


def setup_logging(level: str = "INFO", format_str: Optional[str] = None) -> None:
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


def retry_on_failure(max_retries: int = 3, delay: float = 1.0, backoff: float = 2.0):
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


def timeout_handler(timeout_seconds: float):
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


class ProgressTracker:
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
