"""
Performance Optimizer Component
Responsible for monitoring and optimizing system performance,
ensuring fast response times and efficient resource utilization.
"""

import asyncio
import logging
import time
import gc
import os
import psutil
import functools
from typing import Dict, List, Any, Optional, Callable, TypeVar, Set
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

# Type variable for function result
T = TypeVar('T')

class TaskStats:
    """Tracks statistics for task execution"""
    
    def __init__(self, name: str):
        """
        Initialize task statistics tracker
        
        Args:
            name (str): Task name
        """
        self.name = name
        self.execution_count = 0
        self.total_execution_time = 0.0
        self.min_execution_time = float('inf')
        self.max_execution_time = 0.0
        self.last_execution_time = 0.0
        self.concurrent_executions = 0
        self.max_concurrent_executions = 0
    
    def record_execution(self, execution_time: float):
        """
        Record task execution
        
        Args:
            execution_time (float): Execution time in seconds
        """
        self.execution_count += 1
        self.total_execution_time += execution_time
        self.min_execution_time = min(self.min_execution_time, execution_time)
        self.max_execution_time = max(self.max_execution_time, execution_time)
        self.last_execution_time = execution_time
    
    def start_execution(self):
        """Record start of execution"""
        self.concurrent_executions += 1
        self.max_concurrent_executions = max(
            self.max_concurrent_executions, self.concurrent_executions)
    
    def end_execution(self):
        """Record end of execution"""
        self.concurrent_executions -= 1
    
    def get_average_time(self) -> float:
        """
        Get average execution time
        
        Returns:
            float: Average execution time in seconds
        """
        if self.execution_count == 0:
            return 0.0
        return self.total_execution_time / self.execution_count
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary
        
        Returns:
            Dict[str, Any]: Task statistics
        """
        return {
            'name': self.name,
            'execution_count': self.execution_count,
            'avg_execution_time': self.get_average_time(),
            'min_execution_time': self.min_execution_time if self.min_execution_time != float('inf') else 0,
            'max_execution_time': self.max_execution_time,
            'last_execution_time': self.last_execution_time,
            'concurrent_executions': self.concurrent_executions,
            'max_concurrent_executions': self.max_concurrent_executions
        }


class AsyncBatch:
    """Utility for batching async operations"""
    
    def __init__(self, 
                max_batch_size: int = 10, 
                max_wait_time: float = 0.1):
        """
        Initialize async batch
        
        Args:
            max_batch_size (int): Maximum batch size
            max_wait_time (float): Maximum wait time in seconds
        """
        self.max_batch_size = max_batch_size
        self.max_wait_time = max_wait_time
        self.items = []
        self.batch_event = asyncio.Event()
        self.lock = asyncio.Lock()
        self.processing = False
    
    async def add_item(self, item: Any) -> int:
        """
        Add item to batch
        
        Args:
            item (Any): Item to add
            
        Returns:
            int: Batch ID
        """
        async with self.lock:
            batch_id = len(self.items)
            self.items.append(item)
            
            # Signal if batch is full
            if len(self.items) >= self.max_batch_size:
                self.batch_event.set()
            
            # Start timer for first item
            if len(self.items) == 1 and not self.processing:
                asyncio.create_task(self._wait_for_batch())
            
            return batch_id
    
    async def _wait_for_batch(self):
        """Wait for batch to be ready for processing"""
        self.processing = True
        
        # Wait for batch to fill or timeout
        try:
            await asyncio.wait_for(self.batch_event.wait(), self.max_wait_time)
        except asyncio.TimeoutError:
            pass
        
        # Process the batch
        async with self.lock:
            batch_items = self.items.copy()
            self.items = []
            self.batch_event.clear()
            self.processing = False
        
        # Process the batch (override in subclass)
        await self.process_batch(batch_items)
    
    async def process_batch(self, items: List[Any]):
        """
        Process a batch of items
        
        Args:
            items (List[Any]): Items to process
        """
        # Override in subclass
        pass


class LRUCache:
    """LRU (Least Recently Used) Cache implementation"""
    
    def __init__(self, capacity: int):
        """
        Initialize LRU cache
        
        Args:
            capacity (int): Cache capacity
        """
        self.capacity = capacity
        self.cache = {}
        self.usage_order = []
    
    def get(self, key: Any) -> Optional[Any]:
        """
        Get item from cache
        
        Args:
            key (Any): Cache key
            
        Returns:
            Optional[Any]: Cached item or None if not found
        """
        if key not in self.cache:
            return None
        
        # Update usage order
        self.usage_order.remove(key)
        self.usage_order.append(key)
        
        return self.cache[key]
    
    def put(self, key: Any, value: Any):
        """
        Put item in cache
        
        Args:
            key (Any): Cache key
            value (Any): Value to cache
        """
        if key in self.cache:
            # Update existing key
            self.usage_order.remove(key)
        elif len(self.cache) >= self.capacity:
            # Remove least recently used
            oldest_key = self.usage_order.pop(0)
            del self.cache[oldest_key]
        
        # Add new key
        self.cache[key] = value
        self.usage_order.append(key)
    
    def remove(self, key: Any):
        """
        Remove item from cache
        
        Args:
            key (Any): Cache key
        """
        if key in self.cache:
            del self.cache[key]
            self.usage_order.remove(key)
    
    def clear(self):
        """Clear cache"""
        self.cache = {}
        self.usage_order = []
    
    def __len__(self) -> int:
        """Get cache size"""
        return len(self.cache)


class PerformanceOptimizer:
    """
    Optimizes system performance for high-throughput trading
    
    This component implements various performance optimization techniques:
    - Task prioritization
    - Parallel execution
    - Response time optimization
    - Resource utilization monitoring
    - Adaptive concurrency control
    """
    
    def __init__(self, thread_pool_size: Optional[int] = None):
        """
        Initialize the PerformanceOptimizer
        
        Args:
            thread_pool_size (int, optional): Number of threads in thread pool
        """
        # Initialize thread pool
        if thread_pool_size is None:
            # Use CPU count to determine thread pool size
            thread_pool_size = max(4, os.cpu_count() or 4)
        
        self.thread_pool = ThreadPoolExecutor(max_workers=thread_pool_size)
        
        # Task priority queues
        self.priority_queues = {
            'high': asyncio.PriorityQueue(),
            'normal': asyncio.PriorityQueue(),
            'low': asyncio.PriorityQueue()
        }
        
        # Task statistics
        self.task_stats = {}
        
        # Concurrency control
        self.concurrent_tasks = 0
        self.max_concurrent_tasks = thread_pool_size * 2
        self.concurrency_semaphore = asyncio.Semaphore(self.max_concurrent_tasks)
        
        # Resource monitoring
        self.cpu_usage_history = []
        self.memory_usage_history = []
        self.max_history_length = 100
        
        # Caches
        self.function_cache = LRUCache(capacity=1000)
        self.result_cache = {}
        
        # Performance metrics
        self.task_processing_times = {}
        self.bottleneck_functions = set()
        
        # Task workers
        self.workers = {}
        self.stop_event = asyncio.Event()
        
        logger.info(f"PerformanceOptimizer initialized with {thread_pool_size} threads")
    
    async def start(self):
        """Start the performance optimizer"""
        # Start worker tasks
        for priority in self.priority_queues.keys():
            self.workers[priority] = asyncio.create_task(
                self._worker(priority))
        
        # Start resource monitoring
        asyncio.create_task(self._monitor_resources())
        
        logger.info("PerformanceOptimizer started")
    
    async def stop(self):
        """Stop the performance optimizer"""
        # Signal workers to stop
        self.stop_event.set()
        
        # Wait for workers to complete
        for worker in self.workers.values():
            try:
                await asyncio.wait_for(worker, timeout=5.0)
            except asyncio.TimeoutError:
                pass
        
        # Shutdown thread pool
        self.thread_pool.shutdown(wait=False)
        
        logger.info("PerformanceOptimizer stopped")
    
    async def _worker(self, priority: str):
        """
        Worker task for processing queue
        
        Args:
            priority (str): Queue priority
        """
        queue = self.priority_queues[priority]
        
        while not self.stop_event.is_set():
            try:
                # Get task from queue with timeout
                try:
                    # Use wait_for with a timeout to allow checking stop_event
                    _, (task_id, task_func, args, kwargs, future) = await asyncio.wait_for(
                        queue.get(), timeout=0.5)
                except asyncio.TimeoutError:
                    continue
                
                # Execute task
                await self._execute_task(task_id, task_func, args, kwargs, future)
                
                # Mark task as done
                queue.task_done()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in {priority} priority worker: {str(e)}")
    
    async def _execute_task(self, 
                          task_id: str, 
                          task_func: Callable, 
                          args: tuple, 
                          kwargs: dict,
                          future: asyncio.Future):
        """
        Execute a task
        
        Args:
            task_id (str): Task ID
            task_func (Callable): Task function
            args (tuple): Function arguments
            kwargs (dict): Function keyword arguments
            future (asyncio.Future): Future for result
        """
        try:
            # Acquire semaphore to limit concurrency
            async with self.concurrency_semaphore:
                self.concurrent_tasks += 1
                
                # Record task start
                if task_id in self.task_stats:
                    self.task_stats[task_id].start_execution()
                
                start_time = time.time()
                
                try:
                    # Check if function is async
                    if asyncio.iscoroutinefunction(task_func):
                        # Execute async function
                        result = await task_func(*args, **kwargs)
                    else:
                        # Execute in thread pool
                        loop = asyncio.get_running_loop()
                        result = await loop.run_in_executor(
                            self.thread_pool,
                            lambda: task_func(*args, **kwargs)
                        )
                    
                    # Set result
                    if not future.done():
                        future.set_result(result)
                    
                except Exception as e:
                    # Set exception
                    if not future.done():
                        future.set_exception(e)
                    
                finally:
                    # Record execution time
                    execution_time = time.time() - start_time
                    
                    # Update task stats
                    if task_id in self.task_stats:
                        stats = self.task_stats[task_id]
                        stats.record_execution(execution_time)
                        stats.end_execution()
                    
                    # Update performance metrics
                    self.task_processing_times[task_id] = execution_time
                    
                    # Identify bottlenecks
                    if execution_time > 0.1:  # Threshold for bottleneck identification
                        self.bottleneck_functions.add(task_id)
                    
                    self.concurrent_tasks -= 1
                    
        except asyncio.CancelledError:
            # Task was cancelled
            if not future.done():
                future.cancel()
            raise
            
        except Exception as e:
            logger.error(f"Error executing task {task_id}: {str(e)}")
            # Set exception if not already done
            if not future.done():
                future.set_exception(e)
    
    async def enqueue_task(self, 
                         task_id: str,
                         task_func: Callable, 
                         args: tuple = (), 
                         kwargs: dict = None,
                         priority: str = 'normal') -> asyncio.Future:
        """
        Enqueue a task for execution
        
        Args:
            task_id (str): Task ID
            task_func (Callable): Task function
            args (tuple): Function arguments
            kwargs (dict): Function keyword arguments
            priority (str): Task priority ('high', 'normal', 'low')
            
        Returns:
            asyncio.Future: Future for task result
        """
        if kwargs is None:
            kwargs = {}
        
        # Create future for result
        future = asyncio.Future()
        
        # Create task stats if not exists
        if task_id not in self.task_stats:
            self.task_stats[task_id] = TaskStats(task_id)
        
        # Determine priority queue
        if priority not in self.priority_queues:
            logger.warning(f"Unknown priority: {priority}, using 'normal'")
            priority = 'normal'
        
        queue = self.priority_queues[priority]
        
        # Determine priority value (lower = higher priority)
        priority_value = {
            'high': 0,
            'normal': 1,
            'low': 2
        }[priority]
        
        # Add to queue
        await queue.put((priority_value, (task_id, task_func, args, kwargs, future)))
        
        return future
    
    async def _monitor_resources(self):
        """Monitor system resources"""
        while not self.stop_event.is_set():
            try:
                # Get CPU and memory usage
                cpu_percent = psutil.cpu_percent(interval=None)
                memory_info = psutil.Process(os.getpid()).memory_info()
                memory_percent = memory_info.rss / (psutil.virtual_memory().total * 0.01)
                
                # Add to history
                self.cpu_usage_history.append((time.time(), cpu_percent))
                self.memory_usage_history.append((time.time(), memory_percent))
                
                # Trim history if needed
                if len(self.cpu_usage_history) > self.max_history_length:
                    self.cpu_usage_history = self.cpu_usage_history[-self.max_history_length:]
                if len(self.memory_usage_history) > self.max_history_length:
                    self.memory_usage_history = self.memory_usage_history[-self.max_history_length:]
                
                # Adjust concurrency based on CPU usage
                await self._adjust_concurrency(cpu_percent)
                
                # Wait before next check
                await asyncio.sleep(5.0)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error monitoring resources: {str(e)}")
                await asyncio.sleep(5.0)
    
    async def _adjust_concurrency(self, cpu_percent: float):
        """
        Adjust concurrency based on CPU usage
        
        Args:
            cpu_percent (float): CPU usage percentage
        """
        # Adjust max concurrent tasks based on CPU usage
        if cpu_percent > 90:
            # High CPU usage, reduce concurrency
            new_limit = max(2, self.max_concurrent_tasks - 2)
        elif cpu_percent > 70:
            # Moderately high CPU usage, reduce slightly
            new_limit = max(2, self.max_concurrent_tasks - 1)
        elif cpu_percent < 30:
            # Low CPU usage, increase concurrency
            new_limit = min(os.cpu_count() * 4, self.max_concurrent_tasks + 1)
        else:
            # CPU usage is acceptable, maintain current limit
            return
        
        # Update concurrency limit if changed
        if new_limit != self.max_concurrent_tasks:
            logger.info(f"Adjusting concurrency limit from {self.max_concurrent_tasks} to {new_limit} (CPU: {cpu_percent}%)")
            self.max_concurrent_tasks = new_limit
            
            # Create new semaphore
            old_semaphore = self.concurrency_semaphore
            self.concurrency_semaphore = asyncio.Semaphore(new_limit)
            
            # Release waiting tasks if increasing limit
            if new_limit > old_semaphore._value:
                for _ in range(new_limit - old_semaphore._value):
                    old_semaphore.release()
    
    def get_task_stats(self) -> Dict[str, Dict[str, Any]]:
        """
        Get task statistics
        
        Returns:
            Dict[str, Dict[str, Any]]: Task statistics by task ID
        """
        return {task_id: stats.to_dict() for task_id, stats in self.task_stats.items()}
    
    def get_resource_usage(self) -> Dict[str, Any]:
        """
        Get resource usage
        
        Returns:
            Dict[str, Any]: Resource usage statistics
        """
        # Calculate average CPU and memory usage
        if self.cpu_usage_history:
            avg_cpu = sum(cpu for _, cpu in self.cpu_usage_history) / len(self.cpu_usage_history)
            cpu_trend = self.cpu_usage_history[-1][1] - self.cpu_usage_history[0][1] if len(self.cpu_usage_history) > 1 else 0
        else:
            avg_cpu = 0
            cpu_trend = 0
        
        if self.memory_usage_history:
            avg_memory = sum(mem for _, mem in self.memory_usage_history) / len(self.memory_usage_history)
            memory_trend = self.memory_usage_history[-1][1] - self.memory_usage_history[0][1] if len(self.memory_usage_history) > 1 else 0
        else:
            avg_memory = 0
            memory_trend = 0
        
        return {
            'cpu': {
                'current': self.cpu_usage_history[-1][1] if self.cpu_usage_history else 0,
                'average': avg_cpu,
                'trend': cpu_trend
            },
            'memory': {
                'current': self.memory_usage_history[-1][1] if self.memory_usage_history else 0,
                'average': avg_memory,
                'trend': memory_trend
            },
            'concurrency': {
                'current': self.concurrent_tasks,
                'max': self.max_concurrent_tasks
            }
        }
    
    def get_bottlenecks(self) -> Dict[str, Any]:
        """
        Get system bottlenecks
        
        Returns:
            Dict[str, Any]: Bottleneck information
        """
        # Get top 5 slowest tasks
        slowest_tasks = sorted(
            self.task_processing_times.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        return {
            'slowest_tasks': dict(slowest_tasks),
            'bottleneck_functions': list(self.bottleneck_functions)
        }
    
    def async_batch(self, 
                  max_batch_size: int = 10, 
                  max_wait_time: float = 0.1) -> AsyncBatch:
        """
        Create an async batch processor
        
        Args:
            max_batch_size (int): Maximum batch size
            max_wait_time (float): Maximum wait time in seconds
            
        Returns:
            AsyncBatch: Async batch processor
        """
        return AsyncBatch(max_batch_size, max_wait_time)
    
    def run_in_thread(self, func: Callable) -> Callable:
        """
        Decorator to run a function in a thread pool
        
        Args:
            func (Callable): Function to decorate
            
        Returns:
            Callable: Decorated function
        """
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                self.thread_pool,
                lambda: func(*args, **kwargs)
            )
        return wrapper
    
    def cache(self, ttl: Optional[float] = None) -> Callable:
        """
        Decorator to cache function results
        
        Args:
            ttl (float, optional): Time-to-live in seconds
            
        Returns:
            Callable: Decorator function
        """
        def decorator(func: Callable) -> Callable:
            cache_key = f"{func.__module__}.{func.__qualname__}"
            
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                # Create key from function name and arguments
                key = (cache_key, args, frozenset(kwargs.items()))
                
                # Check cache
                cache_entry = self.function_cache.get(key)
                if cache_entry is not None:
                    timestamp, result = cache_entry
                    if ttl is None or time.time() - timestamp < ttl:
                        return result
                
                # Call function
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    loop = asyncio.get_running_loop()
                    result = await loop.run_in_executor(
                        self.thread_pool,
                        lambda: func(*args, **kwargs)
                    )
                
                # Store in cache
                self.function_cache.put(key, (time.time(), result))
                return result
            
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                # Create key from function name and arguments
                key = (cache_key, args, frozenset(kwargs.items()))
                
                # Check cache
                cache_entry = self.function_cache.get(key)
                if cache_entry is not None:
                    timestamp, result = cache_entry
                    if ttl is None or time.time() - timestamp < ttl:
                        return result
                
                # Call function
                result = func(*args, **kwargs)
                
                # Store in cache
                self.function_cache.put(key, (time.time(), result))
                return result
            
            return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
        
        return decorator
    
    def memoize(self, func: Callable) -> Callable:
        """
        Decorator to memoize function results (permanent cache)
        
        Args:
            func (Callable): Function to decorate
            
        Returns:
            Callable: Decorated function
        """
        cache_key = f"{func.__module__}.{func.__qualname__}"
        
        if cache_key not in self.result_cache:
            self.result_cache[cache_key] = {}
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Create key from arguments
            key = (args, frozenset(kwargs.items()))
            
            # Check cache
            local_cache = self.result_cache[cache_key]
            if key in local_cache:
                return local_cache[key]
            
            # Call function
            result = func(*args, **kwargs)
            
            # Store in cache
            local_cache[key] = result
            return result
        
        return wrapper
    
    def prioritize(self, priority: str = 'normal') -> Callable:
        """
        Decorator to execute a function with specified priority
        
        Args:
            priority (str): Task priority ('high', 'normal', 'low')
            
        Returns:
            Callable: Decorator function
        """
        def decorator(func: Callable) -> Callable:
            func_name = f"{func.__module__}.{func.__qualname__}"
            
            @functools.wraps(func)
            async def wrapper(*args, **kwargs):
                # Enqueue task
                future = await self.enqueue_task(
                    task_id=func_name,
                    task_func=func,
                    args=args,
                    kwargs=kwargs,
                    priority=priority
                )
                
                # Wait for result
                return await future
            
            return wrapper
        
        return decorator