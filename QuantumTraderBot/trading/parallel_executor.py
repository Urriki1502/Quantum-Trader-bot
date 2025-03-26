"""
Parallel Executor Component
Responsible for executing trades in parallel while maintaining
order fairness and ensuring optimal transaction execution.
"""

import asyncio
import logging
import time
import random
from typing import Dict, List, Any, Optional, Callable, Set, Tuple
from enum import Enum

from core.state_manager import StateManager
from core.config_manager import ConfigManager
from trading.flash_executor import FlashExecutor
from utils.performance_optimizer import PerformanceOptimizer

logger = logging.getLogger(__name__)

class ExecutionPriority(Enum):
    """Trade execution priority levels"""
    CRITICAL = 0   # Highest priority, execute immediately (e.g., emergency sell)
    HIGH = 1       # High priority, execute soon (e.g., opportunity with tight time window)
    NORMAL = 2     # Normal priority, standard execution
    LOW = 3        # Low priority, can wait (e.g., position building over time)
    BACKGROUND = 4 # Lowest priority, execute when resources available (e.g., rebalancing)


class TradeTask:
    """Represents a trade task in the execution queue"""
    
    def __init__(self, 
                task_id: str,
                trade_params: Dict[str, Any],
                priority: ExecutionPriority,
                deadline: Optional[float] = None,
                callback: Optional[Callable] = None):
        """
        Initialize trade task
        
        Args:
            task_id (str): Task ID
            trade_params (Dict[str, Any]): Trade parameters
            priority (ExecutionPriority): Execution priority
            deadline (float, optional): Execution deadline (timestamp)
            callback (Callable, optional): Callback function for result
        """
        self.task_id = task_id
        self.trade_params = trade_params
        self.priority = priority
        self.deadline = deadline
        self.callback = callback
        self.future = asyncio.Future()
        
        # Task metadata
        self.creation_time = time.time()
        self.start_time = None
        self.end_time = None
        self.status = "queued"
        self.result = None
        self.error = None
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary
        
        Returns:
            Dict[str, Any]: Task representation
        """
        return {
            'task_id': self.task_id,
            'token_address': self.trade_params.get('token_address'),
            'action': self.trade_params.get('action', 'unknown'),
            'priority': self.priority.name,
            'deadline': self.deadline,
            'creation_time': self.creation_time,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'status': self.status,
            'wait_time': (self.start_time - self.creation_time) if self.start_time else None,
            'execution_time': (self.end_time - self.start_time) if self.end_time and self.start_time else None
        }
    
    def get_sort_key(self) -> Tuple[int, float]:
        """
        Get sort key for priority queue
        
        Returns:
            Tuple[int, float]: Priority value and creation time
        """
        return (self.priority.value, self.creation_time)
    
    def is_expired(self) -> bool:
        """
        Check if task is expired
        
        Returns:
            bool: True if task is expired
        """
        if self.deadline is None:
            return False
        return time.time() > self.deadline


class ParallelExecutor:
    """
    Executes trades in parallel with priority-based scheduling
    
    This component optimizes trade execution by:
    - Prioritizing critical trades
    - Batching similar trades when possible
    - Ensuring fair ordering based on submission time
    - Adapting concurrency based on system load
    """
    
    def __init__(self, 
                config_manager: ConfigManager,
                state_manager: StateManager,
                flash_executor: FlashExecutor,
                performance_optimizer: PerformanceOptimizer):
        """
        Initialize parallel executor
        
        Args:
            config_manager (ConfigManager): Configuration manager instance
            state_manager (StateManager): State manager instance
            flash_executor (FlashExecutor): Flash executor instance
            performance_optimizer (PerformanceOptimizer): Performance optimizer instance
        """
        self.config_manager = config_manager
        self.state_manager = state_manager
        self.flash_executor = flash_executor
        self.performance_optimizer = performance_optimizer
        
        # Execution settings
        self.max_concurrent_executions = self.config_manager.get(
            'parallel_executor.max_concurrent_executions', 8)
        self.execution_timeout = self.config_manager.get(
            'parallel_executor.execution_timeout', 60.0)
        self.queue_capacity = self.config_manager.get(
            'parallel_executor.queue_capacity', 100)
        
        # Execution concurrency control
        self.execution_semaphore = asyncio.Semaphore(self.max_concurrent_executions)
        
        # Trade queues by priority
        self.task_queues = {
            ExecutionPriority.CRITICAL: asyncio.PriorityQueue(),
            ExecutionPriority.HIGH: asyncio.PriorityQueue(),
            ExecutionPriority.NORMAL: asyncio.PriorityQueue(),
            ExecutionPriority.LOW: asyncio.PriorityQueue(),
            ExecutionPriority.BACKGROUND: asyncio.PriorityQueue()
        }
        
        # Active and completed tasks
        self.tasks = {}
        self.active_task_ids = set()
        self.completed_task_ids = set()
        self.max_completed_tasks = 100
        
        # Task processing
        self.is_running = False
        self.worker_tasks = []
        self.worker_count = 5  # One worker per priority level
        
        # Performance metrics
        self.queue_stats = {
            'total_tasks': 0,
            'completed_tasks': 0,
            'failed_tasks': 0,
            'avg_wait_time': 0.0,
            'avg_execution_time': 0.0,
            'peak_queue_size': 0
        }
        
        # Batching support
        self.enable_batching = self.config_manager.get(
            'parallel_executor.enable_batching', True)
        self.batch_size_limit = self.config_manager.get(
            'parallel_executor.batch_size_limit', 5)
        self.batch_wait_time = self.config_manager.get(
            'parallel_executor.batch_wait_time', 0.5)
        
        logger.info("ParallelExecutor initialized")
    
    async def start(self):
        """Start the parallel executor"""
        if self.is_running:
            logger.warning("ParallelExecutor is already running")
            return
        
        self.is_running = True
        
        # Start worker tasks
        for _ in range(self.worker_count):
            worker = asyncio.create_task(self._worker())
            self.worker_tasks.append(worker)
        
        logger.info(f"ParallelExecutor started with {self.worker_count} workers")
        
        # Update component status
        self.state_manager.update_component_status(
            'parallel_executor', 
            'running',
            f"Max concurrent: {self.max_concurrent_executions}, Batching: {self.enable_batching}"
        )
    
    async def stop(self):
        """Stop the parallel executor"""
        if not self.is_running:
            return
            
        self.is_running = False
        
        # Cancel all worker tasks
        for worker in self.worker_tasks:
            worker.cancel()
        
        # Wait for workers to complete
        if self.worker_tasks:
            await asyncio.gather(*self.worker_tasks, return_exceptions=True)
        
        self.worker_tasks = []
        
        logger.info("ParallelExecutor stopped")
        
        # Update component status
        self.state_manager.update_component_status(
            'parallel_executor', 
            'stopped'
        )
    
    async def submit_trade(self, 
                         trade_params: Dict[str, Any],
                         priority: ExecutionPriority = ExecutionPriority.NORMAL,
                         deadline: Optional[float] = None,
                         callback: Optional[Callable] = None) -> str:
        """
        Submit a trade for execution
        
        Args:
            trade_params (Dict[str, Any]): Trade parameters
            priority (ExecutionPriority): Execution priority
            deadline (float, optional): Execution deadline (timestamp)
            callback (Callable, optional): Callback function for result
            
        Returns:
            str: Task ID
        """
        # Check queue capacity
        total_queued = sum(q.qsize() for q in self.task_queues.values())
        if total_queued >= self.queue_capacity:
            raise Exception(f"Execution queue at capacity ({total_queued}/{self.queue_capacity})")
        
        # Generate task ID
        task_id = f"trade_{int(time.time() * 1000)}_{random.randint(1000, 9999)}"
        
        # Create trade task
        task = TradeTask(
            task_id=task_id,
            trade_params=trade_params,
            priority=priority,
            deadline=deadline,
            callback=callback
        )
        
        # Store task
        self.tasks[task_id] = task
        
        # Add to appropriate queue
        priority_queue = self.task_queues[priority]
        await priority_queue.put((task.get_sort_key(), task))
        
        # Update queue statistics
        self.queue_stats['total_tasks'] += 1
        current_size = sum(q.qsize() for q in self.task_queues.values())
        self.queue_stats['peak_queue_size'] = max(
            self.queue_stats['peak_queue_size'], current_size)
        
        # Update component metrics
        self._update_metrics()
        
        logger.info(f"Trade task {task_id} submitted with priority {priority.name}")
        
        return task_id
    
    async def wait_for_completion(self, task_id: str, timeout: Optional[float] = None) -> Dict[str, Any]:
        """
        Wait for task completion
        
        Args:
            task_id (str): Task ID
            timeout (float, optional): Wait timeout in seconds
            
        Returns:
            Dict[str, Any]: Task result
        """
        task = self.tasks.get(task_id)
        if task is None:
            raise ValueError(f"Unknown task ID: {task_id}")
        
        try:
            # Wait for task future
            if timeout is not None:
                await asyncio.wait_for(task.future, timeout)
            else:
                await task.future
            
            # Return result
            return task.result
            
        except asyncio.TimeoutError:
            raise TimeoutError(f"Task {task_id} did not complete within timeout")
    
    async def _worker(self):
        """Worker task for executing trades"""
        while self.is_running:
            try:
                # Check for CRITICAL tasks first
                task = await self._get_next_task()
                
                if task is None:
                    # No tasks in any queue, wait briefly
                    await asyncio.sleep(0.1)
                    continue
                
                # Check if task can be batched
                if self.enable_batching:
                    batch = await self._try_create_batch(task)
                    if len(batch) > 1:
                        # Execute batch
                        await self._execute_batch(batch)
                        continue
                
                # Execute single task
                await self._execute_task(task)
                
            except asyncio.CancelledError:
                # Worker is being cancelled
                break
            except Exception as e:
                logger.error(f"Error in executor worker: {str(e)}")
                await asyncio.sleep(1.0)
    
    async def _get_next_task(self) -> Optional[TradeTask]:
        """
        Get next task from queues, respecting priority
        
        Returns:
            Optional[TradeTask]: Next task or None if all queues empty
        """
        # Check each queue in priority order
        for priority in ExecutionPriority:
            queue = self.task_queues[priority]
            
            if not queue.empty():
                # Get task from queue
                _, task = await queue.get()
                
                # Check if task is expired
                if task.is_expired():
                    # Task expired, mark as failed
                    self._mark_task_failed(
                        task, 
                        Exception(f"Task expired before execution (deadline: {task.deadline})")
                    )
                    
                    # Mark queue task as done
                    queue.task_done()
                    
                    # Continue to next task
                    continue
                
                return task
        
        # All queues empty
        return None
    
    async def _try_create_batch(self, initial_task: TradeTask) -> List[TradeTask]:
        """
        Try to create a batch of similar tasks
        
        Args:
            initial_task (TradeTask): Initial task for batch
            
        Returns:
            List[TradeTask]: Batch of tasks
        """
        # Start batch with initial task
        batch = [initial_task]
        
        # Only batch tasks of same action (buy/sell) and token
        action = initial_task.trade_params.get('action')
        token_address = initial_task.trade_params.get('token_address')
        
        if not action or not token_address:
            # Can't batch without action and token
            return batch
        
        # Check if there are more tasks with same action and token
        for priority in ExecutionPriority:
            queue = self.task_queues[priority]
            
            # Peek at queue (don't remove items yet)
            remaining_slots = self.batch_size_limit - len(batch)
            if remaining_slots <= 0:
                break
                
            # Create a snapshot of the queue
            snapshot = []
            while not queue.empty() and len(snapshot) < 10:  # Limit peek depth
                item = await queue.get()
                snapshot.append(item)
            
            # Put items back in queue
            for item in snapshot:
                await queue.put(item)
            
            # Find matching tasks
            matching_tasks = []
            for _, task in snapshot:
                task_action = task.trade_params.get('action')
                task_token = task.trade_params.get('token_address')
                
                if (task_action == action and 
                    task_token == token_address and 
                    not task.is_expired()):
                    matching_tasks.append(task)
                    
                    if len(matching_tasks) >= remaining_slots:
                        break
            
            # If found matching tasks, add to batch
            if matching_tasks:
                # Remove tasks from queue
                for _ in range(len(matching_tasks)):
                    _, task = await queue.get()
                    
                    # Check if task matches our criteria
                    task_action = task.trade_params.get('action')
                    task_token = task.trade_params.get('token_address')
                    
                    if (task_action == action and 
                        task_token == token_address and 
                        not task.is_expired()):
                        batch.append(task)
                        
                        # Mark as task done in queue
                        queue.task_done()
                        
                        if len(batch) >= self.batch_size_limit:
                            break
                    else:
                        # Put back in queue
                        await queue.put((task.get_sort_key(), task))
        
        logger.info(f"Created batch of {len(batch)} {action} tasks for {token_address}")
        return batch
    
    async def _execute_batch(self, batch: List[TradeTask]):
        """
        Execute a batch of tasks
        
        Args:
            batch (List[TradeTask]): Batch of tasks
        """
        if not batch:
            return
        
        # Mark all tasks as active
        for task in batch:
            task.status = "executing"
            task.start_time = time.time()
            self.active_task_ids.add(task.task_id)
        
        # Get action and token information
        action = batch[0].trade_params.get('action')
        token_address = batch[0].trade_params.get('token_address')
        
        # Prepare batch execution parameters
        if action == 'buy':
            # Aggregate buy amounts
            total_amount_usd = sum(
                task.trade_params.get('amount_usd', 0) 
                for task in batch
            )
            
            # Create consolidated buy parameters
            batch_params = {
                'action': 'buy',
                'token_address': token_address,
                'amount_usd': total_amount_usd,
                'max_slippage': min(
                    task.trade_params.get('max_slippage', 2.0)
                    for task in batch
                )
            }
            
        elif action == 'sell':
            # For sell, we need to handle each sell separately
            # as they may have different percentage values
            batch_params = None
            
            # Mark all tasks as individual execution
            for task in batch:
                await self._execute_task(task)
            
            return
            
        else:
            # Unknown action, execute individually
            logger.warning(f"Unknown batch action: {action}, executing individually")
            
            for task in batch:
                await self._execute_task(task)
            
            return
        
        # Execute batch trade
        try:
            # Acquire semaphore for concurrent execution control
            async with self.execution_semaphore:
                # Execute batch trade with timeout
                logger.info(f"Executing batch {action} for {token_address} ({len(batch)} tasks)")
                
                result = await asyncio.wait_for(
                    self.flash_executor.execute_flash_trade(batch_params),
                    timeout=self.execution_timeout
                )
                
                # Process successful batch execution
                if action == 'buy':
                    # Distribute tokens based on proportional contribution
                    total_tokens = result.get('amount_tokens', 0)
                    avg_price = result.get('entry_price_usd', 0)
                    
                    # Set result for each task
                    for task in batch:
                        amount_usd = task.trade_params.get('amount_usd', 0)
                        proportion = amount_usd / total_amount_usd
                        
                        # Create individual result
                        task_result = {
                            'success': True,
                            'action': 'buy',
                            'token_address': token_address,
                            'amount_usd': amount_usd,
                            'amount_tokens': total_tokens * proportion,
                            'entry_price_usd': avg_price,
                            'timestamp': time.time(),
                            'transaction_hash': result.get('transaction_hash'),
                            'batch_execution': True,
                            'batch_size': len(batch)
                        }
                        
                        # Mark task as completed
                        self._mark_task_completed(task, task_result)
                
        except Exception as e:
            logger.error(f"Error executing batch {action} for {token_address}: {str(e)}")
            
            # Mark all tasks as failed
            for task in batch:
                self._mark_task_failed(task, e)
    
    async def _execute_task(self, task: TradeTask):
        """
        Execute a single task
        
        Args:
            task (TradeTask): Task to execute
        """
        # Mark task as active
        task.status = "executing"
        task.start_time = time.time()
        self.active_task_ids.add(task.task_id)
        
        try:
            # Acquire semaphore for concurrent execution control
            async with self.execution_semaphore:
                # Execute trade with timeout
                result = await asyncio.wait_for(
                    self.flash_executor.execute_flash_trade(task.trade_params),
                    timeout=self.execution_timeout
                )
                
                # Mark task as completed
                self._mark_task_completed(task, result)
                
        except Exception as e:
            logger.error(f"Error executing task {task.task_id}: {str(e)}")
            
            # Mark task as failed
            self._mark_task_failed(task, e)
    
    def _mark_task_completed(self, task: TradeTask, result: Dict[str, Any]):
        """
        Mark task as completed
        
        Args:
            task (TradeTask): Task to mark completed
            result (Dict[str, Any]): Task result
        """
        # Update task
        task.status = "completed"
        task.end_time = time.time()
        task.result = result
        
        # Update tracking
        self.active_task_ids.discard(task.task_id)
        self.completed_task_ids.add(task.task_id)
        
        # Update metrics
        self.queue_stats['completed_tasks'] += 1
        
        # Calculate execution time
        if task.start_time:
            execution_time = task.end_time - task.start_time
            wait_time = task.start_time - task.creation_time
            
            # Update average times
            if self.queue_stats['completed_tasks'] == 1:
                self.queue_stats['avg_execution_time'] = execution_time
                self.queue_stats['avg_wait_time'] = wait_time
            else:
                # Weighted average (more weight to recent executions)
                self.queue_stats['avg_execution_time'] = (
                    self.queue_stats['avg_execution_time'] * 0.8 + execution_time * 0.2
                )
                self.queue_stats['avg_wait_time'] = (
                    self.queue_stats['avg_wait_time'] * 0.8 + wait_time * 0.2
                )
        
        # Update component metrics
        self._update_metrics()
        
        # Set future result
        if not task.future.done():
            task.future.set_result(result)
        
        # Call callback if provided
        if task.callback:
            try:
                task.callback(result)
            except Exception as e:
                logger.error(f"Error in task callback: {str(e)}")
        
        logger.info(f"Task {task.task_id} completed")
        
        # Trim completed tasks if needed
        if len(self.completed_task_ids) > self.max_completed_tasks:
            # Find oldest completed tasks
            to_remove = list(self.completed_task_ids)[:-self.max_completed_tasks]
            
            # Remove from tracking
            for task_id in to_remove:
                self.completed_task_ids.discard(task_id)
                if task_id in self.tasks:
                    del self.tasks[task_id]
    
    def _mark_task_failed(self, task: TradeTask, error: Exception):
        """
        Mark task as failed
        
        Args:
            task (TradeTask): Task to mark failed
            error (Exception): Error that caused failure
        """
        # Update task
        task.status = "failed"
        task.end_time = time.time()
        task.error = str(error)
        
        # Update tracking
        self.active_task_ids.discard(task.task_id)
        self.completed_task_ids.add(task.task_id)
        
        # Update metrics
        self.queue_stats['failed_tasks'] += 1
        
        # Update component metrics
        self._update_metrics()
        
        # Set future exception
        if not task.future.done():
            task.future.set_exception(error)
        
        # Call callback if provided
        if task.callback:
            try:
                task.callback({'success': False, 'error': str(error)})
            except Exception as e:
                logger.error(f"Error in task callback: {str(e)}")
        
        logger.warning(f"Task {task.task_id} failed: {str(error)}")
    
    def _update_metrics(self):
        """Update component metrics in state manager"""
        metrics = {
            'queue_stats': self.queue_stats,
            'active_tasks': len(self.active_task_ids),
            'queued_tasks': sum(q.qsize() for q in self.task_queues.values()),
            'completed_tasks': len(self.completed_task_ids),
            'max_concurrent_executions': self.max_concurrent_executions,
            'batching_enabled': self.enable_batching
        }
        
        # Add queue sizes by priority
        for priority in ExecutionPriority:
            metrics[f"queue_size_{priority.name.lower()}"] = self.task_queues[priority].qsize()
        
        # Update state manager
        self.state_manager.update_component_metrics('parallel_executor', metrics)
    
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Get task status
        
        Args:
            task_id (str): Task ID
            
        Returns:
            Optional[Dict[str, Any]]: Task status or None if not found
        """
        task = self.tasks.get(task_id)
        if task is None:
            return None
        
        return task.to_dict()
    
    def get_active_tasks(self) -> List[Dict[str, Any]]:
        """
        Get active tasks
        
        Returns:
            List[Dict[str, Any]]: Active tasks
        """
        return [
            self.tasks[task_id].to_dict()
            for task_id in self.active_task_ids
            if task_id in self.tasks
        ]
    
    def get_queued_tasks(self) -> List[Dict[str, Any]]:
        """
        Get queued tasks
        
        Returns:
            List[Dict[str, Any]]: Queued tasks
        """
        return [
            task.to_dict()
            for task_id, task in self.tasks.items()
            if task.status == "queued"
        ]
    
    def get_completed_tasks(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get completed tasks
        
        Args:
            limit (int): Maximum number of tasks to return
            
        Returns:
            List[Dict[str, Any]]: Completed tasks
        """
        completed = [
            self.tasks[task_id].to_dict()
            for task_id in self.completed_task_ids
            if task_id in self.tasks
        ]
        
        # Sort by end time (newest first)
        completed.sort(key=lambda t: t.get('end_time', 0), reverse=True)
        
        return completed[:limit]
    
    def cancel_task(self, task_id: str) -> bool:
        """
        Cancel a queued task
        
        Args:
            task_id (str): Task ID
            
        Returns:
            bool: True if task cancelled, False otherwise
        """
        task = self.tasks.get(task_id)
        if task is None:
            return False
        
        # Can only cancel queued tasks
        if task.status != "queued":
            return False
        
        # Mark as cancelled
        task.status = "cancelled"
        task.end_time = time.time()
        
        # Set future result
        if not task.future.done():
            task.future.set_exception(Exception("Task cancelled"))
        
        # Call callback if provided
        if task.callback:
            try:
                task.callback({'success': False, 'error': "Task cancelled"})
            except Exception as e:
                logger.error(f"Error in task callback: {str(e)}")
        
        logger.info(f"Task {task_id} cancelled")
        
        return True