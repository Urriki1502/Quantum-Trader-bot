"""
BaseComponent
Abstract base class for all system components with common functionality.
"""

import asyncio
import logging
import time
import psutil
import traceback
from typing import Dict, Any, Optional, List, Awaitable, Callable

logger = logging.getLogger(__name__)

class BaseComponent:
    """
    BaseComponent provides common functionality for all components:
    - Heartbeat mechanism
    - Error handling
    - Metrics collection
    - State management
    """
    
    def __init__(self, state_manager, name: Optional[str] = None):
        """
        Initialize the BaseComponent
        
        Args:
            state_manager: StateManager instance
            name (str, optional): Component name, defaults to class name
        """
        self.state_manager = state_manager
        self.name = name or self.__class__.__name__
        self.is_running = False
        self._tasks = []
        self._heartbeat_interval = 15  # Seconds between heartbeats
        self._error_count = 0
        self._last_error_time = 0
        self._metrics = {}
        
        # Không đăng ký với state manager ở đây, để tránh đăng ký trùng lặp
        # Việc đăng ký sẽ được thực hiện bởi main.py thông qua phương thức _register_components()
        
    async def start(self):
        """Start the component"""
        if self.is_running:
            logger.warning(f"{self.name} is already running")
            return
            
        try:
            self.is_running = True
            
            # Update status in state manager
            self.state_manager.update_component_status(self.name, "running")
            
            # Start heartbeat task
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
            self._tasks.append(self._heartbeat_task)
            
            logger.info(f"{self.name} started")
            
        except Exception as e:
            logger.error(f"Error starting {self.name}: {str(e)}")
            self.is_running = False
            self.state_manager.update_component_status(self.name, "error", str(e))
            raise
    
    async def stop(self):
        """Stop the component"""
        if not self.is_running:
            logger.warning(f"{self.name} is not running")
            return
            
        try:
            self.is_running = False
            
            # Cancel all tasks
            for task in self._tasks:
                if not task.done():
                    task.cancel()
            
            # Wait for tasks to cancel
            if self._tasks:
                await asyncio.gather(*self._tasks, return_exceptions=True)
            
            self._tasks = []
            
            # Update status in state manager
            self.state_manager.update_component_status(self.name, "stopped")
            
            logger.info(f"{self.name} stopped")
            
        except Exception as e:
            logger.error(f"Error stopping {self.name}: {str(e)}")
            self.state_manager.update_component_status(self.name, "error", str(e))
            raise
    
    async def _heartbeat_loop(self):
        """Task that sends regular heartbeats"""
        try:
            while self.is_running:
                try:
                    # Collect metrics
                    metrics = await self._collect_metrics()
                    
                    # Send heartbeat
                    await self.state_manager.component_heartbeat(
                        self.name, 
                        "healthy" if self._error_count == 0 else "degraded",
                        metrics
                    )
                    
                except Exception as e:
                    logger.error(f"Error in {self.name} heartbeat: {str(e)}")
                
                await asyncio.sleep(self._heartbeat_interval)
                
        except asyncio.CancelledError:
            logger.debug(f"{self.name} heartbeat task cancelled")
            raise
            
        except Exception as e:
            logger.error(f"Unexpected error in {self.name} heartbeat loop: {str(e)}")
    
    async def _collect_metrics(self) -> Dict[str, Any]:
        """
        Collect component metrics
        
        Returns:
            Dict[str, Any]: Component metrics
        """
        try:
            # Get process info
            process = psutil.Process()
            cpu_percent = process.cpu_percent(interval=0.1)
            memory_info = process.memory_info()
            
            # Basic metrics common to all components
            metrics = {
                "cpu_percent": cpu_percent,
                "memory_mb": memory_info.rss / (1024 * 1024),
                "error_count": self._error_count,
                "uptime": time.time() - getattr(self, '_start_time', time.time())
            }
            
            # Add custom metrics
            custom_metrics = await self._collect_component_metrics()
            if custom_metrics:
                metrics.update(custom_metrics)
                
            return metrics
            
        except Exception as e:
            logger.error(f"Error collecting metrics for {self.name}: {str(e)}")
            return {
                "error": str(e),
                "error_count": self._error_count
            }
    
    async def _collect_component_metrics(self) -> Dict[str, Any]:
        """
        Collect component-specific metrics
        Override in subclasses to provide component-specific metrics
        
        Returns:
            Dict[str, Any]: Component-specific metrics
        """
        return {}
        
    def _record_error(self, error: Exception, context: str = ""):
        """
        Record an error
        
        Args:
            error (Exception): The error
            context (str): Error context
        """
        self._error_count += 1
        self._last_error_time = time.time()
        
        error_details = {
            "error": str(error),
            "type": type(error).__name__,
            "context": context,
            "traceback": traceback.format_exc(),
            "count": self._error_count,
            "timestamp": self._last_error_time
        }
        
        self.state_manager.update_component_metric(self.name, "last_error", error_details)
        
        # If too many errors, mark component as degraded
        if self._error_count > 5:
            self.state_manager.update_component_status(
                self.name, 
                "error" if self._error_count > 10 else "running",
                f"Too many errors: {self._error_count} errors recorded"
            )
    
    @classmethod
    async def with_error_boundary(cls, coro, component_name: str, operation: str, state_manager=None):
        """
        Execute coroutine with error boundary
        
        Args:
            coro: Coroutine to execute
            component_name (str): Component name
            operation (str): Operation name
            state_manager: StateManager instance (optional)
            
        Returns:
            Result of coroutine or None if error
        """
        try:
            return await coro
        except Exception as e:
            logger.error(f"Error in {component_name}.{operation}: {str(e)}")
            
            # Record error if state manager provided
            if state_manager:
                error_details = {
                    "error": str(e),
                    "component": component_name,
                    "operation": operation,
                    "traceback": traceback.format_exc(),
                    "timestamp": time.time()
                }
                
                if hasattr(state_manager, "create_alert"):
                    state_manager.create_alert(component_name, "ERROR", f"Error in {operation}: {str(e)}")
                
            return None