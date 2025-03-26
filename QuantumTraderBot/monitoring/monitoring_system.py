"""
MonitoringSystem Component
Responsible for monitoring the health and performance of the system,
collecting metrics, and detecting issues.
"""

import asyncio
import logging
import time
import os
import psutil
import platform
from typing import Dict, Any, List, Optional, Callable, Set
import json
import sys

from core.state_manager import StateManager

logger = logging.getLogger(__name__)

class MonitoringSystem:
    """
    MonitoringSystem handles:
    - Monitoring system resources
    - Checking component health
    - Collecting performance metrics
    - Detecting and reporting issues
    """
    
    def __init__(self, state_manager: StateManager):
        """
        Initialize the MonitoringSystem
        
        Args:
            state_manager (StateManager): State manager instance
        """
        self.state_manager = state_manager
        
        # Monitoring settings
        self.health_check_interval = 60  # seconds
        self.metrics_collection_interval = 300  # seconds
        self.resource_check_interval = 30  # seconds
        
        # System metrics
        self.system_metrics = {
            'cpu_usage': 0,
            'memory_usage': 0,
            'memory_available': 0,
            'disk_usage': 0,
            'disk_available': 0,
            'network_errors': 0,
            'uptime': 0
        }
        
        # Alert thresholds
        self.alert_thresholds = {
            'cpu_usage': 90,  # Percentage
            'memory_usage': 90,  # Percentage
            'disk_usage': 90,  # Percentage
            'component_failures': 1  # Number of consecutive failures
        }
        
        # Component failure counters
        self.component_failures = {}
        
        # Alert callbacks
        self.alert_callbacks = set()
        
        # State
        self.is_running = False
        self.start_time = time.time()
        self.last_health_check = 0
        self.last_metrics_collection = 0
        self.last_resource_check = 0
        
        # Register for alerts from state manager
        self.state_manager.register_alert_listener(self._handle_state_alert)
        
        logger.info("MonitoringSystem initialized")
    
    async def start(self):
        """Start the monitoring system"""
        if self.is_running:
            logger.warning("MonitoringSystem already running")
            return
        
        logger.info("Starting MonitoringSystem")
        self.state_manager.update_component_status('monitoring_system', 'starting')
        
        # Set state
        self.is_running = True
        self.start_time = time.time()
        
        # Start monitoring tasks
        asyncio.create_task(self._monitoring_loop())
        
        self.state_manager.update_component_status('monitoring_system', 'running')
        logger.info("MonitoringSystem started")
    
    async def stop(self):
        """Stop the monitoring system"""
        if not self.is_running:
            logger.warning("MonitoringSystem not running")
            return
        
        logger.info("Stopping MonitoringSystem")
        self.state_manager.update_component_status('monitoring_system', 'stopping')
        
        # Set state
        self.is_running = False
        
        self.state_manager.update_component_status('monitoring_system', 'stopped')
        logger.info("MonitoringSystem stopped")
    
    async def _monitoring_loop(self):
        """Main monitoring loop"""
        logger.debug("Starting monitoring loop")
        
        while self.is_running:
            try:
                # Check if it's time for health check
                if time.time() - self.last_health_check >= self.health_check_interval:
                    await self._check_component_health()
                    self.last_health_check = time.time()
                
                # Check if it's time for metrics collection
                if time.time() - self.last_metrics_collection >= self.metrics_collection_interval:
                    await self._collect_metrics()
                    self.last_metrics_collection = time.time()
                
                # Check if it's time for resource check
                if time.time() - self.last_resource_check >= self.resource_check_interval:
                    await self._check_system_resources()
                    self.last_resource_check = time.time()
                
                # Sleep for a short time
                await asyncio.sleep(5)
                
            except asyncio.CancelledError:
                break
            
            except Exception as e:
                logger.error(f"Error in monitoring loop: {str(e)}")
                await asyncio.sleep(30)  # Sleep longer on error
    
    async def _check_component_health(self):
        """Check the health of all components"""
        logger.debug("Checking component health")
        
        # Get all component states
        component_states = self.state_manager.get_all_component_states()
        
        # Check each component
        unhealthy_components = []
        for name, state in component_states.items():
            # Skip monitoring system itself
            if name == 'monitoring_system':
                continue
            
            if not state.get('is_healthy', False) and state.get('status') != 'stopped':
                # Component is unhealthy and not stopped
                unhealthy_components.append(name)
                
                # Increment failure counter
                if name not in self.component_failures:
                    self.component_failures[name] = 0
                
                self.component_failures[name] += 1
                
                # Check if threshold reached
                if self.component_failures[name] >= self.alert_thresholds['component_failures']:
                    # Create alert if not already alerted
                    self.state_manager.create_alert(
                        'monitoring_system',
                        'ERROR',
                        f"Component {name} is unhealthy: {state.get('error_message', 'Unknown error')}"
                    )
            else:
                # Component is healthy or stopped, reset failure counter
                self.component_failures[name] = 0
        
        # Update metrics
        self.state_manager.update_component_metric(
            'monitoring_system',
            'unhealthy_components',
            len(unhealthy_components)
        )
        
        if unhealthy_components:
            logger.warning(f"Unhealthy components: {', '.join(unhealthy_components)}")
        else:
            logger.debug("All components are healthy")
    
    async def _collect_metrics(self):
        """Collect system and component metrics"""
        logger.debug("Collecting system metrics")
        
        # Calculate uptime
        uptime = time.time() - self.start_time
        
        # Update uptime metric
        self.system_metrics['uptime'] = uptime
        
        # Update global metrics in state manager
        self.state_manager.update_global_metric('system_uptime', uptime)
        self.state_manager.update_global_metric('cpu_usage', self.system_metrics['cpu_usage'])
        self.state_manager.update_global_metric('memory_usage', self.system_metrics['memory_usage'])
        self.state_manager.update_global_metric('disk_usage', self.system_metrics['disk_usage'])
        
        # Get all component states for metrics
        component_states = self.state_manager.get_all_component_states()
        
        # Update component count metric
        self.state_manager.update_global_metric('component_count', len(component_states))
        
        # Collect alert statistics
        alerts = self.state_manager.get_alerts()
        alert_counts = {
            'total': len(alerts),
            'critical': len([a for a in alerts if a.get('level') == 'CRITICAL']),
            'error': len([a for a in alerts if a.get('level') == 'ERROR']),
            'warning': len([a for a in alerts if a.get('level') == 'WARNING']),
            'info': len([a for a in alerts if a.get('level') == 'INFO'])
        }
        
        # Update alert metrics
        for level, count in alert_counts.items():
            self.state_manager.update_global_metric(f'alerts_{level}', count)
        
        # Save system state periodically
        try:
            os.makedirs('./state', exist_ok=True)
            self.state_manager.save_state_to_file('./state/system_state.json')
        except Exception as e:
            logger.error(f"Error saving system state: {str(e)}")
    
    async def _check_system_resources(self):
        """Check system resource usage"""
        logger.debug("Checking system resources")
        
        try:
            # Get CPU usage
            cpu_usage = psutil.cpu_percent(interval=0.5)
            
            # Get memory usage
            memory = psutil.virtual_memory()
            memory_usage = memory.percent
            memory_available = memory.available / (1024 * 1024)  # MB
            
            # Get disk usage
            disk = psutil.disk_usage('/')
            disk_usage = disk.percent
            disk_available = disk.free / (1024 * 1024 * 1024)  # GB
            
            # Update metrics
            self.system_metrics['cpu_usage'] = cpu_usage
            self.system_metrics['memory_usage'] = memory_usage
            self.system_metrics['memory_available'] = memory_available
            self.system_metrics['disk_usage'] = disk_usage
            self.system_metrics['disk_available'] = disk_available
            
            # Update component metrics
            self.state_manager.update_component_metric(
                'monitoring_system',
                'cpu_usage',
                cpu_usage
            )
            self.state_manager.update_component_metric(
                'monitoring_system',
                'memory_usage',
                memory_usage
            )
            self.state_manager.update_component_metric(
                'monitoring_system',
                'disk_usage',
                disk_usage
            )
            
            # Check thresholds and create alerts if necessary
            if cpu_usage > self.alert_thresholds['cpu_usage']:
                self.state_manager.create_alert(
                    'monitoring_system',
                    'WARNING',
                    f"High CPU usage: {cpu_usage:.1f}%"
                )
            
            if memory_usage > self.alert_thresholds['memory_usage']:
                self.state_manager.create_alert(
                    'monitoring_system',
                    'WARNING',
                    f"High memory usage: {memory_usage:.1f}%"
                )
            
            if disk_usage > self.alert_thresholds['disk_usage']:
                self.state_manager.create_alert(
                    'monitoring_system',
                    'WARNING',
                    f"High disk usage: {disk_usage:.1f}%"
                )
            
            logger.debug(f"System resources - CPU: {cpu_usage:.1f}%, Memory: {memory_usage:.1f}%, Disk: {disk_usage:.1f}%")
            
        except Exception as e:
            logger.error(f"Error checking system resources: {str(e)}")
    
    def _handle_state_alert(self, alert):
        """
        Handle alert from state manager
        
        Args:
            alert: Alert object
        """
        # Forward alert to all registered callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Error in alert callback: {str(e)}")
    
    def register_alert_callback(self, callback: Callable):
        """
        Register callback for alerts
        
        Args:
            callback: Callback function that takes an alert object
        """
        self.alert_callbacks.add(callback)
        logger.debug("Registered alert callback")
    
    def unregister_alert_callback(self, callback: Callable):
        """
        Unregister callback for alerts
        
        Args:
            callback: Callback function to unregister
        """
        if callback in self.alert_callbacks:
            self.alert_callbacks.remove(callback)
            logger.debug("Unregistered alert callback")
    
    def get_system_info(self) -> Dict[str, Any]:
        """
        Get system information
        
        Returns:
            Dict[str, Any]: System information
        """
        try:
            info = {
                'platform': platform.system(),
                'platform_release': platform.release(),
                'platform_version': platform.version(),
                'architecture': platform.machine(),
                'processor': platform.processor(),
                'python_version': sys.version,
                'hostname': platform.node(),
                'uptime': time.time() - self.start_time,
                'cpu_count': psutil.cpu_count(logical=True),
                'physical_cpu_count': psutil.cpu_count(logical=False),
                'memory_total': psutil.virtual_memory().total / (1024 * 1024),  # MB
                'disk_total': psutil.disk_usage('/').total / (1024 * 1024 * 1024)  # GB
            }
            return info
        except Exception as e:
            logger.error(f"Error getting system info: {str(e)}")
            return {'error': str(e)}
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """
        Get system metrics
        
        Returns:
            Dict[str, Any]: System metrics
        """
        return self.system_metrics
    
    def get_component_status(self) -> Dict[str, Any]:
        """
        Get status of all components
        
        Returns:
            Dict[str, Any]: Component status
        """
        return self.state_manager.get_all_component_states()
    
    def get_alert_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get alert history
        
        Args:
            limit (int): Maximum number of alerts to return
            
        Returns:
            List[Dict[str, Any]]: Alert history
        """
        return self.state_manager.get_alerts(limit=limit)
