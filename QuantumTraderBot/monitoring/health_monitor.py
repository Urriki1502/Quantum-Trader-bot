"""
Health Monitoring System
Provides comprehensive health monitoring for the trading bot's components,
enabling rapid detection and automatic recovery from issues.
"""

import time
import logging
import asyncio
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Set, Callable

from core.state_manager import StateManager
from core.config_manager import ConfigManager

logger = logging.getLogger(__name__)

class ComponentHealth:
    """Tracks health status of a single component"""
    
    def __init__(self, component_id: str):
        """
        Initialize component health tracker
        
        Args:
            component_id (str): Component identifier
        """
        self.component_id = component_id
        self.last_heartbeat = time.time()
        self.heartbeat_interval = 60  # Default: 60 seconds
        self.status = "healthy"
        self.error_count = 0
        self.warning_count = 0
        self.last_errors = []
        self.metrics = {}
        self.checks = {}
        
    def update_heartbeat(self):
        """Update component heartbeat timestamp"""
        self.last_heartbeat = time.time()
        
    def set_heartbeat_interval(self, interval: float):
        """
        Set expected heartbeat interval
        
        Args:
            interval (float): Heartbeat interval in seconds
        """
        self.heartbeat_interval = interval
        
    def record_error(self, error: str):
        """
        Record a component error
        
        Args:
            error (str): Error description
        """
        self.error_count += 1
        self.last_errors.append({
            'time': time.time(),
            'message': error
        })
        
        # Keep only last 10 errors
        if len(self.last_errors) > 10:
            self.last_errors = self.last_errors[-10:]
            
    def record_warning(self, warning: str):
        """
        Record a component warning
        
        Args:
            warning (str): Warning description
        """
        self.warning_count += 1
        
    def update_status(self, status: str):
        """
        Update component status
        
        Args:
            status (str): New status ('healthy', 'degraded', 'failing', 'offline')
        """
        old_status = self.status
        self.status = status
        
        if old_status != status:
            logger.info(f"Component {self.component_id} status changed: {old_status} → {status}")
            
    def update_metric(self, metric_name: str, value: Any):
        """
        Update a health metric
        
        Args:
            metric_name (str): Metric name
            value (Any): Metric value
        """
        self.metrics[metric_name] = {
            'value': value,
            'timestamp': time.time()
        }
        
    def check_heartbeat(self) -> bool:
        """
        Check if heartbeat is current
        
        Returns:
            bool: True if heartbeat is current
        """
        return (time.time() - self.last_heartbeat) < (self.heartbeat_interval * 2)
    
    def add_check_result(self, check_name: str, passed: bool, details: Optional[Dict[str, Any]] = None):
        """
        Add result of a health check
        
        Args:
            check_name (str): Check name
            passed (bool): Whether check passed
            details (Dict[str, Any], optional): Additional details
        """
        self.checks[check_name] = {
            'passed': passed,
            'timestamp': time.time(),
            'details': details or {}
        }
        
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert health status to dictionary
        
        Returns:
            Dict[str, Any]: Health status
        """
        return {
            'component_id': self.component_id,
            'status': self.status,
            'last_heartbeat': self.last_heartbeat,
            'heartbeat_age': time.time() - self.last_heartbeat,
            'heartbeat_current': self.check_heartbeat(),
            'error_count': self.error_count,
            'warning_count': self.warning_count,
            'last_errors': self.last_errors,
            'metrics': self.metrics,
            'checks': self.checks
        }

class HealthMonitor:
    """
    Comprehensive health monitoring system for all bot components
    
    Provides:
    - Component heartbeat monitoring
    - Health check coordination
    - Anomaly detection
    - Automated recovery triggering
    - Health metrics collection and visualization
    """
    
    def __init__(self, 
                state_manager: StateManager, 
                config_manager: ConfigManager):
        """
        Initialize the health monitor
        
        Args:
            state_manager (StateManager): State manager instance
            config_manager (ConfigManager): Configuration manager instance
        """
        self.state_manager = state_manager
        self.config_manager = config_manager
        
        # Component health tracking
        self.components: Dict[str, ComponentHealth] = {}
        
        # Track automatically restartable components
        self.restartable_components: Set[str] = set()
        
        # Load configuration
        self.check_interval = self.config_manager.get('health.check_interval_sec', 30)
        self.critical_components = self.config_manager.get('health.critical_components', [
            'trading_integration', 'pump_portal_client', 'onchain_analyzer', 
            'risk_manager', 'strategy_manager', 'raydium_client'
        ])
        
        # Auto recovery settings
        self.auto_recovery_enabled = self.config_manager.get('health.auto_recovery.enabled', True)
        self.recovery_cooldown = self.config_manager.get('health.auto_recovery.cooldown_sec', 300)
        self.max_recovery_attempts = self.config_manager.get('health.auto_recovery.max_attempts', 3)
        
        # Track recovery attempts
        self.recovery_attempts: Dict[str, List[float]] = {}
        
        # Initialize metrics
        self.system_metrics = {
            'last_check': 0,
            'system_status': 'starting',
            'healthy_components': 0,
            'total_components': 0,
            'last_recovery': 0
        }
        
        # Register recovery handlers
        self.recovery_handlers: Dict[str, Callable] = {}
        
    def register_component(self, component_id: str, heartbeat_interval: float = 60):
        """
        Register a component for health monitoring
        
        Args:
            component_id (str): Component identifier
            heartbeat_interval (float): Expected heartbeat interval in seconds
        """
        if component_id not in self.components:
            self.components[component_id] = ComponentHealth(component_id)
            
        self.components[component_id].set_heartbeat_interval(heartbeat_interval)
        logger.info(f"Registered component for health monitoring: {component_id}")
        
    def register_restartable_component(self, component_id: str, restart_handler: Callable):
        """
        Register a component that can be automatically restarted
        
        Args:
            component_id (str): Component identifier
            restart_handler (Callable): Function to call for restarting component
        """
        self.restartable_components.add(component_id)
        self.recovery_handlers[component_id] = restart_handler
        logger.info(f"Registered restartable component: {component_id}")
        
    def update_heartbeat(self, component_id: str):
        """
        Update heartbeat for a component
        
        Args:
            component_id (str): Component identifier
        """
        if component_id not in self.components:
            self.register_component(component_id)
            
        self.components[component_id].update_heartbeat()
        
    def record_component_error(self, component_id: str, error: str):
        """
        Record an error for a component
        
        Args:
            component_id (str): Component identifier
            error (str): Error description
        """
        if component_id not in self.components:
            self.register_component(component_id)
            
        self.components[component_id].record_error(error)
        logger.warning(f"Component error: {component_id} - {error}")
        
    def record_health_check(self, component_id: str, check_name: str, 
                          passed: bool, details: Optional[Dict[str, Any]] = None):
        """
        Record result of a health check
        
        Args:
            component_id (str): Component identifier
            check_name (str): Health check name
            passed (bool): Whether check passed
            details (Dict[str, Any], optional): Additional check details
        """
        if component_id not in self.components:
            self.register_component(component_id)
            
        self.components[component_id].add_check_result(check_name, passed, details)
        
    def update_component_metric(self, component_id: str, metric_name: str, value: Any):
        """
        Update a component health metric
        
        Args:
            component_id (str): Component identifier
            metric_name (str): Metric name
            value (Any): Metric value
        """
        if component_id not in self.components:
            self.register_component(component_id)
            
        self.components[component_id].update_metric(metric_name, value)
        
    async def start(self):
        """Start the health monitoring system"""
        logger.info("Starting health monitoring system")
        self.system_metrics['system_status'] = 'running'
        
        # Start monitoring tasks
        asyncio.create_task(self._health_check_loop())
        asyncio.create_task(self._metrics_reporter())
        
    async def _health_check_loop(self):
        """Main health check loop"""
        while True:
            try:
                await self._run_health_checks()
                self.system_metrics['last_check'] = time.time()
                await asyncio.sleep(self.check_interval)
            except Exception as e:
                logger.error(f"Error in health check loop: {e}")
                await asyncio.sleep(5)  # Shorter sleep on error
                
    async def _run_health_checks(self):
        """Run comprehensive health checks on all components"""
        start_time = time.time()
        
        # Check component heartbeats
        healthy_count = 0
        total_count = len(self.components)
        
        for component_id, health in self.components.items():
            # Skip components that haven't been initialized yet
            if health.last_heartbeat == 0:
                continue
                
            # Check heartbeat
            if not health.check_heartbeat():
                heartbeat_age = time.time() - health.last_heartbeat
                logger.warning(f"Stale heartbeat for {component_id}: {heartbeat_age:.1f}s (expected < {health.heartbeat_interval * 2}s)")
                
                if heartbeat_age > health.heartbeat_interval * 5:
                    health.update_status("offline")
                    if component_id in self.critical_components:
                        await self._handle_critical_component_failure(component_id, "Heartbeat timeout")
                elif heartbeat_age > health.heartbeat_interval * 3:
                    health.update_status("failing")
                else:
                    health.update_status("degraded")
            else:
                # Check for excessive errors
                if health.error_count > 20 and time.time() - health.last_heartbeat < 300:
                    health.update_status("degraded")
                else:
                    health.update_status("healthy")
                    healthy_count += 1
        
        # Update system metrics
        self.system_metrics['healthy_components'] = healthy_count
        self.system_metrics['total_components'] = total_count
        
        health_percentage = (healthy_count / total_count * 100) if total_count > 0 else 0
        
        # Update overall system status
        if health_percentage >= 90:
            self.system_metrics['system_status'] = 'healthy'
        elif health_percentage >= 70:
            self.system_metrics['system_status'] = 'degraded'
        else:
            self.system_metrics['system_status'] = 'critical'
            
        # Log check completion
        check_duration = time.time() - start_time
        logger.debug(f"Health check completed in {check_duration:.2f}s: "
                   f"{healthy_count}/{total_count} components healthy ({health_percentage:.1f}%)")
        
    async def _handle_critical_component_failure(self, component_id: str, reason: str):
        """
        Handle failure of a critical component
        
        Args:
            component_id (str): Component identifier
            reason (str): Failure reason
        """
        logger.error(f"Critical component failure: {component_id} - {reason}")
        
        # Check if auto-recovery is enabled
        if not self.auto_recovery_enabled:
            logger.warning(f"Auto-recovery disabled, not attempting to recover {component_id}")
            self.state_manager.create_alert(
                'health_monitor',
                'CRITICAL',
                f"Critical component {component_id} failed, recovery disabled: {reason}"
            )
            return
        
        # Check if component is restartable
        if component_id not in self.restartable_components:
            logger.warning(f"Component {component_id} is not registered as restartable")
            self.state_manager.create_alert(
                'health_monitor',
                'CRITICAL',
                f"Critical component {component_id} failed, no recovery handler: {reason}"
            )
            return
        
        # Check recovery cooldown
        current_time = time.time()
        if component_id in self.recovery_attempts:
            # Remove attempts older than cooldown period
            recent_attempts = [t for t in self.recovery_attempts[component_id] 
                             if current_time - t < self.recovery_cooldown]
            self.recovery_attempts[component_id] = recent_attempts
            
            if len(recent_attempts) >= self.max_recovery_attempts:
                logger.error(f"Max recovery attempts reached for {component_id}, not attempting recovery")
                self.state_manager.create_alert(
                    'health_monitor',
                    'CRITICAL',
                    f"Critical component {component_id} failed, max recovery attempts reached: {reason}"
                )
                return
        else:
            self.recovery_attempts[component_id] = []
        
        # Attempt recovery
        logger.warning(f"Attempting to recover component: {component_id}")
        self.recovery_attempts[component_id].append(current_time)
        self.system_metrics['last_recovery'] = current_time
        
        # Create alert
        self.state_manager.create_alert(
            'health_monitor',
            'WARNING',
            f"Attempting to recover component {component_id}: {reason}"
        )
        
        # Call recovery handler
        try:
            if component_id in self.recovery_handlers:
                handler = self.recovery_handlers[component_id]
                await handler()
                logger.info(f"Recovery initiated for component: {component_id}")
            else:
                logger.error(f"No recovery handler found for component: {component_id}")
        except Exception as e:
            logger.error(f"Error during recovery of {component_id}: {e}")
            self.state_manager.create_alert(
                'health_monitor',
                'ERROR',
                f"Failed to recover component {component_id}: {str(e)}"
            )
    
    async def _metrics_reporter(self):
        """Periodically report health metrics"""
        while True:
            try:
                # Update state manager with health metrics
                self.state_manager.update_component_metric(
                    'health_monitor',
                    'system_status',
                    self.system_metrics['system_status']
                )
                
                self.state_manager.update_component_metric(
                    'health_monitor',
                    'health_percentage',
                    (self.system_metrics['healthy_components'] / self.system_metrics['total_components'] * 100) 
                    if self.system_metrics['total_components'] > 0 else 0
                )
                
                # Save detailed health report
                await self._save_health_report()
                
                await asyncio.sleep(60)  # Report once per minute
            except Exception as e:
                logger.error(f"Error in metrics reporter: {e}")
                await asyncio.sleep(10)
    
    async def _save_health_report(self):
        """Save detailed health report to file"""
        try:
            report_dir = self.config_manager.get('health.report_dir', './data/health')
            os.makedirs(report_dir, exist_ok=True)
            
            report_path = os.path.join(report_dir, 'health_report.json')
            temp_path = report_path + '.tmp'
            
            report = {
                'timestamp': time.time(),
                'system_status': self.system_metrics['system_status'],
                'healthy_components': self.system_metrics['healthy_components'],
                'total_components': self.system_metrics['total_components'],
                'components': {cid: health.to_dict() for cid, health in self.components.items()}
            }
            
            # Write to temp file first
            with open(temp_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            # Atomic move
            os.replace(temp_path, report_path)
        except Exception as e:
            logger.error(f"Error saving health report: {e}")
    
    def get_component_health(self, component_id: str) -> Optional[Dict[str, Any]]:
        """
        Get health status for a specific component
        
        Args:
            component_id (str): Component identifier
            
        Returns:
            Optional[Dict[str, Any]]: Component health status or None if not found
        """
        if component_id in self.components:
            return self.components[component_id].to_dict()
        return None
    
    def get_system_health(self) -> Dict[str, Any]:
        """
        Get overall system health status
        
        Returns:
            Dict[str, Any]: System health status
        """
        return {
            'system_status': self.system_metrics['system_status'],
            'healthy_components': self.system_metrics['healthy_components'],
            'total_components': self.system_metrics['total_components'],
            'health_percentage': (self.system_metrics['healthy_components'] / self.system_metrics['total_components'] * 100) 
                                if self.system_metrics['total_components'] > 0 else 0,
            'last_check': self.system_metrics['last_check'],
            'last_recovery': self.system_metrics['last_recovery'],
            'critical_components': {cid: self.get_component_health(cid) for cid in self.critical_components}
        }