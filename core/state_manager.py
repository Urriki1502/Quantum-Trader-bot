"""
StateManager Component
Responsible for managing and tracking the state of all bot components.
"""

import asyncio
import logging
import time
from typing import Dict, Any, List, Optional, Union
import json

logger = logging.getLogger(__name__)

class Alert:
    """Class representing a system alert"""
    
    def __init__(self, 
                 component: str, 
                 level: str, 
                 message: str, 
                 timestamp: Optional[float] = None):
        """
        Initialize an alert
        
        Args:
            component (str): The component that generated the alert
            level (str): Alert level (INFO, WARNING, ERROR, CRITICAL)
            message (str): Alert message
            timestamp (float, optional): Alert timestamp. Defaults to current time.
        """
        self.component = component
        self.level = level
        self.message = message
        self.timestamp = timestamp if timestamp is not None else time.time()
        self.acknowledged = False
        self.id = f"{self.component}-{self.timestamp}"
    
    def acknowledge(self):
        """Mark the alert as acknowledged"""
        self.acknowledged = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary"""
        return {
            'id': self.id,
            'component': self.component,
            'level': self.level,
            'message': self.message,
            'timestamp': self.timestamp,
            'acknowledged': self.acknowledged
        }


class ComponentState:
    """Class representing a component's state"""
    
    def __init__(self, name: str):
        """
        Initialize component state
        
        Args:
            name (str): Component name
        """
        self.name = name
        # Components start as initializing but we consider them healthy initially
        # to avoid false alerts during startup
        self.status = "initializing"  # initializing, running, error, stopped
        self.is_healthy = True  # Changed to True by default to prevent startup alerts
        self.last_health_check = time.time()
        self.health_check_count = 0  # Track number of health checks
        self.error_message = None
        self.metrics = {}
        self.start_time = time.time()
        self.state_history = []
        # Add grace period for components to properly initialize
        self.initialization_grace_period = 600  # 10 minutes grace period
        # Health check interval expected from components
        self.expected_health_interval = 300  # 5 minutes
        
    def update_status(self, status: str, error_message: Optional[str] = None):
        """
        Update component status
        
        Args:
            status (str): New status
            error_message (str, optional): Error message if status is 'error'
        """
        old_status = self.status
        self.status = status
        self.error_message = error_message
        
        # Add to history
        self.state_history.append({
            'timestamp': time.time(),
            'old_status': old_status,
            'new_status': status,
            'error_message': error_message
        })
        
        # Keep history limited to prevent memory issues
        if len(self.state_history) > 100:
            self.state_history.pop(0)
        
        # Increment health check counter
        self.health_check_count += 1
        
        # Update health based on status
        # Consider running components as healthy
        # For initializing components, check if within grace period
        if status == "running":
            self.is_healthy = True
        elif status == "initializing":
            # During initialization grace period, component is considered healthy
            elapsed_time = time.time() - self.start_time
            self.is_healthy = elapsed_time < self.initialization_grace_period
        else:
            # Error or stopped status
            self.is_healthy = False
            
        self.last_health_check = time.time()
    
    def update_metric(self, metric_name: str, value: Any):
        """
        Update a metric for the component
        
        Args:
            metric_name (str): Metric name
            value (Any): Metric value
        """
        self.metrics[metric_name] = {
            'value': value,
            'timestamp': time.time()
        }
    
    def get_uptime(self) -> float:
        """
        Get component uptime in seconds
        
        Returns:
            float: Uptime in seconds
        """
        return time.time() - self.start_time
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert component state to dictionary
        
        Returns:
            Dict[str, Any]: Component state as dictionary
        """
        return {
            'name': self.name,
            'status': self.status,
            'is_healthy': self.is_healthy,
            'last_health_check': self.last_health_check,
            'error_message': self.error_message,
            'metrics': self.metrics,
            'uptime': self.get_uptime(),
            'state_history': self.state_history[-5:]  # Last 5 state changes
        }


class StateManager:
    """
    StateManager Component
    Responsible for tracking the state of all components in the system.
    """
    
    def __init__(self):
        """Initialize the StateManager"""
        self.components: Dict[str, ComponentState] = {}
        self.alerts: List[Alert] = []
        self.global_metrics = {}
        self.alert_listeners = []
        # Dict to store component instance references
        self._component_instances = {}
        # Dict to store registered component instances
        self._registered_components = {}
        logger.info("StateManager initialized")
        
    def register_component_instance(self, component_id: str, instance: Any) -> bool:
        """
        Register a component instance for reference
        
        Args:
            component_id (str): Component identifier
            instance (Any): Component instance
            
        Returns:
            bool: True if registered successfully
        """
        self._registered_components[component_id] = instance
        # Also store by simple name for easier lookup
        simple_name = component_id.split('.')[-1] if '.' in component_id else component_id
        self._component_instances[simple_name] = instance
        logger.debug(f"Registered component instance: {component_id}")
        return True
    
    def register_component(self, name: str, component: object):
        """
        Register a component to be tracked
        
        Args:
            name (str): Component name
            component (object): Component object
        """
        if name in self.components:
            logger.warning(f"Component {name} already registered, updating reference")
        
        self.components[name] = ComponentState(name)
        logger.debug(f"Registered component: {name}")
        
    async def component_heartbeat(self, component_name: str, status: str = 'healthy', details: Optional[Dict[str, Any]] = None):
        """
        Cập nhật heartbeat cho component
        
        Args:
            component_name (str): Tên component
            status (str): Trạng thái ('healthy', 'degraded', 'error')
            details (Dict[str, Any], optional): Chi tiết bổ sung
        """
        if component_name not in self.components:
            logger.warning(f"Attempted heartbeat for unknown component: {component_name}")
            return
            
        if details is None:
            details = {}
            
        # Calculate consecutive misses
        last_heartbeat = self.components[component_name].last_health_check
        consecutive_misses = 0
        if 'consecutive_misses' in self.components[component_name].metrics:
            consecutive_misses = self.components[component_name].metrics['consecutive_misses'].get('value', 0)
            
        # Reset consecutive misses on successful heartbeat
        self.components[component_name].metrics['consecutive_misses'] = {
            'value': 0,
            'timestamp': time.time()
        }
        
        # Add CPU and memory to metrics if provided
        for key, value in details.items():
            self.components[component_name].update_metric(key, value)
        
        # Update status based on heartbeat status
        component_status = 'running'
        error_message = None
        
        if status == 'degraded':
            component_status = 'running'  # Still running but degraded
            self.create_alert(component_name, "WARNING", f"Component {component_name} is degraded: {details.get('reason', 'Unknown reason')}")
        elif status == 'error':
            component_status = 'error'
            error_message = details.get('error', 'Unknown error')
            self.create_alert(component_name, "ERROR", f"Component {component_name} reported error: {error_message}")
            
        # Update component status
        self.components[component_name].update_status(component_status, error_message)
        
        # Add heartbeat timestamp to metrics
        self.components[component_name].update_metric('last_heartbeat', time.time())
    
    def update_component_status(self, 
                               component_name: str, 
                               status: str, 
                               error_message: Optional[str] = None):
        """
        Update a component's status
        
        Args:
            component_name (str): Component name
            status (str): New status
            error_message (str, optional): Error message if status is 'error'
        """
        if component_name not in self.components:
            logger.warning(f"Attempted to update unknown component: {component_name}")
            return
        
        self.components[component_name].update_status(status, error_message)
        
        # Create alert for error states
        if status == "error":
            self.create_alert(component_name, "ERROR", 
                             f"Component {component_name} reported error: {error_message}")
        
        logger.debug(f"Updated component {component_name} status to {status}")
    
    def update_component_metric(self, component_name: str, metric_name: str, value: Any):
        """
        Update a component's metric
        
        Args:
            component_name (str): Component name
            metric_name (str): Metric name
            value (Any): Metric value
        """
        if component_name not in self.components:
            logger.warning(f"Attempted to update metrics for unknown component: {component_name}")
            return
        
        self.components[component_name].update_metric(metric_name, value)
        logger.debug(f"Updated metric {metric_name} for {component_name}: {value}")
    
    def update_global_metric(self, metric_name: str, value: Any):
        """
        Update a global system metric
        
        Args:
            metric_name (str): Metric name
            value (Any): Metric value
        """
        self.global_metrics[metric_name] = {
            'value': value,
            'timestamp': time.time()
        }
        logger.debug(f"Updated global metric {metric_name}: {value}")
    
    def create_alert(self, component: str, level: str, message: str) -> Alert:
        """
        Create a new system alert
        
        Args:
            component (str): Component that generated the alert
            level (str): Alert level
            message (str): Alert message
            
        Returns:
            Alert: The created alert object
        """
        alert = Alert(component, level, message)
        self.alerts.append(alert)
        
        # Log based on alert level
        if level == "INFO":
            logger.info(f"ALERT [{component}]: {message}")
        elif level == "WARNING":
            logger.warning(f"ALERT [{component}]: {message}")
        elif level == "ERROR":
            logger.error(f"ALERT [{component}]: {message}")
        elif level == "CRITICAL":
            logger.critical(f"ALERT [{component}]: {message}")
        
        # Notify listeners
        for listener in self.alert_listeners:
            try:
                listener(alert)
            except Exception as e:
                logger.error(f"Error notifying alert listener: {str(e)}")
        
        # Keep alerts limited to prevent memory issues
        if len(self.alerts) > 1000:
            self.alerts.pop(0)
            
        return alert
    
    def get_component(self, component_name: str) -> Optional[ComponentState]:
        """
        Get a component's state object
        
        Args:
            component_name (str): Component name
            
        Returns:
            Optional[ComponentState]: Component state object or None if not found
        """
        if component_name not in self.components:
            return None
        
        return self.components[component_name]
        
    def get_component_state(self, component_name: str) -> Optional[Dict[str, Any]]:
        """
        Get a component's state
        
        Args:
            component_name (str): Component name
            
        Returns:
            Optional[Dict[str, Any]]: Component state as dictionary or None if not found
        """
        if component_name not in self.components:
            return None
        
        return self.components[component_name].to_dict()
    
    def get_all_components(self) -> Dict[str, ComponentState]:
        """
        Get all component state objects
        
        Returns:
            Dict[str, ComponentState]: All component objects
        """
        return self.components
    
    def get_component_instance(self, component_name: str) -> Optional[Any]:
        """
        Get a component instance
        
        Args:
            component_name (str): Component name
            
        Returns:
            Optional[Any]: Component instance or None if not found
        """
        # Retrieve stored component reference if available
        if component_name in self._component_instances:
            return self._component_instances[component_name]
            
        # Otherwise look for registered components
        for component_id, component in self._registered_components.items():
            if component_id.endswith(component_name) or component_id == component_name:
                return component
                
        return None
    
    def get_component_metrics(self, component_name: str) -> Dict[str, Any]:
        """
        Get a component's metrics
        
        Args:
            component_name (str): Component name
            
        Returns:
            Dict[str, Any]: Component metrics dictionary or empty dict if not found
        """
        if component_name not in self.components:
            logger.warning(f"Attempted to get metrics for unknown component: {component_name}")
            return {}
        
        return self.components[component_name].metrics
    
    def update_component_metrics(self, component_name: str, metrics: Dict[str, Any]):
        """
        Update component metrics
        
        Args:
            component_name (str): Component name
            metrics (Dict[str, Any]): Metrics dictionary
        """
        if component_name not in self.components:
            logger.warning(f"Attempted to update metrics for unknown component: {component_name}")
            return
        
        for metric_name, value in metrics.items():
            self.components[component_name].update_metric(metric_name, value)
    
    def update_component_health_check(self, component_name: str):
        """
        Update component health check timestamp
        
        Args:
            component_name (str): Component name
        """
        if component_name not in self.components:
            logger.warning(f"Attempted to update health check for unknown component: {component_name}")
            return
        
        self.components[component_name].last_health_check = time.time()
    
    def add_alert(self, component: str, level: str, message: str) -> Alert:
        """
        Add a new alert (alias for create_alert)
        
        Args:
            component (str): Component name
            level (str): Alert level
            message (str): Alert message
            
        Returns:
            Alert: Created alert
        """
        return self.create_alert(component, level, message)
    
    def get_all_component_states(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all component states
        
        Returns:
            Dict[str, Dict[str, Any]]: All component states
        """
        return {name: component.to_dict() for name, component in self.components.items()}
    
    def get_alerts(self, 
                  component: Optional[str] = None, 
                  level: Optional[str] = None, 
                  acknowledged: Optional[bool] = None,
                  limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get filtered alerts
        
        Args:
            component (str, optional): Filter by component
            level (str, optional): Filter by level
            acknowledged (bool, optional): Filter by acknowledged status
            limit (int): Maximum number of alerts to return
            
        Returns:
            List[Dict[str, Any]]: List of filtered alerts
        """
        filtered = self.alerts
        
        if component is not None:
            filtered = [a for a in filtered if a.component == component]
        
        if level is not None:
            filtered = [a for a in filtered if a.level == level]
        
        if acknowledged is not None:
            filtered = [a for a in filtered if a.acknowledged == acknowledged]
        
        # Return most recent alerts first
        filtered.sort(key=lambda a: a.timestamp, reverse=True)
        
        return [alert.to_dict() for alert in filtered[:limit]]
    
    def acknowledge_alert(self, alert_id: str) -> bool:
        """
        Acknowledge an alert
        
        Args:
            alert_id (str): Alert ID
            
        Returns:
            bool: True if alert was found and acknowledged, False otherwise
        """
        for alert in self.alerts:
            if alert.id == alert_id:
                alert.acknowledge()
                logger.debug(f"Alert {alert_id} acknowledged")
                return True
        
        logger.warning(f"Attempted to acknowledge unknown alert: {alert_id}")
        return False
    
    def get_system_health(self) -> Dict[str, Any]:
        """
        Get overall system health status
        
        Returns:
            Dict[str, Any]: System health information
        """
        components_healthy = all(c.is_healthy for c in self.components.values())
        unhealthy_components = [name for name, c in self.components.items() if not c.is_healthy]
        
        # Count alerts by level
        alert_counts = {
            'total': len(self.alerts),
            'unacknowledged': len([a for a in self.alerts if not a.acknowledged]),
            'critical': len([a for a in self.alerts if a.level == "CRITICAL" and not a.acknowledged]),
            'error': len([a for a in self.alerts if a.level == "ERROR" and not a.acknowledged]),
            'warning': len([a for a in self.alerts if a.level == "WARNING" and not a.acknowledged]),
            'info': len([a for a in self.alerts if a.level == "INFO" and not a.acknowledged])
        }
        
        return {
            'healthy': components_healthy,
            'unhealthy_components': unhealthy_components,
            'component_count': len(self.components),
            'alerts': alert_counts,
            'global_metrics': self.global_metrics,
            'timestamp': time.time()
        }
    
    def register_alert_listener(self, listener):
        """
        Register a function to be called when new alerts are created
        
        Args:
            listener: Function that takes an Alert object as parameter
        """
        self.alert_listeners.append(listener)
        logger.debug(f"Registered new alert listener")
    
    def export_state(self) -> Dict[str, Any]:
        """
        Export complete system state
        
        Returns:
            Dict[str, Any]: Complete system state
        """
        return {
            'health': self.get_system_health(),
            'components': self.get_all_component_states(),
            'alerts': [alert.to_dict() for alert in self.alerts[-100:]],  # Last 100 alerts
            'global_metrics': self.global_metrics,
            'timestamp': time.time()
        }
    
    def save_state_to_file(self, filename: str) -> bool:
        """
        Save system state to a file
        
        Args:
            filename (str): Filename to save to
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            with open(filename, 'w') as f:
                json.dump(self.export_state(), f, indent=2)
            
            logger.debug(f"System state saved to {filename}")
            return True
        except Exception as e:
            logger.error(f"Error saving system state: {str(e)}")
            return False
    
    async def check_components_health(self):
        """
        Check the health of all components and create alerts for unhealthy ones.
        Also attempt automatic recovery for unhealthy components where possible.
        """
        now = time.time()
        unhealthy_components = []
        
        for name, component in self.components.items():
            # For components in initialization, consider them healthy during grace period
            if component.status == "initializing":
                elapsed_time = now - component.start_time
                if elapsed_time < component.initialization_grace_period:
                    component.is_healthy = True
                    continue
                elif elapsed_time >= component.initialization_grace_period:
                    # Component has been initializing for too long
                    logger.warning(f"Component {name} has been initializing for {elapsed_time:.1f}s, exceeding grace period")
            
            # Check if health check is stale based on expected interval
            if now - component.last_health_check > component.expected_health_interval:
                # Don't immediately mark as unhealthy, just warn first after 5 minutes
                if component.health_check_count > 0:  # Skip for components that have never reported
                    self.create_alert(
                        name, 
                        "WARNING", 
                        f"Component {name} health check is stale, no updates for >{component.expected_health_interval//60} minutes"
                    )
                
                # Only mark as unhealthy if it's significantly overdue (twice the expected interval)
                if now - component.last_health_check > component.expected_health_interval * 2:
                    component.is_healthy = False
                    if component.status != "stopped" and component.health_check_count > 0:
                        self.create_alert(
                            name, 
                            "ERROR", 
                            f"Component {name} is unhealthy due to missed health checks, status: {component.status}"
                        )
                        # Add to list of unhealthy components for recovery attempt
                        unhealthy_components.append(name)
            
            # Create alert for components in error state
            if component.status == "error" and not component.is_healthy:
                self.create_alert(
                    name, 
                    "ERROR", 
                    f"Component {name} is in error state: {component.error_message or 'Unknown error'}"
                )
                # Add to list of unhealthy components for recovery attempt
                unhealthy_components.append(name)
        
        # Attempt automatic recovery for unhealthy components
        if unhealthy_components:
            logger.warning(f"Attempting to recover unhealthy components: {', '.join(unhealthy_components)}")
            for component_name in unhealthy_components:
                await self.attempt_component_recovery(component_name)
        
        # Check for unusual system metrics - detect CPU, memory, or disk issues
        if 'cpu_usage' in self.global_metrics and self.global_metrics['cpu_usage']['value'] > 90:
            self.create_alert(
                'system', 
                'WARNING', 
                f"High CPU usage: {self.global_metrics['cpu_usage']['value']}%"
            )
            
        if 'memory_usage' in self.global_metrics and self.global_metrics['memory_usage']['value'] > 90:
            self.create_alert(
                'system', 
                'WARNING', 
                f"High memory usage: {self.global_metrics['memory_usage']['value']}%"
            )
            
        if 'disk_usage' in self.global_metrics and self.global_metrics['disk_usage']['value'] > 90:
            self.create_alert(
                'system', 
                'WARNING', 
                f"High disk usage: {self.global_metrics['disk_usage']['value']}%"
            )
    
    async def attempt_component_recovery(self, component_name: str) -> bool:
        """
        Attempt to automatically recover an unhealthy component
        
        Args:
            component_name (str): Name of the component to recover
            
        Returns:
            bool: True if recovery was attempted, False otherwise
        """
        if component_name not in self.components:
            logger.warning(f"Cannot recover unknown component: {component_name}")
            return False
            
        component = self.components[component_name]
        component_instance = self.get_component_instance(component_name)
        
        # If we don't have a reference to the component instance, we can't restart it
        if not component_instance:
            logger.warning(f"No instance reference for component {component_name}, cannot attempt recovery")
            return False
            
        # Only attempt recovery for certain critical components
        recovery_candidates = [
            'pump_portal_client', 
            'onchain_analyzer', 
            'trading_integration',
            'connection_pool'
        ]
        
        if component_name not in recovery_candidates:
            logger.debug(f"Component {component_name} is not eligible for automatic recovery")
            return False
            
        try:
            # Kiểm tra xem component có phương thức restart không
            has_restart = hasattr(component_instance, 'restart') and callable(getattr(component_instance, 'restart'))
            
            # Hoặc phương thức stop và start
            has_stop_start = (hasattr(component_instance, 'stop') and callable(getattr(component_instance, 'stop')) and
                             hasattr(component_instance, 'start') and callable(getattr(component_instance, 'start')))
            
            if has_restart:
                logger.info(f"Attempting to restart component {component_name}")
                restart_method = getattr(component_instance, 'restart')
                # Kiểm tra xem restart là hàm đồng bộ hay bất đồng bộ
                if asyncio.iscoroutinefunction(restart_method):
                    await restart_method()
                else:
                    restart_method()
                    
                # Cập nhật trạng thái
                self.update_component_status(component_name, 'initializing', "Auto-recovery initiated")
                self.create_alert(
                    component_name,
                    "INFO",
                    f"Auto-recovery initiated for component {component_name}"
                )
                
                return True
                
            elif has_stop_start:
                logger.info(f"Attempting stop/start recovery for component {component_name}")
                stop_method = getattr(component_instance, 'stop')
                start_method = getattr(component_instance, 'start')
                
                # Stop component
                if asyncio.iscoroutinefunction(stop_method):
                    await stop_method()
                else:
                    stop_method()
                    
                # Short delay between stop and start
                await asyncio.sleep(2)
                
                # Start component
                if asyncio.iscoroutinefunction(start_method):
                    await start_method()
                else:
                    start_method()
                
                # Cập nhật trạng thái
                self.update_component_status(component_name, 'initializing', "Auto-recovery initiated")
                self.create_alert(
                    component_name,
                    "INFO",
                    f"Auto-recovery initiated for component {component_name} with stop/start sequence"
                )
                
                return True
                
            else:
                logger.warning(f"Component {component_name} doesn't have restart or stop/start methods")
                return False
                
        except Exception as e:
            logger.error(f"Error during recovery attempt for {component_name}: {str(e)}")
            self.create_alert(
                component_name,
                "ERROR",
                f"Auto-recovery failed for component {component_name}: {str(e)}"
            )
            return False
