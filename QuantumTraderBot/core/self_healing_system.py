"""
Self-Healing System Component
Responsible for automatically detecting and resolving issues in the bot,
improving resilience and reducing downtime.
"""

import asyncio
import logging
import time
import traceback
from typing import Dict, List, Any, Optional, Tuple, Callable
import random

from core.state_manager import StateManager
from core.config_manager import ConfigManager

logger = logging.getLogger(__name__)

class DiagnosticEngine:
    """Performs system diagnostics and issue detection"""
    
    def __init__(self, state_manager: StateManager):
        """
        Initialize the DiagnosticEngine
        
        Args:
            state_manager (StateManager): State manager instance
        """
        self.state_manager = state_manager
        
        # Define health checks
        self.health_checks = {
            'component_status': self._check_component_status,
            'resource_usage': self._check_resource_usage,
            'connection_status': self._check_connection_status,
            'error_rate': self._check_error_rate,
            'heartbeat': self._check_heartbeat
        }
    
    async def check_system_health(self) -> Dict[str, Any]:
        """
        Check the health of the entire system
        
        Returns:
            Dict[str, Any]: Health check results
        """
        health_results = {
            'timestamp': time.time(),
            'check_results': {},
            'issues': []
        }
        
        # Run all health checks
        for check_name, check_func in self.health_checks.items():
            try:
                check_result = await check_func()
                health_results['check_results'][check_name] = check_result
                
                # Add issues if check failed
                if check_result['status'] == 'failed':
                    for issue in check_result.get('issues', []):
                        health_results['issues'].append({
                            'check': check_name,
                            'component': issue.get('component', 'unknown'),
                            'type': issue.get('type', 'unknown'),
                            'details': issue.get('details', ''),
                            'severity': issue.get('severity', 'medium'),
                            'timestamp': time.time()
                        })
                        
            except Exception as e:
                logger.error(f"Error in health check {check_name}: {str(e)}")
                health_results['check_results'][check_name] = {
                    'status': 'error',
                    'error': str(e)
                }
        
        # Calculate overall health
        issue_count = len(health_results['issues'])
        critical_issues = sum(1 for i in health_results['issues'] if i['severity'] == 'critical')
        
        if critical_issues > 0:
            health_results['overall_status'] = 'critical'
        elif issue_count > 3:
            health_results['overall_status'] = 'unhealthy'
        elif issue_count > 0:
            health_results['overall_status'] = 'degraded'
        else:
            health_results['overall_status'] = 'healthy'
            
        health_results['issue_count'] = issue_count
        health_results['critical_issues'] = critical_issues
        
        return health_results
    
    async def _check_component_status(self) -> Dict[str, Any]:
        """
        Check the status of all components
        
        Returns:
            Dict[str, Any]: Check result
        """
        result = {
            'status': 'passed',
            'issues': []
        }
        
        # Get all components
        components = self.state_manager.get_all_components()
        
        for component_name, component_state in components.items():
            # Check if component is in error state
            if component_state.status == 'error':
                result['status'] = 'failed'
                result['issues'].append({
                    'component': component_name,
                    'type': 'component_error',
                    'details': component_state.error_message or 'Unknown error',
                    'severity': 'high'
                })
            # Check if component is stalled
            elif component_state.status == 'running' and not component_state.is_healthy:
                result['status'] = 'failed'
                result['issues'].append({
                    'component': component_name,
                    'type': 'component_stalled',
                    'details': f"Component appears stalled, last health check: {component_state.last_health_check}",
                    'severity': 'medium'
                })
                
        result['component_count'] = len(components)
        result['error_components'] = len([c for c in components.values() if c.status == 'error'])
        result['unhealthy_components'] = len([c for c in components.values() if not c.is_healthy])
        
        return result
    
    async def _check_resource_usage(self) -> Dict[str, Any]:
        """
        Check system resource usage
        
        Returns:
            Dict[str, Any]: Check result
        """
        result = {
            'status': 'passed',
            'issues': []
        }
        
        # Get memory metrics if available
        memory_component = self.state_manager.get_component('memory_manager')
        if memory_component:
            memory_metrics = memory_component.metrics.get('memory_usage', {})
            
            # Check for high memory usage
            if memory_metrics.get('percent', 0) > 90:
                result['status'] = 'failed'
                result['issues'].append({
                    'component': 'memory_manager',
                    'type': 'high_memory_usage',
                    'details': f"Memory usage at {memory_metrics.get('percent', 0)}%",
                    'severity': 'high'
                })
            elif memory_metrics.get('percent', 0) > 80:
                result['status'] = 'failed'
                result['issues'].append({
                    'component': 'memory_manager',
                    'type': 'elevated_memory_usage',
                    'details': f"Memory usage at {memory_metrics.get('percent', 0)}%",
                    'severity': 'medium'
                })
                
            result['memory_usage'] = memory_metrics
        
        # In production, would check CPU, disk, network, etc.
        # For demonstration, just use memory
        
        return result
    
    async def _check_connection_status(self) -> Dict[str, Any]:
        """
        Check connection status to external services
        
        Returns:
            Dict[str, Any]: Check result
        """
        result = {
            'status': 'passed',
            'issues': []
        }
        
        # Check API client connections
        components_to_check = ['pump_portal_client', 'raydium_client']
        for component_name in components_to_check:
            component = self.state_manager.get_component(component_name)
            if not component:
                continue
                
            # Check if connected
            if component.status != 'running' or not component.metrics.get('connected', False):
                result['status'] = 'failed'
                result['issues'].append({
                    'component': component_name,
                    'type': 'connection_failure',
                    'details': f"Component not connected: {component.error_message or 'Unknown error'}",
                    'severity': 'high' if component_name == 'pump_portal_client' else 'medium'
                })
        
        return result
    
    async def _check_error_rate(self) -> Dict[str, Any]:
        """
        Check error rates in the system
        
        Returns:
            Dict[str, Any]: Check result
        """
        result = {
            'status': 'passed',
            'issues': []
        }
        
        # Get all components
        components = self.state_manager.get_all_components()
        
        for component_name, component_state in components.items():
            # Check error rates in metrics
            error_rate = component_state.metrics.get('error_rate', 0)
            if error_rate > 50:  # Over 50% error rate
                result['status'] = 'failed'
                result['issues'].append({
                    'component': component_name,
                    'type': 'high_error_rate',
                    'details': f"High error rate: {error_rate}%",
                    'severity': 'high'
                })
            elif error_rate > 20:  # Over 20% error rate
                result['status'] = 'failed'
                result['issues'].append({
                    'component': component_name,
                    'type': 'elevated_error_rate',
                    'details': f"Elevated error rate: {error_rate}%",
                    'severity': 'medium'
                })
        
        return result
    
    async def _check_heartbeat(self) -> Dict[str, Any]:
        """
        Check heartbeat of all components
        
        Returns:
            Dict[str, Any]: Check result
        """
        result = {
            'status': 'passed',
            'issues': []
        }
        
        # Get all components
        components = self.state_manager.get_all_components()
        current_time = time.time()
        
        for component_name, component_state in components.items():
            # Skip stopped components
            if component_state.status == 'stopped':
                continue
                
            # Check time since last health check
            time_since_last = current_time - component_state.last_health_check
            
            # Get expected interval from component state
            expected_interval = component_state.expected_health_interval
            
            # Account for initialization grace period
            grace_period = component_state.initialization_grace_period
            if (current_time - component_state.start_time) < grace_period:
                # Component still in grace period, skip heartbeat check
                continue
            
            # Check if heartbeat is missing
            if time_since_last > expected_interval * 3:  # 3x the expected interval
                result['status'] = 'failed'
                result['issues'].append({
                    'component': component_name,
                    'type': 'missing_heartbeat',
                    'details': f"No heartbeat for {time_since_last:.1f}s (expected interval: {expected_interval}s)",
                    'severity': 'high'
                })
            elif time_since_last > expected_interval * 2:  # 2x the expected interval
                result['status'] = 'failed'
                result['issues'].append({
                    'component': component_name,
                    'type': 'delayed_heartbeat',
                    'details': f"Delayed heartbeat by {time_since_last - expected_interval:.1f}s (expected interval: {expected_interval}s)",
                    'severity': 'medium'
                })
        
        return result


class RepairEngine:
    """Performs repairs for detected issues"""
    
    def __init__(self, 
                state_manager: StateManager,
                config_manager: ConfigManager):
        """
        Initialize the RepairEngine
        
        Args:
            state_manager (StateManager): State manager instance
            config_manager (ConfigManager): Configuration manager instance
        """
        self.state_manager = state_manager
        self.config_manager = config_manager
        
        # Define repair strategies
        self.repair_strategies = {
            'component_error': self._repair_component_error,
            'component_stalled': self._repair_component_stalled,
            'high_memory_usage': self._repair_high_memory_usage,
            'connection_failure': self._repair_connection_failure,
            'missing_heartbeat': self._repair_missing_heartbeat,
            'delayed_heartbeat': self._repair_delayed_heartbeat
        }
        
        # Track repair history
        self.repair_history = []
        self.max_history = 100
        
        # Repair limits to prevent repair loops
        self.repair_limits = {}
        self.max_repairs_per_component = 5
        self.repair_cooldown = 300  # 5 minutes
        
        logger.info("RepairEngine initialized")
    
    def create_repair_plan(self, issue: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a repair plan for an issue
        
        Args:
            issue (Dict[str, Any]): Issue details
            
        Returns:
            Dict[str, Any]: Repair plan
        """
        issue_type = issue.get('type', 'unknown')
        component = issue.get('component', 'unknown')
        
        # Check repair limits
        if not self._check_repair_limits(component, issue_type):
            logger.warning(f"Repair limit reached for {component} ({issue_type})")
            return {
                'issue': issue,
                'plan': 'none',
                'reason': 'repair_limit_reached',
                'can_repair': False
            }
        
        # Get repair strategy
        repair_func = self.repair_strategies.get(issue_type)
        if not repair_func:
            return {
                'issue': issue,
                'plan': 'none',
                'reason': 'no_repair_strategy',
                'can_repair': False
            }
        
        # Create repair plan
        repair_plan = {
            'issue': issue,
            'plan': issue_type,
            'component': component,
            'timestamp': time.time(),
            'can_repair': True,
            'repair_function': repair_func
        }
        
        return repair_plan
    
    async def execute_repair(self, repair_plan: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a repair plan
        
        Args:
            repair_plan (Dict[str, Any]): Repair plan
            
        Returns:
            Dict[str, Any]: Repair result
        """
        if not repair_plan.get('can_repair', False):
            logger.warning(f"Cannot execute repair: {repair_plan.get('reason', 'unknown')}")
            return {
                'success': False,
                'repair_plan': repair_plan,
                'reason': repair_plan.get('reason', 'unknown'),
                'timestamp': time.time()
            }
        
        component = repair_plan.get('component', 'unknown')
        issue_type = repair_plan.get('plan', 'unknown')
        repair_func = repair_plan.get('repair_function')
        
        if not repair_func:
            logger.error(f"No repair function in plan for {component} ({issue_type})")
            return {
                'success': False,
                'repair_plan': repair_plan,
                'reason': 'no_repair_function',
                'timestamp': time.time()
            }
        
        try:
            # Execute repair function
            repair_result = await repair_func(repair_plan)
            
            # Update repair history
            self._update_repair_history(component, issue_type, repair_result)
            
            return {
                'success': repair_result.get('success', False),
                'component': component,
                'issue_type': issue_type,
                'details': repair_result.get('details', ''),
                'actions_taken': repair_result.get('actions', []),
                'timestamp': time.time()
            }
            
        except Exception as e:
            logger.error(f"Error executing repair for {component} ({issue_type}): {str(e)}")
            
            # Update repair history as failed
            self._update_repair_history(component, issue_type, {'success': False, 'error': str(e)})
            
            return {
                'success': False,
                'component': component,
                'issue_type': issue_type,
                'error': str(e),
                'traceback': traceback.format_exc(),
                'timestamp': time.time()
            }
    
    def _check_repair_limits(self, component: str, issue_type: str) -> bool:
        """
        Check if repair should be allowed based on limits
        
        Args:
            component (str): Component name
            issue_type (str): Issue type
            
        Returns:
            bool: True if repair is allowed, False otherwise
        """
        # Create component key
        component_key = f"{component}_{issue_type}"
        
        # Initialize if not exists
        if component_key not in self.repair_limits:
            self.repair_limits[component_key] = {
                'count': 0,
                'last_repair': 0
            }
        
        limits = self.repair_limits[component_key]
        current_time = time.time()
        
        # Check cooldown
        if (current_time - limits['last_repair']) < self.repair_cooldown:
            # Only if we've already tried to repair recently
            if limits['count'] > 0:
                return False
        
        # Check count
        if limits['count'] >= self.max_repairs_per_component:
            # Reset count if it's been a long time since last repair
            if (current_time - limits['last_repair']) > self.repair_cooldown * 5:
                limits['count'] = 0
            else:
                return False
        
        return True
    
    def _update_repair_history(self, 
                             component: str, 
                             issue_type: str, 
                             result: Dict[str, Any]):
        """
        Update repair history and limits
        
        Args:
            component (str): Component name
            issue_type (str): Issue type
            result (Dict[str, Any]): Repair result
        """
        # Update repair limits
        component_key = f"{component}_{issue_type}"
        if component_key not in self.repair_limits:
            self.repair_limits[component_key] = {
                'count': 0,
                'last_repair': 0
            }
        
        self.repair_limits[component_key]['count'] += 1
        self.repair_limits[component_key]['last_repair'] = time.time()
        
        # Add to history
        history_entry = {
            'component': component,
            'issue_type': issue_type,
            'success': result.get('success', False),
            'timestamp': time.time(),
            'details': result.get('details', ''),
            'actions': result.get('actions', [])
        }
        
        self.repair_history.append(history_entry)
        
        # Trim history if needed
        if len(self.repair_history) > self.max_history:
            self.repair_history = self.repair_history[-self.max_history:]
        
        # Update state manager with repair metrics
        component_state = self.state_manager.get_component(component)
        if component_state:
            # Update repair count in metrics
            metrics = component_state.metrics.copy()
            repair_count = metrics.get('repair_count', 0) + 1
            metrics['repair_count'] = repair_count
            metrics['last_repair'] = time.time()
            metrics['last_repair_success'] = result.get('success', False)
            
            # Update state manager
            self.state_manager.update_component_metrics(component, metrics)
    
    async def _repair_component_error(self, repair_plan: Dict[str, Any]) -> Dict[str, Any]:
        """
        Repair a component in error state
        
        Args:
            repair_plan (Dict[str, Any]): Repair plan
            
        Returns:
            Dict[str, Any]: Repair result
        """
        component = repair_plan.get('component')
        issue = repair_plan.get('issue', {})
        
        # Get component from state manager
        component_state = self.state_manager.get_component(component)
        if not component_state:
            return {
                'success': False,
                'details': f"Component {component} not found",
                'actions': []
            }
        
        # Get component instance
        component_instance = self.state_manager.get_component_instance(component)
        
        actions = []
        
        # Check if component has a reset method
        if hasattr(component_instance, 'reset'):
            try:
                # Call reset method
                await component_instance.reset()
                actions.append(f"Called reset() on {component}")
            except Exception as e:
                logger.error(f"Error resetting {component}: {str(e)}")
                return {
                    'success': False,
                    'details': f"Error resetting component: {str(e)}",
                    'actions': actions
                }
        
        # If component has restart method, try that
        if hasattr(component_instance, 'restart'):
            try:
                # Call restart method
                await component_instance.restart()
                actions.append(f"Called restart() on {component}")
            except Exception as e:
                logger.error(f"Error restarting {component}: {str(e)}")
                return {
                    'success': False,
                    'details': f"Error restarting component: {str(e)}",
                    'actions': actions
                }
        
        # If no special methods, try to stop and start again
        elif hasattr(component_instance, 'stop') and hasattr(component_instance, 'start'):
            try:
                # Stop component
                await component_instance.stop()
                actions.append(f"Called stop() on {component}")
                
                # Small delay
                await asyncio.sleep(1)
                
                # Start component
                await component_instance.start()
                actions.append(f"Called start() on {component}")
            except Exception as e:
                logger.error(f"Error restarting {component}: {str(e)}")
                return {
                    'success': False,
                    'details': f"Error restarting component: {str(e)}",
                    'actions': actions
                }
        
        # Update state
        self.state_manager.update_component_status(
            component, 
            'running', 
            'Reset by self-healing system'
        )
        
        return {
            'success': True,
            'details': f"Component {component} restarted successfully",
            'actions': actions
        }
    
    async def _repair_component_stalled(self, repair_plan: Dict[str, Any]) -> Dict[str, Any]:
        """
        Repair a stalled component
        
        Args:
            repair_plan (Dict[str, Any]): Repair plan
            
        Returns:
            Dict[str, Any]: Repair result
        """
        # For stalled components, use the same repair strategy as for errors
        return await self._repair_component_error(repair_plan)
    
    async def _repair_high_memory_usage(self, repair_plan: Dict[str, Any]) -> Dict[str, Any]:
        """
        Repair high memory usage
        
        Args:
            repair_plan (Dict[str, Any]): Repair plan
            
        Returns:
            Dict[str, Any]: Repair result
        """
        actions = []
        
        # Find memory manager
        memory_manager = self.state_manager.get_component_instance('memory_manager')
        
        if memory_manager and hasattr(memory_manager, 'cleanup'):
            try:
                # Force memory cleanup
                cleanup_result = await memory_manager.cleanup()
                actions.append("Triggered memory cleanup")
                
                # Force garbage collection
                if hasattr(memory_manager, 'perform_gc'):
                    gc_result = await memory_manager.perform_gc(force=True)
                    actions.append("Triggered forced garbage collection")
                
                return {
                    'success': True,
                    'details': "Memory cleanup performed",
                    'actions': actions
                }
            except Exception as e:
                logger.error(f"Error performing memory cleanup: {str(e)}")
                return {
                    'success': False,
                    'details': f"Error performing memory cleanup: {str(e)}",
                    'actions': actions
                }
        else:
            return {
                'success': False,
                'details': "Memory manager not available or does not support cleanup",
                'actions': []
            }
    
    async def _repair_connection_failure(self, repair_plan: Dict[str, Any]) -> Dict[str, Any]:
        """
        Repair connection failure
        
        Args:
            repair_plan (Dict[str, Any]): Repair plan
            
        Returns:
            Dict[str, Any]: Repair result
        """
        component = repair_plan.get('component')
        
        # Get component instance
        component_instance = self.state_manager.get_component_instance(component)
        if not component_instance:
            return {
                'success': False,
                'details': f"Component {component} not found",
                'actions': []
            }
        
        actions = []
        
        # Check for reconnect method
        if hasattr(component_instance, 'reconnect'):
            try:
                # Call reconnect method
                await component_instance.reconnect()
                actions.append(f"Called reconnect() on {component}")
                
                # Wait a bit to see if connection is established
                await asyncio.sleep(2)
                
                # Check if reconnect was successful
                if hasattr(component_instance, 'is_connected') and component_instance.is_connected:
                    return {
                        'success': True,
                        'details': f"Successfully reconnected {component}",
                        'actions': actions
                    }
                else:
                    # Try a second time
                    await component_instance.reconnect()
                    actions.append(f"Called reconnect() on {component} (second attempt)")
                    
                    await asyncio.sleep(3)
                    
                    if hasattr(component_instance, 'is_connected') and component_instance.is_connected:
                        return {
                            'success': True,
                            'details': f"Successfully reconnected {component} on second attempt",
                            'actions': actions
                        }
                    else:
                        return {
                            'success': False,
                            'details': f"Failed to reconnect {component} after multiple attempts",
                            'actions': actions
                        }
            except Exception as e:
                logger.error(f"Error reconnecting {component}: {str(e)}")
                return {
                    'success': False,
                    'details': f"Error reconnecting {component}: {str(e)}",
                    'actions': actions
                }
        else:
            # Fall back to restart
            return await self._repair_component_error(repair_plan)
    
    async def _repair_missing_heartbeat(self, repair_plan: Dict[str, Any]) -> Dict[str, Any]:
        """
        Repair missing heartbeat
        
        Args:
            repair_plan (Dict[str, Any]): Repair plan
            
        Returns:
            Dict[str, Any]: Repair result
        """
        # For missing heartbeat, use the same repair strategy as for errors
        # This is typically a sign that the component is frozen
        return await self._repair_component_error(repair_plan)
    
    async def _repair_delayed_heartbeat(self, repair_plan: Dict[str, Any]) -> Dict[str, Any]:
        """
        Repair delayed heartbeat
        
        Args:
            repair_plan (Dict[str, Any]): Repair plan
            
        Returns:
            Dict[str, Any]: Repair result
        """
        component = repair_plan.get('component')
        
        # Get component state
        component_state = self.state_manager.get_component(component)
        if not component_state:
            return {
                'success': False,
                'details': f"Component {component} not found",
                'actions': []
            }
        
        # For delayed heartbeat, just update the health check timestamp
        # This gives the component a chance to recover on its own
        self.state_manager.update_component_health_check(component)
        
        return {
            'success': True,
            'details': f"Reset health check timestamp for {component}, allowing time to recover",
            'actions': [f"Reset health check timestamp for {component}"]
        }


class SelfHealingSystem:
    """
    Self-healing system for automatically detecting and fixing issues
    
    This component monitors the health of the system and attempts to
    repair issues automatically, improving resilience and reducing downtime.
    """
    
    def __init__(self, 
                state_manager: StateManager,
                config_manager: ConfigManager):
        """
        Initialize the SelfHealingSystem
        
        Args:
            state_manager (StateManager): State manager instance
            config_manager (ConfigManager): Configuration manager instance
        """
        self.state_manager = state_manager
        self.config_manager = config_manager
        
        # Load configuration
        self.healing_enabled = self.config_manager.get('self_healing.enabled', True)
        self.check_interval = self.config_manager.get('self_healing.check_interval', 60)
        self.healing_mode = self.config_manager.get('self_healing.mode', 'auto')
        
        # Initialize components
        self.diagnostic_engine = DiagnosticEngine(state_manager)
        self.repair_engine = RepairEngine(state_manager, config_manager)
        
        # Task management
        self.monitoring_task = None
        self.is_running = False
        
        logger.info(f"SelfHealingSystem initialized (enabled: {self.healing_enabled}, mode: {self.healing_mode})")
    
    async def start(self):
        """Start the self-healing system"""
        if self.is_running:
            logger.warning("SelfHealingSystem is already running")
            return
        
        self.is_running = True
        
        # Start monitoring task
        if self.healing_enabled:
            self.monitoring_task = asyncio.create_task(self.monitor_and_heal())
            
        logger.info("SelfHealingSystem started")
        
        # Update component status
        self.state_manager.update_component_status(
            'self_healing_system', 
            'running',
            f"Mode: {self.healing_mode}, Check interval: {self.check_interval}s"
        )
    
    async def stop(self):
        """Stop the self-healing system"""
        if not self.is_running:
            return
            
        self.is_running = False
        
        # Cancel monitoring task
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
            self.monitoring_task = None
            
        logger.info("SelfHealingSystem stopped")
        
        # Update component status
        self.state_manager.update_component_status(
            'self_healing_system', 
            'stopped'
        )
    
    async def monitor_and_heal(self):
        """Monitor the system and automatically heal issues"""
        logger.info(f"Starting monitoring and healing loop (interval: {self.check_interval}s)")
        
        while self.is_running:
            try:
                # Check system health
                health_check = await self.diagnostic_engine.check_system_health()
                
                # Update state
                self.state_manager.update_component_metrics(
                    'self_healing_system',
                    {
                        'last_check': time.time(),
                        'overall_status': health_check['overall_status'],
                        'issue_count': health_check['issue_count'],
                        'critical_issues': health_check['critical_issues']
                    }
                )
                
                # Handle issues if any
                if health_check['issues']:
                    await self._handle_issues(health_check['issues'])
                    
            except Exception as e:
                logger.error(f"Error in monitoring loop: {str(e)}")
                
            # Sleep until next check
            await asyncio.sleep(self.check_interval)
    
    async def _handle_issues(self, issues: List[Dict[str, Any]]):
        """
        Handle detected issues
        
        Args:
            issues (List[Dict[str, Any]]): List of detected issues
        """
        logger.info(f"Handling {len(issues)} issues")
        
        # Process each issue
        for issue in issues:
            issue_type = issue.get('type', 'unknown')
            component = issue.get('component', 'unknown')
            severity = issue.get('severity', 'medium')
            
            logger.info(f"Processing issue: {issue_type} in {component} (severity: {severity})")
            
            # Skip if healing is not automatic and issue is not critical
            if self.healing_mode != 'auto' and severity != 'critical':
                logger.info(f"Skipping non-critical issue in {self.healing_mode} mode")
                continue
            
            # Create repair plan
            repair_plan = self.repair_engine.create_repair_plan(issue)
            
            # Skip if plan says we cannot repair
            if not repair_plan.get('can_repair', False):
                logger.warning(f"Cannot repair issue: {repair_plan.get('reason', 'unknown')}")
                continue
            
            # Execute repair
            logger.info(f"Executing repair plan for {component} ({issue_type})")
            repair_result = await self.repair_engine.execute_repair(repair_plan)
            
            # Log result
            if repair_result.get('success', False):
                logger.info(f"Repair successful for {component} ({issue_type})")
            else:
                logger.warning(f"Repair failed for {component} ({issue_type}): {repair_result.get('reason', 'unknown')}")
                
            # Brief pause between repairs
            await asyncio.sleep(2)
    
    async def run_health_check(self) -> Dict[str, Any]:
        """
        Run a system health check manually
        
        Returns:
            Dict[str, Any]: Health check results
        """
        return await self.diagnostic_engine.check_system_health()
    
    async def repair_issue(self, 
                         component: str, 
                         issue_type: str) -> Dict[str, Any]:
        """
        Repair a specific issue manually
        
        Args:
            component (str): Component name
            issue_type (str): Issue type
            
        Returns:
            Dict[str, Any]: Repair result
        """
        # Create artificial issue
        issue = {
            'component': component,
            'type': issue_type,
            'details': 'Manually triggered repair',
            'severity': 'medium'
        }
        
        # Create repair plan
        repair_plan = self.repair_engine.create_repair_plan(issue)
        
        if not repair_plan.get('can_repair', False):
            return {
                'success': False,
                'component': component,
                'issue_type': issue_type,
                'reason': repair_plan.get('reason', 'unknown'),
                'message': f"Cannot repair {issue_type} for {component}: {repair_plan.get('reason', 'unknown')}"
            }
        
        # Execute repair
        repair_result = await self.repair_engine.execute_repair(repair_plan)
        
        return repair_result
    
    def get_repair_history(self) -> List[Dict[str, Any]]:
        """
        Get repair history
        
        Returns:
            List[Dict[str, Any]]: Repair history
        """
        return self.repair_engine.repair_history