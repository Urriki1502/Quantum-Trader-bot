"""
Health Check Component
Performs comprehensive health checks on all components and dependencies
to ensure the trading bot is functioning properly.
"""

import asyncio
import logging
import time
import json
import os
from typing import Dict, List, Any, Optional, Set, Tuple

from core.state_manager import StateManager
from core.config_manager import ConfigManager

logger = logging.getLogger(__name__)

class HealthCheck:
    """
    Performs health checks on all components and system dependencies
    
    This component helps identify and troubleshoot potential issues
    before real trading begins, ensuring system stability and reliability.
    """
    
    def __init__(self, 
                config_manager: ConfigManager,
                state_manager: StateManager):
        """
        Initialize health check
        
        Args:
            config_manager (ConfigManager): Configuration manager instance
            state_manager (StateManager): State manager instance
        """
        self.config_manager = config_manager
        self.state_manager = state_manager
        
        # Health check settings
        self.log_directory = 'logs'
        self.health_check_log = os.path.join(self.log_directory, 'health_check.log')
        
        # Ensure log directory exists
        os.makedirs(self.log_directory, exist_ok=True)
        
        logger.info("HealthCheck initialized")
    
    async def run_comprehensive_check(self) -> Dict[str, Any]:
        """
        Run a comprehensive health check on all components
        
        Returns:
            Dict[str, Any]: Health check results
        """
        start_time = time.time()
        
        results = {
            'timestamp': start_time,
            'overall_status': 'healthy',
            'components': {},
            'dependencies': {},
            'issues': [],
            'recommendations': []
        }
        
        # Check components
        components_result = await self._check_components()
        results['components'] = components_result['components']
        
        if components_result['issues']:
            results['issues'].extend(components_result['issues'])
            
        # Check dependencies
        dependencies_result = await self._check_dependencies()
        results['dependencies'] = dependencies_result['dependencies']
        
        if dependencies_result['issues']:
            results['issues'].extend(dependencies_result['issues'])
        
        # Check connections
        connections_result = await self._check_connections()
        results['connections'] = connections_result['connections']
        
        if connections_result['issues']:
            results['issues'].extend(connections_result['issues'])
        
        # Check configuration
        config_result = await self._check_configuration()
        results['configuration'] = config_result['configuration']
        
        if config_result['issues']:
            results['issues'].extend(config_result['issues'])
        
        # Generate recommendations
        if results['issues']:
            results['recommendations'] = self._generate_recommendations(results['issues'])
            results['overall_status'] = 'degraded' if len(results['issues']) < 3 else 'unhealthy'
        
        # Calculate execution time
        results['execution_time_ms'] = (time.time() - start_time) * 1000
        
        # Save results to log
        self._save_results(results)
        
        return results
    
    async def _check_components(self) -> Dict[str, Any]:
        """
        Check all components
        
        Returns:
            Dict[str, Any]: Component check results
        """
        result = {
            'components': {},
            'issues': []
        }
        
        # Get all components from state manager
        components = self.state_manager.get_all_components()
        
        for component_name, component_state in components.items():
            # Check component status
            component_result = {
                'status': component_state.status,
                'is_healthy': component_state.is_healthy,
                'last_health_check': component_state.last_health_check,
                'error_message': component_state.error_message,
                'metrics': component_state.metrics
            }
            
            result['components'][component_name] = component_result
            
            # Check for issues
            if component_state.status != 'running':
                result['issues'].append({
                    'component': component_name,
                    'type': 'component_not_running',
                    'description': f"Component {component_name} is not running",
                    'severity': 'high'
                })
            elif not component_state.is_healthy:
                result['issues'].append({
                    'component': component_name,
                    'type': 'component_unhealthy',
                    'description': f"Component {component_name} is not healthy",
                    'severity': 'medium'
                })
            elif component_state.error_message:
                result['issues'].append({
                    'component': component_name,
                    'type': 'component_error',
                    'description': f"Component {component_name} has error: {component_state.error_message}",
                    'severity': 'medium'
                })
        
        return result
    
    async def _check_dependencies(self) -> Dict[str, Any]:
        """
        Check all dependencies
        
        Returns:
            Dict[str, Any]: Dependency check results
        """
        result = {
            'dependencies': {},
            'issues': []
        }
        
        # Check Python dependencies
        try:
            # Check key dependencies
            import solana
            result['dependencies']['solana'] = {'version': solana.__version__, 'status': 'installed'}
            
            import flask
            result['dependencies']['flask'] = {'version': flask.__version__, 'status': 'installed'}
            
            import sqlalchemy
            result['dependencies']['sqlalchemy'] = {'version': sqlalchemy.__version__, 'status': 'installed'}
            
            import aiohttp
            result['dependencies']['aiohttp'] = {'version': aiohttp.__version__, 'status': 'installed'}
            
        except ImportError as e:
            missing_dependency = str(e).split("'")[1] if "'" in str(e) else str(e)
            result['issues'].append({
                'component': 'dependencies',
                'type': 'missing_dependency',
                'description': f"Missing dependency: {missing_dependency}",
                'severity': 'high'
            })
        
        # Check external dependencies
        # Database
        try:
            result['dependencies']['database'] = {'status': 'available'}
            
            # Check environment variables
            if not os.environ.get('DATABASE_URL'):
                result['dependencies']['database']['status'] = 'no_connection_string'
                result['issues'].append({
                    'component': 'dependencies',
                    'type': 'missing_database_url',
                    'description': "DATABASE_URL environment variable not set",
                    'severity': 'high'
                })
        except Exception as e:
            result['dependencies']['database'] = {'status': 'error', 'error': str(e)}
            result['issues'].append({
                'component': 'dependencies',
                'type': 'database_error',
                'description': f"Database error: {str(e)}",
                'severity': 'high'
            })
        
        # Check Telegram
        try:
            telegram_token = os.environ.get('TELEGRAM_BOT_TOKEN')
            if telegram_token:
                result['dependencies']['telegram'] = {'status': 'token_available'}
            else:
                result['dependencies']['telegram'] = {'status': 'no_token'}
                result['issues'].append({
                    'component': 'dependencies',
                    'type': 'missing_telegram_token',
                    'description': "TELEGRAM_BOT_TOKEN environment variable not set",
                    'severity': 'medium'
                })
        except Exception as e:
            result['dependencies']['telegram'] = {'status': 'error', 'error': str(e)}
        
        return result
    
    async def _check_connections(self) -> Dict[str, Any]:
        """
        Check external connections
        
        Returns:
            Dict[str, Any]: Connection check results
        """
        result = {
            'connections': {},
            'issues': []
        }
        
        # Check Solana RPC connection
        try:
            import solana
            from solana.rpc.api import Client
            
            rpc_url = os.environ.get('SOLANA_RPC_URL')
            if not rpc_url:
                result['connections']['solana_rpc'] = {'status': 'no_url'}
                result['issues'].append({
                    'component': 'connections',
                    'type': 'missing_solana_rpc_url',
                    'description': "SOLANA_RPC_URL environment variable not set",
                    'severity': 'high'
                })
            else:
                # Create client
                client = Client(rpc_url)
                
                # Test connection
                response = client.get_health()
                
                if response['result'] == 'ok':
                    result['connections']['solana_rpc'] = {'status': 'connected', 'url': rpc_url}
                else:
                    result['connections']['solana_rpc'] = {'status': 'error', 'response': response}
                    result['issues'].append({
                        'component': 'connections',
                        'type': 'solana_rpc_error',
                        'description': f"Solana RPC error: {response}",
                        'severity': 'high'
                    })
                    
        except Exception as e:
            result['connections']['solana_rpc'] = {'status': 'error', 'error': str(e)}
            result['issues'].append({
                'component': 'connections',
                'type': 'solana_rpc_error',
                'description': f"Solana RPC error: {str(e)}",
                'severity': 'high'
            })
        
        # Check PumpPortal API
        try:
            pump_portal_api_key = os.environ.get('PUMP_PORTAL_API_KEY')
            if not pump_portal_api_key:
                result['connections']['pump_portal'] = {'status': 'no_api_key'}
                result['issues'].append({
                    'component': 'connections',
                    'type': 'missing_pump_portal_api_key',
                    'description': "PUMP_PORTAL_API_KEY environment variable not set",
                    'severity': 'high'
                })
            else:
                # We can't actually test the connection without making an API call
                # which is beyond the scope of a health check
                result['connections']['pump_portal'] = {'status': 'api_key_available'}
        except Exception as e:
            result['connections']['pump_portal'] = {'status': 'error', 'error': str(e)}
        
        # Check wallet
        try:
            wallet_private_key = os.environ.get('WALLET_PRIVATE_KEY')
            if not wallet_private_key:
                result['connections']['wallet'] = {'status': 'no_private_key'}
                result['issues'].append({
                    'component': 'connections',
                    'type': 'missing_wallet_private_key',
                    'description': "WALLET_PRIVATE_KEY environment variable not set",
                    'severity': 'high'
                })
            else:
                # Don't test wallet directly for security reasons
                result['connections']['wallet'] = {'status': 'private_key_available'}
        except Exception as e:
            result['connections']['wallet'] = {'status': 'error', 'error': str(e)}
        
        return result
    
    async def _check_configuration(self) -> Dict[str, Any]:
        """
        Check configuration
        
        Returns:
            Dict[str, Any]: Configuration check results
        """
        result = {
            'configuration': {},
            'issues': []
        }
        
        # Get selected configuration values
        try:
            # Trading configuration
            result['configuration']['trading'] = {
                'enabled': self.config_manager.get('trading.enabled', False),
                'max_position_size_usd': self.config_manager.get('trading.max_position_size_usd', 0),
                'network': self.config_manager.get('trading.network', 'mainnet')
            }
            
            if not result['configuration']['trading']['enabled']:
                result['issues'].append({
                    'component': 'configuration',
                    'type': 'trading_disabled',
                    'description': "Trading is disabled in configuration",
                    'severity': 'medium'
                })
            
            if result['configuration']['trading']['max_position_size_usd'] <= 0:
                result['issues'].append({
                    'component': 'configuration',
                    'type': 'invalid_position_size',
                    'description': "Maximum position size is zero or negative",
                    'severity': 'high'
                })
            
            # Risk configuration
            result['configuration']['risk'] = {
                'stop_loss_percentage': self.config_manager.get('risk.stop_loss_percentage', 0),
                'take_profit_percentage': self.config_manager.get('risk.take_profit_percentage', 0),
                'max_slippage_percentage': self.config_manager.get('risk.max_slippage_percentage', 0)
            }
            
            if result['configuration']['risk']['stop_loss_percentage'] <= 0:
                result['issues'].append({
                    'component': 'configuration',
                    'type': 'invalid_stop_loss',
                    'description': "Stop loss percentage is zero or negative",
                    'severity': 'medium'
                })
            
            if result['configuration']['risk']['take_profit_percentage'] <= 0:
                result['issues'].append({
                    'component': 'configuration',
                    'type': 'invalid_take_profit',
                    'description': "Take profit percentage is zero or negative",
                    'severity': 'medium'
                })
            
            # Monitoring configuration
            result['configuration']['monitoring'] = {
                'telegram_enabled': self.config_manager.get('monitoring.telegram.enabled', False),
                'log_level': self.config_manager.get('logging.level', 'INFO')
            }
            
        except Exception as e:
            result['configuration']['error'] = str(e)
            result['issues'].append({
                'component': 'configuration',
                'type': 'configuration_error',
                'description': f"Error reading configuration: {str(e)}",
                'severity': 'high'
            })
        
        return result
    
    def _generate_recommendations(self, issues: List[Dict[str, Any]]) -> List[str]:
        """
        Generate recommendations based on issues
        
        Args:
            issues (List[Dict[str, Any]]): List of issues
            
        Returns:
            List[str]: Recommendations
        """
        recommendations = []
        
        for issue in issues:
            issue_type = issue.get('type')
            
            if issue_type == 'component_not_running':
                recommendations.append(f"Restart component {issue.get('component')}")
            elif issue_type == 'component_unhealthy':
                recommendations.append(f"Check logs for component {issue.get('component')}")
            elif issue_type == 'component_error':
                recommendations.append(f"Fix error in component {issue.get('component')}: {issue.get('description')}")
            elif issue_type == 'missing_dependency':
                recommendations.append(f"Install missing dependency: {issue.get('description')}")
            elif issue_type in ['missing_database_url', 'missing_solana_rpc_url', 
                               'missing_pump_portal_api_key', 'missing_wallet_private_key']:
                env_var = issue_type.replace('missing_', '').upper()
                recommendations.append(f"Set environment variable {env_var}")
            elif issue_type == 'trading_disabled':
                recommendations.append("Enable trading in configuration")
            elif issue_type in ['invalid_position_size', 'invalid_stop_loss', 'invalid_take_profit']:
                config_key = issue_type.replace('invalid_', '')
                recommendations.append(f"Set valid value for {config_key} in configuration")
        
        return recommendations
    
    def _save_results(self, results: Dict[str, Any]):
        """
        Save health check results to log file
        
        Args:
            results (Dict[str, Any]): Health check results
        """
        try:
            with open(self.health_check_log, 'w') as f:
                json.dump(results, f, indent=2)
                
            logger.info(f"Health check results saved to {self.health_check_log}")
        except Exception as e:
            logger.error(f"Error saving health check results: {str(e)}")
    
    async def verify_component_health(self, component_name: str) -> Dict[str, Any]:
        """
        Verify health of a specific component
        
        Args:
            component_name (str): Component name
            
        Returns:
            Dict[str, Any]: Component health check results
        """
        component_state = self.state_manager.get_component(component_name)
        
        if not component_state:
            return {
                'component': component_name,
                'status': 'not_found',
                'is_healthy': False,
                'message': f"Component {component_name} not found"
            }
        
        return {
            'component': component_name,
            'status': component_state.status,
            'is_healthy': component_state.is_healthy,
            'last_health_check': component_state.last_health_check,
            'error_message': component_state.error_message,
            'metrics': component_state.metrics
        }