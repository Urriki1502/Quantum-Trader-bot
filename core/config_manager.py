"""
ConfigManager Component
Responsible for loading and managing configuration from various sources.
"""

import os
import json
import yaml
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path

logger = logging.getLogger(__name__)

class ConfigManager:
    """
    ConfigManager loads and validates configuration from:
    - Environment variables
    - JSON/YAML config files
    - Default values
    """
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize the ConfigManager
        
        Args:
            config_file (str, optional): Path to a config file to load
        """
        # Initialize config store
        self.config: Dict[str, Any] = {}
        
        # Default paths to check for config files
        self.config_paths = [
            './config.json',
            './config.yaml',
            './config.yml',
            os.path.expanduser('~/.quantum_memecoin/config.json'),
            os.path.expanduser('~/.quantum_memecoin/config.yaml')
        ]
        
        # Add specific config file if provided
        if config_file:
            self.config_paths.insert(0, config_file)
        
        # Load configuration from various sources
        self._load_defaults()
        self._load_config_files()
        self._load_environment_variables()
        
        # Validate required configuration
        self._validate_config()
        
        logger.info("ConfigManager initialized")
        logger.debug(f"Loaded configuration for {len(self.config)} settings")
    
    def _load_defaults(self):
        """Load default configuration values"""
        self.config = {
            # Core configuration
            'log_level': 'INFO',
            'log_dir': './logs',
            'state_persistence_enabled': True,
            'state_persistence_interval': 900,  # 15 minutes
            'state_persistence_file': './state/system_state.json',
            'environment': 'production',  # Set to production mode for real trading
            
            # Network configuration
            'pump_portal': {
                'base_url': 'wss://pumpportal.fun/api/data',  # URL chính xác từ tài liệu chính thức
                'rest_url': 'https://pumpportal.fun/api',     # URL REST API chính thức
                'reconnect_interval': 30,
                'timeout': 60,
                'max_retries': 5
            },
            
            # Trading configuration
            'raydium': {
                'network': 'mainnet',  # mainnet or devnet
                'endpoint': 'https://api.mainnet-beta.solana.com',
                'timeout': 60,
                'max_retries': 3,
                'auto_retry': True
            },
            
            # Risk management configuration
            'risk': {
                'max_position_size_usd': 1000,  # Maximum position size in USD
                'max_exposure_percentage': 5,   # Maximum % of portfolio in one token
                'stop_loss_percentage': 5,      # Stop loss trigger percentage
                'take_profit_percentage': 10,   # Take profit trigger percentage
                'max_slippage_percentage': 2,   # Maximum allowed slippage
                'var_confidence_level': 0.95    # VaR confidence level
            },
            
            # Strategy configuration
            'strategy': {
                'default_strategy': 'momentum',
                'max_active_positions': 10,
                'min_liquidity_usd': 10000,     # Minimum liquidity in USD
                'min_volume_usd': 5000,         # Minimum 24h volume in USD
                'blacklisted_keywords': ['scam', 'honeypot', 'rug']
            },
            
            # Monitoring configuration
            'monitoring': {
                'health_check_interval': 60,    # Health check interval in seconds
                'metrics_collection_interval': 300,  # Metrics collection interval
                'system_alert_levels': ['WARNING', 'ERROR', 'CRITICAL']
            },
            
            # Notification configuration
            'telegram': {
                'enabled': True,
                'notification_levels': ['ERROR', 'CRITICAL'],
                'trade_notifications': True,
                'system_notifications': True
            },
            
            # Memory management
            'memory': {
                'gc_interval': 3600,  # Garbage collection interval in seconds
                'max_memory_usage_percentage': 75,  # Maximum memory usage percentage
                'cache_ttl': 1800  # Cache time-to-live in seconds
            }
        }
    
    def _load_config_files(self):
        """Load configuration from files"""
        for config_path in self.config_paths:
            if os.path.exists(config_path):
                try:
                    with open(config_path, 'r') as file:
                        if config_path.endswith('.json'):
                            file_config = json.load(file)
                        elif config_path.endswith(('.yaml', '.yml')):
                            file_config = yaml.safe_load(file)
                        else:
                            logger.warning(f"Unsupported config file format: {config_path}")
                            continue
                    
                    # Deep merge file config with existing config
                    self._deep_merge(self.config, file_config)
                    logger.info(f"Loaded configuration from {config_path}")
                    
                    # Once we've loaded a config file successfully, we can stop
                    break
                except Exception as e:
                    logger.error(f"Error loading config from {config_path}: {str(e)}")
    
    def _load_environment_variables(self):
        """Load configuration from environment variables"""
        # Define mappings from env vars to config keys
        env_mappings = {
            'QUANTUM_LOG_LEVEL': 'log_level',
            'QUANTUM_LOG_DIR': 'log_dir',
            
            # PumpPortal (will be added in future)
            'PUMP_PORTAL_API_KEY': 'pump_portal.api_key',
            'PUMP_PORTAL_BASE_URL': 'pump_portal.base_url',
            
            # Raydium/Solana - Use environment secret from Replit
            'SOLANA_RPC_URL': 'raydium.endpoint',
            'RAYDIUM_NETWORK': 'raydium.network',
            
            # Wallet - Use environment secret from Replit
            'WALLET_PRIVATE_KEY': 'wallet.private_key',
            
            # Risk
            'RISK_MAX_POSITION_SIZE': 'risk.max_position_size_usd',
            'RISK_MAX_EXPOSURE': 'risk.max_exposure_percentage',
            'RISK_STOP_LOSS': 'risk.stop_loss_percentage',
            'RISK_TAKE_PROFIT': 'risk.take_profit_percentage',
            
            # Telegram - Use environment secret from Replit
            'TELEGRAM_BOT_TOKEN': 'telegram.bot_token',
            'TELEGRAM_CHAT_ID': 'telegram.chat_id',
            'TELEGRAM_ENABLED': 'telegram.enabled'
        }
        
        # Also check for DATABASE_URL and other Replit-specific environment variables
        additional_env = {
            'DATABASE_URL': 'database.url',
            'REPLIT_DB_URL': 'database.replit_db_url',
            'PGUSER': 'database.username',
            'PGPASSWORD': 'database.password',
            'PGDATABASE': 'database.name',
            'PGHOST': 'database.host',
            'PGPORT': 'database.port'
        }
        
        # Combine all environment variable mappings
        env_mappings.update(additional_env)
        
        # Process environment variables
        for env_var, config_path in env_mappings.items():
            value = os.environ.get(env_var)
            if value is not None:
                # Convert value to appropriate type
                if value.lower() in ('true', 'yes', '1'):
                    value = True
                elif value.lower() in ('false', 'no', '0'):
                    value = False
                elif value.isdigit():
                    value = int(value)
                elif value.replace('.', '', 1).isdigit() and value.count('.') <= 1:
                    value = float(value)
                
                # Set the value in the nested config dictionary
                self.set(config_path, value)
                logger.debug(f"Set {config_path} from environment variable {env_var}")
    
    def _validate_config(self):
        """Validate required configuration values"""
        # Check if we have the required environment variables
        has_wallet_key = os.environ.get('WALLET_PRIVATE_KEY') is not None
        has_pumpportal_key = os.environ.get('PUMP_PORTAL_API_KEY') is not None
        
        # Define required configuration
        required_configs = []
        
        # PumpPortal is required for production use
        if not has_pumpportal_key and not self.get('pump_portal.api_key'):
            logger.warning("PumpPortal API key is required for production use")
            # Force set this value if not already set
            if self.get('environment') == 'production':
                self.set('pump_portal.api_key', os.environ.get('PUMP_PORTAL_API_KEY', ''))
                logger.info("Using PumpPortal API key from environment")
        
        # Wallet private key is essential, but might be available via environment even
        # if not in config directly
        if not has_wallet_key and not self.get('wallet.private_key'):
            logger.warning("Wallet private key is required")
        
        # Verify we have access to Solana RPC
        if not os.environ.get('SOLANA_RPC_URL') and not self.get('raydium.endpoint'):
            logger.warning("Solana RPC URL is required for blockchain operations")
        
        # If telegram is enabled but missing API key, warn and try to set from environment
        if self.get('telegram.enabled', True):
            if not self.get('telegram.bot_token'):
                logger.warning("Telegram bot token not configured")
                self.set('telegram.bot_token', os.environ.get('TELEGRAM_BOT_TOKEN', ''))
            
            if not self.get('telegram.chat_id'):
                logger.warning("Telegram chat ID not configured")
                self.set('telegram.chat_id', os.environ.get('TELEGRAM_CHAT_ID', ''))
    
    def _deep_merge(self, base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
        """
        Deep merge two dictionaries
        
        Args:
            base (Dict[str, Any]): Base dictionary
            update (Dict[str, Any]): Dictionary to merge on top
            
        Returns:
            Dict[str, Any]: Merged dictionary
        """
        for key, value in update.items():
            if isinstance(value, dict) and key in base and isinstance(base[key], dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value
        return base
    
    def get(self, path: str, default: Any = None) -> Any:
        """
        Get a configuration value by path
        
        Args:
            path (str): Dot-separated path to the config value
            default (Any, optional): Default value if path not found
            
        Returns:
            Any: Configuration value
        """
        # Split the path into parts
        parts = path.split('.')
        
        # Navigate through the config dictionary
        current = self.config
        try:
            for part in parts:
                current = current[part]
            return current
        except (KeyError, TypeError):
            return default
    
    def set(self, path: str, value: Any):
        """
        Set a configuration value by path
        
        Args:
            path (str): Dot-separated path to the config value
            value (Any): Value to set
        """
        # Split the path into parts
        parts = path.split('.')
        
        # Navigate through the config dictionary, creating missing parts
        current = self.config
        for i, part in enumerate(parts[:-1]):
            if part not in current or not isinstance(current[part], dict):
                current[part] = {}
            current = current[part]
        
        # Set the value
        current[parts[-1]] = value
    
    def get_all(self) -> Dict[str, Any]:
        """
        Get the entire configuration dictionary
        
        Returns:
            Dict[str, Any]: Complete configuration
        """
        return self.config
    
    def save_to_file(self, file_path: str) -> bool:
        """
        Save current configuration to a file
        
        Args:
            file_path (str): Path to save the configuration
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
            
            # Determine file format from extension
            if file_path.endswith('.json'):
                with open(file_path, 'w') as f:
                    json.dump(self.config, f, indent=2)
            elif file_path.endswith(('.yaml', '.yml')):
                with open(file_path, 'w') as f:
                    yaml.dump(self.config, f, default_flow_style=False)
            else:
                logger.error(f"Unsupported file format for {file_path}")
                return False
            
            logger.info(f"Configuration saved to {file_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving configuration to {file_path}: {str(e)}")
            return False
    
    def get_network_endpoint(self) -> str:
        """
        Get the current Solana network endpoint based on configuration
        
        Returns:
            str: Solana network endpoint URL
        """
        network = self.get('raydium.network', 'mainnet')
        
        # Use the configured endpoint if available
        endpoint = self.get('raydium.endpoint')
        if endpoint:
            return endpoint
        
        # Otherwise use default endpoints based on network
        if network == 'devnet':
            return 'https://api.devnet.solana.com'
        else:
            return 'https://api.mainnet-beta.solana.com'
