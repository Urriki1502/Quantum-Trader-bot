"""
Mainnet Validator Component
Provides validation and safeguards for mainnet operation,
ensuring safe operation with real funds.
"""

import time
import logging
import json
import asyncio
from typing import Dict, List, Any, Optional, Callable, Tuple, Set

from core.config_manager import ConfigManager
from core.state_manager import StateManager

logger = logging.getLogger(__name__)

class MainnetValidator:
    """
    MainnetValidator provides:
    - Production safeguards
    - Transaction validation
    - Risk thresholds for mainnet
    - Emergency circuit breakers
    - Transaction/time limits
    
    This component ensures safer operation with real funds.
    """
    
    def __init__(self, 
                config_manager: ConfigManager,
                state_manager: StateManager):
        """
        Initialize MainnetValidator
        
        Args:
            config_manager (ConfigManager): Configuration manager instance
            state_manager (StateManager): State manager instance
        """
        self.config_manager = config_manager
        self.state_manager = state_manager
        
        # Load validation settings
        self._load_validation_settings()
        
        # Transaction tracking
        self.transaction_history = []
        self.pending_transactions = {}
        
        # Emergency state
        self.emergency_mode = False
        self.emergency_reason = None
        
        # Validation handlers
        self.pre_transaction_validators = []
        self.post_transaction_validators = []
        
        # Register default validators
        self._register_default_validators()
        
        # Start time for tracking age-based limits
        self.start_time = time.time()
        
        # Token allowlist (if enabled)
        self.token_allowlist = set()
        if self.allowlist_only:
            self._load_allowlist()
    
    def _load_validation_settings(self):
        """Load validation settings from configuration"""
        # Global validation settings
        self.validation_enabled = self.config_manager.get('mainnet.validation_enabled', True)
        self.dry_run_mode = self.config_manager.get('mainnet.dry_run_mode', False)
        
        # Warmup period - restrict trading early after deployment
        self.warmup_period_hours = self.config_manager.get('mainnet.warmup_period_hours', 24)
        self.warmup_tx_limit = self.config_manager.get('mainnet.warmup_tx_limit', 10)
        self.warmup_max_position_usd = self.config_manager.get('mainnet.warmup_max_position_usd', 100)
        
        # Transaction limits
        self.max_single_tx_value_usd = self.config_manager.get('mainnet.max_single_tx_value_usd', 1000)
        self.max_hourly_tx_value_usd = self.config_manager.get('mainnet.max_hourly_tx_value_usd', 5000)
        self.max_daily_tx_value_usd = self.config_manager.get('mainnet.max_daily_tx_value_usd', 10000)
        
        # Token validation
        self.min_token_age_hours = self.config_manager.get('mainnet.min_token_age_hours', 2)
        self.min_token_liquidity_usd = self.config_manager.get('mainnet.min_token_liquidity_usd', 20000)
        self.allowlist_only = self.config_manager.get('mainnet.allowlist_only', False)
        
        # Market protection
        self.max_slippage_percent = self.config_manager.get('mainnet.max_slippage_percent', 3)
        self.max_position_tokens_percent = self.config_manager.get('mainnet.max_position_tokens_percent', 2)
        
        # Circuit breaker settings
        self.circuit_breaker_enabled = self.config_manager.get('mainnet.circuit_breaker.enabled', True)
        self.max_consecutive_errors = self.config_manager.get('mainnet.circuit_breaker.max_consecutive_errors', 3)
        self.consecutive_errors = 0
    
    def _register_default_validators(self):
        """Register default transaction validators"""
        
        # Pre-transaction validators
        self.register_pre_transaction_validator(self._validate_warmup_period)
        self.register_pre_transaction_validator(self._validate_transaction_value)
        self.register_pre_transaction_validator(self._validate_token_eligibility)
        self.register_pre_transaction_validator(self._validate_transaction_limits)
        self.register_pre_transaction_validator(self._validate_market_conditions)
        
        # Post-transaction validators
        self.register_post_transaction_validator(self._validate_transaction_result)
    
    def _load_allowlist(self):
        """Load token allowlist"""
        try:
            allowlist_path = self.config_manager.get('mainnet.token_allowlist_file', './data/token_allowlist.json')
            
            try:
                with open(allowlist_path, 'r') as f:
                    allowlist_data = json.load(f)
                    self.token_allowlist = set(allowlist_data.get('tokens', []))
                    logger.info(f"Loaded {len(self.token_allowlist)} allowlisted tokens")
            except FileNotFoundError:
                logger.warning(f"Allowlist file not found: {allowlist_path}")
                # Initialize empty allowlist file
                import os
                os.makedirs(os.path.dirname(allowlist_path), exist_ok=True)
                with open(allowlist_path, 'w') as f:
                    json.dump({'tokens': []}, f)
        except Exception as e:
            logger.error(f"Error loading token allowlist: {e}")
    
    def add_to_allowlist(self, token_address: str, reason: str):
        """
        Add a token to the allowlist
        
        Args:
            token_address (str): Token address
            reason (str): Reason for allowlisting
        """
        if token_address in self.token_allowlist:
            return
        
        self.token_allowlist.add(token_address)
        
        try:
            allowlist_path = self.config_manager.get('mainnet.token_allowlist_file', './data/token_allowlist.json')
            
            # Load existing allowlist
            try:
                with open(allowlist_path, 'r') as f:
                    allowlist_data = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                allowlist_data = {'tokens': []}
            
            # Update allowlist
            if 'tokens' not in allowlist_data:
                allowlist_data['tokens'] = []
            
            if token_address not in allowlist_data['tokens']:
                allowlist_data['tokens'].append(token_address)
            
            # Add reason if tracking reasons
            if 'reasons' not in allowlist_data:
                allowlist_data['reasons'] = {}
            
            allowlist_data['reasons'][token_address] = {
                'reason': reason,
                'timestamp': time.time()
            }
            
            # Save allowlist
            with open(allowlist_path, 'w') as f:
                json.dump(allowlist_data, f, indent=2)
            
            logger.info(f"Added token to allowlist: {token_address} (Reason: {reason})")
        except Exception as e:
            logger.error(f"Error adding token to allowlist: {e}")
    
    def register_pre_transaction_validator(self, validator_func: Callable[[Dict[str, Any]], Tuple[bool, str]]):
        """
        Register a pre-transaction validator function
        
        Args:
            validator_func (Callable): Validator function that returns (is_valid, reason)
        """
        self.pre_transaction_validators.append(validator_func)
    
    def register_post_transaction_validator(self, validator_func: Callable[[Dict[str, Any], Dict[str, Any]], Tuple[bool, str]]):
        """
        Register a post-transaction validator function
        
        Args:
            validator_func (Callable): Validator function that returns (is_valid, reason)
        """
        self.post_transaction_validators.append(validator_func)
    
    async def validate_transaction(self, transaction: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate a transaction before execution
        
        Args:
            transaction (Dict[str, Any]): Transaction data
            
        Returns:
            Dict[str, Any]: Validation result
        """
        # Skip validation if disabled
        if not self.validation_enabled:
            return {
                'valid': True,
                'reason': 'Validation disabled',
                'warnings': []
            }
        
        # Check if in emergency mode
        if self.emergency_mode:
            return {
                'valid': False,
                'reason': f"Emergency mode active: {self.emergency_reason}",
                'warnings': []
            }
        
        # Skip straight to dry run if mode is enabled
        if self.dry_run_mode:
            return {
                'valid': True,
                'reason': 'Dry run mode - no real transactions will be executed',
                'dry_run': True,
                'warnings': ['System is in dry run mode - transaction would be simulated only']
            }
        
        valid = True
        reason = "Transaction valid"
        warnings = []
        
        # Run pre-transaction validators
        for validator in self.pre_transaction_validators:
            try:
                validator_valid, validator_reason = validator(transaction)
                if not validator_valid:
                    valid = False
                    reason = validator_reason
                    break
                elif validator_reason:
                    warnings.append(validator_reason)
            except Exception as e:
                logger.error(f"Error in transaction validator: {e}")
                valid = False
                reason = f"Validator error: {str(e)}"
                break
        
        # If valid, track pending transaction
        if valid:
            tx_id = transaction.get('tx_id', str(time.time()))
            self.pending_transactions[tx_id] = {
                'timestamp': time.time(),
                'transaction': transaction
            }
        
        return {
            'valid': valid,
            'reason': reason,
            'warnings': warnings
        }
    
    async def record_transaction_result(self, 
                                      transaction: Dict[str, Any], 
                                      result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Record and validate transaction result
        
        Args:
            transaction (Dict[str, Any]): Transaction data
            result (Dict[str, Any]): Transaction result
            
        Returns:
            Dict[str, Any]: Validation result
        """
        tx_id = transaction.get('tx_id', str(time.time()))
        
        # Record transaction
        tx_record = {
            'tx_id': tx_id,
            'timestamp': time.time(),
            'transaction': transaction,
            'result': result,
            'success': result.get('success', False)
        }
        
        self.transaction_history.append(tx_record)
        
        # Limit history size
        if len(self.transaction_history) > 1000:
            self.transaction_history = self.transaction_history[-1000:]
        
        # Remove from pending
        if tx_id in self.pending_transactions:
            del self.pending_transactions[tx_id]
        
        # Track consecutive errors for circuit breaker
        if self.circuit_breaker_enabled:
            if not result.get('success', False):
                self.consecutive_errors += 1
                
                if self.consecutive_errors >= self.max_consecutive_errors:
                    await self._trigger_circuit_breaker(
                        f"Circuit breaker triggered: {self.consecutive_errors} consecutive errors"
                    )
            else:
                self.consecutive_errors = 0
        
        # Run post-transaction validators
        valid = True
        reason = "Transaction result valid"
        warnings = []
        
        for validator in self.post_transaction_validators:
            try:
                validator_valid, validator_reason = validator(transaction, result)
                if not validator_valid:
                    valid = False
                    reason = validator_reason
                    break
                elif validator_reason:
                    warnings.append(validator_reason)
            except Exception as e:
                logger.error(f"Error in transaction result validator: {e}")
                valid = False
                reason = f"Validator error: {str(e)}"
                break
        
        # Update state metrics
        if result.get('success', False):
            self.state_manager.update_component_metric(
                'mainnet_validator',
                'successful_transactions',
                1,
                increment=True
            )
        else:
            self.state_manager.update_component_metric(
                'mainnet_validator',
                'failed_transactions',
                1,
                increment=True
            )
        
        return {
            'valid': valid,
            'reason': reason,
            'warnings': warnings
        }
    
    async def _trigger_circuit_breaker(self, reason: str):
        """
        Trigger circuit breaker and enter emergency mode
        
        Args:
            reason (str): Reason for circuit breaker
        """
        logger.critical(f"CIRCUIT BREAKER TRIGGERED: {reason}")
        
        self.emergency_mode = True
        self.emergency_reason = reason
        
        # Create critical alert
        self.state_manager.create_alert(
            'mainnet_validator', 
            'CRITICAL', 
            f"Circuit breaker triggered: {reason}"
        )
        
        # Update state
        self.state_manager.update_component_state(
            'mainnet_validator',
            {
                'emergency_mode': True,
                'emergency_reason': reason,
                'emergency_time': time.time()
            }
        )
        
        # In production, this would send notifications, alerts, etc.
    
    def reset_circuit_breaker(self):
        """Reset circuit breaker and exit emergency mode"""
        if not self.emergency_mode:
            return
        
        logger.warning("Resetting circuit breaker and exiting emergency mode")
        
        self.emergency_mode = False
        self.emergency_reason = None
        self.consecutive_errors = 0
        
        # Update state
        self.state_manager.update_component_state(
            'mainnet_validator',
            {
                'emergency_mode': False,
                'emergency_reset_time': time.time()
            }
        )
        
        # Create alert
        self.state_manager.create_alert(
            'mainnet_validator', 
            'WARNING', 
            f"Circuit breaker reset - resuming normal operation"
        )
    
    def _validate_warmup_period(self, transaction: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Validate transaction against warmup period restrictions
        
        Args:
            transaction (Dict[str, Any]): Transaction data
            
        Returns:
            Tuple[bool, str]: (is_valid, reason)
        """
        # Check if still in warmup period
        system_age_hours = (time.time() - self.start_time) / 3600
        
        if system_age_hours < self.warmup_period_hours:
            # Check transaction count
            if len(self.transaction_history) >= self.warmup_tx_limit:
                return False, f"Exceeded transaction limit during warmup period ({self.warmup_tx_limit})"
            
            # Check transaction value
            tx_value_usd = transaction.get('value_usd', 0)
            if tx_value_usd > self.warmup_max_position_usd:
                return False, f"Transaction value ${tx_value_usd} exceeds warmup limit ${self.warmup_max_position_usd}"
            
            return True, f"System in warmup period ({system_age_hours:.1f}/{self.warmup_period_hours}h) - limits applied"
        
        return True, None
    
    def _validate_transaction_value(self, transaction: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Validate transaction value
        
        Args:
            transaction (Dict[str, Any]): Transaction data
            
        Returns:
            Tuple[bool, str]: (is_valid, reason)
        """
        tx_value_usd = transaction.get('value_usd', 0)
        
        if tx_value_usd <= 0:
            return True, None  # Zero-value transaction or value not specified
        
        if tx_value_usd > self.max_single_tx_value_usd:
            return False, f"Transaction value ${tx_value_usd} exceeds maximum ${self.max_single_tx_value_usd}"
        
        return True, None
    
    def _validate_token_eligibility(self, transaction: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Validate token eligibility
        
        Args:
            transaction (Dict[str, Any]): Transaction data
            
        Returns:
            Tuple[bool, str]: (is_valid, reason)
        """
        token_address = transaction.get('token_address')
        token_data = transaction.get('token_data', {})
        
        # Skip validation for non-token transactions
        if not token_address:
            return True, None
        
        # Check allowlist if enabled
        if self.allowlist_only and token_address not in self.token_allowlist:
            return False, f"Token {token_address} not in allowlist"
        
        # Check token age
        discovery_time = token_data.get('discovery_time', 0)
        if discovery_time > 0:
            token_age_hours = (time.time() - discovery_time) / 3600
            if token_age_hours < self.min_token_age_hours:
                return False, f"Token age ({token_age_hours:.1f}h) below minimum ({self.min_token_age_hours}h)"
        
        # Check liquidity
        liquidity_usd = token_data.get('liquidity_usd', 0)
        if liquidity_usd < self.min_token_liquidity_usd:
            return False, f"Token liquidity (${liquidity_usd}) below minimum (${self.min_token_liquidity_usd})"
        
        return True, None
    
    def _validate_transaction_limits(self, transaction: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Validate transaction against time-based limits
        
        Args:
            transaction (Dict[str, Any]): Transaction data
            
        Returns:
            Tuple[bool, str]: (is_valid, reason)
        """
        tx_value_usd = transaction.get('value_usd', 0)
        
        if tx_value_usd <= 0:
            return True, None  # Zero-value transaction or value not specified
        
        # Calculate recent transaction totals
        current_time = time.time()
        hour_ago = current_time - 3600
        day_ago = current_time - 86400
        
        hourly_total = sum(
            tx['transaction'].get('value_usd', 0)
            for tx in self.transaction_history
            if tx['timestamp'] > hour_ago
        )
        
        daily_total = sum(
            tx['transaction'].get('value_usd', 0)
            for tx in self.transaction_history
            if tx['timestamp'] > day_ago
        )
        
        # Check hourly limit
        if hourly_total + tx_value_usd > self.max_hourly_tx_value_usd:
            return False, f"Transaction would exceed hourly limit (${hourly_total + tx_value_usd} > ${self.max_hourly_tx_value_usd})"
        
        # Check daily limit
        if daily_total + tx_value_usd > self.max_daily_tx_value_usd:
            return False, f"Transaction would exceed daily limit (${daily_total + tx_value_usd} > ${self.max_daily_tx_value_usd})"
        
        return True, None
    
    def _validate_market_conditions(self, transaction: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Validate market conditions for transaction
        
        Args:
            transaction (Dict[str, Any]): Transaction data
            
        Returns:
            Tuple[bool, str]: (is_valid, reason)
        """
        # Check slippage
        expected_slippage = transaction.get('expected_slippage', 0)
        if expected_slippage > self.max_slippage_percent:
            return False, f"Expected slippage ({expected_slippage}%) exceeds maximum ({self.max_slippage_percent}%)"
        
        # Check position size relative to token supply
        token_data = transaction.get('token_data', {})
        position_tokens_percent = transaction.get('position_tokens_percent', 0)
        
        if position_tokens_percent > self.max_position_tokens_percent:
            return False, f"Position size ({position_tokens_percent}% of supply) exceeds maximum ({self.max_position_tokens_percent}%)"
        
        return True, None
    
    def _validate_transaction_result(self, 
                                   transaction: Dict[str, Any], 
                                   result: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Validate transaction result
        
        Args:
            transaction (Dict[str, Any]): Transaction data
            result (Dict[str, Any]): Transaction result
            
        Returns:
            Tuple[bool, str]: (is_valid, reason)
        """
        # Check for success
        if not result.get('success', False):
            return False, f"Transaction failed: {result.get('error', 'Unknown error')}"
        
        # Check for excessive slippage
        expected_slippage = transaction.get('expected_slippage', 0)
        actual_slippage = result.get('slippage', 0)
        
        if actual_slippage > expected_slippage * 1.5:
            return False, f"Actual slippage ({actual_slippage}%) significantly exceeded expected ({expected_slippage}%)"
        
        return True, None
    
    def get_transaction_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get transaction history
        
        Args:
            limit (int): Maximum number of transactions to return
            
        Returns:
            List[Dict[str, Any]]: Transaction history
        """
        return self.transaction_history[-limit:]
    
    def get_validator_status(self) -> Dict[str, Any]:
        """
        Get validator status
        
        Returns:
            Dict[str, Any]: Validator status
        """
        return {
            'validation_enabled': self.validation_enabled,
            'dry_run_mode': self.dry_run_mode,
            'emergency_mode': self.emergency_mode,
            'emergency_reason': self.emergency_reason,
            'system_age_hours': (time.time() - self.start_time) / 3600,
            'warmup_active': (time.time() - self.start_time) / 3600 < self.warmup_period_hours,
            'transaction_count': len(self.transaction_history),
            'pending_transactions': len(self.pending_transactions),
            'consecutive_errors': self.consecutive_errors,
            'max_consecutive_errors': self.max_consecutive_errors,
            'allowlist_enabled': self.allowlist_only,
            'allowlist_size': len(self.token_allowlist),
            'limits': {
                'max_single_tx_value_usd': self.max_single_tx_value_usd,
                'max_hourly_tx_value_usd': self.max_hourly_tx_value_usd,
                'max_daily_tx_value_usd': self.max_daily_tx_value_usd,
                'min_token_age_hours': self.min_token_age_hours,
                'min_token_liquidity_usd': self.min_token_liquidity_usd
            }
        }