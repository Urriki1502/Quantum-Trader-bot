"""
Wallet Security Module
Provides critical security measures for wallet management and protection
in a real-world trading environment.
"""

import os
import time
import json
import base64
import logging
import hashlib
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes
from cryptography.fernet import Fernet

logger = logging.getLogger(__name__)

class WalletSecurityManager:
    """
    Handles wallet security with these critical components:
    - Secure key storage with strong encryption
    - Transaction limits and thresholds
    - Rate limiting for sensitive operations
    - Emergency security measures
    """
    
    def __init__(self, config_manager, security_manager):
        """
        Initialize WalletSecurityManager
        
        Args:
            config_manager: Configuration manager instance
            security_manager: Security manager instance
        """
        self.config_manager = config_manager
        self.security_manager = security_manager
        
        # Load settings
        self._load_security_settings()
        
        # Initialize transaction tracking
        self.transaction_history = []
        self.last_tx_time = 0
        
        # Load or initialize limits
        self._initialize_limits()
    
    def _load_security_settings(self):
        """Load security settings from configuration"""
        # Key protection settings
        self.encryption_enabled = self.config_manager.get('wallet.security.encryption_enabled', True)
        
        # Transaction limits
        self.max_tx_per_minute = self.config_manager.get('wallet.security.max_tx_per_minute', 10)
        self.max_tx_per_hour = self.config_manager.get('wallet.security.max_tx_per_hour', 30)
        self.max_tx_per_day = self.config_manager.get('wallet.security.max_tx_per_day', 100)
        
        # Value limits
        self.max_tx_value_usd = self.config_manager.get('wallet.security.max_tx_value_usd', 1000)
        self.max_daily_value_usd = self.config_manager.get('wallet.security.max_daily_value_usd', 5000)
        
        # Emergency settings
        self.emergency_pause_threshold = self.config_manager.get('wallet.security.emergency_pause_threshold', 3)
    
    def _initialize_limits(self):
        """Initialize transaction limits and counters"""
        self.tx_count = {
            'minute': 0,
            'hour': 0,
            'day': 0
        }
        
        self.tx_value = {
            'minute': 0,
            'hour': 0,
            'day': 0
        }
        
        self.tx_reset_time = {
            'minute': time.time() + 60,
            'hour': time.time() + 3600,
            'day': time.time() + 86400
        }
        
        # Suspicious activity counter
        self.suspicious_activity_count = 0
    
    def store_wallet_key(self, key_data: bytes, password: str) -> str:
        """
        Securely store wallet key with strong encryption
        
        Args:
            key_data (bytes): Wallet key data to store
            password (str): Password for encryption
            
        Returns:
            str: Identifier for stored key
        """
        if not self.encryption_enabled:
            raise SecurityError("Wallet key encryption is disabled, cannot store keys securely")
        
        # Generate a strong salt
        salt = os.urandom(16)
        
        # Key derivation with high iteration count for security
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        
        key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
        
        # Encrypt the key data
        f = Fernet(key)
        encrypted_data = f.encrypt(key_data)
        
        # Create storage structure with salt
        storage_data = {
            'salt': base64.b64encode(salt).decode('utf-8'),
            'data': base64.b64encode(encrypted_data).decode('utf-8'),
            'created_at': int(time.time())
        }
        
        # Generate identifier
        identifier = hashlib.sha256(encrypted_data).hexdigest()[:12]
        
        # Store in secure location (in production this would use secure storage)
        key_file = f"wallet_key_{identifier}.enc"
        key_path = os.path.join(self.config_manager.get('security.key_storage_path', './data/keys'), key_file)
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(key_path), exist_ok=True)
        
        with open(key_path, 'w') as f:
            json.dump(storage_data, f)
        
        logger.info(f"Wallet key stored securely with identifier {identifier}")
        return identifier
    
    def retrieve_wallet_key(self, identifier: str, password: str) -> bytes:
        """
        Retrieve and decrypt wallet key
        
        Args:
            identifier (str): Key identifier
            password (str): Password for decryption
            
        Returns:
            bytes: Decrypted wallet key data
        """
        # Locate key file
        key_file = f"wallet_key_{identifier}.enc"
        key_path = os.path.join(self.config_manager.get('security.key_storage_path', './data/keys'), key_file)
        
        if not os.path.exists(key_path):
            raise KeyError(f"No wallet key found with identifier {identifier}")
        
        # Load encrypted data
        with open(key_path, 'r') as f:
            storage_data = json.load(f)
        
        # Extract components
        salt = base64.b64decode(storage_data['salt'])
        encrypted_data = base64.b64decode(storage_data['data'])
        
        # Derive key
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        
        key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
        
        # Decrypt
        try:
            f = Fernet(key)
            decrypted_data = f.decrypt(encrypted_data)
            
            # Log access (for audit trail)
            logger.info(f"Wallet key {identifier} accessed")
            
            return decrypted_data
        except Exception as e:
            # Increment suspicious counter for failed decryption
            self.suspicious_activity_count += 1
            logger.warning(f"Failed wallet key decryption attempt: {e}")
            raise SecurityError("Failed to decrypt wallet key - incorrect password")
    
    def check_transaction_limits(self, tx_value_usd: float) -> Dict[str, Any]:
        """
        Check if a transaction is within security limits
        
        Args:
            tx_value_usd (float): Transaction value in USD
            
        Returns:
            Dict[str, Any]: Limit check results
        """
        current_time = time.time()
        
        # Reset counters if time periods have elapsed
        for period in ['minute', 'hour', 'day']:
            if current_time > self.tx_reset_time[period]:
                self.tx_count[period] = 0
                self.tx_value[period] = 0
                if period == 'minute':
                    self.tx_reset_time[period] = current_time + 60
                elif period == 'hour':
                    self.tx_reset_time[period] = current_time + 3600
                else:  # day
                    self.tx_reset_time[period] = current_time + 86400
        
        # Check transaction value limit
        if tx_value_usd > self.max_tx_value_usd:
            return {
                'allowed': False,
                'reason': f"Transaction value ${tx_value_usd} exceeds maximum ${self.max_tx_value_usd}"
            }
        
        # Check transaction count limits
        if self.tx_count['minute'] >= self.max_tx_per_minute:
            return {
                'allowed': False,
                'reason': f"Exceeded maximum transactions per minute ({self.max_tx_per_minute})"
            }
        
        if self.tx_count['hour'] >= self.max_tx_per_hour:
            return {
                'allowed': False,
                'reason': f"Exceeded maximum transactions per hour ({self.max_tx_per_hour})"
            }
        
        if self.tx_count['day'] >= self.max_tx_per_day:
            return {
                'allowed': False,
                'reason': f"Exceeded maximum transactions per day ({self.max_tx_per_day})"
            }
        
        # Check daily value limit
        if self.tx_value['day'] + tx_value_usd > self.max_daily_value_usd:
            return {
                'allowed': False,
                'reason': f"Transaction would exceed daily value limit ${self.max_daily_value_usd}"
            }
        
        # Anti-flooding protection (minimum time between transactions)
        min_tx_interval = self.config_manager.get('wallet.security.min_tx_interval_sec', 2)
        if current_time - self.last_tx_time < min_tx_interval:
            return {
                'allowed': False,
                'reason': f"Transaction too soon after previous transaction"
            }
        
        # All checks passed
        return {
            'allowed': True,
            'reason': "Transaction within limits"
        }
    
    def record_transaction(self, tx_data: Dict[str, Any]):
        """
        Record a completed transaction for security tracking
        
        Args:
            tx_data (Dict[str, Any]): Transaction data
        """
        tx_value_usd = tx_data.get('value_usd', 0)
        current_time = time.time()
        
        # Update counters
        self.tx_count['minute'] += 1
        self.tx_count['hour'] += 1
        self.tx_count['day'] += 1
        
        self.tx_value['minute'] += tx_value_usd
        self.tx_value['hour'] += tx_value_usd
        self.tx_value['day'] += tx_value_usd
        
        # Record timestamp
        self.last_tx_time = current_time
        
        # Add to history (limited to last 100 transactions)
        tx_record = {
            'time': current_time,
            'value_usd': tx_value_usd,
            'token': tx_data.get('token_address', ''),
            'type': tx_data.get('type', 'unknown')
        }
        
        self.transaction_history.append(tx_record)
        if len(self.transaction_history) > 100:
            self.transaction_history = self.transaction_history[-100:]
        
        logger.info(f"Transaction recorded: {tx_data.get('type', 'unknown')} "
                   f"{tx_data.get('token_symbol', '')} ${tx_value_usd:.2f}")
    
    def detect_unusual_activity(self, tx_data: Dict[str, Any]) -> bool:
        """
        Detect potentially unusual or suspicious activity
        
        Args:
            tx_data (Dict[str, Any]): Transaction data
            
        Returns:
            bool: True if unusual activity detected
        """
        # Check for unusual transaction patterns
        unusual = False
        reason = None
        
        # 1. Check for rapid succession of transactions
        if len(self.transaction_history) >= 3:
            last_three = self.transaction_history[-3:]
            time_span = last_three[-1]['time'] - last_three[0]['time']
            
            # If 3 transactions in less than 5 seconds
            if time_span < 5 and time_span > 0:
                unusual = True
                reason = "Rapid succession of transactions"
        
        # 2. Check for unusual value pattern
        if len(self.transaction_history) >= 5:
            last_five = self.transaction_history[-5:]
            avg_value = sum(tx['value_usd'] for tx in last_five[:-1]) / 4
            
            # If current transaction is 5x the average of the last 4
            if tx_data.get('value_usd', 0) > avg_value * 5 and avg_value > 0:
                unusual = True
                reason = "Transaction value significantly higher than recent average"
        
        # 3. Excessive value in short time
        if self.tx_value['minute'] > self.max_daily_value_usd * 0.25:
            unusual = True
            reason = "High transaction volume in short period"
        
        if unusual:
            self.suspicious_activity_count += 1
            logger.warning(f"Unusual activity detected: {reason}")
            
            # Check if we should trigger emergency pause
            if self.suspicious_activity_count >= self.emergency_pause_threshold:
                self._trigger_emergency_pause(reason)
        
        return unusual
    
    def _trigger_emergency_pause(self, reason: str):
        """
        Trigger emergency pause of trading activity
        
        Args:
            reason (str): Reason for emergency pause
        """
        logger.critical(f"EMERGENCY PAUSE triggered: {reason}")
        
        # Signal emergency to other components
        self.security_manager.trigger_security_event(
            "EMERGENCY_PAUSE", 
            {"reason": reason, "triggered_by": "wallet_security"}
        )
        
        # Reset the threshold counter
        self.suspicious_activity_count = 0
    
    def get_transaction_stats(self) -> Dict[str, Any]:
        """
        Get transaction statistics for monitoring
        
        Returns:
            Dict[str, Any]: Transaction stats
        """
        return {
            'counts': {
                'minute': self.tx_count['minute'],
                'hour': self.tx_count['hour'],
                'day': self.tx_count['day'],
            },
            'values': {
                'minute': self.tx_value['minute'],
                'hour': self.tx_value['hour'],
                'day': self.tx_value['day'],
            },
            'limits': {
                'max_tx_per_minute': self.max_tx_per_minute,
                'max_tx_per_hour': self.max_tx_per_hour,
                'max_tx_per_day': self.max_tx_per_day,
                'max_tx_value_usd': self.max_tx_value_usd,
                'max_daily_value_usd': self.max_daily_value_usd,
            },
            'reset_times': {
                'minute': self.tx_reset_time['minute'] - time.time(),
                'hour': self.tx_reset_time['hour'] - time.time(),
                'day': self.tx_reset_time['day'] - time.time(),
            }
        }

class SecurityError(Exception):
    """Exception raised for security violations"""
    pass