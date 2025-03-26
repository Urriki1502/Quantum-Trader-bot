"""
SecurityManager Component
Responsible for secure storage and handling of sensitive information,
as well as authentication and authorization.
"""

import os
import base64
import logging
import json
import hashlib
import time
from typing import Dict, Any, Optional, List, Tuple
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from core.config_manager import ConfigManager

logger = logging.getLogger(__name__)

class SecurityManager:
    """
    SecurityManager handles:
    - Secure storage of API keys and wallet keys
    - Encryption and decryption of sensitive data
    - Authentication of operations
    - Management of blacklist and whitelist
    """
    
    def __init__(self, config_manager: ConfigManager):
        """
        Initialize the SecurityManager
        
        Args:
            config_manager (ConfigManager): Configuration manager instance
        """
        self.config_manager = config_manager
        self.encrypted_storage = {}
        self.blacklisted_addresses = set()
        self.blacklisted_tokens = set()
        self.whitelisted_addresses = set()
        self.whitelisted_tokens = set()
        
        # Initialize encryption key
        self._init_encryption_key()
        
        # Load blacklists and whitelists
        self._load_address_lists()
        
        logger.info("SecurityManager initialized")
    
    def _init_encryption_key(self):
        """Initialize or load the encryption key"""
        # Try to get a persistent key from environment or config
        key = os.environ.get("ENCRYPTION_KEY")
        
        if not key:
            key = self.config_manager.get("security.encryption_key")
        
        if key:
            # Use existing key
            try:
                self.fernet = Fernet(key.encode())
                logger.debug("Using existing encryption key")
            except Exception as e:
                logger.error(f"Invalid encryption key: {str(e)}")
                self._generate_new_key()
        else:
            # Generate a new key
            self._generate_new_key()
    
    def _generate_new_key(self):
        """Generate a new Fernet encryption key"""
        key = Fernet.generate_key()
        self.fernet = Fernet(key)
        
        # Store key in config for persistence
        self.config_manager.set("security.encryption_key", key.decode())
        logger.info("Generated new encryption key")
    
    def _load_address_lists(self):
        """Load blacklisted and whitelisted addresses and tokens"""
        # Load from config
        blacklist_addresses = self.config_manager.get("security.blacklist.addresses", [])
        blacklist_tokens = self.config_manager.get("security.blacklist.tokens", [])
        whitelist_addresses = self.config_manager.get("security.whitelist.addresses", [])
        whitelist_tokens = self.config_manager.get("security.whitelist.tokens", [])
        
        # Convert to sets for O(1) lookups
        self.blacklisted_addresses = set(blacklist_addresses)
        self.blacklisted_tokens = set(blacklist_tokens)
        self.whitelisted_addresses = set(whitelist_addresses)
        self.whitelisted_tokens = set(whitelist_tokens)
        
        logger.debug(f"Loaded {len(self.blacklisted_addresses)} blacklisted addresses")
        logger.debug(f"Loaded {len(self.blacklisted_tokens)} blacklisted tokens")
        logger.debug(f"Loaded {len(self.whitelisted_addresses)} whitelisted addresses")
        logger.debug(f"Loaded {len(self.whitelisted_tokens)} whitelisted tokens")
    
    def encrypt(self, data: str) -> str:
        """
        Encrypt sensitive data
        
        Args:
            data (str): The data to encrypt
            
        Returns:
            str: Base64 encoded encrypted data
        """
        try:
            encrypted = self.fernet.encrypt(data.encode())
            return encrypted.decode()
        except Exception as e:
            logger.error(f"Encryption error: {str(e)}")
            return ""
    
    def decrypt(self, encrypted_data: str) -> str:
        """
        Decrypt encrypted data
        
        Args:
            encrypted_data (str): Base64 encoded encrypted data
            
        Returns:
            str: Decrypted data
        """
        try:
            decrypted = self.fernet.decrypt(encrypted_data.encode())
            return decrypted.decode()
        except Exception as e:
            logger.error(f"Decryption error: {str(e)}")
            return ""
    
    def store_api_key(self, service: str, api_key: str):
        """
        Securely store an API key
        
        Args:
            service (str): Service name
            api_key (str): API key to store
        """
        encrypted_key = self.encrypt(api_key)
        self.encrypted_storage[f"api_key_{service}"] = encrypted_key
        logger.debug(f"Stored API key for {service}")
    
    def get_api_key(self, service: str) -> str:
        """
        Retrieve an API key
        
        Args:
            service (str): Service name
            
        Returns:
            str: Decrypted API key or empty string if not found
        """
        encrypted_key = self.encrypted_storage.get(f"api_key_{service}")
        
        if not encrypted_key:
            # Try to get from environment
            env_var = f"{service.upper()}_API_KEY"
            api_key = os.environ.get(env_var)
            
            if not api_key:
                # Try from config
                api_key = self.config_manager.get(f"{service.lower()}.api_key")
            
            if api_key:
                # Store for future use
                self.store_api_key(service, api_key)
                return api_key
            
            logger.warning(f"API key for {service} not found")
            return ""
        
        return self.decrypt(encrypted_key)
    
    def store_wallet_key(self, wallet_name: str, private_key: str):
        """
        Securely store a wallet private key
        
        Args:
            wallet_name (str): Wallet name or identifier
            private_key (str): Private key to store
        """
        encrypted_key = self.encrypt(private_key)
        self.encrypted_storage[f"wallet_key_{wallet_name}"] = encrypted_key
        logger.debug(f"Stored wallet key for {wallet_name}")
    
    def get_wallet_key(self, wallet_name: str = "default") -> str:
        """
        Retrieve a wallet private key
        
        Args:
            wallet_name (str): Wallet name or identifier
            
        Returns:
            str: Decrypted private key or empty string if not found
        """
        encrypted_key = self.encrypted_storage.get(f"wallet_key_{wallet_name}")
        
        if not encrypted_key:
            # Try to get from environment
            env_var = "WALLET_PRIVATE_KEY"
            private_key = os.environ.get(env_var)
            
            if not private_key:
                # Try from config
                private_key = self.config_manager.get("wallet.private_key")
            
            if private_key:
                # Store for future use
                self.store_wallet_key(wallet_name, private_key)
                return private_key
            
            logger.warning(f"Wallet key for {wallet_name} not found")
            return ""
        
        return self.decrypt(encrypted_key)
    
    def verify_transaction_signature(self, transaction: Dict[str, Any], signature: str) -> bool:
        """
        Verify a transaction signature
        
        Args:
            transaction (Dict[str, Any]): Transaction data
            signature (str): Transaction signature
            
        Returns:
            bool: True if signature is valid, False otherwise
        """
        # This is a placeholder for actual signature verification
        # In a real implementation, this would use the appropriate blockchain library
        # to verify the signature against the transaction data
        
        logger.debug("Transaction signature verification placeholder")
        return True
    
    def is_address_blacklisted(self, address: str) -> bool:
        """
        Check if an address is blacklisted
        
        Args:
            address (str): Address to check
            
        Returns:
            bool: True if address is blacklisted, False otherwise
        """
        return address in self.blacklisted_addresses
    
    def is_token_blacklisted(self, token_address: str) -> bool:
        """
        Check if a token is blacklisted
        
        Args:
            token_address (str): Token address to check
            
        Returns:
            bool: True if token is blacklisted, False otherwise
        """
        return token_address in self.blacklisted_tokens
    
    def is_address_whitelisted(self, address: str) -> bool:
        """
        Check if an address is whitelisted
        
        Args:
            address (str): Address to check
            
        Returns:
            bool: True if address is whitelisted, False otherwise
        """
        # If whitelist is empty, all addresses are allowed (except blacklisted ones)
        if not self.whitelisted_addresses:
            return not self.is_address_blacklisted(address)
        
        return address in self.whitelisted_addresses
    
    def is_token_whitelisted(self, token_address: str) -> bool:
        """
        Check if a token is whitelisted
        
        Args:
            token_address (str): Token address to check
            
        Returns:
            bool: True if token is whitelisted, False otherwise
        """
        # If whitelist is empty, all tokens are allowed (except blacklisted ones)
        if not self.whitelisted_tokens:
            return not self.is_token_blacklisted(token_address)
        
        return token_address in self.whitelisted_tokens
    
    def add_to_blacklist(self, item: str, item_type: str = "address"):
        """
        Add an item to the blacklist
        
        Args:
            item (str): Address or token to blacklist
            item_type (str): Type of item ('address' or 'token')
        """
        if item_type == "address":
            self.blacklisted_addresses.add(item)
            logger.info(f"Added address to blacklist: {item}")
        elif item_type == "token":
            self.blacklisted_tokens.add(item)
            logger.info(f"Added token to blacklist: {item}")
        else:
            logger.warning(f"Unknown blacklist item type: {item_type}")
    
    def add_to_whitelist(self, item: str, item_type: str = "address"):
        """
        Add an item to the whitelist
        
        Args:
            item (str): Address or token to whitelist
            item_type (str): Type of item ('address' or 'token')
        """
        if item_type == "address":
            self.whitelisted_addresses.add(item)
            logger.info(f"Added address to whitelist: {item}")
        elif item_type == "token":
            self.whitelisted_tokens.add(item)
            logger.info(f"Added token to whitelist: {item}")
        else:
            logger.warning(f"Unknown whitelist item type: {item_type}")
    
    def remove_from_blacklist(self, item: str, item_type: str = "address"):
        """
        Remove an item from the blacklist
        
        Args:
            item (str): Address or token to remove
            item_type (str): Type of item ('address' or 'token')
        """
        if item_type == "address":
            self.blacklisted_addresses.discard(item)
            logger.info(f"Removed address from blacklist: {item}")
        elif item_type == "token":
            self.blacklisted_tokens.discard(item)
            logger.info(f"Removed token from blacklist: {item}")
        else:
            logger.warning(f"Unknown blacklist item type: {item_type}")
    
    def remove_from_whitelist(self, item: str, item_type: str = "address"):
        """
        Remove an item from the whitelist
        
        Args:
            item (str): Address or token to remove
            item_type (str): Type of item ('address' or 'token')
        """
        if item_type == "address":
            self.whitelisted_addresses.discard(item)
            logger.info(f"Removed address from whitelist: {item}")
        elif item_type == "token":
            self.whitelisted_tokens.discard(item)
            logger.info(f"Removed token from whitelist: {item}")
        else:
            logger.warning(f"Unknown whitelist item type: {item_type}")
    
    def save_lists(self):
        """Save blacklists and whitelists to configuration"""
        self.config_manager.set("security.blacklist.addresses", list(self.blacklisted_addresses))
        self.config_manager.set("security.blacklist.tokens", list(self.blacklisted_tokens))
        self.config_manager.set("security.whitelist.addresses", list(self.whitelisted_addresses))
        self.config_manager.set("security.whitelist.tokens", list(self.whitelisted_tokens))
        logger.debug("Saved blacklists and whitelists to configuration")
    
    def detect_suspicious_activity(self, activity: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Detect suspicious activity based on various parameters
        
        Args:
            activity (Dict[str, Any]): Activity data
            
        Returns:
            Tuple[bool, str]: (is_suspicious, reason)
        """
        # This is a placeholder for actual suspicious activity detection
        # In a real implementation, this would use various heuristics to
        # detect potential security threats
        
        # Example check: Unusually large transaction amount
        if activity.get('amount', 0) > self.config_manager.get('security.max_transaction_amount', 1000000):
            return True, "Unusually large transaction amount"
        
        # Example check: Transaction to blacklisted address
        to_address = activity.get('to')
        if to_address and self.is_address_blacklisted(to_address):
            return True, f"Transaction to blacklisted address: {to_address}"
        
        # No suspicious activity detected
        return False, ""
