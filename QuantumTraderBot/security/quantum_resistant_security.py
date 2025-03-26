"""
Quantum-Resistant Security Component
Provides advanced security measures to protect against current and future 
quantum computing threats, ensuring long-term security for the trading bot.
"""

import logging
import os
import time
import base64
import hashlib
import hmac
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.fernet import Fernet

from core.state_manager import StateManager
from core.config_manager import ConfigManager
from core.security_manager import SecurityManager

logger = logging.getLogger(__name__)

class PostQuantumCryptoEngine:
    """
    Implements post-quantum cryptographic operations
    
    While true quantum-resistant algorithms like CRYSTALS-Kyber and CRYSTALS-Dilithium
    would be used in production, this implementation uses hybrid approaches with
    conventional crypto as a practical compromise.
    """
    
    def __init__(self, config_manager: ConfigManager):
        """
        Initialize the PostQuantumCryptoEngine
        
        Args:
            config_manager (ConfigManager): Configuration manager instance
        """
        self.config_manager = config_manager
        
        # Load security settings
        self.security_level = self.config_manager.get('security.quantum_resistant.level', 'standard')
        
        # Key derivation settings
        self.kdf_iterations = self._get_kdf_iterations()
        self.hash_algorithm = self._get_hash_algorithm()
        
        # Initialize key cache
        self.key_cache = {}
        
        logger.info(f"PostQuantumCryptoEngine initialized with security level: {self.security_level}")
    
    def _get_kdf_iterations(self) -> int:
        """
        Get KDF iteration count based on security level
        
        Returns:
            int: KDF iteration count
        """
        if self.security_level == 'high':
            return 1000000  # High security
        elif self.security_level == 'medium':
            return 500000   # Medium security
        else:
            return 100000   # Standard security
    
    def _get_hash_algorithm(self) -> Callable:
        """
        Get hash algorithm based on security level
        
        Returns:
            Callable: Hash algorithm
        """
        if self.security_level == 'high':
            return hashes.SHA512()
        elif self.security_level == 'medium':
            return hashes.SHA384()
        else:
            return hashes.SHA256()
    
    def derive_key(self, 
                  base_key: bytes, 
                  salt: Optional[bytes] = None,
                  info: Optional[bytes] = None) -> bytes:
        """
        Derive a cryptographic key using PBKDF2
        
        Args:
            base_key (bytes): Base key material
            salt (bytes, optional): Salt for key derivation
            info (bytes, optional): Additional context info
            
        Returns:
            bytes: Derived key
        """
        # Generate salt if not provided
        if salt is None:
            salt = os.urandom(16)
        
        # Use info as additional salt if provided
        if info is not None:
            salt = hashlib.sha256(salt + info).digest()
        
        # Create cache key
        cache_key = base64.b64encode(base_key + salt).decode('utf-8')
        
        # Check cache
        if cache_key in self.key_cache:
            return self.key_cache[cache_key]
        
        # Derive key using PBKDF2
        kdf = PBKDF2HMAC(
            algorithm=self.hash_algorithm,
            length=32,  # 256-bit key
            salt=salt,
            iterations=self.kdf_iterations
        )
        
        derived_key = kdf.derive(base_key)
        
        # Cache derived key
        self.key_cache[cache_key] = derived_key
        
        return derived_key
    
    def hybrid_sign(self, 
                   data: Union[str, bytes], 
                   private_key: bytes,
                   key_info: Optional[bytes] = None) -> Dict[str, bytes]:
        """
        Create a hybrid signature using both conventional and hash-based methods
        
        Args:
            data (Union[str, bytes]): Data to sign
            private_key (bytes): Private key for signing
            key_info (bytes, optional): Additional key context
            
        Returns:
            Dict[str, bytes]: Signature data
        """
        # Convert string to bytes if needed
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        # Get current timestamp for freshness
        timestamp = int(time.time()).to_bytes(8, byteorder='big')
        
        # Derive signing keys
        signing_key = self.derive_key(private_key, salt=timestamp, info=key_info)
        
        # Create conventional signature (HMAC-SHA256)
        conventional_sig = hmac.new(
            key=signing_key,
            msg=data + timestamp,
            digestmod=hashlib.sha256
        ).digest()
        
        # Create hash-based signature
        # In a real implementation, this would use a true post-quantum algorithm
        # For now, use a different hash function for demonstration
        hash_based_sig = hmac.new(
            key=signing_key,
            msg=data + conventional_sig,
            digestmod=hashlib.sha512
        ).digest()
        
        # Combine signatures
        return {
            'data_hash': hashlib.sha256(data).digest(),
            'timestamp': timestamp,
            'conventional_sig': conventional_sig,
            'hash_based_sig': hash_based_sig
        }
    
    def verify_hybrid_signature(self, 
                               data: Union[str, bytes], 
                               signature: Dict[str, bytes],
                               public_key: bytes,
                               key_info: Optional[bytes] = None,
                               max_age_seconds: int = 300) -> bool:
        """
        Verify a hybrid signature
        
        Args:
            data (Union[str, bytes]): Data to verify
            signature (Dict[str, bytes]): Signature data
            public_key (bytes): Public key for verification
            key_info (bytes, optional): Additional key context
            max_age_seconds (int): Maximum age of signature in seconds
            
        Returns:
            bool: True if signature is valid, False otherwise
        """
        # Convert string to bytes if needed
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        # Check data hash
        if hashlib.sha256(data).digest() != signature.get('data_hash'):
            logger.warning("Data hash mismatch in signature verification")
            return False
        
        # Check timestamp freshness
        timestamp = signature.get('timestamp')
        if timestamp:
            sig_time = int.from_bytes(timestamp, byteorder='big')
            current_time = int(time.time())
            
            if current_time - sig_time > max_age_seconds:
                logger.warning(f"Signature too old: {current_time - sig_time} seconds")
                return False
        
        # Derive verification key
        verification_key = self.derive_key(public_key, salt=timestamp, info=key_info)
        
        # Verify conventional signature
        conventional_sig = signature.get('conventional_sig')
        expected_conventional_sig = hmac.new(
            key=verification_key,
            msg=data + timestamp,
            digestmod=hashlib.sha256
        ).digest()
        
        if not hmac.compare_digest(conventional_sig, expected_conventional_sig):
            logger.warning("Conventional signature verification failed")
            return False
        
        # Verify hash-based signature
        hash_based_sig = signature.get('hash_based_sig')
        expected_hash_based_sig = hmac.new(
            key=verification_key,
            msg=data + conventional_sig,
            digestmod=hashlib.sha512
        ).digest()
        
        if not hmac.compare_digest(hash_based_sig, expected_hash_based_sig):
            logger.warning("Hash-based signature verification failed")
            return False
        
        return True
    
    def lattice_encrypt(self, 
                      data: Union[str, bytes], 
                      key: bytes) -> Dict[str, bytes]:
        """
        Encrypt data using a hybrid approach (symmetric + key derivation)
        
        In a real implementation, this would use lattice-based encryption
        
        Args:
            data (Union[str, bytes]): Data to encrypt
            key (bytes): Encryption key
            
        Returns:
            Dict[str, bytes]: Encrypted data with metadata
        """
        # Convert string to bytes if needed
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        # Generate a random salt
        salt = os.urandom(16)
        
        # Derive encryption key
        encryption_key = self.derive_key(key, salt=salt)
        
        # Create a Fernet key
        fernet_key = base64.urlsafe_b64encode(encryption_key)
        
        # Encrypt data
        f = Fernet(fernet_key)
        ciphertext = f.encrypt(data)
        
        return {
            'ciphertext': ciphertext,
            'salt': salt,
            'version': b'v1'  # For future compatibility
        }
    
    def lattice_decrypt(self, 
                      encrypted_data: Dict[str, bytes], 
                      key: bytes) -> bytes:
        """
        Decrypt data encrypted with lattice_encrypt
        
        Args:
            encrypted_data (Dict[str, bytes]): Encrypted data with metadata
            key (bytes): Decryption key
            
        Returns:
            bytes: Decrypted data
        """
        # Extract components
        ciphertext = encrypted_data.get('ciphertext')
        salt = encrypted_data.get('salt')
        
        if not ciphertext or not salt:
            raise ValueError("Invalid encrypted data format")
        
        # Derive decryption key
        decryption_key = self.derive_key(key, salt=salt)
        
        # Create a Fernet key
        fernet_key = base64.urlsafe_b64encode(decryption_key)
        
        # Decrypt data
        f = Fernet(fernet_key)
        
        try:
            plaintext = f.decrypt(ciphertext)
            return plaintext
        except Exception as e:
            logger.error(f"Decryption failed: {str(e)}")
            raise ValueError("Decryption failed") from e


class QuantumResistantSecurity:
    """
    Provides quantum-resistant security features
    
    This component adds an additional layer of security to protect
    against future quantum computing threats.
    """
    
    def __init__(self, 
                config_manager: ConfigManager,
                state_manager: StateManager,
                security_manager: SecurityManager):
        """
        Initialize the QuantumResistantSecurity
        
        Args:
            config_manager (ConfigManager): Configuration manager instance
            state_manager (StateManager): State manager instance
            security_manager (SecurityManager): Security manager instance
        """
        self.config_manager = config_manager
        self.state_manager = state_manager
        self.security_manager = security_manager
        
        # Initialize crypto engine
        self.crypto_engine = PostQuantumCryptoEngine(config_manager)
        
        # Security indicators
        self.intrusion_attempts = 0
        self.last_security_check = time.time()
        
        logger.info("QuantumResistantSecurity initialized")
    
    def secure_key_operations(self, 
                            operation_type: str, 
                            data: Any,
                            key_identifier: Optional[str] = None) -> Any:
        """
        Perform secure key operations with quantum-resistant protection
        
        Args:
            operation_type (str): Operation type ('sign', 'verify', 'encrypt', 'decrypt')
            data (Any): Data for the operation
            key_identifier (str, optional): Key identifier, defaults to wallet key
            
        Returns:
            Any: Operation result
        """
        # Get the appropriate key
        if key_identifier:
            key = self.security_manager.get_raw_key(key_identifier)
        else:
            # Default to wallet private key
            key = self.security_manager.get_wallet_private_key()
        
        if not key:
            logger.error(f"No key found for identifier: {key_identifier}")
            raise ValueError(f"No key found for identifier: {key_identifier}")
        
        # Perform the requested operation
        if operation_type == 'sign':
            return self.crypto_engine.hybrid_sign(data, key)
        elif operation_type == 'verify':
            signature, verify_data = data
            return self.crypto_engine.verify_hybrid_signature(verify_data, signature, key)
        elif operation_type == 'encrypt':
            return self.crypto_engine.lattice_encrypt(data, key)
        elif operation_type == 'decrypt':
            return self.crypto_engine.lattice_decrypt(data, key)
        else:
            logger.error(f"Unknown operation type: {operation_type}")
            raise ValueError(f"Unknown operation type: {operation_type}")
    
    def secure_transaction(self, 
                         transaction_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply quantum-resistant security to a transaction
        
        Args:
            transaction_data (Dict[str, Any]): Transaction data
            
        Returns:
            Dict[str, Any]: Secured transaction data
        """
        # Create a secure copy of the transaction data
        secured_data = transaction_data.copy()
        
        # Add security metadata
        secured_data['security'] = {
            'timestamp': int(time.time()),
            'version': '1.0'
        }
        
        # Convert to bytes for signing
        tx_bytes = self._serialize_transaction(secured_data)
        
        # Sign the transaction
        signature = self.crypto_engine.hybrid_sign(
            tx_bytes, 
            self.security_manager.get_wallet_private_key()
        )
        
        # Add signature to transaction data
        secured_data['signature'] = {
            'timestamp': base64.b64encode(signature['timestamp']).decode('utf-8'),
            'data_hash': base64.b64encode(signature['data_hash']).decode('utf-8'),
            'conventional_sig': base64.b64encode(signature['conventional_sig']).decode('utf-8'),
            'hash_based_sig': base64.b64encode(signature['hash_based_sig']).decode('utf-8')
        }
        
        return secured_data
    
    def verify_secured_transaction(self, transaction_data: Dict[str, Any]) -> bool:
        """
        Verify a secured transaction
        
        Args:
            transaction_data (Dict[str, Any]): Secured transaction data
            
        Returns:
            bool: True if transaction is valid, False otherwise
        """
        # Extract signature
        signature_data = transaction_data.get('signature')
        if not signature_data:
            logger.warning("No signature data in transaction")
            self._record_intrusion_attempt()
            return False
        
        # Create a copy without the signature
        tx_data = transaction_data.copy()
        tx_data.pop('signature', None)
        
        # Serialize for verification
        tx_bytes = self._serialize_transaction(tx_data)
        
        # Decode signature components
        signature = {
            'timestamp': base64.b64decode(signature_data['timestamp']),
            'data_hash': base64.b64decode(signature_data['data_hash']),
            'conventional_sig': base64.b64decode(signature_data['conventional_sig']),
            'hash_based_sig': base64.b64decode(signature_data['hash_based_sig'])
        }
        
        # Verify signature
        return self.crypto_engine.verify_hybrid_signature(
            tx_bytes,
            signature,
            self.security_manager.get_wallet_public_key()
        )
    
    def _serialize_transaction(self, tx_data: Dict[str, Any]) -> bytes:
        """
        Serialize transaction data for signing/verification
        
        Args:
            tx_data (Dict[str, Any]): Transaction data
            
        Returns:
            bytes: Serialized transaction
        """
        # Simple serialization for demonstration
        # In production, would use a standardized format like CBOR or MessagePack
        tx_str = str(sorted([(k, str(v)) for k, v in tx_data.items()]))
        return tx_str.encode('utf-8')
    
    def secure_api_key(self, api_key: str) -> str:
        """
        Apply additional security to an API key
        
        Args:
            api_key (str): API key
            
        Returns:
            str: Secured API key
        """
        # Mix with hardware-specific information
        salt = self.security_manager.get_hardware_fingerprint().encode('utf-8')
        
        # Derive a more secure key
        secured_key = self.crypto_engine.derive_key(
            api_key.encode('utf-8'), 
            salt=salt
        )
        
        # Convert to usable format
        return base64.urlsafe_b64encode(secured_key).decode('utf-8')
    
    def perform_security_audit(self) -> Dict[str, Any]:
        """
        Perform a security audit of the system
        
        Returns:
            Dict[str, Any]: Audit results
        """
        audit_results = {
            'timestamp': time.time(),
            'intrusion_attempts': self.intrusion_attempts,
            'last_security_check': self.last_security_check,
            'checks': {}
        }
        
        # Check key integrity
        wallet_key_status = self._check_wallet_key_integrity()
        audit_results['checks']['wallet_key_integrity'] = wallet_key_status
        
        # Check for suspicious activities
        suspicious_activity = self._check_suspicious_activity()
        audit_results['checks']['suspicious_activity'] = suspicious_activity
        
        # Update last check time
        self.last_security_check = time.time()
        
        # Update state
        self.state_manager.update_component_metrics(
            'quantum_resistant_security', 
            {
                'last_audit': self.last_security_check,
                'intrusion_attempts': self.intrusion_attempts,
                'wallet_key_integrity': wallet_key_status['status'],
                'suspicious_activity': suspicious_activity['status']
            }
        )
        
        return audit_results
    
    def _check_wallet_key_integrity(self) -> Dict[str, Any]:
        """
        Check wallet key integrity
        
        Returns:
            Dict[str, Any]: Check results
        """
        # Get wallet keys
        private_key = self.security_manager.get_wallet_private_key()
        public_key = self.security_manager.get_wallet_public_key()
        
        if not private_key or not public_key:
            return {
                'status': 'failed',
                'reason': 'Wallet keys not available'
            }
        
        # Create test data
        test_data = f"integrity_check_{time.time()}".encode('utf-8')
        
        try:
            # Sign data
            signature = self.crypto_engine.hybrid_sign(test_data, private_key)
            
            # Verify signature
            valid = self.crypto_engine.verify_hybrid_signature(
                test_data, signature, public_key)
            
            if valid:
                return {
                    'status': 'passed',
                    'details': 'Wallet key integrity verified'
                }
            else:
                self._record_intrusion_attempt()
                return {
                    'status': 'failed',
                    'reason': 'Signature verification failed'
                }
                
        except Exception as e:
            logger.error(f"Wallet key integrity check failed: {str(e)}")
            return {
                'status': 'failed',
                'reason': f"Error during check: {str(e)}"
            }
    
    def _check_suspicious_activity(self) -> Dict[str, Any]:
        """
        Check for suspicious activities
        
        Returns:
            Dict[str, Any]: Check results
        """
        # In production, would implement more sophisticated checks
        # For now, just check intrusion attempts
        
        if self.intrusion_attempts > 5:
            return {
                'status': 'warning',
                'reason': f"High number of intrusion attempts: {self.intrusion_attempts}"
            }
        
        return {
            'status': 'passed',
            'details': 'No suspicious activity detected'
        }
    
    def _record_intrusion_attempt(self):
        """Record a potential intrusion attempt"""
        self.intrusion_attempts += 1
        logger.warning(f"Potential intrusion attempt detected (total: {self.intrusion_attempts})")
        
        # Update state
        self.state_manager.update_component_status(
            'quantum_resistant_security',
            'warning' if self.intrusion_attempts > 5 else 'running',
            f"Intrusion attempts: {self.intrusion_attempts}"
        )
        
        # Alert if threshold exceeded
        if self.intrusion_attempts > 10:
            self.state_manager.add_alert(
                'quantum_resistant_security',
                'CRITICAL',
                f"High number of intrusion attempts detected: {self.intrusion_attempts}"
            )
    
    def reset_security_state(self):
        """Reset security state (use with caution)"""
        self.intrusion_attempts = 0
        self.last_security_check = time.time()
        
        # Update state
        self.state_manager.update_component_status(
            'quantum_resistant_security',
            'running',
            'Security state reset'
        )