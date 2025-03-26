"""
MEVProtection Component
Responsible for protecting trades from MEV (Maximal Extractable Value) attacks,
such as front-running and sandwich attacks.
"""

import asyncio
import logging
import random
import time
from typing import Dict, Any, List, Optional, Tuple

from core.config_manager import ConfigManager

logger = logging.getLogger(__name__)

class MEVProtection:
    """
    MEVProtection handles:
    - Detection of potential sandwich attacks
    - Protection against front-running
    - Transaction timing optimization
    - Safe transaction routing
    """
    
    def __init__(self, config_manager: ConfigManager):
        """
        Initialize the MEVProtection
        
        Args:
            config_manager (ConfigManager): Configuration manager instance
        """
        self.config_manager = config_manager
        
        # Settings
        self.protection_enabled = self.config_manager.get('mev.protection_enabled', True)
        self.randomization_enabled = self.config_manager.get('mev.randomize_amounts', True)
        self.randomization_percent = self.config_manager.get('mev.randomization_percentage', 2)
        self.delay_enabled = self.config_manager.get('mev.delay_enabled', True)
        self.min_delay = self.config_manager.get('mev.min_delay_seconds', 0)
        self.max_delay = self.config_manager.get('mev.max_delay_seconds', 5)
        
        # Tracking data
        self.sandwich_detections = {}  # token_address -> detection count
        self.protected_transactions = 0
        
        logger.info("MEVProtection initialized")
    
    async def protect_transaction(self, 
                                token_address: str, 
                                amount_usd: float) -> Dict[str, Any]:
        """
        Apply MEV protection to a transaction
        
        Args:
            token_address (str): Token address
            amount_usd (float): Transaction amount in USD
            
        Returns:
            Dict[str, Any]: Protection result with potentially modified parameters
        """
        if not self.protection_enabled:
            return {
                'protected': False,
                'modified_amount': amount_usd,
                'delay': 0,
                'reason': 'Protection disabled'
            }
        
        logger.debug(f"Applying MEV protection for {token_address}")
        
        modified_amount = amount_usd
        delay_seconds = 0
        protection_applied = False
        applied_techniques = []
        
        # Apply amount randomization if enabled
        if self.randomization_enabled and amount_usd > 0:
            original_amount = amount_usd
            
            # Calculate random adjustment within the randomization percentage
            random_factor = 1.0 + random.uniform(
                -self.randomization_percent / 100, 
                self.randomization_percent / 100
            )
            modified_amount = amount_usd * random_factor
            
            # Ensure minimum trade size (at least $10)
            modified_amount = max(10, modified_amount)
            
            logger.debug(f"Randomized amount: ${amount_usd:.2f} -> ${modified_amount:.2f}")
            protection_applied = True
            applied_techniques.append('amount_randomization')
        
        # Apply transaction delay if enabled
        if self.delay_enabled:
            delay_seconds = random.uniform(self.min_delay, self.max_delay)
            
            if delay_seconds > 0:
                logger.debug(f"Applying transaction delay: {delay_seconds:.2f} seconds")
                await asyncio.sleep(delay_seconds)
                protection_applied = True
                applied_techniques.append('timing_delay')
        
        # Record protection
        if protection_applied:
            self.protected_transactions += 1
        
        return {
            'protected': protection_applied,
            'modified_amount': modified_amount,
            'delay': delay_seconds,
            'techniques': applied_techniques,
            'original_amount': amount_usd
        }
    
    async def detect_sandwich_attack(self, 
                                   token_address: str, 
                                   price_before: float,
                                   price_after: float) -> bool:
        """
        Detect potential sandwich attack based on price movement
        
        Args:
            token_address (str): Token address
            price_before (float): Price before transaction
            price_after (float): Price after transaction
            
        Returns:
            bool: True if sandwich attack detected, False otherwise
        """
        # Calculate price impact percentage
        price_change_pct = ((price_after - price_before) / price_before) * 100
        
        # Define threshold for suspicious price movement
        threshold = self.config_manager.get('mev.sandwich_detection_threshold', 3)
        
        # Check if price movement exceeds threshold
        is_suspicious = abs(price_change_pct) > threshold
        
        if is_suspicious:
            logger.warning(f"Potential sandwich attack detected for {token_address}: {price_change_pct:.2f}% price change")
            
            # Record detection
            if token_address in self.sandwich_detections:
                self.sandwich_detections[token_address] += 1
            else:
                self.sandwich_detections[token_address] = 1
        
        return is_suspicious
    
    async def optimize_transaction_timing(self, block_rate: float = 1.0) -> float:
        """
        Optimize transaction timing to minimize MEV exposure
        
        Args:
            block_rate (float): Average block time in seconds
            
        Returns:
            float: Recommended delay in seconds
        """
        # Simple transaction timing optimization based on random delays
        # to avoid predictable transaction patterns
        
        # Calculate optimal timing
        optimal_delay = random.uniform(0, block_rate * 0.8)
        
        logger.debug(f"Optimal transaction timing delay: {optimal_delay:.2f} seconds")
        return optimal_delay
    
    def get_protection_stats(self) -> Dict[str, Any]:
        """
        Get protection statistics
        
        Returns:
            Dict[str, Any]: Protection statistics
        """
        return {
            'protected_transactions': self.protected_transactions,
            'sandwich_detections': sum(self.sandwich_detections.values()),
            'tokens_with_sandwich_attacks': len(self.sandwich_detections),
            'protection_enabled': self.protection_enabled,
            'randomization_enabled': self.randomization_enabled,
            'delay_enabled': self.delay_enabled
        }
    
    def get_high_risk_tokens(self, threshold: int = 2) -> List[str]:
        """
        Get tokens with high risk of MEV attacks
        
        Args:
            threshold (int): Minimum number of detections to consider high risk
            
        Returns:
            List[str]: List of high-risk token addresses
        """
        return [
            token for token, count in self.sandwich_detections.items()
            if count >= threshold
        ]
