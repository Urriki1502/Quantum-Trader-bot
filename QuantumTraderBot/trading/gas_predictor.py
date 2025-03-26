"""
GasPredictor Component
Responsible for predicting optimal gas prices for transactions,
balancing cost and speed based on network conditions.
"""

import asyncio
import logging
import time
import statistics
from typing import Dict, Any, List, Optional, Tuple, Deque
from collections import deque

from core.config_manager import ConfigManager

logger = logging.getLogger(__name__)

class GasPredictor:
    """
    GasPredictor handles:
    - Predicting optimal gas prices
    - Monitoring network conditions
    - Balancing transaction speed and cost
    - Learning from past transactions
    """
    
    def __init__(self, config_manager: ConfigManager):
        """
        Initialize the GasPredictor
        
        Args:
            config_manager (ConfigManager): Configuration manager instance
        """
        self.config_manager = config_manager
        
        # Settings
        self.default_priority = self.config_manager.get('gas.default_priority', 'medium')
        self.max_gas_price = self.config_manager.get('gas.max_price', 1000)  # in lamports
        self.gas_price_history_size = 100
        
        # Gas price settings for different priority levels
        self.gas_multipliers = {
            'low': self.config_manager.get('gas.low_multiplier', 0.8),
            'medium': self.config_manager.get('gas.medium_multiplier', 1.0),
            'high': self.config_manager.get('gas.high_multiplier', 1.3),
            'urgent': self.config_manager.get('gas.urgent_multiplier', 2.0)
        }
        
        # Historical data
        self.gas_price_history = deque(maxlen=self.gas_price_history_size)
        self.last_network_check = 0
        self.current_base_gas_price = None
        self.network_congestion = 'normal'  # normal, high, extreme
        
        logger.info("GasPredictor initialized")
    
    async def predict_gas_price(self, priority: str = None) -> float:
        """
        Predict optimal gas price based on network conditions and priority
        
        Args:
            priority (str, optional): Transaction priority ('low', 'medium', 'high', 'urgent')
            
        Returns:
            float: Predicted gas price
        """
        if priority is None:
            priority = self.default_priority
        
        # Get network conditions if stale (>60 seconds)
        if time.time() - self.last_network_check > 60:
            await self._check_network_conditions()
        
        # If we still don't have a base gas price, use default
        if self.current_base_gas_price is None:
            logger.warning("No base gas price available, using default")
            self.current_base_gas_price = 20  # Default base gas price for Solana
        
        # Get multiplier for priority
        multiplier = self.gas_multipliers.get(priority, 1.0)
        
        # Apply network congestion adjustment
        if self.network_congestion == 'high':
            multiplier *= 1.5
        elif self.network_congestion == 'extreme':
            multiplier *= 2.5
        
        # Calculate predicted gas price
        predicted_gas = self.current_base_gas_price * multiplier
        
        # Cap at maximum gas price
        predicted_gas = min(predicted_gas, self.max_gas_price)
        
        logger.debug(f"Predicted gas price for {priority} priority: {predicted_gas:.2f}")
        return predicted_gas
    
    async def _check_network_conditions(self):
        """Check current network conditions and update base gas price"""
        try:
            # In a real implementation, this would query the Solana network
            # for current gas prices and network congestion
            
            # For now, simulate network conditions based on time patterns
            hour = time.localtime().tm_hour
            
            # Simulate network congestion based on time of day
            # Busier during business hours
            if 8 <= hour <= 17:
                # Business hours - higher gas prices
                self.current_base_gas_price = 25
                
                # Simulate random congestion events
                if time.time() % 3600 < 600:  # 10 minutes of each hour
                    self.network_congestion = 'high'
                else:
                    self.network_congestion = 'normal'
            else:
                # Off hours - lower gas prices
                self.current_base_gas_price = 15
                self.network_congestion = 'normal'
            
            # Add to history
            self.gas_price_history.append({
                'timestamp': time.time(),
                'base_price': self.current_base_gas_price,
                'congestion': self.network_congestion
            })
            
            self.last_network_check = time.time()
            logger.debug(f"Updated network conditions - Base gas: {self.current_base_gas_price}, Congestion: {self.network_congestion}")
            
        except Exception as e:
            logger.error(f"Error checking network conditions: {str(e)}")
    
    async def estimate_transaction_cost(self, 
                                      instruction_count: int, 
                                      priority: str = None) -> Dict[str, float]:
        """
        Estimate transaction cost based on instruction count and gas price
        
        Args:
            instruction_count (int): Number of instructions in the transaction
            priority (str, optional): Transaction priority
            
        Returns:
            Dict[str, float]: Estimated cost in SOL and USD
        """
        # Get predicted gas price
        gas_price = await self.predict_gas_price(priority)
        
        # Basic transaction cost model for Solana
        # Base cost + per-instruction cost
        base_cost = 5000  # Base transaction cost in lamports
        per_instruction = 2000  # Additional cost per instruction in lamports
        
        # Calculate total cost in lamports
        total_lamports = base_cost + (instruction_count * per_instruction)
        
        # Convert to SOL (1 SOL = 1,000,000,000 lamports)
        total_sol = total_lamports / 1_000_000_000
        
        # Estimate USD cost (would use actual SOL price in a real implementation)
        sol_price_usd = 20  # Placeholder SOL price
        total_usd = total_sol * sol_price_usd
        
        logger.debug(f"Estimated transaction cost: {total_sol:.9f} SOL (${total_usd:.4f})")
        
        return {
            'lamports': total_lamports,
            'sol': total_sol,
            'usd': total_usd,
            'gas_price': gas_price
        }
    
    async def optimize_batch_timing(self, transactions_count: int) -> Dict[str, Any]:
        """
        Optimize timing for batch transactions
        
        Args:
            transactions_count (int): Number of transactions in batch
            
        Returns:
            Dict[str, Any]: Optimization recommendations
        """
        # Get current network conditions
        if time.time() - self.last_network_check > 60:
            await self._check_network_conditions()
        
        # Calculate optimal timing
        if self.network_congestion == 'extreme':
            # During extreme congestion, spread transactions over longer time
            delay_between = 5.0
            priority = 'high'
        elif self.network_congestion == 'high':
            # During high congestion, moderate delay
            delay_between = 2.0
            priority = 'medium'
        else:
            # Normal conditions, minimal delay
            delay_between = 0.5
            priority = 'medium'
        
        # Calculate total time
        total_time = delay_between * (transactions_count - 1)
        
        return {
            'delay_between_tx': delay_between,
            'total_time': total_time,
            'recommended_priority': priority,
            'network_congestion': self.network_congestion
        }
    
    def get_gas_price_stats(self) -> Dict[str, Any]:
        """
        Get statistics about gas prices
        
        Returns:
            Dict[str, Any]: Gas price statistics
        """
        if not self.gas_price_history:
            return {
                'current': self.current_base_gas_price or 0,
                'min': 0,
                'max': 0,
                'avg': 0,
                'median': 0
            }
        
        prices = [entry['base_price'] for entry in self.gas_price_history]
        
        return {
            'current': self.current_base_gas_price,
            'min': min(prices),
            'max': max(prices),
            'avg': statistics.mean(prices),
            'median': statistics.median(prices),
            'congestion': self.network_congestion,
            'history_count': len(self.gas_price_history)
        }
    
    def learn_from_transaction(self, 
                              gas_used: float, 
                              confirmation_time: float, 
                              transaction_priority: str):
        """
        Learn from completed transaction to improve predictions
        
        Args:
            gas_used (float): Actual gas used
            confirmation_time (float): Time to confirmation in seconds
            transaction_priority (str): Transaction priority used
        """
        logger.debug(f"Learning from transaction - Gas: {gas_used}, Time: {confirmation_time}s, Priority: {transaction_priority}")
        
        # In a real implementation, this would adjust prediction models
        # based on actual transaction outcomes
        
        # Simple learning: adjust multipliers based on confirmation time
        target_times = {
            'low': 30,  # 30 seconds for low priority
            'medium': 15,  # 15 seconds for medium priority
            'high': 5,  # 5 seconds for high priority
            'urgent': 2  # 2 seconds for urgent priority
        }
        
        if transaction_priority in target_times:
            target_time = target_times[transaction_priority]
            current_multiplier = self.gas_multipliers[transaction_priority]
            
            # If confirmation took longer than target, increase multiplier
            if confirmation_time > target_time * 1.5:
                new_multiplier = current_multiplier * 1.05  # Increase by 5%
                self.gas_multipliers[transaction_priority] = min(new_multiplier, 4.0)  # Cap at 4x
            
            # If confirmation was much faster than target, decrease multiplier
            elif confirmation_time < target_time * 0.5:
                new_multiplier = current_multiplier * 0.95  # Decrease by 5%
                self.gas_multipliers[transaction_priority] = max(new_multiplier, 0.5)  # Floor at 0.5x
