"""
DEX Interface Module
Provides abstract base class and utilities for interfacing with decentralized exchanges.
This module makes it possible to support multiple DEXes with a unified interface.
"""

import logging
import asyncio
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple, Union

from utils.api_resilience import with_retry, with_timeout
from core.state_manager import StateManager
from core.config_manager import ConfigManager
from core.security_manager import SecurityManager

logger = logging.getLogger(__name__)

class DEXInterface(ABC):
    """
    Abstract base class for DEX interfaces
    Defines the required methods that all DEX implementations must provide
    """
    
    def __init__(self, 
                config_manager: ConfigManager,
                state_manager: StateManager,
                security_manager: SecurityManager):
        """
        Initialize the DEX Interface
        
        Args:
            config_manager (ConfigManager): Configuration manager instance
            state_manager (StateManager): State manager instance
            security_manager (SecurityManager): Security manager instance
        """
        self.config_manager = config_manager
        self.state_manager = state_manager
        self.security_manager = security_manager
        
        # DEX identification
        self.dex_name = "abstract"
        self.dex_display_name = "Abstract DEX"
        
        # State
        self.is_running = False
        self.is_connected = False
        
    @abstractmethod
    async def start(self):
        """Start the DEX interface and establish connections"""
        pass
        
    @abstractmethod
    async def stop(self):
        """Stop the DEX interface and close all connections"""
        pass
        
    @abstractmethod
    async def get_token_info(self, token_address: str) -> Optional[Dict[str, Any]]:
        """
        Get token information
        
        Args:
            token_address (str): Token address
            
        Returns:
            Optional[Dict[str, Any]]: Token information or None if not found
        """
        pass
        
    @abstractmethod
    async def get_token_price(self, token_address: str) -> Optional[float]:
        """
        Get current token price in USD
        
        Args:
            token_address (str): Token address
            
        Returns:
            Optional[float]: Token price in USD or None if not available
        """
        pass
        
    @abstractmethod
    async def get_token_liquidity(self, token_address: str) -> Optional[float]:
        """
        Get token liquidity in USD
        
        Args:
            token_address (str): Token address
            
        Returns:
            Optional[float]: Token liquidity in USD or None if not available
        """
        pass
        
    @abstractmethod
    async def buy_token(self, 
                       token_address: str, 
                       amount_usd: float,
                       max_slippage: float = 2.0) -> Dict[str, Any]:
        """
        Buy a token
        
        Args:
            token_address (str): Token address
            amount_usd (float): Amount to spend in USD
            max_slippage (float): Maximum allowed slippage percentage
            
        Returns:
            Dict[str, Any]: Buy result
        """
        pass
        
    @abstractmethod
    async def sell_token(self, 
                        token_address: str, 
                        percent_of_holdings: float = 100.0,
                        max_slippage: float = 2.0) -> Dict[str, Any]:
        """
        Sell a token
        
        Args:
            token_address (str): Token address
            percent_of_holdings (float): Percentage of holdings to sell (0-100)
            max_slippage (float): Maximum allowed slippage percentage
            
        Returns:
            Dict[str, Any]: Sell result
        """
        pass
        
    @abstractmethod
    async def get_wallet_tokens(self) -> List[Dict[str, Any]]:
        """
        Get list of tokens in the wallet
        
        Returns:
            List[Dict[str, Any]]: List of token data
        """
        pass
        
    @abstractmethod
    async def get_wallet_balance(self, token_address: Optional[str] = None) -> float:
        """
        Get wallet balance for a token or native coin
        
        Args:
            token_address (str, optional): Token address or None for native coin
            
        Returns:
            float: Token balance or native coin balance
        """
        pass
        
    @abstractmethod
    async def estimate_price_impact(self, 
                                  token_address: str, 
                                  amount_usd: float) -> float:
        """
        Estimate price impact for a trade
        
        Args:
            token_address (str): Token address
            amount_usd (float): Trade amount in USD
            
        Returns:
            float: Estimated price impact as a percentage
        """
        pass
        
    @abstractmethod
    async def get_historical_prices(self, 
                                  token_address: str,
                                  timeframe: str = '1h',
                                  limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get historical price data for a token
        
        Args:
            token_address (str): Token address
            timeframe (str): Timeframe (e.g., '1m', '5m', '1h', '1d')
            limit (int): Maximum number of data points
            
        Returns:
            List[Dict[str, Any]]: List of OHLC price data
        """
        pass
        
    @abstractmethod
    async def calculate_atr(self, 
                          token_address: str, 
                          period: int = 14,
                          timeframe: str = '1m') -> Optional[float]:
        """
        Calculate Average True Range (ATR) for a token
        
        Args:
            token_address (str): Token address
            period (int): ATR period
            timeframe (str): Timeframe for price data
            
        Returns:
            Optional[float]: ATR value or None if not available
        """
        pass
    
    def _update_component_status(self, status: str, error_message: Optional[str] = None):
        """
        Update component status in state manager
        
        Args:
            status (str): New status
            error_message (str, optional): Error message if status is 'error'
        """
        component_name = f"dex_{self.dex_name}"
        self.state_manager.update_component_status(component_name, status, error_message)
        
    def _update_metric(self, metric_name: str, value: Any):
        """
        Update a metric in state manager
        
        Args:
            metric_name (str): Metric name
            value (Any): Metric value
        """
        component_name = f"dex_{self.dex_name}"
        self.state_manager.update_component_metric(component_name, metric_name, value)


class DEXManager:
    """
    Manages and coordinates multiple DEX interfaces
    Provides unified access to the best available DEX for each operation
    """
    
    def __init__(self, 
                config_manager: ConfigManager,
                state_manager: StateManager):
        """
        Initialize the DEX Manager
        
        Args:
            config_manager (ConfigManager): Configuration manager instance
            state_manager (StateManager): State manager instance
        """
        self.config_manager = config_manager
        self.state_manager = state_manager
        
        # DEX interfaces
        self.dex_interfaces: Dict[str, DEXInterface] = {}
        
        # Default/preferred DEX
        self.default_dex_name = self.config_manager.get('dex.default', 'raydium')
        
        # DEX priority order (fallback sequence)
        self.dex_priority = self.config_manager.get('dex.priority', ['raydium', 'jupiter', 'orca'])
        
    def register_dex(self, dex_interface: DEXInterface):
        """
        Register a DEX interface
        
        Args:
            dex_interface (DEXInterface): DEX interface instance
        """
        self.dex_interfaces[dex_interface.dex_name] = dex_interface
        logger.info(f"Registered DEX interface: {dex_interface.dex_display_name}")
        
    async def start_all(self):
        """Start all registered DEX interfaces"""
        for dex_name, dex in self.dex_interfaces.items():
            try:
                logger.info(f"Starting DEX interface: {dex.dex_display_name}")
                await dex.start()
            except Exception as e:
                logger.error(f"Failed to start DEX interface {dex.dex_display_name}: {str(e)}")
                
    async def stop_all(self):
        """Stop all registered DEX interfaces"""
        for dex_name, dex in self.dex_interfaces.items():
            try:
                await dex.stop()
            except Exception as e:
                logger.error(f"Error stopping DEX interface {dex.dex_display_name}: {str(e)}")
                
    def get_dex(self, dex_name: Optional[str] = None) -> Optional[DEXInterface]:
        """
        Get a specific DEX interface by name
        
        Args:
            dex_name (str, optional): DEX name or None for default
            
        Returns:
            Optional[DEXInterface]: DEX interface or None if not found
        """
        if dex_name is None:
            dex_name = self.default_dex_name
            
        # Make sure we have a valid string key
        if dex_name and isinstance(dex_name, str):
            return self.dex_interfaces.get(dex_name)
        else:
            logger.warning(f"Invalid DEX name: {dex_name}")
            return None
        
    async def get_best_dex_for_token(self, token_address: str) -> Optional[DEXInterface]:
        """
        Get the best DEX for a specific token based on liquidity
        
        Args:
            token_address (str): Token address
            
        Returns:
            Optional[DEXInterface]: Best DEX interface or None if not found
        """
        best_dex = None
        best_liquidity = 0
        
        # Check liquidity on all DEXes
        for dex_name, dex in self.dex_interfaces.items():
            try:
                if dex.is_running and dex.is_connected:
                    liquidity = await dex.get_token_liquidity(token_address)
                    if liquidity and liquidity > best_liquidity:
                        best_liquidity = liquidity
                        best_dex = dex
            except Exception as e:
                logger.warning(f"Error checking liquidity on {dex.dex_display_name}: {str(e)}")
                
        # If no liquidity found, fallback to default
        if best_dex is None:
            return self.get_dex()
            
        return best_dex
        
    async def execute_with_fallback(self, method_name: str, *args, **kwargs) -> Any:
        """
        Execute a method on DEXes with fallback
        
        Args:
            method_name (str): Method name to execute
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Any: Method result
        """
        # Try with each DEX in priority order
        for dex_name in self.dex_priority:
            dex = self.get_dex(dex_name)
            if dex and dex.is_running and dex.is_connected:
                try:
                    method = getattr(dex, method_name)
                    return await method(*args, **kwargs)
                except Exception as e:
                    logger.warning(f"Error executing {method_name} on {dex.dex_display_name}: {str(e)}")
                    
        # If all DEXes failed, raise exception
        raise Exception(f"Failed to execute {method_name} on any available DEX")
        
    # Convenience methods that use the best available DEX
    
    async def get_token_info(self, token_address: str) -> Optional[Dict[str, Any]]:
        """Get token information using best available DEX"""
        return await self.execute_with_fallback('get_token_info', token_address)
        
    async def get_token_price(self, token_address: str) -> Optional[float]:
        """Get token price using best available DEX"""
        return await self.execute_with_fallback('get_token_price', token_address)
        
    async def get_token_liquidity(self, token_address: str) -> Optional[float]:
        """Get token liquidity using best available DEX"""
        return await self.execute_with_fallback('get_token_liquidity', token_address)
        
    async def buy_token(self, token_address: str, amount_usd: float, max_slippage: float = 2.0) -> Dict[str, Any]:
        """Buy token using best available DEX"""
        # Use DEX with best liquidity for this token
        best_dex = await self.get_best_dex_for_token(token_address)
        if best_dex:
            return await best_dex.buy_token(token_address, amount_usd, max_slippage)
        else:
            return await self.execute_with_fallback('buy_token', token_address, amount_usd, max_slippage)
        
    async def sell_token(self, token_address: str, percent_of_holdings: float = 100.0, max_slippage: float = 2.0) -> Dict[str, Any]:
        """Sell token using best available DEX"""
        # Use DEX with best liquidity for this token
        best_dex = await self.get_best_dex_for_token(token_address)
        if best_dex:
            return await best_dex.sell_token(token_address, percent_of_holdings, max_slippage)
        else:
            return await self.execute_with_fallback('sell_token', token_address, percent_of_holdings, max_slippage)
        
    async def get_wallet_tokens(self) -> List[Dict[str, Any]]:
        """Get wallet tokens using default DEX"""
        return await self.execute_with_fallback('get_wallet_tokens')
        
    async def get_wallet_balance(self, token_address: Optional[str] = None) -> float:
        """Get wallet balance using default DEX"""
        return await self.execute_with_fallback('get_wallet_balance', token_address)
        
    async def get_all_dex_statuses(self) -> Dict[str, Dict[str, Any]]:
        """
        Get status of all registered DEXes
        
        Returns:
            Dict[str, Dict[str, Any]]: DEX statuses
        """
        statuses = {}
        for dex_name, dex in self.dex_interfaces.items():
            statuses[dex_name] = {
                'name': dex.dex_display_name,
                'is_running': dex.is_running,
                'is_connected': dex.is_connected
            }
        return statuses