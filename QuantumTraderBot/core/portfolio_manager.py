"""
Portfolio Manager Component
Manages the trading portfolio and assets, handling allocation,
rebalancing, and optimizing returns based on risk profiles.
"""

import time
import logging
import asyncio
import json
import math
from enum import Enum
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Set, Union

from core.config_manager import ConfigManager
from core.state_manager import StateManager

logger = logging.getLogger(__name__)

class AssetType(Enum):
    """Types of assets in the portfolio"""
    NATIVE = "native"  # SOL
    TOKEN = "token"    # SPL tokens
    NFT = "nft"        # NFTs
    LP = "lp"          # Liquidity pool positions

class PositionStatus(Enum):
    """Status of a trading position"""
    OPEN = "open"
    CLOSED = "closed"
    PARTIAL = "partial"  # Partially closed
    PENDING = "pending"  # Waiting for execution
    FAILED = "failed"    # Failed to execute

class Position:
    """Trading position with comprehensive tracking"""
    
    def __init__(self, position_id: str, token_address: str):
        """
        Initialize position
        
        Args:
            position_id (str): Unique position identifier
            token_address (str): Token address
        """
        self.position_id = position_id
        self.token_address = token_address
        self.token_symbol = ""
        self.token_name = ""
        
        self.strategy_id = None
        self.status = PositionStatus.PENDING
        
        self.entry_price_usd = 0
        self.current_price_usd = 0
        self.exit_price_usd = 0
        
        self.amount_tokens = 0
        self.entry_value_usd = 0
        self.current_value_usd = 0
        self.exit_value_usd = 0
        
        self.entry_timestamp = time.time()
        self.last_update_timestamp = time.time()
        self.exit_timestamp = None
        
        self.pnl_usd = 0
        self.pnl_percent = 0
        
        self.stop_loss_price = 0
        self.take_profit_prices = []
        self.exit_reason = None
        
        self.partial_exits = []
        self.notes = []
        self.transactions = []
    
    def update_price(self, current_price: float):
        """
        Update current price and calculated values
        
        Args:
            current_price (float): Current token price in USD
        """
        old_price = self.current_price_usd
        self.current_price_usd = current_price
        self.last_update_timestamp = time.time()
        
        # Calculate current value
        self.current_value_usd = self.amount_tokens * current_price
        
        # Calculate P&L
        if self.entry_price_usd > 0:
            self.pnl_usd = self.current_value_usd - self.entry_value_usd
            self.pnl_percent = (self.current_price_usd / self.entry_price_usd - 1) * 100
        
        # Log significant price movements
        if old_price > 0:
            change_percent = (current_price / old_price - 1) * 100
            if abs(change_percent) > 5:
                logger.info(f"Position {self.position_id} ({self.token_symbol}): "
                           f"Price moved {change_percent:.2f}% from ${old_price:.6f} to ${current_price:.6f}")
    
    def add_transaction(self, tx_type: str, tx_data: Dict[str, Any]):
        """
        Add transaction to position history
        
        Args:
            tx_type (str): Transaction type
            tx_data (Dict[str, Any]): Transaction data
        """
        tx_record = {
            'type': tx_type,
            'timestamp': time.time(),
            'data': tx_data
        }
        
        self.transactions.append(tx_record)
    
    def add_note(self, note: str):
        """
        Add note to position
        
        Args:
            note (str): Note text
        """
        self.notes.append({
            'timestamp': time.time(),
            'text': note
        })
    
    def record_exit(self, exit_price: float, exit_value: float, reason: str):
        """
        Record position exit
        
        Args:
            exit_price (float): Exit price in USD
            exit_value (float): Exit value in USD
            reason (str): Exit reason
        """
        self.exit_price_usd = exit_price
        self.exit_value_usd = exit_value
        self.exit_timestamp = time.time()
        self.exit_reason = reason
        self.status = PositionStatus.CLOSED
        
        # Final P&L calculation
        self.pnl_usd = self.exit_value_usd - self.entry_value_usd
        self.pnl_percent = (self.exit_price_usd / self.entry_price_usd - 1) * 100
        
        logger.info(f"Position {self.position_id} ({self.token_symbol}) closed: "
                   f"P&L ${self.pnl_usd:.2f} ({self.pnl_percent:.2f}%), "
                   f"Reason: {reason}")
    
    def record_partial_exit(self, amount: float, price: float, value: float, reason: str):
        """
        Record partial position exit
        
        Args:
            amount (float): Token amount sold
            price (float): Exit price in USD
            value (float): Exit value in USD
            reason (str): Exit reason
        """
        partial_exit = {
            'timestamp': time.time(),
            'amount': amount,
            'price': price,
            'value': value,
            'reason': reason
        }
        
        self.partial_exits.append(partial_exit)
        
        # Update remaining amount
        self.amount_tokens -= amount
        
        # Update status
        self.status = PositionStatus.PARTIAL
        
        # Update current value
        self.current_value_usd = self.amount_tokens * self.current_price_usd
        
        logger.info(f"Partial exit for position {self.position_id} ({self.token_symbol}): "
                   f"{amount} tokens at ${price:.6f}, Reason: {reason}")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert position to dictionary
        
        Returns:
            Dict[str, Any]: Position data
        """
        return {
            'position_id': self.position_id,
            'token_address': self.token_address,
            'token_symbol': self.token_symbol,
            'token_name': self.token_name,
            'strategy_id': self.strategy_id,
            'status': self.status.value,
            'entry_price_usd': self.entry_price_usd,
            'current_price_usd': self.current_price_usd,
            'exit_price_usd': self.exit_price_usd,
            'amount_tokens': self.amount_tokens,
            'entry_value_usd': self.entry_value_usd,
            'current_value_usd': self.current_value_usd,
            'exit_value_usd': self.exit_value_usd,
            'entry_timestamp': self.entry_timestamp,
            'last_update_timestamp': self.last_update_timestamp,
            'exit_timestamp': self.exit_timestamp,
            'pnl_usd': self.pnl_usd,
            'pnl_percent': self.pnl_percent,
            'stop_loss_price': self.stop_loss_price,
            'take_profit_prices': self.take_profit_prices,
            'exit_reason': self.exit_reason,
            'partial_exits': self.partial_exits,
            'notes': self.notes,
            'transactions': self.transactions
        }

class PortfolioManager:
    """
    Manages the trading portfolio with comprehensive features:
    - Asset tracking and valuation
    - Position management
    - Risk-based allocation
    - Performance analysis
    - Portfolio rebalancing
    """
    
    def __init__(self, 
                config_manager: ConfigManager,
                state_manager: StateManager,
                raydium_client=None):
        """
        Initialize PortfolioManager
        
        Args:
            config_manager (ConfigManager): Configuration manager instance
            state_manager (StateManager): State manager instance
            raydium_client: DEX client for market operations
        """
        self.config_manager = config_manager
        self.state_manager = state_manager
        self.raydium_client = raydium_client
        
        # Position tracking
        self.active_positions: Dict[str, Position] = {}
        self.closed_positions: Dict[str, Position] = {}
        self.next_position_id = 1
        
        # Portfolio state
        self.portfolio_value_usd = 0
        self.assets: Dict[str, Dict[str, Any]] = {}
        self.portfolio_history: List[Dict[str, Any]] = []
        
        # Load settings
        self.max_positions = self.config_manager.get('portfolio.max_positions', 10)
        self.rebalance_threshold = self.config_manager.get('portfolio.rebalance_threshold_percent', 10)
        self.performance_fee_percent = self.config_manager.get('portfolio.performance_fee_percent', 0)
        
        # Initialize positions from disk
        self._load_positions()
        
        # Tasks
        self.tasks = {}
    
    async def start(self):
        """Start the PortfolioManager"""
        logger.info("Starting portfolio manager")
        
        # Start background tasks
        self.tasks['update_portfolio'] = asyncio.create_task(self._update_portfolio_loop())
        self.tasks['save_positions'] = asyncio.create_task(self._save_positions_loop())
        
        # Update initial portfolio value
        await self.update_portfolio_value()
    
    async def stop(self):
        """Stop the PortfolioManager"""
        logger.info("Stopping portfolio manager")
        
        # Cancel tasks
        for task_name, task in self.tasks.items():
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        # Save positions
        await self._save_positions()
    
    async def _update_portfolio_loop(self):
        """Background task to update portfolio regularly"""
        while True:
            try:
                await self.update_portfolio_value()
                await self.update_position_prices()
                await asyncio.sleep(60)  # Update once per minute
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in portfolio update loop: {e}")
                await asyncio.sleep(10)  # Sleep shorter time on error
    
    async def _save_positions_loop(self):
        """Background task to save positions regularly"""
        while True:
            try:
                await self._save_positions()
                await asyncio.sleep(300)  # Save every 5 minutes
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in position save loop: {e}")
                await asyncio.sleep(60)  # Sleep shorter time on error
    
    def _load_positions(self):
        """Load positions from disk"""
        try:
            positions_file = self.config_manager.get('portfolio.positions_file', './data/positions.json')
            
            try:
                with open(positions_file, 'r') as f:
                    position_data = json.load(f)
                    
                    # Load active positions
                    for pos_id, pos_dict in position_data.get('active_positions', {}).items():
                        position = Position(pos_id, pos_dict['token_address'])
                        
                        # Load fields
                        for key, value in pos_dict.items():
                            if key != 'status':  # Status is handled specially
                                setattr(position, key, value)
                        
                        # Fix status enum
                        position.status = PositionStatus(pos_dict.get('status', 'pending'))
                        
                        self.active_positions[pos_id] = position
                    
                    # Load closed positions
                    for pos_id, pos_dict in position_data.get('closed_positions', {}).items():
                        position = Position(pos_id, pos_dict['token_address'])
                        
                        # Load fields
                        for key, value in pos_dict.items():
                            if key != 'status':  # Status is handled specially
                                setattr(position, key, value)
                        
                        # Fix status enum
                        position.status = PositionStatus(pos_dict.get('status', 'closed'))
                        
                        self.closed_positions[pos_id] = position
                    
                    # Load next position ID
                    self.next_position_id = position_data.get('next_position_id', 1)
                    
                    logger.info(f"Loaded {len(self.active_positions)} active positions and "
                               f"{len(self.closed_positions)} closed positions")
            except FileNotFoundError:
                logger.info("No positions file found, starting with empty portfolio")
            
        except Exception as e:
            logger.error(f"Error loading positions: {e}")
    
    async def _save_positions(self):
        """Save positions to disk"""
        try:
            positions_file = self.config_manager.get('portfolio.positions_file', './data/positions.json')
            
            # Create parent directory if it doesn't exist
            import os
            os.makedirs(os.path.dirname(positions_file), exist_ok=True)
            
            position_data = {
                'active_positions': {pos_id: pos.to_dict() for pos_id, pos in self.active_positions.items()},
                'closed_positions': {pos_id: pos.to_dict() for pos_id, pos in self.closed_positions.items()},
                'next_position_id': self.next_position_id,
                'saved_at': time.time()
            }
            
            # Save to temporary file first
            temp_file = positions_file + '.tmp'
            
            with open(temp_file, 'w') as f:
                json.dump(position_data, f, indent=2)
            
            # Atomic replace
            os.replace(temp_file, positions_file)
            
            logger.debug(f"Saved {len(self.active_positions)} active positions and "
                       f"{len(self.closed_positions)} closed positions")
        except Exception as e:
            logger.error(f"Error saving positions: {e}")
    
    async def update_portfolio_value(self) -> float:
        """
        Update total portfolio value
        
        Returns:
            float: Portfolio value in USD
        """
        try:
            # In production, this would query actual asset balances
            # Start with native SOL
            sol_balance = 0
            if self.raydium_client:
                sol_balance = await self.raydium_client.get_wallet_balance()
            
            # Get SOL price
            sol_price = 0
            if self.raydium_client:
                sol_price_info = await self.raydium_client.get_token_price(
                    "So11111111111111111111111111111111111111112"  # SOL mint address
                )
                if sol_price_info:
                    sol_price = sol_price_info
            
            if sol_price <= 0:
                # Fallback price
                sol_price = self.config_manager.get('portfolio.sol_price_usd', 100)
            
            # Calculate SOL value
            sol_value = sol_balance * sol_price
            
            # Add SOL to assets
            self.assets["So11111111111111111111111111111111111111112"] = {
                'token_address': "So11111111111111111111111111111111111111112",
                'token_symbol': "SOL",
                'token_name': "Solana",
                'balance': sol_balance,
                'price_usd': sol_price,
                'value_usd': sol_value,
                'updated_at': time.time(),
                'asset_type': AssetType.NATIVE.value
            }
            
            # Get token balances
            token_assets = []
            if self.raydium_client:
                wallet_tokens = await self.raydium_client.get_wallet_tokens()
                
                for token in wallet_tokens:
                    token_address = token.get('address')
                    balance = token.get('balance', 0)
                    price = token.get('price_usd', 0)
                    
                    if token_address == "So11111111111111111111111111111111111111112":
                        continue  # Skip SOL, already accounted for
                    
                    token_assets.append({
                        'token_address': token_address,
                        'token_symbol': token.get('symbol', ''),
                        'token_name': token.get('name', ''),
                        'balance': balance,
                        'price_usd': price,
                        'value_usd': balance * price,
                        'updated_at': time.time(),
                        'asset_type': AssetType.TOKEN.value
                    })
            
            # Add token assets
            for asset in token_assets:
                self.assets[asset['token_address']] = asset
            
            # Calculate total portfolio value
            total_value = sum(asset['value_usd'] for asset in self.assets.values())
            
            # Add active position values
            position_value = sum(pos.current_value_usd for pos in self.active_positions.values())
            
            # Calculate total
            self.portfolio_value_usd = total_value + position_value
            
            # Record history point
            self.portfolio_history.append({
                'timestamp': time.time(),
                'value_usd': self.portfolio_value_usd,
                'asset_count': len(self.assets),
                'position_count': len(self.active_positions)
            })
            
            # Keep history to reasonable size
            if len(self.portfolio_history) > 1440:  # 1 day at 1-minute interval
                self.portfolio_history = self.portfolio_history[-1440:]
            
            # Update state metrics
            self.state_manager.update_component_metric(
                'portfolio_manager',
                'portfolio_value_usd',
                self.portfolio_value_usd
            )
            
            self.state_manager.update_component_metric(
                'portfolio_manager',
                'active_positions_count',
                len(self.active_positions)
            )
            
            return self.portfolio_value_usd
        
        except Exception as e:
            logger.error(f"Error updating portfolio value: {e}")
            return self.portfolio_value_usd
    
    async def update_position_prices(self):
        """Update prices for all active positions"""
        if not self.raydium_client:
            logger.warning("Raydium client not available for position price updates")
            return
        
        for position_id, position in list(self.active_positions.items()):
            try:
                # Skip closed positions
                if position.status == PositionStatus.CLOSED:
                    continue
                
                # Get current price
                price = await self.raydium_client.get_token_price(position.token_address)
                
                if price is not None and price > 0:
                    # Update position
                    position.update_price(price)
                    
                    # Check for stop loss / take profit
                    await self._check_exit_conditions(position_id)
            
            except Exception as e:
                logger.error(f"Error updating position {position_id}: {e}")
    
    async def _check_exit_conditions(self, position_id: str):
        """
        Check exit conditions for a position
        
        Args:
            position_id (str): Position ID
        """
        if position_id not in self.active_positions:
            return
        
        position = self.active_positions[position_id]
        
        # Skip if position is not open
        if position.status != PositionStatus.OPEN and position.status != PositionStatus.PARTIAL:
            return
        
        # Check stop loss
        if position.stop_loss_price > 0 and position.current_price_usd <= position.stop_loss_price:
            logger.warning(f"Stop loss triggered for position {position_id} ({position.token_symbol}): "
                         f"Current price ${position.current_price_usd} <= stop loss ${position.stop_loss_price}")
            
            # In production, this would execute the sell order
            # For now, just update position status
            await self.exit_position(
                position_id=position_id,
                price=position.current_price_usd,
                reason="stop_loss"
            )
            return
        
        # Check take profit levels
        for idx, tp_price in enumerate(position.take_profit_prices):
            if position.current_price_usd >= tp_price:
                logger.info(f"Take profit triggered for position {position_id} ({position.token_symbol}): "
                          f"Current price ${position.current_price_usd} >= take profit ${tp_price}")
                
                # Determine exit percentage based on take profit level
                exit_percent = self._get_take_profit_exit_percent(idx, len(position.take_profit_prices))
                
                if exit_percent >= 100:
                    # Full exit
                    await self.exit_position(
                        position_id=position_id,
                        price=position.current_price_usd,
                        reason=f"take_profit_level_{idx+1}"
                    )
                else:
                    # Partial exit
                    exit_amount = position.amount_tokens * (exit_percent / 100)
                    await self.partial_exit_position(
                        position_id=position_id,
                        amount=exit_amount,
                        price=position.current_price_usd,
                        reason=f"take_profit_level_{idx+1}"
                    )
                
                # Remove this take profit level
                position.take_profit_prices.pop(idx)
                
                # Add note about take profit
                position.add_note(f"Take profit level {idx+1} reached at ${position.current_price_usd}")
                
                return
    
    def _get_take_profit_exit_percent(self, level_idx: int, total_levels: int) -> float:
        """
        Determine percentage to exit at a take profit level
        
        Args:
            level_idx (int): Take profit level index (0-based)
            total_levels (int): Total number of take profit levels
            
        Returns:
            float: Percentage of position to exit (0-100)
        """
        # Simple implementation: equal distribution
        # First level: exit 1/N of position
        # Last level: exit remainder (100%)
        
        if level_idx == total_levels - 1:
            return 100  # Exit entire remaining position at final level
        
        return 100 / total_levels
    
    async def create_position(self, 
                            token_address: str,
                            entry_price: float,
                            amount: float,
                            strategy_id: Optional[str] = None,
                            token_data: Optional[Dict[str, Any]] = None) -> str:
        """
        Create a new trading position
        
        Args:
            token_address (str): Token address
            entry_price (float): Entry price in USD
            amount (float): Token amount
            strategy_id (str, optional): Strategy ID
            token_data (Dict[str, Any], optional): Additional token data
            
        Returns:
            str: Position ID
        """
        # Generate position ID
        position_id = str(self.next_position_id)
        self.next_position_id += 1
        
        # Create position
        position = Position(position_id, token_address)
        position.entry_price_usd = entry_price
        position.current_price_usd = entry_price
        position.amount_tokens = amount
        position.entry_value_usd = entry_price * amount
        position.current_value_usd = entry_price * amount
        position.strategy_id = strategy_id
        position.status = PositionStatus.OPEN
        
        # Add token info if available
        if token_data:
            position.token_symbol = token_data.get('symbol', '')
            position.token_name = token_data.get('name', '')
        
        # Add to active positions
        self.active_positions[position_id] = position
        
        # Log
        logger.info(f"Created position {position_id} for {position.token_symbol or token_address}: "
                   f"{amount} tokens at ${entry_price}")
        
        # Update portfolio value
        await self.update_portfolio_value()
        
        # Save positions
        await self._save_positions()
        
        return position_id
    
    async def update_position(self, 
                            position_id: str,
                            token_data: Optional[Dict[str, Any]] = None,
                            stop_loss: Optional[float] = None,
                            take_profits: Optional[List[float]] = None) -> bool:
        """
        Update position parameters
        
        Args:
            position_id (str): Position ID
            token_data (Dict[str, Any], optional): Updated token data
            stop_loss (float, optional): Stop loss price
            take_profits (List[float], optional): Take profit prices
            
        Returns:
            bool: True if position was updated
        """
        if position_id not in self.active_positions:
            logger.warning(f"Position {position_id} not found")
            return False
        
        position = self.active_positions[position_id]
        
        # Update token data
        if token_data:
            position.token_symbol = token_data.get('symbol', position.token_symbol)
            position.token_name = token_data.get('name', position.token_name)
        
        # Update stop loss
        if stop_loss is not None:
            position.stop_loss_price = stop_loss
            position.add_note(f"Stop loss updated to ${stop_loss}")
        
        # Update take profits
        if take_profits is not None:
            position.take_profit_prices = sorted(take_profits)
            position.add_note(f"Take profit levels updated to {position.take_profit_prices}")
        
        logger.info(f"Updated position {position_id} ({position.token_symbol})")
        
        # Save positions
        await self._save_positions()
        
        return True
    
    async def exit_position(self, 
                          position_id: str,
                          price: Optional[float] = None,
                          reason: str = "manual") -> bool:
        """
        Exit a trading position
        
        Args:
            position_id (str): Position ID
            price (float, optional): Exit price (uses current price if None)
            reason (str): Exit reason
            
        Returns:
            bool: True if position was exited
        """
        if position_id not in self.active_positions:
            logger.warning(f"Position {position_id} not found")
            return False
        
        position = self.active_positions[position_id]
        
        # Skip if already closed
        if position.status == PositionStatus.CLOSED:
            logger.warning(f"Position {position_id} already closed")
            return False
        
        # Use provided price or current price
        exit_price = price if price is not None else position.current_price_usd
        exit_value = position.amount_tokens * exit_price
        
        # In production, this would execute the sell order
        # For now, just update position
        position.record_exit(exit_price, exit_value, reason)
        
        # Move to closed positions
        self.closed_positions[position_id] = position
        del self.active_positions[position_id]
        
        # Update portfolio value
        await self.update_portfolio_value()
        
        # Save positions
        await self._save_positions()
        
        # Log performance
        self.state_manager.update_component_metric(
            'portfolio_manager',
            'realized_pnl_usd',
            position.pnl_usd
        )
        
        return True
    
    async def partial_exit_position(self, 
                                  position_id: str,
                                  amount: float,
                                  price: Optional[float] = None,
                                  reason: str = "manual") -> bool:
        """
        Partially exit a trading position
        
        Args:
            position_id (str): Position ID
            amount (float): Token amount to exit
            price (float, optional): Exit price (uses current price if None)
            reason (str): Exit reason
            
        Returns:
            bool: True if position was partially exited
        """
        if position_id not in self.active_positions:
            logger.warning(f"Position {position_id} not found")
            return False
        
        position = self.active_positions[position_id]
        
        # Skip if closed
        if position.status == PositionStatus.CLOSED:
            logger.warning(f"Position {position_id} already closed")
            return False
        
        # Check amount
        if amount > position.amount_tokens:
            logger.warning(f"Cannot exit {amount} tokens from position {position_id}, only {position.amount_tokens} available")
            amount = position.amount_tokens
        
        # Use provided price or current price
        exit_price = price if price is not None else position.current_price_usd
        exit_value = amount * exit_price
        
        # In production, this would execute the sell order
        # For now, just update position
        position.record_partial_exit(amount, exit_price, exit_value, reason)
        
        # If all tokens exited, close position
        if position.amount_tokens <= 0:
            await self.exit_position(position_id, exit_price, f"{reason}_full")
        
        # Update portfolio value
        await self.update_portfolio_value()
        
        # Save positions
        await self._save_positions()
        
        return True
    
    def get_position(self, position_id: str) -> Optional[Dict[str, Any]]:
        """
        Get position details
        
        Args:
            position_id (str): Position ID
            
        Returns:
            Optional[Dict[str, Any]]: Position details or None if not found
        """
        if position_id in self.active_positions:
            return self.active_positions[position_id].to_dict()
        
        if position_id in self.closed_positions:
            return self.closed_positions[position_id].to_dict()
        
        return None
    
    def get_active_positions(self) -> List[Dict[str, Any]]:
        """
        Get all active positions
        
        Returns:
            List[Dict[str, Any]]: Active positions
        """
        return [pos.to_dict() for pos in self.active_positions.values()]
    
    def get_closed_positions(self, 
                           limit: int = 100, 
                           offset: int = 0) -> List[Dict[str, Any]]:
        """
        Get closed positions
        
        Args:
            limit (int): Maximum number of positions to return
            offset (int): Offset for pagination
            
        Returns:
            List[Dict[str, Any]]: Closed positions
        """
        # Sort by exit time, newest first
        sorted_positions = sorted(
            self.closed_positions.values(),
            key=lambda p: p.exit_timestamp or 0,
            reverse=True
        )
        
        # Apply pagination
        paginated = sorted_positions[offset:offset+limit]
        
        return [pos.to_dict() for pos in paginated]
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """
        Get portfolio summary
        
        Returns:
            Dict[str, Any]: Portfolio summary
        """
        # Active positions value
        active_value = sum(pos.current_value_usd for pos in self.active_positions.values())
        
        # Calculate performance metrics
        total_realized_pnl = sum(
            pos.pnl_usd for pos in self.closed_positions.values()
        )
        
        total_unrealized_pnl = sum(
            pos.pnl_usd for pos in self.active_positions.values()
        )
        
        # Calculate win rate (percentage of profitable closed positions)
        profitable_positions = sum(
            1 for pos in self.closed_positions.values() if pos.pnl_usd > 0
        )
        
        total_closed = len(self.closed_positions)
        win_rate = (profitable_positions / total_closed * 100) if total_closed > 0 else 0
        
        # Get performance by strategy
        strategy_performance = {}
        for pos in self.closed_positions.values():
            strategy = pos.strategy_id or "unknown"
            
            if strategy not in strategy_performance:
                strategy_performance[strategy] = {
                    'realized_pnl': 0,
                    'position_count': 0,
                    'profitable_count': 0
                }
            
            strategy_performance[strategy]['realized_pnl'] += pos.pnl_usd
            strategy_performance[strategy]['position_count'] += 1
            
            if pos.pnl_usd > 0:
                strategy_performance[strategy]['profitable_count'] += 1
        
        # Calculate strategy win rates
        for strategy, perf in strategy_performance.items():
            perf['win_rate'] = (
                perf['profitable_count'] / perf['position_count'] * 100
            ) if perf['position_count'] > 0 else 0
        
        # Calculate assets by type
        assets_by_type = {}
        for asset in self.assets.values():
            asset_type = asset['asset_type']
            
            if asset_type not in assets_by_type:
                assets_by_type[asset_type] = {
                    'count': 0,
                    'value_usd': 0
                }
            
            assets_by_type[asset_type]['count'] += 1
            assets_by_type[asset_type]['value_usd'] += asset['value_usd']
        
        return {
            'portfolio_value_usd': self.portfolio_value_usd,
            'active_positions_count': len(self.active_positions),
            'active_positions_value_usd': active_value,
            'closed_positions_count': len(self.closed_positions),
            'total_realized_pnl_usd': total_realized_pnl,
            'total_unrealized_pnl_usd': total_unrealized_pnl,
            'win_rate': win_rate,
            'assets_count': len(self.assets),
            'assets_by_type': assets_by_type,
            'strategy_performance': strategy_performance,
            'last_updated': time.time()
        }
    
    async def allocate_capital(self, strategy_allocations: Dict[str, float]) -> Dict[str, float]:
        """
        Allocate capital to strategies based on target percentages
        
        Args:
            strategy_allocations (Dict[str, float]): Strategy allocations (percentage)
            
        Returns:
            Dict[str, float]: Actual allocation amounts in USD
        """
        # Update portfolio value
        await self.update_portfolio_value()
        
        # Calculate allocatable capital (subtract active positions)
        active_value = sum(pos.current_value_usd for pos in self.active_positions.values())
        allocatable_capital = max(0, self.portfolio_value_usd - active_value)
        
        # Calculate allocation amounts
        allocation_amounts = {}
        for strategy, percentage in strategy_allocations.items():
            allocation_amounts[strategy] = allocatable_capital * (percentage / 100)
        
        logger.info(f"Capital allocation: {allocation_amounts}")
        
        return allocation_amounts
    
    async def rebalance_portfolio(self) -> Dict[str, Any]:
        """
        Rebalance portfolio according to target allocations
        
        Returns:
            Dict[str, Any]: Rebalance results
        """
        logger.info("Starting portfolio rebalance")
        
        # Update portfolio value
        await self.update_portfolio_value()
        
        # Get target allocations
        target_allocations = self.config_manager.get('portfolio.target_allocations', {
            'native': 30,  # 30% in SOL
            'memecoins': 50,  # 50% in memecoins
            'reserve': 20   # 20% reserve
        })
        
        # Calculate current allocations
        current_allocations = {
            'native': 0,
            'memecoins': 0,
            'reserve': 0
        }
        
        # Native SOL
        sol_asset = self.assets.get("So11111111111111111111111111111111111111112")
        if sol_asset:
            current_allocations['native'] = sol_asset['value_usd']
        
        # Memecoins (active positions)
        memecoin_value = sum(pos.current_value_usd for pos in self.active_positions.values())
        current_allocations['memecoins'] = memecoin_value
        
        # Reserve (all other assets)
        reserve_value = self.portfolio_value_usd - current_allocations['native'] - current_allocations['memecoins']
        current_allocations['reserve'] = max(0, reserve_value)
        
        # Calculate percentages
        current_percentages = {}
        for category, value in current_allocations.items():
            current_percentages[category] = (value / self.portfolio_value_usd * 100) if self.portfolio_value_usd > 0 else 0
        
        # Check for imbalances
        imbalances = {}
        for category, target_pct in target_allocations.items():
            current_pct = current_percentages.get(category, 0)
            diff = abs(current_pct - target_pct)
            
            if diff > self.rebalance_threshold:
                imbalances[category] = {
                    'current_percent': current_pct,
                    'target_percent': target_pct,
                    'difference': current_pct - target_pct,
                    'requires_action': True
                }
            else:
                imbalances[category] = {
                    'current_percent': current_pct,
                    'target_percent': target_pct,
                    'difference': current_pct - target_pct,
                    'requires_action': False
                }
        
        # Build rebalance report
        rebalance_report = {
            'portfolio_value_usd': self.portfolio_value_usd,
            'current_allocations': current_allocations,
            'current_percentages': current_percentages,
            'target_allocations': target_allocations,
            'imbalances': imbalances,
            'timestamp': time.time(),
            'recommended_actions': []
        }
        
        # Add recommended actions
        for category, imbalance in imbalances.items():
            if imbalance['requires_action']:
                diff_value = (imbalance['target_percent'] - imbalance['current_percent']) / 100 * self.portfolio_value_usd
                
                if diff_value > 0:
                    rebalance_report['recommended_actions'].append({
                        'action': 'increase',
                        'category': category,
                        'amount_usd': diff_value,
                        'description': f"Increase {category} allocation by ${diff_value:.2f}"
                    })
                else:
                    rebalance_report['recommended_actions'].append({
                        'action': 'decrease',
                        'category': category,
                        'amount_usd': abs(diff_value),
                        'description': f"Decrease {category} allocation by ${abs(diff_value):.2f}"
                    })
        
        # In a production system, this would execute the rebalance actions
        # For now, just report the recommendations
        
        logger.info(f"Portfolio rebalance complete: {len(rebalance_report['recommended_actions'])} recommended actions")
        
        return rebalance_report
    
    def get_portfolio_history(self, 
                            period: str = 'day') -> List[Dict[str, Any]]:
        """
        Get portfolio value history
        
        Args:
            period (str): Time period ('hour', 'day', 'week', 'month', 'all')
            
        Returns:
            List[Dict[str, Any]]: Portfolio history
        """
        if not self.portfolio_history:
            return []
        
        now = time.time()
        
        if period == 'hour':
            cutoff = now - 3600  # 1 hour
        elif period == 'day':
            cutoff = now - 86400  # 1 day
        elif period == 'week':
            cutoff = now - 604800  # 1 week
        elif period == 'month':
            cutoff = now - 2592000  # 30 days
        else:
            cutoff = 0  # All history
        
        # Filter by time
        filtered = [point for point in self.portfolio_history if point['timestamp'] >= cutoff]
        
        # Downsample if needed
        if period == 'week' and len(filtered) > 168:  # >1h points for a week
            # Downsample to ~1 hour points
            step = len(filtered) // 168
            filtered = filtered[::step]
        elif period == 'month' and len(filtered) > 720:  # >1h points for a month
            # Downsample to ~1 hour points
            step = len(filtered) // 720
            filtered = filtered[::step]
        
        return filtered