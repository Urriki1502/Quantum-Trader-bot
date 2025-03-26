"""
DynamicProfitManager Component
Responsible for dynamically adjusting take profit and stop loss levels
based on market conditions, volatility, and trading patterns.
"""

import asyncio
import logging
import math
import time
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta

from core.state_manager import StateManager
from core.config_manager import ConfigManager
from trading.risk_manager import RiskManager

logger = logging.getLogger(__name__)

class DynamicProfitManager:
    """
    DynamicProfitManager handles:
    - Dynamic adjustment of take profit (TP) levels
    - Adaptive stop loss (SL) management
    - Multi-tier profit extraction
    - Market condition-based profit strategies
    - Win rate optimization
    """
    
    def __init__(self, 
                config_manager: ConfigManager,
                state_manager: StateManager,
                risk_manager: RiskManager):
        """
        Initialize the DynamicProfitManager
        
        Args:
            config_manager (ConfigManager): Configuration manager instance
            state_manager (StateManager): State manager instance
            risk_manager (RiskManager): Risk manager instance
        """
        self.config_manager = config_manager
        self.state_manager = state_manager
        self.risk_manager = risk_manager
        
        # Default profit settings
        self.base_take_profit = self.config_manager.get('profit.base_take_profit_percentage', 10)
        self.min_take_profit = self.config_manager.get('profit.min_take_profit_percentage', 5)
        self.max_take_profit = self.config_manager.get('profit.max_take_profit_percentage', 50)
        
        # Trailing settings
        self.enable_trailing = self.config_manager.get('profit.enable_trailing_stop', True)
        self.trailing_activation = self.config_manager.get('profit.trailing_activation_percentage', 5)
        self.trailing_distance = self.config_manager.get('profit.trailing_distance_percentage', 2)
        
        # Multi-tier profit settings
        self.enable_tiered_profit = self.config_manager.get('profit.enable_tiered_profit', True)
        self.profit_tiers = self.config_manager.get('profit.tiers', [
            {'percentage': 30, 'target': 5},  # Sell 30% when price reaches +5%
            {'percentage': 30, 'target': 15}, # Sell another 30% at +15%
            {'percentage': 40, 'target': 30}  # Sell remaining 40% at +30%
        ])
        
        # Win rate optimization
        self.target_win_rate = self.config_manager.get('profit.target_win_rate', 70)  # 70%
        self.win_rate_adjustment_factor = self.config_manager.get('profit.win_rate_adjustment_factor', 0.1)
        
        # Historical data
        self.token_volatility = {}  # token_address -> volatility data
        self.market_conditions = {}  # token_address -> market condition
        self.price_movement_history = {}  # token_address -> price movements
        self.profit_history = {}  # token_address -> profit history
        
        # Active trades
        self.active_trades = {}  # token_address -> trade data with profit settings
        
        logger.info("DynamicProfitManager initialized")
    
    async def calculate_optimal_profit_levels(self, token_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate optimal take profit and stop loss levels based on token data
        
        Args:
            token_data (Dict[str, Any]): Token data including price, volatility, etc.
            
        Returns:
            Dict[str, Any]: Calculated profit levels
        """
        token_address = token_data.get('address')
        token_symbol = token_data.get('symbol', '')
        
        logger.debug(f"Calculating profit levels for {token_symbol} ({token_address})")
        
        # Get risk assessment
        risk_assessment = await self.risk_manager.assess_token_risk(token_data)
        risk_score = risk_assessment.get('risk_score', 50)
        
        # Get volatility (ATR or other volatility measure)
        volatility = token_data.get('atr', 0)
        if not volatility or volatility <= 0:
            # Use historical volatility from our tracking or default
            volatility = self.token_volatility.get(token_address, {}).get('value', 0.05)
        
        # Determine market condition
        market_condition = await self._detect_market_condition(token_data)
        
        # Calculate optimal take profit based on risk, volatility, and market condition
        optimal_tp = self._calculate_take_profit(risk_score, volatility, market_condition)
        
        # Calculate optimal stop loss based on risk, volatility, and market condition
        optimal_sl = self._calculate_stop_loss(risk_score, volatility, market_condition)
        
        # Determine if trailing stop should be used
        use_trailing = self._should_use_trailing(market_condition, volatility)
        
        # Determine if tiered profit-taking should be used
        use_tiered = self._should_use_tiered(market_condition, volatility)
        
        # Create profit tiers if using tiered approach
        profit_tiers = []
        if use_tiered:
            profit_tiers = self._create_profit_tiers(optimal_tp, volatility, market_condition)
        
        # Create result
        profit_levels = {
            'take_profit_percentage': optimal_tp,
            'stop_loss_percentage': optimal_sl,
            'use_trailing_stop': use_trailing,
            'trailing_activation_percentage': self.trailing_activation,
            'trailing_distance_percentage': self.trailing_distance,
            'use_tiered_profit': use_tiered,
            'profit_tiers': profit_tiers,
            'market_condition': market_condition,
            'volatility': volatility,
            'risk_score': risk_score,
            'timestamp': time.time()
        }
        
        logger.info(f"Calculated profit levels for {token_symbol}: TP={optimal_tp:.2f}%, SL={optimal_sl:.2f}%, Market={market_condition}")
        return profit_levels
    
    def register_trade(self, token_address: str, trade_data: Dict[str, Any], profit_levels: Dict[str, Any]):
        """
        Register a new trade with calculated profit levels
        
        Args:
            token_address (str): Token address
            trade_data (Dict[str, Any]): Trade data
            profit_levels (Dict[str, Any]): Calculated profit levels
        """
        logger.debug(f"Registering trade for {token_address} with profit levels")
        
        # Store trade with profit levels
        self.active_trades[token_address] = {
            'trade_data': trade_data,
            'profit_levels': profit_levels,
            'entry_time': time.time(),
            'entry_price': trade_data.get('entry_price_usd', 0),
            'position_size': trade_data.get('position_size_usd', 0),
            'current_price': trade_data.get('entry_price_usd', 0),
            'highest_price': trade_data.get('entry_price_usd', 0),
            'trailing_activated': False,
            'completed_tiers': [],
            'last_update': time.time()
        }
        
        # Update state metrics
        self.state_manager.update_component_metric(
            'dynamic_profit_manager',
            'active_trades_count',
            len(self.active_trades)
        )
    
    def unregister_trade(self, token_address: str, exit_data: Optional[Dict[str, Any]] = None):
        """
        Unregister a trade and record results
        
        Args:
            token_address (str): Token address
            exit_data (Dict[str, Any], optional): Exit data including profit/loss
        """
        if token_address not in self.active_trades:
            logger.warning(f"Attempted to unregister unknown trade: {token_address}")
            return
        
        logger.debug(f"Unregistering trade for {token_address}")
        
        # Get trade data
        trade = self.active_trades[token_address]
        
        # Record profit/loss if provided
        if exit_data:
            profit_percentage = exit_data.get('profit_loss_percent', 0)
            exit_reason = exit_data.get('exit_reason', 'unknown')
            
            # Store in profit history
            if token_address not in self.profit_history:
                self.profit_history[token_address] = []
            
            self.profit_history[token_address].append({
                'entry_time': trade['entry_time'],
                'exit_time': time.time(),
                'profit_percentage': profit_percentage,
                'exit_reason': exit_reason,
                'profit_levels': trade['profit_levels'],
                'market_condition': trade['profit_levels']['market_condition']
            })
            
            # Update win rate metrics
            self._update_win_rate_metrics()
            
            # Learn from this trade result
            self._learn_from_trade_result(
                token_address, 
                profit_percentage, 
                exit_reason,
                trade['profit_levels']
            )
        
        # Remove from active trades
        del self.active_trades[token_address]
        
        # Update state metrics
        self.state_manager.update_component_metric(
            'dynamic_profit_manager',
            'active_trades_count',
            len(self.active_trades)
        )
    
    async def update_price(self, token_address: str, current_price: float) -> Dict[str, Any]:
        """
        Update current price for an active trade and check for exits
        
        Args:
            token_address (str): Token address
            current_price (float): Current token price
            
        Returns:
            Dict[str, Any]: Exit signal if triggered, otherwise None
        """
        if token_address not in self.active_trades:
            return {}
        
        trade = self.active_trades[token_address]
        entry_price = trade['entry_price']
        
        # Skip if price hasn't changed
        if current_price == trade['current_price']:
            return {}
        
        # Update current price
        trade['current_price'] = current_price
        trade['last_update'] = time.time()
        
        # Calculate price change percentage
        price_change_percent = ((current_price / entry_price) - 1) * 100
        
        # Update highest price if new high
        if current_price > trade['highest_price']:
            trade['highest_price'] = current_price
        
        # Check for tiered profit exits
        if trade['profit_levels']['use_tiered_profit']:
            tier_exit = self._check_tiered_exit(token_address, current_price, entry_price)
            if tier_exit:
                return tier_exit
        
        # Check for trailing stop activation and trailing stop exits
        if trade['profit_levels']['use_trailing_stop']:
            # Check if we've reached the trailing activation threshold
            if not trade['trailing_activated']:
                activation_threshold = entry_price * (1 + trade['profit_levels']['trailing_activation_percentage'] / 100)
                if current_price >= activation_threshold:
                    trade['trailing_activated'] = True
                    logger.info(f"Trailing stop activated for {token_address} at {current_price:.6f} (+{price_change_percent:.2f}%)")
            
            # Check if trailing stop has been hit
            if trade['trailing_activated']:
                trailing_exit = self._check_trailing_exit(token_address, current_price, trade['highest_price'])
                if trailing_exit:
                    return trailing_exit
        
        # Check regular take profit
        take_profit_level = entry_price * (1 + trade['profit_levels']['take_profit_percentage'] / 100)
        if current_price >= take_profit_level:
            return {
                'should_exit': True,
                'exit_type': 'take_profit',
                'exit_percentage': 100,  # Exit full position
                'price_change_percent': price_change_percent,
                'reason': f"Take profit reached: {price_change_percent:.2f}% > {trade['profit_levels']['take_profit_percentage']:.2f}%"
            }
        
        # Check stop loss
        stop_loss_level = entry_price * (1 - trade['profit_levels']['stop_loss_percentage'] / 100)
        if current_price <= stop_loss_level:
            return {
                'should_exit': True,
                'exit_type': 'stop_loss',
                'exit_percentage': 100,  # Exit full position
                'price_change_percent': price_change_percent,
                'reason': f"Stop loss reached: {price_change_percent:.2f}% < -{trade['profit_levels']['stop_loss_percentage']:.2f}%"
            }
        
        # No exit triggered
        return {}
    
    async def adjust_profit_levels(self, token_address: str) -> Dict[str, Any]:
        """
        Adjust profit levels for an active trade based on current market conditions
        
        Args:
            token_address (str): Token address
            
        Returns:
            Dict[str, Any]: Adjusted profit levels if trade exists
        """
        if token_address not in self.active_trades:
            logger.warning(f"Attempted to adjust profit levels for unknown trade: {token_address}")
            return {}
        
        trade = self.active_trades[token_address]
        
        # Get current market condition
        token_data = {
            'address': token_address,
            'price_usd': trade['current_price'],
            'atr': trade['profit_levels']['volatility']
        }
        
        # Get updated market condition
        market_condition = await self._detect_market_condition(token_data)
        
        # Only adjust if market condition has changed
        if market_condition == trade['profit_levels']['market_condition']:
            return trade['profit_levels']
        
        logger.info(f"Market condition changed for {token_address}: {trade['profit_levels']['market_condition']} -> {market_condition}")
        
        # Recalculate profit levels
        risk_score = trade['profit_levels']['risk_score']
        volatility = trade['profit_levels']['volatility']
        
        # Calculate new take profit based on risk, volatility, and market condition
        new_tp = self._calculate_take_profit(risk_score, volatility, market_condition)
        
        # Calculate new stop loss based on risk, volatility, and market condition
        new_sl = self._calculate_stop_loss(risk_score, volatility, market_condition)
        
        # Determine if trailing stop should be used
        use_trailing = self._should_use_trailing(market_condition, volatility)
        
        # Determine if tiered profit-taking should be used
        use_tiered = self._should_use_tiered(market_condition, volatility)
        
        # Create profit tiers if using tiered approach
        profit_tiers = []
        if use_tiered:
            profit_tiers = self._create_profit_tiers(new_tp, volatility, market_condition)
        
        # Create result
        new_profit_levels = {
            'take_profit_percentage': new_tp,
            'stop_loss_percentage': new_sl,
            'use_trailing_stop': use_trailing,
            'trailing_activation_percentage': self.trailing_activation,
            'trailing_distance_percentage': self.trailing_distance,
            'use_tiered_profit': use_tiered,
            'profit_tiers': profit_tiers,
            'market_condition': market_condition,
            'volatility': volatility,
            'risk_score': risk_score,
            'timestamp': time.time()
        }
        
        # Update trade
        trade['profit_levels'] = new_profit_levels
        logger.info(f"Adjusted profit levels for {token_address}: TP={new_tp:.2f}%, SL={new_sl:.2f}%, Market={market_condition}")
        
        return new_profit_levels
    
    async def _detect_market_condition(self, token_data: Dict[str, Any]) -> str:
        """
        Detect current market condition based on price action and volume
        
        Args:
            token_data (Dict[str, Any]): Token data
            
        Returns:
            str: Market condition ('trending', 'volatile', 'ranging', 'unknown')
        """
        token_address = token_data.get('address')
        
        # Use historical price data if available
        if token_address in self.price_movement_history:
            price_data = self.price_movement_history[token_address]
            
            # Calculate metrics from price data
            if len(price_data) >= 10:
                # Calculate price velocity (rate of change)
                recent_changes = [p['price_change'] for p in price_data[-10:]]
                avg_change = sum(recent_changes) / len(recent_changes)
                
                # Calculate price acceleration (change in velocity)
                if len(recent_changes) >= 2:
                    accelerations = [recent_changes[i] - recent_changes[i-1] for i in range(1, len(recent_changes))]
                    avg_acceleration = sum(accelerations) / len(accelerations)
                else:
                    avg_acceleration = 0
                
                # Calculate volatility (standard deviation of changes)
                if len(recent_changes) >= 2:
                    volatility = math.sqrt(sum((x - avg_change) ** 2 for x in recent_changes) / len(recent_changes))
                else:
                    volatility = 0
                
                # Determine market condition
                if abs(avg_change) > 0.5 and abs(avg_acceleration) < 0.2:
                    # Strong consistent movement in one direction
                    return 'trending'
                elif volatility > 1.0:
                    # High variability in price movements
                    return 'volatile'
                else:
                    # No strong trend, low volatility
                    return 'ranging'
        
        # Default based on available metrics from token_data
        recent_price_change = token_data.get('price_change_24h', 0)
        volatility = token_data.get('atr', 0)
        volume_change = token_data.get('volume_change_24h', 0)
        
        if abs(recent_price_change) > 20:
            return 'trending'
        elif volatility > 0.1 or abs(volume_change) > 30:
            return 'volatile'
        else:
            return 'ranging'
    
    def _calculate_take_profit(self, risk_score: float, volatility: float, market_condition: str) -> float:
        """
        Calculate optimal take profit percentage based on risk, volatility and market condition
        
        Args:
            risk_score (float): Risk score (0-100)
            volatility (float): Volatility value
            market_condition (str): Detected market condition
            
        Returns:
            float: Take profit percentage
        """
        # Base take profit is adjusted by the volatility
        base_tp = self.base_take_profit * (1 + volatility * 2)
        
        # Adjust based on market condition
        if market_condition == 'trending':
            # In trending markets, set higher take profits
            base_tp *= 1.5
        elif market_condition == 'volatile':
            # In volatile markets, set more conservative take profits
            base_tp *= 0.8
        elif market_condition == 'ranging':
            # In ranging markets, set moderate take profits
            base_tp *= 1.2
            
        # Risk adjustment - higher risk means lower take profit targets
        # to exit positions more quickly
        risk_multiplier = 1 - (risk_score / 200)  # 0.5 to 1.0 range
        tp = base_tp * risk_multiplier
        
        # Ensure within min/max bounds
        tp = max(tp, self.min_take_profit)
        tp = min(tp, self.max_take_profit)
        
        return round(tp, 2)
    
    def _calculate_stop_loss(self, risk_score: float, volatility: float, market_condition: str) -> float:
        """
        Calculate optimal stop loss percentage based on risk, volatility and market condition
        
        Args:
            risk_score (float): Risk score (0-100)
            volatility (float): Volatility value
            market_condition (str): Detected market condition
            
        Returns:
            float: Stop loss percentage
        """
        # Base stop loss is higher for volatile tokens
        base_sl = max(2, volatility * 100 * 0.5)  # Convert volatility to percentage, use 50% of it
        
        # Adjust based on market condition
        if market_condition == 'trending':
            # In trending markets, give more room
            base_sl *= 1.2
        elif market_condition == 'volatile':
            # In volatile markets, wider stop loss to avoid noise
            base_sl *= 1.5
        elif market_condition == 'ranging':
            # In ranging markets, tighter stop loss
            base_sl *= 0.8
            
        # Risk adjustment - higher risk means tighter stop loss
        risk_multiplier = 1 - (risk_score / 200)  # 0.5 to 1.0 range
        sl = base_sl / risk_multiplier  # Higher risk -> lower multiplier -> tighter stop loss
        
        # Ensure stop loss is reasonable
        sl = max(sl, 2)  # Minimum 2%
        sl = min(sl, 15)  # Maximum 15%
        
        return round(sl, 2)
    
    def _should_use_trailing(self, market_condition: str, volatility: float) -> bool:
        """
        Determine if trailing stop should be used based on market condition and volatility
        
        Args:
            market_condition (str): Detected market condition
            volatility (float): Volatility value
            
        Returns:
            bool: True if trailing stop should be used
        """
        if not self.enable_trailing:
            return False
            
        # Trailing stops work best in trending markets
        if market_condition == 'trending':
            return True
        
        # For volatile markets, only use trailing stops if volatility is manageable
        if market_condition == 'volatile' and volatility < 0.1:
            return True
            
        # Don't use trailing stops in ranging markets
        if market_condition == 'ranging':
            return False
            
        # Default
        return False
    
    def _should_use_tiered(self, market_condition: str, volatility: float) -> bool:
        """
        Determine if tiered profit-taking should be used
        
        Args:
            market_condition (str): Detected market condition
            volatility (float): Volatility value
            
        Returns:
            bool: True if tiered profit-taking should be used
        """
        if not self.enable_tiered_profit:
            return False
        
        # Tiered profit-taking works well in trending and volatile markets
        if market_condition in ('trending', 'volatile'):
            return True
            
        # For ranging markets, only use tiered if volatility is high enough
        if market_condition == 'ranging' and volatility > 0.05:
            return True
            
        # Default
        return False
    
    def _create_profit_tiers(self, optimal_tp: float, volatility: float, market_condition: str) -> List[Dict[str, Any]]:
        """
        Create profit tiers based on optimal take profit, volatility, and market condition
        
        Args:
            optimal_tp (float): Optimal take profit percentage
            volatility (float): Volatility value
            market_condition (str): Detected market condition
            
        Returns:
            List[Dict[str, Any]]: List of profit tiers
        """
        # Create tiers based on market condition
        if market_condition == 'trending':
            # In trending markets, progressive tiers to ride the trend
            tiers = [
                {'percentage': 20, 'target': round(optimal_tp * 0.3, 2)},
                {'percentage': 30, 'target': round(optimal_tp * 0.7, 2)},
                {'percentage': 50, 'target': round(optimal_tp * 1.5, 2)}
            ]
        elif market_condition == 'volatile':
            # In volatile markets, take profits more quickly
            tiers = [
                {'percentage': 40, 'target': round(optimal_tp * 0.4, 2)},
                {'percentage': 40, 'target': round(optimal_tp * 0.8, 2)},
                {'percentage': 20, 'target': round(optimal_tp * 1.2, 2)}
            ]
        else:  # ranging
            # In ranging markets, balanced approach
            tiers = [
                {'percentage': 30, 'target': round(optimal_tp * 0.5, 2)},
                {'percentage': 40, 'target': round(optimal_tp, 2)},
                {'percentage': 30, 'target': round(optimal_tp * 1.3, 2)}
            ]
            
        return tiers
    
    def _check_tiered_exit(self, token_address: str, current_price: float, entry_price: float) -> Optional[Dict[str, Any]]:
        """
        Check if a tiered exit should be triggered
        
        Args:
            token_address (str): Token address
            current_price (float): Current token price
            entry_price (float): Entry price
            
        Returns:
            Optional[Dict[str, Any]]: Exit signal if triggered, otherwise None
        """
        trade = self.active_trades[token_address]
        tiers = trade['profit_levels']['profit_tiers']
        completed_tiers = trade['completed_tiers']
        
        # Calculate price change percentage
        price_change_percent = ((current_price / entry_price) - 1) * 100
        
        # Check each tier that hasn't been completed
        for i, tier in enumerate(tiers):
            tier_id = f"tier_{i}"
            
            # Skip completed tiers
            if tier_id in completed_tiers:
                continue
                
            # Check if this tier's target has been reached
            if price_change_percent >= tier['target']:
                # Mark this tier as completed
                trade['completed_tiers'].append(tier_id)
                
                return {
                    'should_exit': True,
                    'exit_type': 'tiered_profit',
                    'tier_id': tier_id,
                    'exit_percentage': tier['percentage'],
                    'price_change_percent': price_change_percent,
                    'reason': f"Profit tier {i+1} reached: {price_change_percent:.2f}% > {tier['target']}%"
                }
        
        return None
    
    def _check_trailing_exit(self, token_address: str, current_price: float, highest_price: float) -> Optional[Dict[str, Any]]:
        """
        Check if trailing stop has been triggered
        
        Args:
            token_address (str): Token address
            current_price (float): Current token price
            highest_price (float): Highest recorded price
            
        Returns:
            Optional[Dict[str, Any]]: Exit signal if trailing stop triggered, otherwise None
        """
        trade = self.active_trades[token_address]
        trailing_distance = trade['profit_levels']['trailing_distance_percentage']
        
        # Calculate trailing stop level
        trailing_stop_level = highest_price * (1 - trailing_distance / 100)
        
        # Check if price has fallen below trailing stop
        if current_price < trailing_stop_level:
            entry_price = trade['entry_price']
            price_change_percent = ((current_price / entry_price) - 1) * 100
            
            return {
                'should_exit': True,
                'exit_type': 'trailing_stop',
                'exit_percentage': 100,  # Exit full position
                'price_change_percent': price_change_percent,
                'highest_reached': ((highest_price / entry_price) - 1) * 100,
                'reason': f"Trailing stop triggered: {price_change_percent:.2f}% (from high of {((highest_price / entry_price) - 1) * 100:.2f}%)"
            }
        
        return None
    
    def _update_win_rate_metrics(self):
        """
        Update win rate metrics based on trade history
        """
        # Count total trades and winning trades
        total_trades = 0
        winning_trades = 0
        
        for token_address, trades in self.profit_history.items():
            total_trades += len(trades)
            winning_trades += sum(1 for t in trades if t['profit_percentage'] > 0)
        
        # Calculate win rate
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        # Update metrics
        self.state_manager.update_component_metric(
            'dynamic_profit_manager',
            'win_rate',
            round(win_rate, 2)
        )
        
        self.state_manager.update_component_metric(
            'dynamic_profit_manager',
            'total_trades',
            total_trades
        )
        
        # Update by market condition
        market_conditions = set()
        for token_address, trades in self.profit_history.items():
            for trade in trades:
                market_conditions.add(trade.get('market_condition', 'unknown'))
        
        for condition in market_conditions:
            condition_trades = 0
            condition_wins = 0
            
            for token_address, trades in self.profit_history.items():
                condition_trades += sum(1 for t in trades if t.get('market_condition') == condition)
                condition_wins += sum(1 for t in trades if t.get('market_condition') == condition and t['profit_percentage'] > 0)
            
            condition_win_rate = (condition_wins / condition_trades * 100) if condition_trades > 0 else 0
            
            self.state_manager.update_component_metric(
                'dynamic_profit_manager',
                f'win_rate_{condition}',
                round(condition_win_rate, 2)
            )
    
    def _learn_from_trade_result(self, token_address: str, profit_percentage: float, exit_reason: str, profit_levels: Dict[str, Any]):
        """
        Learn from trade result to improve future profit calculations
        
        Args:
            token_address (str): Token address
            profit_percentage (float): Profit/loss percentage
            exit_reason (str): Reason for exit
            profit_levels (Dict[str, Any]): Profit levels used
        """
        # Skip if we have very few trades
        total_trades = sum(len(trades) for trades in self.profit_history.values())
        if total_trades < 5:
            return
        
        # Calculate current win rate
        winning_trades = sum(sum(1 for t in trades if t['profit_percentage'] > 0) for trades in self.profit_history.values())
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        # Check if win rate is below target
        if win_rate < self.target_win_rate:
            # Losing too many trades, adjust take profit and stop loss
            market_condition = profit_levels['market_condition']
            
            # Analyze trades by market condition
            condition_trades = []
            for trades in self.profit_history.values():
                condition_trades.extend([t for t in trades if t.get('market_condition') == market_condition])
            
            condition_win_rate = 0
            if condition_trades:
                condition_wins = sum(1 for t in condition_trades if t['profit_percentage'] > 0)
                condition_win_rate = (condition_wins / len(condition_trades) * 100)
            
            # If this market condition has a poor win rate, adjust settings
            if condition_win_rate < self.target_win_rate:
                # Analyze exits to determine what to adjust
                sl_exits = sum(1 for t in condition_trades if t.get('exit_reason', '').startswith('stop_loss'))
                tp_exits = sum(1 for t in condition_trades if t.get('exit_reason', '').startswith('take_profit'))
                
                # If we're hitting stop losses too often, adjust stop loss configuration
                if sl_exits > tp_exits:
                    # Adjust stop loss parameters to be more conservative
                    if market_condition == 'volatile':
                        # For volatile markets, increase stop loss distance
                        self._adjust_config_value('risk.volatile_market_sl_multiplier', 0.1, 0.5, 2.0)
                    else:
                        # For other markets, generally widen stop losses
                        self._adjust_config_value('risk.stop_loss_percentage', 0.5, 2.0, 15.0)
                else:
                    # If not hitting take profits enough, lower targets
                    if market_condition == 'trending':
                        # For trending markets, be less aggressive
                        self._adjust_config_value('profit.trending_market_tp_multiplier', -0.1, 0.5, 2.0)
                    else:
                        # For other markets, generally lower take profit targets
                        self._adjust_config_value('profit.base_take_profit_percentage', -1.0, self.min_take_profit, self.max_take_profit)
    
    def _adjust_config_value(self, config_path: str, adjustment: float, min_value: float, max_value: float):
        """
        Adjust a configuration value within bounds
        
        Args:
            config_path (str): Configuration path
            adjustment (float): Adjustment amount
            min_value (float): Minimum allowed value
            max_value (float): Maximum allowed value
        """
        current_value = self.config_manager.get(config_path, 0)
        new_value = current_value + adjustment
        
        # Ensure within bounds
        new_value = max(min_value, min(new_value, max_value))
        
        # Update config
        self.config_manager.set(config_path, new_value)
        logger.info(f"Adjusted {config_path}: {current_value} -> {new_value} (learning from trade results)")
    
    def get_profit_settings(self, token_address: str) -> Optional[Dict[str, Any]]:
        """
        Get stored profit settings for a token
        
        Args:
            token_address (str): Token address
            
        Returns:
            Optional[Dict[str, Any]]: Profit settings or None if not found
        """
        if token_address in self.active_trades:
            return self.active_trades[token_address]['profit_levels']
        return None
    
    def get_win_rate_stats(self) -> Dict[str, Any]:
        """
        Get win rate statistics
        
        Returns:
            Dict[str, Any]: Win rate statistics
        """
        # Count total trades and winning trades
        total_trades = 0
        winning_trades = 0
        
        for token_address, trades in self.profit_history.items():
            total_trades += len(trades)
            winning_trades += sum(1 for t in trades if t['profit_percentage'] > 0)
        
        # Calculate overall win rate
        overall_win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        # Calculate by market condition
        market_condition_stats = {}
        market_conditions = set()
        
        for token_address, trades in self.profit_history.items():
            for trade in trades:
                market_conditions.add(trade.get('market_condition', 'unknown'))
        
        for condition in market_conditions:
            condition_trades = 0
            condition_wins = 0
            profits = []
            
            for token_address, trades in self.profit_history.items():
                condition_trades += sum(1 for t in trades if t.get('market_condition') == condition)
                condition_wins += sum(1 for t in trades if t.get('market_condition') == condition and t['profit_percentage'] > 0)
                profits.extend([t['profit_percentage'] for t in trades if t.get('market_condition') == condition])
            
            condition_win_rate = (condition_wins / condition_trades * 100) if condition_trades > 0 else 0
            avg_profit = sum(profits) / len(profits) if profits else 0
            
            market_condition_stats[condition] = {
                'trades': condition_trades,
                'wins': condition_wins,
                'win_rate': round(condition_win_rate, 2),
                'avg_profit': round(avg_profit, 2)
            }
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'overall_win_rate': round(overall_win_rate, 2),
            'market_conditions': market_condition_stats,
            'active_trades': len(self.active_trades)
        }