"""
AdaptiveStrategy Component
Responsible for adapting trading strategies based on market conditions,
optimizing parameters, and learning from past performance.
"""

import asyncio
import logging
import time
import random
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)

class AdaptiveStrategy:
    """
    AdaptiveStrategy handles:
    - Adapting to market conditions
    - Switching between different strategies
    - Optimizing strategy parameters
    - Learning from successful patterns
    """
    
    def __init__(self):
        """Initialize the AdaptiveStrategy"""
        # Available strategies
        self.strategies = {
            'momentum': self._momentum_strategy,
            'breakout': self._breakout_strategy,
            'reversal': self._reversal_strategy,
            'volume_spike': self._volume_spike_strategy,
            'new_listing': self._new_listing_strategy
        }
        
        # Strategy parameter ranges for optimization
        self.parameter_ranges = {
            'momentum': {
                'lookback_period': (5, 30),
                'threshold': (0.5, 5.0),
                'exit_threshold': (0.3, 3.0)
            },
            'breakout': {
                'resistance_periods': (10, 50),
                'breakout_threshold': (1.0, 5.0),
                'confirmation_candles': (1, 3)
            },
            'reversal': {
                'overbought_threshold': (70, 85),
                'oversold_threshold': (15, 30),
                'confirmation_periods': (1, 5)
            },
            'volume_spike': {
                'volume_multiplier': (2.0, 10.0),
                'price_change_threshold': (0.5, 5.0),
                'lookback_periods': (3, 20)
            },
            'new_listing': {
                'max_age_hours': (1, 48),
                'min_liquidity_usd': (5000, 50000),
                'initial_pump_threshold': (3.0, 20.0)
            }
        }
        
        # Current parameters for each strategy
        self.current_parameters = {
            'momentum': {
                'lookback_period': 15,
                'threshold': 2.0,
                'exit_threshold': 1.0
            },
            'breakout': {
                'resistance_periods': 20,
                'breakout_threshold': 3.0,
                'confirmation_candles': 2
            },
            'reversal': {
                'overbought_threshold': 75,
                'oversold_threshold': 25,
                'confirmation_periods': 2
            },
            'volume_spike': {
                'volume_multiplier': 3.0,
                'price_change_threshold': 2.0,
                'lookback_periods': 10
            },
            'new_listing': {
                'max_age_hours': 24,
                'min_liquidity_usd': 10000,
                'initial_pump_threshold': 5.0
            }
        }
        
        # Strategy performance tracking
        self.strategy_performance = {
            'momentum': {'wins': 0, 'losses': 0, 'total_profit': 0},
            'breakout': {'wins': 0, 'losses': 0, 'total_profit': 0},
            'reversal': {'wins': 0, 'losses': 0, 'total_profit': 0},
            'volume_spike': {'wins': 0, 'losses': 0, 'total_profit': 0},
            'new_listing': {'wins': 0, 'losses': 0, 'total_profit': 0}
        }
        
        # Market condition detection
        self.current_market_condition = 'normal'  # normal, trending, volatile, sideways
        self.market_history = []
        
        # Optimization settings
        self.optimization_frequency = 100  # Optimize after every 100 trades
        self.trades_since_optimization = 0
        
        # Success pattern recognition
        self.success_patterns = []
        
        logger.info("AdaptiveStrategy initialized")
    
    async def select_strategy(self, token_data: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """
        Select best strategy based on market conditions and token data
        
        Args:
            token_data (Dict[str, Any]): Token data
            
        Returns:
            Tuple[str, Dict[str, Any]]: Selected strategy name and parameters
        """
        logger.debug(f"Selecting strategy for token {token_data.get('symbol', '')}")
        
        # Detect market condition
        market_condition = await self._detect_market_condition(token_data)
        self.current_market_condition = market_condition
        
        # Strategy scores
        scores = {
            'momentum': 0,
            'breakout': 0,
            'reversal': 0,
            'volume_spike': 0,
            'new_listing': 0
        }
        
        # Adjust scores based on market condition
        if market_condition == 'trending':
            scores['momentum'] += 30
            scores['breakout'] += 20
        elif market_condition == 'volatile':
            scores['reversal'] += 30
            scores['volume_spike'] += 20
        elif market_condition == 'sideways':
            scores['breakout'] += 10
            scores['reversal'] += 20
            
        # Check for new listing
        discovery_time = token_data.get('discovery_time', 0)
        if discovery_time > 0:
            hours_since_discovery = (time.time() - discovery_time) / 3600
            if hours_since_discovery < self.current_parameters['new_listing']['max_age_hours']:
                scores['new_listing'] += 50 - (hours_since_discovery * 2)  # Score decreases with age
        
        # Check for volume spike
        volume_change = token_data.get('volume_change', 0)
        if volume_change > self.current_parameters['volume_spike']['volume_multiplier']:
            scores['volume_spike'] += 25
        
        # Check for price momentum
        price_change = token_data.get('price_change', 0)
        if abs(price_change) > self.current_parameters['momentum']['threshold']:
            scores['momentum'] += 15
        
        # Add performance-based bias
        for strategy, perf in self.strategy_performance.items():
            if perf['wins'] + perf['losses'] > 0:
                win_rate = perf['wins'] / (perf['wins'] + perf['losses'])
                avg_profit = perf['total_profit'] / (perf['wins'] + perf['losses']) if perf['wins'] + perf['losses'] > 0 else 0
                
                # Bias score based on win rate and average profit
                performance_score = win_rate * 20 + (avg_profit * 5)
                scores[strategy] += performance_score
        
        # Get strategy with highest score
        best_strategy = max(scores.items(), key=lambda x: x[1])
        strategy_name = best_strategy[0]
        
        # Get parameters for selected strategy
        parameters = self.current_parameters[strategy_name]
        
        logger.info(f"Selected strategy: {strategy_name} for market condition: {market_condition}")
        return strategy_name, parameters
    
    async def execute_strategy(self, 
                             strategy_name: str, 
                             parameters: Dict[str, Any], 
                             token_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the selected strategy
        
        Args:
            strategy_name (str): Strategy name
            parameters (Dict[str, Any]): Strategy parameters
            token_data (Dict[str, Any]): Token data
            
        Returns:
            Dict[str, Any]: Strategy results
        """
        logger.debug(f"Executing strategy {strategy_name} for {token_data.get('symbol', '')}")
        
        # Get strategy function
        strategy_func = self.strategies.get(strategy_name)
        
        if not strategy_func:
            logger.warning(f"Unknown strategy: {strategy_name}")
            return {
                'should_trade': False,
                'reason': f"Unknown strategy: {strategy_name}",
                'signals': {}
            }
        
        # Execute strategy
        result = await strategy_func(parameters, token_data)
        
        # Add strategy metadata
        result['strategy_name'] = strategy_name
        result['market_condition'] = self.current_market_condition
        result['parameters'] = parameters
        
        return result
    
    async def update_performance(self, 
                               strategy_name: str, 
                               is_win: bool, 
                               profit_pct: float, 
                               trade_data: Dict[str, Any]):
        """
        Update strategy performance records
        
        Args:
            strategy_name (str): Strategy name
            is_win (bool): Whether trade was successful
            profit_pct (float): Profit/loss percentage
            trade_data (Dict[str, Any]): Complete trade data
        """
        logger.debug(f"Updating performance for {strategy_name}: Win: {is_win}, Profit: {profit_pct}%")
        
        if strategy_name not in self.strategy_performance:
            logger.warning(f"Unknown strategy for performance update: {strategy_name}")
            return
        
        # Update performance records
        if is_win:
            self.strategy_performance[strategy_name]['wins'] += 1
        else:
            self.strategy_performance[strategy_name]['losses'] += 1
        
        self.strategy_performance[strategy_name]['total_profit'] += profit_pct
        
        # Record success pattern if significant profit
        if profit_pct > 10:
            self._record_success_pattern(strategy_name, trade_data)
        
        # Increment counter for optimization
        self.trades_since_optimization += 1
        
        # Check if optimization is needed
        if self.trades_since_optimization >= self.optimization_frequency:
            await self._optimize_parameters()
            self.trades_since_optimization = 0
    
    def _record_success_pattern(self, strategy_name: str, trade_data: Dict[str, Any]):
        """
        Record successful trade pattern for learning
        
        Args:
            strategy_name (str): Strategy name
            trade_data (Dict[str, Any]): Trade data
        """
        # Extract key metrics from trade data
        pattern = {
            'strategy': strategy_name,
            'market_condition': self.current_market_condition,
            'price_change': trade_data.get('price_change', 0),
            'volume_change': trade_data.get('volume_change', 0),
            'liquidity_usd': trade_data.get('liquidity_usd', 0),
            'entry_time': trade_data.get('entry_time', 0),
            'exit_time': trade_data.get('exit_time', 0),
            'profit_pct': trade_data.get('final_pnl_percentage', 0),
            'parameters': self.current_parameters[strategy_name].copy()
        }
        
        self.success_patterns.append(pattern)
        logger.debug(f"Recorded success pattern for {strategy_name}")
        
        # Limit pattern history size
        if len(self.success_patterns) > 100:
            self.success_patterns.pop(0)
    
    async def _optimize_parameters(self):
        """Optimize strategy parameters based on performance data"""
        logger.info("Optimizing strategy parameters")
        
        for strategy_name, performance in self.strategy_performance.items():
            # Only optimize if we have enough data
            if performance['wins'] + performance['losses'] < 5:
                continue
            
            # Calculate win rate
            win_rate = performance['wins'] / (performance['wins'] + performance['losses'])
            
            # Find successful patterns for this strategy
            strategy_patterns = [p for p in self.success_patterns if p['strategy'] == strategy_name]
            
            if strategy_patterns:
                # Extract parameters from successful patterns
                successful_params = {}
                
                for param_name in self.current_parameters[strategy_name].keys():
                    values = [p['parameters'].get(param_name) for p in strategy_patterns]
                    if values:
                        # Use weighted average leaning toward recent patterns
                        weights = np.linspace(0.5, 1.0, len(values))
                        weighted_avg = np.average(values, weights=weights)
                        successful_params[param_name] = weighted_avg
                
                # Adjust current parameters toward successful patterns
                for param_name, value in successful_params.items():
                    current = self.current_parameters[strategy_name][param_name]
                    # Move 20% toward optimal values
                    new_value = current + (value - current) * 0.2
                    
                    # Ensure within valid range
                    param_range = self.parameter_ranges[strategy_name][param_name]
                    new_value = max(param_range[0], min(param_range[1], new_value))
                    
                    self.current_parameters[strategy_name][param_name] = new_value
                    
                logger.debug(f"Optimized parameters for {strategy_name}")
            
            # If strategy is performing poorly, add some randomness
            if win_rate < 0.3:
                self._randomize_parameters(strategy_name, factor=0.3)
                logger.debug(f"Added randomness to parameters for poorly performing strategy: {strategy_name}")
    
    def _randomize_parameters(self, strategy_name: str, factor: float = 0.1):
        """
        Add randomness to strategy parameters
        
        Args:
            strategy_name (str): Strategy name
            factor (float): Randomization factor (0.0-1.0)
        """
        for param_name, param_range in self.parameter_ranges[strategy_name].items():
            current = self.current_parameters[strategy_name][param_name]
            
            # Calculate maximum adjustment
            adjustment_range = (param_range[1] - param_range[0]) * factor
            
            # Apply random adjustment
            adjustment = random.uniform(-adjustment_range, adjustment_range)
            new_value = current + adjustment
            
            # Ensure within valid range
            new_value = max(param_range[0], min(param_range[1], new_value))
            
            self.current_parameters[strategy_name][param_name] = new_value
    
    async def _detect_market_condition(self, token_data: Dict[str, Any]) -> str:
        """
        Detect current market condition
        
        Args:
            token_data (Dict[str, Any]): Token data
            
        Returns:
            str: Market condition ('normal', 'trending', 'volatile', 'sideways')
        """
        # Extract price history if available
        price_history = token_data.get('historical_prices', [])
        
        if not price_history or len(price_history) < 10:
            # Not enough data, assume normal market
            return 'normal'
        
        try:
            # Extract prices
            prices = [float(p['close']) for p in price_history]
            
            # Calculate volatility (standard deviation of returns)
            returns = [((prices[i] - prices[i-1]) / prices[i-1]) for i in range(1, len(prices))]
            volatility = np.std(returns)
            
            # Calculate trend strength (absolute linear regression slope)
            x = np.arange(len(prices))
            slope, _ = np.polyfit(x, prices, 1)
            trend_strength = abs(slope / np.mean(prices))
            
            # Calculate range bound (max - min) / mean
            price_range = (max(prices) - min(prices)) / np.mean(prices)
            
            # Determine market condition
            if volatility > 0.03:  # High volatility
                return 'volatile'
            elif trend_strength > 0.01:  # Strong trend
                return 'trending'
            elif price_range < 0.05:  # Small range
                return 'sideways'
            else:
                return 'normal'
        
        except Exception as e:
            logger.error(f"Error detecting market condition: {str(e)}")
            return 'normal'
    
    async def _momentum_strategy(self, 
                               parameters: Dict[str, Any], 
                               token_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Momentum trading strategy
        
        Args:
            parameters (Dict[str, Any]): Strategy parameters
            token_data (Dict[str, Any]): Token data
            
        Returns:
            Dict[str, Any]: Strategy results
        """
        # Extract parameters
        lookback_period = int(parameters['lookback_period'])
        threshold = parameters['threshold']
        
        # Extract price history
        price_history = token_data.get('historical_prices', [])
        
        if not price_history or len(price_history) < lookback_period:
            return {
                'should_trade': False,
                'reason': "Insufficient price history for momentum strategy",
                'signals': {}
            }
        
        # Calculate price change over lookback period
        current_price = float(price_history[-1]['close'])
        lookback_price = float(price_history[-lookback_period]['close'])
        
        price_change_pct = ((current_price - lookback_price) / lookback_price) * 100
        
        # Check if price change exceeds threshold
        if price_change_pct > threshold:
            # Bullish momentum
            return {
                'should_trade': True,
                'reason': f"Strong bullish momentum: {price_change_pct:.2f}% increase over {lookback_period} periods",
                'signals': {
                    'momentum': price_change_pct,
                    'direction': 'bullish',
                    'strength': min(1.0, price_change_pct / (threshold * 2))
                }
            }
        elif price_change_pct < -threshold:
            # Bearish momentum (we might not want to trade this in a long-only strategy)
            return {
                'should_trade': False,
                'reason': f"Bearish momentum detected: {price_change_pct:.2f}% decrease over {lookback_period} periods",
                'signals': {
                    'momentum': price_change_pct,
                    'direction': 'bearish',
                    'strength': min(1.0, abs(price_change_pct) / (threshold * 2))
                }
            }
        else:
            # No significant momentum
            return {
                'should_trade': False,
                'reason': f"No significant momentum: {price_change_pct:.2f}% change over {lookback_period} periods",
                'signals': {}
            }
    
    async def _breakout_strategy(self, 
                              parameters: Dict[str, Any], 
                              token_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Breakout trading strategy
        
        Args:
            parameters (Dict[str, Any]): Strategy parameters
            token_data (Dict[str, Any]): Token data
            
        Returns:
            Dict[str, Any]: Strategy results
        """
        # Extract parameters
        resistance_periods = int(parameters['resistance_periods'])
        breakout_threshold = parameters['breakout_threshold']
        confirmation_candles = int(parameters['confirmation_candles'])
        
        # Extract price history
        price_history = token_data.get('historical_prices', [])
        
        if not price_history or len(price_history) < resistance_periods + confirmation_candles:
            return {
                'should_trade': False,
                'reason': "Insufficient price history for breakout strategy",
                'signals': {}
            }
        
        # Find resistance level (highest high in the period)
        resistance_history = price_history[-(resistance_periods + confirmation_candles):-confirmation_candles]
        resistance_level = max([float(p['high']) for p in resistance_history])
        
        # Check recent candles for breakout
        recent_candles = price_history[-confirmation_candles:]
        current_price = float(recent_candles[-1]['close'])
        
        # Check if all recent candles closed above resistance
        breakout_confirmed = all([float(p['close']) > resistance_level for p in recent_candles])
        
        # Calculate percentage breakout
        breakout_pct = ((current_price - resistance_level) / resistance_level) * 100
        
        if breakout_confirmed and breakout_pct > breakout_threshold:
            return {
                'should_trade': True,
                'reason': f"Confirmed breakout: {breakout_pct:.2f}% above resistance level",
                'signals': {
                    'breakout': breakout_pct,
                    'resistance_level': resistance_level,
                    'strength': min(1.0, breakout_pct / (breakout_threshold * 2))
                }
            }
        else:
            return {
                'should_trade': False,
                'reason': "No confirmed breakout detected",
                'signals': {}
            }
    
    async def _reversal_strategy(self, 
                               parameters: Dict[str, Any], 
                               token_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Reversal trading strategy
        
        Args:
            parameters (Dict[str, Any]): Strategy parameters
            token_data (Dict[str, Any]): Token data
            
        Returns:
            Dict[str, Any]: Strategy results
        """
        # Extract parameters
        overbought_threshold = parameters['overbought_threshold']
        oversold_threshold = parameters['oversold_threshold']
        confirmation_periods = int(parameters['confirmation_periods'])
        
        # Extract price history
        price_history = token_data.get('historical_prices', [])
        
        if not price_history or len(price_history) < 14 + confirmation_periods:
            return {
                'should_trade': False,
                'reason': "Insufficient price history for reversal strategy",
                'signals': {}
            }
        
        # Calculate RSI (14-period)
        prices = [float(p['close']) for p in price_history]
        rsi = self._calculate_rsi(prices)
        
        # Check for oversold condition with price confirmation
        if rsi < oversold_threshold:
            # Check for confirmation (price increasing for confirmation_periods)
            recent_prices = prices[-confirmation_periods:]
            price_increasing = all(recent_prices[i] > recent_prices[i-1] for i in range(1, len(recent_prices)))
            
            if price_increasing:
                return {
                    'should_trade': True,
                    'reason': f"Oversold reversal: RSI {rsi:.2f} with {confirmation_periods} periods of price increase",
                    'signals': {
                        'rsi': rsi,
                        'condition': 'oversold',
                        'strength': min(1.0, (oversold_threshold - rsi) / oversold_threshold)
                    }
                }
        
        # We could add overbought reversals for short selling, but for a long-only strategy we'll skip that
        
        return {
            'should_trade': False,
            'reason': "No reversal pattern detected",
            'signals': {}
        }
    
    async def _volume_spike_strategy(self, 
                                   parameters: Dict[str, Any], 
                                   token_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Volume spike trading strategy
        
        Args:
            parameters (Dict[str, Any]): Strategy parameters
            token_data (Dict[str, Any]): Token data
            
        Returns:
            Dict[str, Any]: Strategy results
        """
        # Extract parameters
        volume_multiplier = parameters['volume_multiplier']
        price_change_threshold = parameters['price_change_threshold']
        lookback_periods = int(parameters['lookback_periods'])
        
        # Extract price/volume history
        price_history = token_data.get('historical_prices', [])
        
        if not price_history or len(price_history) < lookback_periods + 1:
            return {
                'should_trade': False,
                'reason': "Insufficient price/volume history for volume spike strategy",
                'signals': {}
            }
        
        # Calculate average volume over lookback period
        lookback_volumes = [float(p['volume']) for p in price_history[-lookback_periods-1:-1]]
        avg_volume = sum(lookback_volumes) / len(lookback_volumes)
        
        # Get current volume and price change
        current_volume = float(price_history[-1]['volume'])
        current_price = float(price_history[-1]['close'])
        previous_price = float(price_history[-2]['close'])
        
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 0
        price_change_pct = ((current_price - previous_price) / previous_price) * 100
        
        # Check for volume spike with positive price movement
        if volume_ratio > volume_multiplier and price_change_pct > price_change_threshold:
            return {
                'should_trade': True,
                'reason': f"Volume spike: {volume_ratio:.2f}x average with {price_change_pct:.2f}% price increase",
                'signals': {
                    'volume_ratio': volume_ratio,
                    'price_change': price_change_pct,
                    'strength': min(1.0, (volume_ratio / volume_multiplier) * (price_change_pct / price_change_threshold) / 2)
                }
            }
        
        return {
            'should_trade': False,
            'reason': "No significant volume spike with price movement detected",
            'signals': {}
        }
    
    async def _new_listing_strategy(self, 
                                  parameters: Dict[str, Any], 
                                  token_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        New token listing strategy
        
        Args:
            parameters (Dict[str, Any]): Strategy parameters
            token_data (Dict[str, Any]): Token data
            
        Returns:
            Dict[str, Any]: Strategy results
        """
        # Extract parameters
        max_age_hours = parameters['max_age_hours']
        min_liquidity_usd = parameters['min_liquidity_usd']
        initial_pump_threshold = parameters['initial_pump_threshold']
        
        # Check token age
        discovery_time = token_data.get('discovery_time', 0)
        if discovery_time == 0:
            return {
                'should_trade': False,
                'reason': "Unknown discovery time for new listing strategy",
                'signals': {}
            }
        
        hours_since_discovery = (time.time() - discovery_time) / 3600
        
        # Check liquidity
        liquidity_usd = token_data.get('liquidity_usd', 0)
        
        # Extract price history if available
        price_history = token_data.get('historical_prices', [])
        initial_pump = False
        
        if price_history and len(price_history) >= 2:
            # Check if there's been an initial pump
            initial_price = float(price_history[0]['close'])
            current_price = float(price_history[-1]['close'])
            price_change_pct = ((current_price - initial_price) / initial_price) * 100
            
            initial_pump = price_change_pct > initial_pump_threshold
        
        # Check strategy conditions
        if hours_since_discovery <= max_age_hours and liquidity_usd >= min_liquidity_usd:
            freshness_score = 1.0 - (hours_since_discovery / max_age_hours)
            
            return {
                'should_trade': True,
                'reason': f"New token ({hours_since_discovery:.1f} hours old) with adequate liquidity (${liquidity_usd:.2f})",
                'signals': {
                    'token_age_hours': hours_since_discovery,
                    'liquidity_usd': liquidity_usd,
                    'initial_pump': initial_pump,
                    'freshness': freshness_score,
                    'strength': freshness_score * 0.8 + 0.2
                }
            }
        
        return {
            'should_trade': False,
            'reason': f"Token doesn't meet new listing criteria (age: {hours_since_discovery:.1f}h, liquidity: ${liquidity_usd})",
            'signals': {}
        }
    
    def _calculate_rsi(self, prices: List[float], period: int = 14) -> float:
        """
        Calculate Relative Strength Index
        
        Args:
            prices (List[float]): Price series
            period (int): RSI period
            
        Returns:
            float: RSI value
        """
        if len(prices) < period + 1:
            return 50  # Not enough data, return neutral
        
        # Calculate price changes
        deltas = np.diff(prices)
        
        # Get gains and losses
        gain = np.where(deltas > 0, deltas, 0)
        loss = np.where(deltas < 0, -deltas, 0)
        
        # Calculate average gain and loss
        avg_gain = np.sum(gain[-period:]) / period
        avg_loss = np.sum(loss[-period:]) / period
        
        if avg_loss == 0:
            return 100  # No losses, RSI = 100
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return float(rsi)
    
    def get_strategy_parameters(self, strategy_name: str) -> Dict[str, Any]:
        """
        Get current parameters for a strategy
        
        Args:
            strategy_name (str): Strategy name
            
        Returns:
            Dict[str, Any]: Strategy parameters
        """
        return self.current_parameters.get(strategy_name, {})
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get performance statistics for all strategies
        
        Returns:
            Dict[str, Any]: Performance statistics
        """
        stats = {}
        
        for strategy_name, perf in self.strategy_performance.items():
            total_trades = perf['wins'] + perf['losses']
            win_rate = perf['wins'] / total_trades if total_trades > 0 else 0
            
            stats[strategy_name] = {
                'total_trades': total_trades,
                'wins': perf['wins'],
                'losses': perf['losses'],
                'win_rate': win_rate,
                'total_profit': perf['total_profit'],
                'avg_profit': perf['total_profit'] / total_trades if total_trades > 0 else 0
            }
        
        return stats
