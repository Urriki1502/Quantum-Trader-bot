"""
StrategyManager Component
Responsible for coordinating and executing trading strategies,
managing strategy allocation, and tracking performance.
"""

import asyncio
import logging
import time
from typing import Dict, Any, List, Optional, Tuple

from core.state_manager import StateManager
from core.config_manager import ConfigManager
from strategy.adaptive_strategy import AdaptiveStrategy

logger = logging.getLogger(__name__)

class StrategyManager:
    """
    StrategyManager handles:
    - Managing multiple trading strategies
    - Coordinating strategy execution
    - Tracking strategy performance
    - Allocating capital across strategies
    """
    
    def __init__(self, 
                config_manager: ConfigManager,
                state_manager: StateManager):
        """
        Initialize the StrategyManager
        
        Args:
            config_manager (ConfigManager): Configuration manager instance
            state_manager (StateManager): State manager instance
        """
        self.config_manager = config_manager
        self.state_manager = state_manager
        
        # Initialize adaptive strategy
        self.adaptive_strategy = AdaptiveStrategy()
        
        # Strategy settings
        self.default_strategy = self.config_manager.get('strategy.default_strategy', 'momentum')
        self.active_strategies = self.config_manager.get('strategy.active_strategies', 
                                                        ['momentum', 'breakout', 'new_listing'])
        
        # Strategy allocations (percentage of capital)
        self.strategy_allocations = {
            'momentum': 30,
            'breakout': 20,
            'reversal': 15,
            'volume_spike': 15,
            'new_listing': 20
        }
        
        # Performance tracking
        self.strategy_performance = {}
        
        # Active trades by strategy
        self.strategy_trades = {}  # strategy_name -> list of active trades
        
        logger.info("StrategyManager initialized")
    
    async def evaluate_token(self, token_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate token with the appropriate strategy
        
        Args:
            token_data (Dict[str, Any]): Token data
            
        Returns:
            Dict[str, Any]: Evaluation result
        """
        token_address = token_data.get('address')
        token_symbol = token_data.get('symbol', '')
        
        logger.info(f"Evaluating token: {token_symbol} ({token_address})")
        
        try:
            # Get best strategy for this token
            strategy_name, parameters = await self.adaptive_strategy.select_strategy(token_data)
            
            # If selected strategy isn't in active strategies, use default
            if strategy_name not in self.active_strategies:
                logger.debug(f"Selected strategy {strategy_name} not active, using {self.default_strategy}")
                strategy_name = self.default_strategy
                parameters = self.adaptive_strategy.get_strategy_parameters(strategy_name)
            
            # Execute strategy
            strategy_result = await self.adaptive_strategy.execute_strategy(
                strategy_name, parameters, token_data
            )
            
            # Update state manager with metrics
            self.state_manager.update_component_metric(
                'strategy_manager', 
                'strategies_executed', 
                self.state_manager.get_component_state('strategy_manager')
                .get('metrics', {}).get('strategies_executed', 0) + 1
            )
            
            logger.info(f"Strategy {strategy_name} evaluation for {token_symbol}: Trade: {strategy_result['should_trade']}")
            return strategy_result
            
        except Exception as e:
            logger.error(f"Error evaluating token with strategy: {str(e)}")
            return {
                'should_trade': False,
                'reason': f"Strategy error: {str(e)}",
                'signals': {},
                'strategy_name': 'error'
            }
    
    async def record_trade_result(self, 
                                trade_data: Dict[str, Any], 
                                final_pnl: float):
        """
        Record trade result for performance tracking
        
        Args:
            trade_data (Dict[str, Any]): Trade data
            final_pnl (float): Final profit/loss percentage
        """
        strategy_name = trade_data.get('strategy', self.default_strategy)
        token_symbol = trade_data.get('token_symbol', '')
        
        logger.info(f"Recording trade result for {token_symbol} using {strategy_name}: PnL {final_pnl:.2f}%")
        
        # Update adaptive strategy performance
        await self.adaptive_strategy.update_performance(
            strategy_name, 
            is_win=final_pnl > 0, 
            profit_pct=final_pnl,
            trade_data=trade_data
        )
        
        # Update performance tracking
        if strategy_name not in self.strategy_performance:
            self.strategy_performance[strategy_name] = {
                'trades': 0,
                'wins': 0,
                'losses': 0,
                'total_pnl': 0.0,
                'best_trade': 0.0,
                'worst_trade': 0.0
            }
        
        perf = self.strategy_performance[strategy_name]
        perf['trades'] += 1
        perf['total_pnl'] += final_pnl
        
        if final_pnl > 0:
            perf['wins'] += 1
            if final_pnl > perf['best_trade']:
                perf['best_trade'] = final_pnl
        else:
            perf['losses'] += 1
            if final_pnl < perf['worst_trade']:
                perf['worst_trade'] = final_pnl
        
        # Update metrics in state manager
        self.state_manager.update_component_metric(
            'strategy_manager', 
            f'strategy_{strategy_name}_win_rate', 
            perf['wins'] / perf['trades'] if perf['trades'] > 0 else 0
        )
        
        # Update strategy allocation periodically based on performance
        if sum(p['trades'] for p in self.strategy_performance.values()) % 10 == 0:
            await self._adjust_strategy_allocations()
    
    async def _adjust_strategy_allocations(self):
        """Adjust strategy allocations based on performance"""
        logger.debug("Adjusting strategy allocations based on performance")
        
        # Only adjust if we have enough performance data
        total_trades = sum(p['trades'] for p in self.strategy_performance.values())
        if total_trades < 20:
            logger.debug(f"Not enough trades ({total_trades}) to adjust allocations")
            return
        
        # Calculate performance score for each strategy
        scores = {}
        total_score = 0
        
        for strategy_name, perf in self.strategy_performance.items():
            if perf['trades'] < 5:
                # Not enough data for this strategy
                scores[strategy_name] = 10  # Base score
            else:
                win_rate = perf['wins'] / perf['trades'] if perf['trades'] > 0 else 0
                avg_pnl = perf['total_pnl'] / perf['trades']
                
                # Score based on win rate and average P&L
                # Higher win rate and higher average P&L = higher score
                score = (win_rate * 50) + (avg_pnl * 5)
                scores[strategy_name] = max(10, score)  # Minimum score of 10
            
            total_score += scores[strategy_name]
        
        # Calculate new allocations
        if total_score > 0:
            new_allocations = {}
            
            for strategy_name, score in scores.items():
                # Calculate allocation percentage (minimum 5%)
                allocation = max(5, int((score / total_score) * 100))
                new_allocations[strategy_name] = allocation
            
            # Normalize allocations to total 100%
            total_allocation = sum(new_allocations.values())
            if total_allocation != 100:
                scaling_factor = 100 / total_allocation
                for strategy_name in new_allocations:
                    new_allocations[strategy_name] = int(new_allocations[strategy_name] * scaling_factor)
                
                # Adjust rounding errors
                diff = 100 - sum(new_allocations.values())
                if diff != 0:
                    # Add/subtract the difference from the strategy with highest/lowest allocation
                    if diff > 0:
                        strategy_name = max(new_allocations.items(), key=lambda x: x[1])[0]
                    else:
                        strategy_name = min(new_allocations.items(), key=lambda x: x[1])[0]
                    
                    new_allocations[strategy_name] += diff
            
            logger.info(f"Updated strategy allocations: {new_allocations}")
            self.strategy_allocations = new_allocations
    
    async def calculate_position_size(self, 
                                    strategy_name: str, 
                                    base_size: float) -> float:
        """
        Calculate position size based on strategy allocation
        
        Args:
            strategy_name (str): Strategy name
            base_size (float): Base position size
            
        Returns:
            float: Adjusted position size
        """
        # Get allocation for strategy
        allocation = self.strategy_allocations.get(strategy_name, 10)  # Default 10%
        
        # Adjust base size by allocation percentage
        adjusted_size = base_size * (allocation / 100)
        
        logger.debug(f"Adjusted position size for {strategy_name}: ${base_size:.2f} -> ${adjusted_size:.2f} ({allocation}% allocation)")
        
        return adjusted_size
    
    def register_active_trade(self, strategy_name: str, trade_data: Dict[str, Any]):
        """
        Register an active trade for a strategy
        
        Args:
            strategy_name (str): Strategy name
            trade_data (Dict[str, Any]): Trade data
        """
        if strategy_name not in self.strategy_trades:
            self.strategy_trades[strategy_name] = []
        
        self.strategy_trades[strategy_name].append(trade_data)
        
        # Update metrics
        self.state_manager.update_component_metric(
            'strategy_manager', 
            f'active_trades_{strategy_name}', 
            len(self.strategy_trades[strategy_name])
        )
    
    def unregister_active_trade(self, strategy_name: str, token_address: str):
        """
        Unregister an active trade for a strategy
        
        Args:
            strategy_name (str): Strategy name
            token_address (str): Token address
        """
        if strategy_name in self.strategy_trades:
            self.strategy_trades[strategy_name] = [
                t for t in self.strategy_trades[strategy_name] 
                if t.get('token_address') != token_address
            ]
            
            # Update metrics
            self.state_manager.update_component_metric(
                'strategy_manager', 
                f'active_trades_{strategy_name}', 
                len(self.strategy_trades[strategy_name])
            )
    
    def get_active_strategies(self) -> List[str]:
        """
        Get list of active strategies
        
        Returns:
            List[str]: Active strategy names
        """
        return self.active_strategies
    
    def set_active_strategies(self, strategies: List[str]):
        """
        Set active strategies
        
        Args:
            strategies (List[str]): List of strategy names to activate
        """
        # Validate strategies
        valid_strategies = []
        for strategy in strategies:
            if strategy in self.adaptive_strategy.strategies:
                valid_strategies.append(strategy)
            else:
                logger.warning(f"Unknown strategy: {strategy}")
        
        if not valid_strategies:
            logger.warning("No valid strategies specified, keeping current active strategies")
            return
        
        self.active_strategies = valid_strategies
        logger.info(f"Set active strategies: {valid_strategies}")
    
    def get_strategy_allocations(self) -> Dict[str, int]:
        """
        Get strategy allocations
        
        Returns:
            Dict[str, int]: Strategy allocations
        """
        return self.strategy_allocations
    
    def set_strategy_allocations(self, allocations: Dict[str, int]):
        """
        Set strategy allocations
        
        Args:
            allocations (Dict[str, int]): Strategy allocations
        """
        # Validate allocations
        total = sum(allocations.values())
        if total != 100:
            logger.warning(f"Strategy allocations do not sum to 100% (total: {total}%)")
            # Normalize to 100%
            factor = 100 / total
            for strategy in allocations:
                allocations[strategy] = int(allocations[strategy] * factor)
        
        self.strategy_allocations = allocations
        logger.info(f"Set strategy allocations: {allocations}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get performance summary for all strategies
        
        Returns:
            Dict[str, Any]: Performance summary
        """
        # Get adaptive strategy performance stats
        adaptive_stats = self.adaptive_strategy.get_performance_stats()
        
        # Combine with our tracking
        summary = {
            'by_strategy': adaptive_stats,
            'allocations': self.strategy_allocations,
            'active_strategies': self.active_strategies,
            'total_trades': sum(p.get('trades', 0) for p in self.strategy_performance.values()),
            'total_wins': sum(p.get('wins', 0) for p in self.strategy_performance.values()),
            'total_losses': sum(p.get('losses', 0) for p in self.strategy_performance.values()),
            'active_trades_by_strategy': {s: len(t) for s, t in self.strategy_trades.items()}
        }
        
        # Calculate overall win rate
        if summary['total_trades'] > 0:
            summary['overall_win_rate'] = summary['total_wins'] / summary['total_trades']
        else:
            summary['overall_win_rate'] = 0
        
        return summary
