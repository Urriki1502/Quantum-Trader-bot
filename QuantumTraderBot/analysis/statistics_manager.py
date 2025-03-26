"""
StatisticsManager Component
Responsible for tracking and analyzing trading statistics,
performance metrics, and other system metrics.
"""

import asyncio
import logging
import time
import json
import os
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class StatisticsManager:
    """
    StatisticsManager handles:
    - Collecting trading statistics
    - Calculating performance metrics
    - Analyzing win/loss ratios
    - Tracking portfolio performance
    """
    
    def __init__(self):
        """Initialize the StatisticsManager"""
        # Trading statistics
        self.trades = []
        self.daily_stats = {}
        self.token_stats = {}
        self.strategy_stats = {}
        
        # Performance metrics
        self.win_count = 0
        self.loss_count = 0
        self.total_profit_usd = 0
        self.total_loss_usd = 0
        self.largest_win_usd = 0
        self.largest_loss_usd = 0
        
        # Portfolio tracking
        self.portfolio_history = []
        self.initial_portfolio_value = 0
        self.current_portfolio_value = 0
        
        # Other metrics
        self.last_stats_save = 0
        self.stats_save_path = './data/trading_stats.json'
        
        # Load any existing stats
        self._load_stats()
        
        logger.info("StatisticsManager initialized")
    
    def _load_stats(self):
        """Load statistics from saved file if available"""
        try:
            if os.path.exists(self.stats_save_path):
                with open(self.stats_save_path, 'r') as f:
                    data = json.load(f)
                
                # Load saved stats
                self.trades = data.get('trades', [])
                self.daily_stats = data.get('daily_stats', {})
                self.token_stats = data.get('token_stats', {})
                self.strategy_stats = data.get('strategy_stats', {})
                self.win_count = data.get('win_count', 0)
                self.loss_count = data.get('loss_count', 0)
                self.total_profit_usd = data.get('total_profit_usd', 0)
                self.total_loss_usd = data.get('total_loss_usd', 0)
                self.largest_win_usd = data.get('largest_win_usd', 0)
                self.largest_loss_usd = data.get('largest_loss_usd', 0)
                self.portfolio_history = data.get('portfolio_history', [])
                
                if self.portfolio_history:
                    self.initial_portfolio_value = self.portfolio_history[0].get('value', 0)
                    self.current_portfolio_value = self.portfolio_history[-1].get('value', 0)
                
                logger.info(f"Loaded statistics: {len(self.trades)} trades, {len(self.portfolio_history)} portfolio snapshots")
        
        except Exception as e:
            logger.error(f"Error loading statistics: {str(e)}")
    
    async def save_stats(self):
        """Save statistics to file"""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.stats_save_path), exist_ok=True)
            
            # Prepare data
            data = {
                'trades': self.trades,
                'daily_stats': self.daily_stats,
                'token_stats': self.token_stats,
                'strategy_stats': self.strategy_stats,
                'win_count': self.win_count,
                'loss_count': self.loss_count,
                'total_profit_usd': self.total_profit_usd,
                'total_loss_usd': self.total_loss_usd,
                'largest_win_usd': self.largest_win_usd,
                'largest_loss_usd': self.largest_loss_usd,
                'portfolio_history': self.portfolio_history,
                'last_updated': time.time()
            }
            
            # Save to file
            with open(self.stats_save_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            self.last_stats_save = time.time()
            logger.debug("Statistics saved to file")
            
        except Exception as e:
            logger.error(f"Error saving statistics: {str(e)}")
    
    async def record_trade(self, trade_data: Dict[str, Any]):
        """
        Record a completed trade
        
        Args:
            trade_data (Dict[str, Any]): Trade data
        """
        # Extract trade details
        token_address = trade_data.get('token_address', '')
        token_symbol = trade_data.get('token_symbol', '')
        amount_usd = trade_data.get('amount_usd', 0)
        entry_price = trade_data.get('entry_price', 0)
        exit_price = trade_data.get('exit_price', 0)
        pnl_percentage = trade_data.get('final_pnl_percentage', 0)
        strategy = trade_data.get('strategy', 'unknown')
        entry_time = trade_data.get('entry_time', time.time())
        exit_time = trade_data.get('exit_time', time.time())
        
        logger.info(f"Recording trade for {token_symbol}: PnL {pnl_percentage:.2f}%")
        
        # Calculate PnL in USD
        pnl_usd = amount_usd * (pnl_percentage / 100)
        
        # Determine if win or loss
        is_win = pnl_percentage > 0
        
        # Create trade record
        trade_record = {
            'token_address': token_address,
            'token_symbol': token_symbol,
            'amount_usd': amount_usd,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'entry_time': entry_time,
            'exit_time': exit_time,
            'holding_time_seconds': exit_time - entry_time,
            'pnl_percentage': pnl_percentage,
            'pnl_usd': pnl_usd,
            'is_win': is_win,
            'strategy': strategy,
            'record_time': time.time()
        }
        
        # Add to trades list
        self.trades.append(trade_record)
        
        # Update win/loss counters
        if is_win:
            self.win_count += 1
            self.total_profit_usd += pnl_usd
            if pnl_usd > self.largest_win_usd:
                self.largest_win_usd = pnl_usd
        else:
            self.loss_count += 1
            self.total_loss_usd += abs(pnl_usd)
            if abs(pnl_usd) > self.largest_loss_usd:
                self.largest_loss_usd = abs(pnl_usd)
        
        # Update token stats
        if token_address not in self.token_stats:
            self.token_stats[token_address] = {
                'token_symbol': token_symbol,
                'trades_count': 0,
                'win_count': 0,
                'loss_count': 0,
                'total_pnl_usd': 0,
                'avg_pnl_percentage': 0,
                'total_volume_usd': 0
            }
        
        token_stat = self.token_stats[token_address]
        token_stat['trades_count'] += 1
        token_stat['win_count'] += 1 if is_win else 0
        token_stat['loss_count'] += 1 if not is_win else 0
        token_stat['total_pnl_usd'] += pnl_usd
        token_stat['total_volume_usd'] += amount_usd
        token_stat['avg_pnl_percentage'] = (token_stat['total_pnl_usd'] / token_stat['total_volume_usd']) * 100 if token_stat['total_volume_usd'] > 0 else 0
        
        # Update strategy stats
        if strategy not in self.strategy_stats:
            self.strategy_stats[strategy] = {
                'trades_count': 0,
                'win_count': 0,
                'loss_count': 0,
                'total_pnl_usd': 0,
                'avg_pnl_percentage': 0,
                'total_volume_usd': 0
            }
        
        strategy_stat = self.strategy_stats[strategy]
        strategy_stat['trades_count'] += 1
        strategy_stat['win_count'] += 1 if is_win else 0
        strategy_stat['loss_count'] += 1 if not is_win else 0
        strategy_stat['total_pnl_usd'] += pnl_usd
        strategy_stat['total_volume_usd'] += amount_usd
        strategy_stat['avg_pnl_percentage'] = (strategy_stat['total_pnl_usd'] / strategy_stat['total_volume_usd']) * 100 if strategy_stat['total_volume_usd'] > 0 else 0
        
        # Update daily stats
        date_key = datetime.fromtimestamp(exit_time).strftime('%Y-%m-%d')
        if date_key not in self.daily_stats:
            self.daily_stats[date_key] = {
                'trades_count': 0,
                'win_count': 0,
                'loss_count': 0,
                'total_pnl_usd': 0,
                'total_volume_usd': 0
            }
        
        daily_stat = self.daily_stats[date_key]
        daily_stat['trades_count'] += 1
        daily_stat['win_count'] += 1 if is_win else 0
        daily_stat['loss_count'] += 1 if not is_win else 0
        daily_stat['total_pnl_usd'] += pnl_usd
        daily_stat['total_volume_usd'] += amount_usd
        
        # Save stats periodically
        if time.time() - self.last_stats_save > 300:  # Save every 5 minutes
            await self.save_stats()
    
    async def update_portfolio_value(self, portfolio_value: float):
        """
        Update portfolio value
        
        Args:
            portfolio_value (float): Current portfolio value in USD
        """
        logger.debug(f"Updating portfolio value: ${portfolio_value:.2f}")
        
        # Set initial value if this is the first update
        if not self.portfolio_history:
            self.initial_portfolio_value = portfolio_value
        
        self.current_portfolio_value = portfolio_value
        
        # Add to history
        self.portfolio_history.append({
            'timestamp': time.time(),
            'value': portfolio_value
        })
        
        # Limit history size
        if len(self.portfolio_history) > 1000:
            self.portfolio_history = self.portfolio_history[-1000:]
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get summary of trading performance
        
        Returns:
            Dict[str, Any]: Performance summary
        """
        total_trades = self.win_count + self.loss_count
        win_rate = self.win_count / total_trades if total_trades > 0 else 0
        
        profit_factor = self.total_profit_usd / self.total_loss_usd if self.total_loss_usd > 0 else float('inf')
        
        net_profit = self.total_profit_usd - self.total_loss_usd
        
        # Calculate ROI
        portfolio_roi = ((self.current_portfolio_value - self.initial_portfolio_value) / 
                         self.initial_portfolio_value * 100) if self.initial_portfolio_value > 0 else 0
        
        # Calculate average trade metrics
        avg_win = self.total_profit_usd / self.win_count if self.win_count > 0 else 0
        avg_loss = self.total_loss_usd / self.loss_count if self.loss_count > 0 else 0
        
        # Calculate expectancy
        expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
        
        return {
            'total_trades': total_trades,
            'win_count': self.win_count,
            'loss_count': self.loss_count,
            'win_rate': win_rate,
            'total_profit_usd': self.total_profit_usd,
            'total_loss_usd': self.total_loss_usd,
            'net_profit_usd': net_profit,
            'profit_factor': profit_factor,
            'portfolio_roi': portfolio_roi,
            'largest_win_usd': self.largest_win_usd,
            'largest_loss_usd': self.largest_loss_usd,
            'avg_win_usd': avg_win,
            'avg_loss_usd': avg_loss,
            'expectancy_usd': expectancy,
            'initial_portfolio_value': self.initial_portfolio_value,
            'current_portfolio_value': self.current_portfolio_value,
            'last_updated': time.time()
        }
    
    def get_daily_performance(self, days: int = 7) -> List[Dict[str, Any]]:
        """
        Get daily performance for the last N days
        
        Args:
            days (int): Number of days to return
            
        Returns:
            List[Dict[str, Any]]: Daily performance data
        """
        # Generate list of dates for the last N days
        today = datetime.now().date()
        date_list = [(today - timedelta(days=i)).strftime('%Y-%m-%d') for i in range(days)]
        
        daily_performance = []
        
        for date_key in date_list:
            if date_key in self.daily_stats:
                stats = self.daily_stats[date_key]
                daily_performance.append({
                    'date': date_key,
                    'trades_count': stats['trades_count'],
                    'win_count': stats['win_count'],
                    'loss_count': stats['loss_count'],
                    'win_rate': stats['win_count'] / stats['trades_count'] if stats['trades_count'] > 0 else 0,
                    'total_pnl_usd': stats['total_pnl_usd'],
                    'total_volume_usd': stats['total_volume_usd']
                })
            else:
                # No trades on this day
                daily_performance.append({
                    'date': date_key,
                    'trades_count': 0,
                    'win_count': 0,
                    'loss_count': 0,
                    'win_rate': 0,
                    'total_pnl_usd': 0,
                    'total_volume_usd': 0
                })
        
        # Reverse to get chronological order
        daily_performance.reverse()
        return daily_performance
    
    def get_strategy_performance(self) -> List[Dict[str, Any]]:
        """
        Get performance breakdown by strategy
        
        Returns:
            List[Dict[str, Any]]: Strategy performance data
        """
        strategy_performance = []
        
        for strategy, stats in self.strategy_stats.items():
            strategy_performance.append({
                'strategy': strategy,
                'trades_count': stats['trades_count'],
                'win_count': stats['win_count'],
                'loss_count': stats['loss_count'],
                'win_rate': stats['win_count'] / stats['trades_count'] if stats['trades_count'] > 0 else 0,
                'total_pnl_usd': stats['total_pnl_usd'],
                'avg_pnl_percentage': stats['avg_pnl_percentage'],
                'total_volume_usd': stats['total_volume_usd']
            })
        
        # Sort by total PnL (best performing first)
        strategy_performance.sort(key=lambda x: x['total_pnl_usd'], reverse=True)
        return strategy_performance
    
    def get_token_performance(self, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Get performance breakdown by token
        
        Args:
            limit (int): Maximum number of tokens to return
            
        Returns:
            List[Dict[str, Any]]: Token performance data
        """
        token_performance = []
        
        for token_address, stats in self.token_stats.items():
            token_performance.append({
                'token_address': token_address,
                'token_symbol': stats['token_symbol'],
                'trades_count': stats['trades_count'],
                'win_count': stats['win_count'],
                'loss_count': stats['loss_count'],
                'win_rate': stats['win_count'] / stats['trades_count'] if stats['trades_count'] > 0 else 0,
                'total_pnl_usd': stats['total_pnl_usd'],
                'avg_pnl_percentage': stats['avg_pnl_percentage'],
                'total_volume_usd': stats['total_volume_usd']
            })
        
        # Sort by total PnL (best performing first)
        token_performance.sort(key=lambda x: x['total_pnl_usd'], reverse=True)
        
        # Limit the number of tokens returned
        return token_performance[:limit]
    
    def get_portfolio_history(self, 
                             interval: str = 'hourly', 
                             limit: int = 168) -> List[Dict[str, Any]]:
        """
        Get portfolio value history with specified interval
        
        Args:
            interval (str): 'hourly', 'daily', or 'raw'
            limit (int): Maximum number of data points
            
        Returns:
            List[Dict[str, Any]]: Portfolio history
        """
        if not self.portfolio_history:
            return []
        
        if interval == 'raw':
            # Return raw data, limited to the requested number of points
            return self.portfolio_history[-limit:]
        
        # Process data based on interval
        processed_history = []
        
        if interval == 'hourly':
            # Group by hour
            hour_buckets = {}
            
            for entry in self.portfolio_history:
                hour_key = datetime.fromtimestamp(entry['timestamp']).strftime('%Y-%m-%d %H:00:00')
                if hour_key not in hour_buckets:
                    hour_buckets[hour_key] = []
                hour_buckets[hour_key].append(entry['value'])
            
            for hour_key, values in hour_buckets.items():
                avg_value = sum(values) / len(values)
                processed_history.append({
                    'timestamp': datetime.strptime(hour_key, '%Y-%m-%d %H:00:00').timestamp(),
                    'value': avg_value
                })
        
        elif interval == 'daily':
            # Group by day
            day_buckets = {}
            
            for entry in self.portfolio_history:
                day_key = datetime.fromtimestamp(entry['timestamp']).strftime('%Y-%m-%d')
                if day_key not in day_buckets:
                    day_buckets[day_key] = []
                day_buckets[day_key].append(entry['value'])
            
            for day_key, values in day_buckets.items():
                avg_value = sum(values) / len(values)
                processed_history.append({
                    'timestamp': datetime.strptime(day_key, '%Y-%m-%d').timestamp(),
                    'value': avg_value
                })
        
        # Sort by timestamp
        processed_history.sort(key=lambda x: x['timestamp'])
        
        # Limit to requested number of points
        return processed_history[-limit:]
    
    def get_recent_trades(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get most recent trades
        
        Args:
            limit (int): Maximum number of trades to return
            
        Returns:
            List[Dict[str, Any]]: Recent trades
        """
        # Sort trades by exit time (most recent first)
        sorted_trades = sorted(self.trades, key=lambda x: x.get('exit_time', 0), reverse=True)
        
        # Return limited number of trades
        return sorted_trades[:limit]
    
    def get_trade_by_id(self, trade_id: str) -> Optional[Dict[str, Any]]:
        """
        Get trade by ID
        
        Args:
            trade_id (str): Trade ID
            
        Returns:
            Optional[Dict[str, Any]]: Trade data or None if not found
        """
        for trade in self.trades:
            if trade.get('id') == trade_id:
                return trade
        return None
