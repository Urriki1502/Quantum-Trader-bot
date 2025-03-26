"""
Flash Executor Component
Responsible for high-speed trade execution with optimal routing
and transaction optimization.
"""

import asyncio
import logging
import time
from typing import Dict, List, Any, Optional, Tuple
import base58

from core.state_manager import StateManager
from core.config_manager import ConfigManager
from trading.mev_protection import MEVProtection
from trading.gas_predictor import GasPredictor

logger = logging.getLogger(__name__)

class ExecutionTimer:
    """Utility class to measure execution time"""
    
    def __init__(self):
        """Initialize the timer"""
        self.start_time = time.time()
        self.last_checkpoint = self.start_time
        self.checkpoints = {}
    
    def checkpoint(self, name: str):
        """Record a checkpoint"""
        now = time.time()
        duration = now - self.last_checkpoint
        self.checkpoints[name] = duration
        self.last_checkpoint = now
    
    def elapsed(self) -> float:
        """Get total elapsed time in ms"""
        return (time.time() - self.start_time) * 1000
    
    def get_checkpoints(self) -> Dict[str, float]:
        """Get all checkpoints in ms"""
        return {k: v * 1000 for k, v in self.checkpoints.items()}


class OptimalRouteCalculator:
    """Calculates optimal trading routes for best execution"""
    
    def __init__(self, config_manager: ConfigManager):
        """
        Initialize the OptimalRouteCalculator
        
        Args:
            config_manager (ConfigManager): Configuration manager instance
        """
        self.config_manager = config_manager
        
        # Route optimization settings
        self.min_liquidity_threshold = self.config_manager.get(
            'trading.min_liquidity_threshold', 1000)
        self.max_price_impact_threshold = self.config_manager.get(
            'trading.max_price_impact_threshold', 5.0)
        self.split_threshold_usd = self.config_manager.get(
            'trading.split_threshold_usd', 5000)
    
    async def calculate_optimal_route(self, 
                                    token_address: str, 
                                    amount_usd: float,
                                    dex_client) -> Dict[str, Any]:
        """
        Calculate optimal trading route
        
        Args:
            token_address (str): Token address
            amount_usd (float): Trade amount in USD
            dex_client: DEX client instance
            
        Returns:
            Dict[str, Any]: Optimal route data
        """
        # Check if trade should be split
        should_split = amount_usd > self.split_threshold_usd
        
        # Get pool data for the token
        pools = await dex_client.get_token_pools(token_address)
        
        if not pools:
            logger.warning(f"No pools found for token {token_address}")
            return {
                'token_address': token_address,
                'amount_usd': amount_usd,
                'route_type': 'direct',
                'pools': [],
                'estimated_price_impact': 100,
                'success': False,
                'error': 'No pools found'
            }
        
        # Filter pools by liquidity
        valid_pools = [p for p in pools if p.get('liquidity_usd', 0) >= self.min_liquidity_threshold]
        
        if not valid_pools:
            logger.warning(f"No pools with sufficient liquidity for token {token_address}")
            return {
                'token_address': token_address,
                'amount_usd': amount_usd,
                'route_type': 'direct',
                'pools': [pools[0]] if pools else [],
                'estimated_price_impact': 100,
                'success': False,
                'error': 'Insufficient liquidity'
            }
        
        # Sort pools by liquidity (descending)
        sorted_pools = sorted(valid_pools, key=lambda p: p.get('liquidity_usd', 0), reverse=True)
        
        # Calculate price impact for best pool
        best_pool = sorted_pools[0]
        price_impact = await dex_client.estimate_price_impact(token_address, amount_usd)
        
        # If price impact is too high and should split, use multiple pools
        if price_impact > self.max_price_impact_threshold and should_split:
            return await self._calculate_split_route(token_address, amount_usd, sorted_pools, dex_client)
        
        # Otherwise use direct route with best pool
        return {
            'token_address': token_address,
            'amount_usd': amount_usd,
            'route_type': 'direct',
            'pools': [best_pool],
            'estimated_price_impact': price_impact,
            'success': True
        }
    
    async def _calculate_split_route(self, 
                                   token_address: str, 
                                   amount_usd: float,
                                   sorted_pools: List[Dict[str, Any]],
                                   dex_client) -> Dict[str, Any]:
        """
        Calculate split route across multiple pools
        
        Args:
            token_address (str): Token address
            amount_usd (float): Trade amount in USD
            sorted_pools (List[Dict[str, Any]]): Sorted list of pools
            dex_client: DEX client instance
            
        Returns:
            Dict[str, Any]: Split route data
        """
        # Calculate optimal split to minimize price impact
        splits = []
        remaining_amount = amount_usd
        total_impact = 0
        
        for pool in sorted_pools:
            # Skip if no more amount to allocate
            if remaining_amount <= 0:
                break
                
            # Calculate optimal amount for this pool
            pool_liquidity = pool.get('liquidity_usd', 0)
            
            # Target amount is based on pool liquidity and remaining amount
            # Try to use no more than 2% of pool liquidity to limit impact
            target_amount = min(remaining_amount, pool_liquidity * 0.02)
            
            if target_amount < 10:  # Skip if too small
                continue
                
            # Calculate price impact for this amount
            impact = await dex_client.estimate_price_impact(token_address, target_amount)
            
            # Add to splits
            splits.append({
                'pool': pool,
                'amount_usd': target_amount,
                'price_impact': impact
            })
            
            # Update remaining amount and total impact
            remaining_amount -= target_amount
            
            # Weighted impact
            total_impact += impact * (target_amount / amount_usd)
        
        # Check if we allocated all the amount
        if remaining_amount > 0 and splits:
            # Distribute remaining among existing splits
            for split in splits:
                additional = remaining_amount / len(splits)
                split['amount_usd'] += additional
            remaining_amount = 0
        
        if not splits:
            # Fall back to direct route with best pool
            return {
                'token_address': token_address,
                'amount_usd': amount_usd,
                'route_type': 'direct',
                'pools': [sorted_pools[0]],
                'estimated_price_impact': await dex_client.estimate_price_impact(token_address, amount_usd),
                'success': True,
                'warning': 'Could not calculate optimal split, using direct route'
            }
        
        return {
            'token_address': token_address,
            'amount_usd': amount_usd,
            'route_type': 'split',
            'splits': splits,
            'total_split_count': len(splits),
            'estimated_price_impact': total_impact,
            'success': True
        }


class FlashExecutor:
    """
    FlashExecutor handles high-speed trade execution with:
    - Optimal routing
    - Transaction batching
    - MEV protection
    - Gas optimization
    """
    
    def __init__(self, 
                config_manager: ConfigManager,
                state_manager: StateManager,
                dex_client,
                rpc_client,
                mev_protection: MEVProtection,
                gas_predictor: GasPredictor):
        """
        Initialize the FlashExecutor
        
        Args:
            config_manager (ConfigManager): Configuration manager instance
            state_manager (StateManager): State manager instance
            dex_client: DEX client instance
            rpc_client: RPC client instance
            mev_protection (MEVProtection): MEV protection instance
            gas_predictor (GasPredictor): Gas predictor instance
        """
        self.config_manager = config_manager
        self.state_manager = state_manager
        self.dex_client = dex_client
        self.rpc_client = rpc_client
        self.mev_protection = mev_protection
        self.gas_predictor = gas_predictor
        
        # Initialize route calculator
        self.router = OptimalRouteCalculator(config_manager)
        
        # Execution settings
        self.max_retries = self.config_manager.get('trading.max_retries', 3)
        self.retry_delay = self.config_manager.get('trading.retry_delay', 1.0)
        self.priority_execution = self.config_manager.get('trading.priority_execution', True)
        
        # Success rate tracking
        self.execution_count = 0
        self.success_count = 0
        
        logger.info("FlashExecutor initialized")
    
    async def execute_flash_trade(self, trade_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a flash trade with optimal execution
        
        Args:
            trade_params (Dict[str, Any]): Trade parameters
            
        Returns:
            Dict[str, Any]: Trade result
        """
        token_address = trade_params.get('token_address')
        amount_usd = trade_params.get('amount_usd', 0)
        action = trade_params.get('action', 'buy')
        max_slippage = trade_params.get('max_slippage', 1.0)
        
        if not token_address or amount_usd <= 0:
            logger.error(f"Invalid trade parameters: {trade_params}")
            return {
                'success': False,
                'error': 'Invalid trade parameters',
                'params': trade_params
            }
        
        # Start execution timer
        timer = ExecutionTimer()
        
        try:
            # Calculate optimal route
            route = await self.router.calculate_optimal_route(
                token_address, amount_usd, self.dex_client)
            timer.checkpoint('route_calculation')
            
            if not route['success']:
                return {
                    'success': False,
                    'error': route.get('error', 'Route calculation failed'),
                    'route': route,
                    'execution_time_ms': timer.elapsed()
                }
            
            # Apply MEV protection
            protected_params = await self.mev_protection.protect_transaction(
                token_address, amount_usd)
            timer.checkpoint('mev_protection')
            
            # Adjust parameters based on protection
            if protected_params and 'adjusted_slippage' in protected_params:
                max_slippage = protected_params['adjusted_slippage']
            
            # Prepare transaction(s)
            if route['route_type'] == 'direct':
                # Single transaction
                tx_result = await self._execute_single_trade(
                    token_address, amount_usd, action, max_slippage, route)
            else:
                # Split transaction
                tx_result = await self._execute_split_trade(
                    token_address, action, max_slippage, route)
            
            timer.checkpoint('transaction_execution')
            
            # Update success rate
            self.execution_count += 1
            if tx_result['success']:
                self.success_count += 1
            
            # Add timing information
            tx_result['timing'] = {
                'total_ms': timer.elapsed(),
                'checkpoints': timer.get_checkpoints()
            }
            
            # Update metrics
            self._update_execution_metrics(tx_result)
            
            return tx_result
            
        except Exception as e:
            logger.error(f"Error executing flash trade: {str(e)}")
            error_result = {
                'success': False,
                'error': str(e),
                'execution_time_ms': timer.elapsed(),
                'params': trade_params
            }
            
            # Update metrics even for errors
            self._update_execution_metrics(error_result)
            
            return error_result
    
    async def _execute_single_trade(self, 
                                  token_address: str, 
                                  amount_usd: float,
                                  action: str,
                                  max_slippage: float,
                                  route: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a single trade transaction
        
        Args:
            token_address (str): Token address
            amount_usd (float): Trade amount in USD
            action (str): Trade action ('buy' or 'sell')
            max_slippage (float): Maximum allowed slippage
            route (Dict[str, Any]): Route data
            
        Returns:
            Dict[str, Any]: Transaction result
        """
        # Get optimal gas price
        priority = 'high' if self.priority_execution else 'medium'
        gas_price = await self.gas_predictor.predict_gas_price(priority)
        
        # Create and sign transaction
        tx_params = {
            'token_address': token_address,
            'amount_usd': amount_usd,
            'max_slippage': max_slippage,
            'gas_price': gas_price,
            'priority': priority
        }
        
        # Execute transaction based on action
        for retry in range(self.max_retries):
            try:
                if action == 'buy':
                    result = await self.dex_client.buy_token(
                        token_address, amount_usd, max_slippage)
                else:  # sell
                    result = await self.dex_client.sell_token(
                        token_address, 100.0, max_slippage)  # 100% of holdings
                
                # If successful, return result
                if result and result.get('success'):
                    return {
                        'success': True,
                        'transaction_id': result.get('transaction_id'),
                        'action': action,
                        'token_address': token_address,
                        'amount_usd': amount_usd,
                        'amount_token': result.get('amount_token', 0),
                        'price_impact': result.get('price_impact', 0),
                        'actual_slippage': result.get('slippage', 0),
                        'retry_count': retry
                    }
                
                # If failed but can retry, wait and retry
                if retry < self.max_retries - 1:
                    logger.warning(f"Transaction failed, retrying ({retry+1}/{self.max_retries})")
                    await asyncio.sleep(self.retry_delay)
                    
                    # Update gas price for retry
                    gas_price = await self.gas_predictor.predict_gas_price('high')
                    tx_params['gas_price'] = gas_price
            
            except Exception as e:
                logger.error(f"Error in transaction attempt {retry+1}: {str(e)}")
                if retry < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay)
        
        # If we got here, all retries failed
        return {
            'success': False,
            'error': 'All transaction attempts failed',
            'action': action,
            'token_address': token_address,
            'amount_usd': amount_usd,
            'retry_count': self.max_retries
        }
    
    async def _execute_split_trade(self, 
                                 token_address: str,
                                 action: str,
                                 max_slippage: float,
                                 route: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a split trade across multiple pools
        
        Args:
            token_address (str): Token address
            action (str): Trade action ('buy' or 'sell')
            max_slippage (float): Maximum allowed slippage
            route (Dict[str, Any]): Route data with splits
            
        Returns:
            Dict[str, Any]: Transaction result
        """
        splits = route.get('splits', [])
        if not splits:
            return {
                'success': False,
                'error': 'No splits in route',
                'route': route
            }
        
        # Execute each split as a separate transaction
        results = []
        total_amount_token = 0
        successful_splits = 0
        
        for split in splits:
            split_amount = split.get('amount_usd', 0)
            if split_amount <= 0:
                continue
                
            # Execute the split trade
            result = await self._execute_single_trade(
                token_address, split_amount, action, max_slippage, 
                {'route_type': 'direct', 'pools': [split['pool']], 'success': True})
            
            results.append(result)
            
            if result['success']:
                total_amount_token += result.get('amount_token', 0)
                successful_splits += 1
        
        # Determine overall success
        success = successful_splits > 0
        
        return {
            'success': success,
            'action': action,
            'token_address': token_address,
            'total_amount_usd': route['amount_usd'],
            'executed_amount_usd': sum(r.get('amount_usd', 0) for r in results if r['success']),
            'total_amount_token': total_amount_token,
            'split_count': len(splits),
            'successful_splits': successful_splits,
            'split_results': results,
            'error': None if success else 'Some or all splits failed'
        }
    
    def _update_execution_metrics(self, result: Dict[str, Any]):
        """
        Update execution metrics in state manager
        
        Args:
            result (Dict[str, Any]): Execution result
        """
        # Calculate success rate
        success_rate = (self.success_count / max(1, self.execution_count)) * 100
        
        # Update component metrics
        metrics = {
            'execution_count': self.execution_count,
            'success_count': self.success_count,
            'success_rate': success_rate,
            'last_execution_time_ms': result.get('timing', {}).get('total_ms', 0)
        }
        
        # Add result-specific metrics
        if result.get('success'):
            if 'price_impact' in result:
                metrics['last_price_impact'] = result['price_impact']
            if 'actual_slippage' in result:
                metrics['last_slippage'] = result['actual_slippage']
        
        self.state_manager.update_component_metrics('flash_executor', metrics)