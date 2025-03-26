"""
TradingIntegration Component
Responsible for orchestrating trading operations, integrating data from both PumpPortal
and direct on-chain analysis with Raydium trading functionality, and implementing trading strategies.
"""

import asyncio
import logging
import time
import traceback
from typing import Dict, Any, List, Optional, Set, Tuple
import json

from network.pump_portal_client import PumpPortalClient
from network.onchain_analyzer import OnchainAnalyzer
from trading.raydium_client import RaydiumClient
from trading.risk_manager import RiskManager
from trading.dynamic_profit_manager import DynamicProfitManager
from strategy.strategy_manager import StrategyManager
from core.state_manager import StateManager
from core.config_manager import ConfigManager
from trading.mev_protection import MEVProtection
from trading.gas_predictor import GasPredictor

logger = logging.getLogger(__name__)

class TradingIntegration:
    """
    TradingIntegration orchestrates:
    - Integration between PumpPortal and Raydium
    - Coordinating trading operations
    - Managing trade queue and execution
    - Applying trading strategies
    """
    
    def __init__(self,
                pump_portal_client: PumpPortalClient,
                onchain_analyzer: OnchainAnalyzer,
                raydium_client: RaydiumClient,
                risk_manager: RiskManager,
                strategy_manager: StrategyManager,
                state_manager: StateManager,
                config_manager: ConfigManager):
        """
        Initialize the TradingIntegration
        
        Args:
            pump_portal_client (PumpPortalClient): PumpPortal client instance
            onchain_analyzer (OnchainAnalyzer): On-chain data analyzer instance
            raydium_client (RaydiumClient): Raydium client instance
            risk_manager (RiskManager): Risk manager instance
            strategy_manager (StrategyManager): Strategy manager instance
            state_manager (StateManager): State manager instance
            config_manager (ConfigManager): Configuration manager instance
        """
        # Component integration settings
        self.enable_connection_coordination = config_manager.get('trading_integration.enable_connection_coordination', True)
        self.status_check_interval = config_manager.get('trading_integration.status_check_interval', 30)  # seconds
        self.pump_portal_client = pump_portal_client
        self.onchain_analyzer = onchain_analyzer
        self.raydium_client = raydium_client
        self.risk_manager = risk_manager
        self.strategy_manager = strategy_manager
        self.state_manager = state_manager
        self.config_manager = config_manager
        
        # Initialize Dynamic Profit Manager
        self.dynamic_profit_manager = DynamicProfitManager(
            config_manager,
            state_manager,
            risk_manager
        )
        
        # Initialize MEV Protection and Gas Predictor
        self.mev_protection = MEVProtection(config_manager)
        self.gas_predictor = GasPredictor(config_manager)
        
        # Token data cache for faster lookup and reduced API calls
        self.token_data_cache = {}
        
        # Telegram notifier for sending alerts (shared from monitoring system)
        # Sử dụng get_component_instance để lấy instance thay vì ComponentState
        self.telegram_notifier = self.state_manager.get_component_instance('telegram_notifier')
        
        # Trade management
        self.trade_queue = asyncio.Queue()
        self.active_trades = {}  # trade_id -> trade data
        self.completed_trades = []
        self.trade_history = []  # Full trade history
        
        # Settings
        self.min_liquidity_usd = self.config_manager.get('strategy.min_liquidity_usd', 10000)
        self.min_volume_usd = self.config_manager.get('strategy.min_volume_usd', 5000)
        self.max_active_positions = self.config_manager.get('strategy.max_active_positions', 10)
        
        # Token blacklist based on keywords
        self.blacklisted_keywords = self.config_manager.get('strategy.blacklisted_keywords', 
                                                           ['scam', 'honeypot', 'rug'])
        
        # State
        self.is_running = False
        self.worker_task = None
        self.connection_monitor_task = None
        
        # Connection status tracking
        self.last_pumpportal_status_check = 0
        self.last_pumpportal_status = None
        
        logger.info("TradingIntegration initialized")
    
    async def start(self):
        """Start the TradingIntegration component"""
        if self.is_running:
            logger.warning("TradingIntegration already running")
            return
        
        logger.info("Starting TradingIntegration")
        self.state_manager.update_component_status('trading_integration', 'starting')
        
        # Đăng ký callback với PumpPortalClient trước khi khởi động các tác vụ khác
        self.pump_portal_client.register_new_token_callback(self._handle_new_token)
        self.pump_portal_client.register_liquidity_change_callback(self._handle_liquidity_change)
        self.pump_portal_client.register_price_change_callback(self._handle_price_change)
        
        # Đăng ký callback cho OnchainAnalyzer
        self.onchain_analyzer.register_new_token_callback(self._handle_onchain_new_token)
        self.onchain_analyzer.register_liquidity_change_callback(self._handle_liquidity_change)
        self.onchain_analyzer.register_whale_activity_callback(self._handle_whale_activity)
        
        # Khởi động các tác vụ sau khi đăng ký callback
        self.is_running = True
        self.worker_task = asyncio.create_task(self._trade_worker())
        self.connection_monitor_task = asyncio.create_task(self._connection_status_monitor())
        
        # Thêm tác vụ kiểm tra sức khỏe định kỳ sẽ được triển khai sau
        # self.health_check_task = asyncio.create_task(self._health_check())
        
        # Xác nhận số lượng callback đã đăng ký
        logger.info(f"Registered {len(self.pump_portal_client.new_token_callbacks)} new token callbacks with PumpPortalClient")
        
        self.state_manager.update_component_status('trading_integration', 'running')
        logger.info("TradingIntegration started")
    
    async def stop(self):
        """Stop the TradingIntegration component safely"""
        if not self.is_running:
            logger.warning("TradingIntegration not running")
            return
        
        logger.info("Stopping TradingIntegration")
        self.state_manager.update_component_status('trading_integration', 'stopping')
        
        # Lưu danh sách các giao dịch đang hoạt động nếu có
        if self.active_trades:
            try:
                active_count = len(self.active_trades)
                logger.info(f"Saving state for {active_count} active trades")
                # TODO: Lưu trạng thái của các giao dịch đang hoạt động vào cơ sở dữ liệu
            except Exception as e:
                logger.error(f"Error saving active trades: {str(e)}")
        
        # Đánh dấu là đã dừng để ngăn chặn các task mới
        self.is_running = False
        
        # Hủy worker task an toàn
        if self.worker_task:
            try:
                logger.info("Canceling trade worker task")
                self.worker_task.cancel()
                await asyncio.wait_for(self.worker_task, timeout=5)
            except (asyncio.CancelledError, asyncio.TimeoutError) as e:
                logger.warning(f"Worker task cancel timeout: {str(e)}")
            except Exception as e:
                logger.error(f"Error canceling worker task: {str(e)}")
                
        # Cancel connection monitor task
        if self.connection_monitor_task:
            try:
                logger.info("Canceling connection monitor task")
                self.connection_monitor_task.cancel()
                await asyncio.wait_for(self.connection_monitor_task, timeout=5)
            except (asyncio.CancelledError, asyncio.TimeoutError) as e:
                logger.warning(f"Connection monitor task cancel timeout: {str(e)}")
            except Exception as e:
                logger.error(f"Error canceling connection monitor task: {str(e)}")
        
        # Cancel health check task
        if hasattr(self, 'health_check_task') and self.health_check_task:
            try:
                logger.info("Canceling health check task")
                self.health_check_task.cancel()
                await asyncio.wait_for(self.health_check_task, timeout=5)
            except (asyncio.CancelledError, asyncio.TimeoutError) as e:
                logger.warning(f"Health check task cancel timeout: {str(e)}")
            except Exception as e:
                logger.error(f"Error canceling health check task: {str(e)}")
        
        # Hủy các task khác đang chạy nếu có
        pending_tasks = self._get_pending_tasks()
        if pending_tasks:
            logger.info(f"Canceling {len(pending_tasks)} pending trading tasks")
            for task_name, task in pending_tasks.items():
                try:
                    logger.debug(f"Canceling task: {task_name}")
                    task.cancel()
                except Exception as e:
                    logger.warning(f"Error canceling task {task_name}: {str(e)}")
        
        # Xóa cache
        self.token_data_cache.clear()
        
        # Phát hành thông báo về việc dừng
        try:
            if self.telegram_notifier:
                await self.telegram_notifier.send_message("Dừng hệ thống giao dịch")
            else:
                logger.warning("No telegram_notifier instance available for sending message")
        except Exception as e:
            logger.warning(f"Error sending stop notification: {str(e)}")
        
        self.state_manager.update_component_status('trading_integration', 'stopped')
        logger.info("TradingIntegration stopped")
    
    def _get_pending_tasks(self):
        """Get a dictionary of pending tasks related to trading"""
        pending_tasks = {}
        for task in asyncio.all_tasks():
            task_name = task.get_name()
            # Kiểm tra các task liên quan đến trading
            if ('trade' in task_name or 'token' in task_name) and not task.done():
                pending_tasks[task_name] = task
        return pending_tasks
    
    async def _handle_new_token(self, token_data: Dict[str, Any]):
        """
        Handle new token event from PumpPortal
        
        Args:
            token_data (Dict[str, Any]): Token data
        """
        try:
            token_address = token_data.get('address')
            token_symbol = token_data.get('symbol', '')
            token_name = token_data.get('name', '')
            liquidity_usd = token_data.get('liquidity_usd', 0)
            
            logger.info(f"Trading Integration received new token event: {token_symbol} ({token_address}) - Liquidity: ${liquidity_usd}")
            
            # Validate token data
            if not token_address:
                logger.warning(f"Received invalid token data without address: {token_data}")
                return
                
            # Update metrics
            self.state_manager.update_component_metric(
                'trading_integration', 
                'new_tokens_analyzed', 
                len(self.trade_history)
            )
            
            # Basic filtering
            if await self._should_skip_token(token_data):
                logger.info(f"Skipping token {token_symbol}: did not pass initial filters")
                return
            
            logger.info(f"Token {token_symbol} passed initial filters, proceeding to full analysis")
            
            # Analyze token further and consider for trading
            await self._analyze_token_for_trading(token_data)
            
        except Exception as e:
            logger.error(f"Error processing new token event: {str(e)}")
            logger.debug(f"Token data causing error: {token_data}")
    
    async def _handle_liquidity_change(self, data: Dict[str, Any]):
        """
        Handle liquidity change event from PumpPortal
        
        Args:
            data (Dict[str, Any]): Liquidity change data
        """
        try:
            token_address = data.get('token_address')
            old_liquidity = data.get('old_liquidity_usd', 0)
            new_liquidity = data.get('new_liquidity_usd', 0)
            percentage_change = data.get('percentage_change', 0)
            
            logger.debug(f"Trading Integration received liquidity change for {token_address}: {old_liquidity:.2f} -> {new_liquidity:.2f} USD ({percentage_change:.2f}%)")
            
            # Validate data
            if not token_address:
                logger.warning(f"Received invalid liquidity change data without token address: {data}")
                return
            
            # Check if this token is in our active trades
            if token_address in self.active_trades:
                trade = self.active_trades[token_address]
                token_symbol = trade.get('token_symbol', 'Unknown')
                
                # Large liquidity drop might indicate potential rugpull
                if percentage_change < -20:
                    logger.warning(f"Significant liquidity drop (-{abs(percentage_change):.2f}%) for {token_symbol} ({token_address})")
                    
                    # Consider emergency exit
                    if percentage_change < -50:
                        logger.critical(f"Emergency: Major liquidity drop (-{abs(percentage_change):.2f}%) for {token_symbol} ({token_address})")
                        await self._emergency_exit(token_address, "Major liquidity drop detected")
            else:
                logger.debug(f"Liquidity change for non-active token {token_address}: {old_liquidity:.2f} -> {new_liquidity:.2f} USD")
                
        except Exception as e:
            logger.error(f"Error processing liquidity change event: {str(e)}")
            logger.debug(f"Data causing error: {data}")
    
    async def _handle_onchain_new_token(self, token_data: Dict[str, Any]):
        """
        Handle new token event from OnchainAnalyzer
        
        Args:
            token_data (Dict[str, Any]): Token data from on-chain analysis
        """
        try:
            token_address = token_data.get('address')
            liquidity_usd = token_data.get('liquidity_usd', 0)
            
            # For tokens discovered on-chain, we might not have a symbol yet
            # so we'll use a shortened address version as a placeholder
            token_symbol = token_data.get('symbol', f"UNK:{token_address[:6]}")
            
            logger.info(f"Trading Integration received on-chain new token event: {token_symbol} ({token_address}) - Liquidity: ${liquidity_usd}")
            
            # Validate token data
            if not token_address:
                logger.warning(f"Received invalid on-chain token data without address: {token_data}")
                return
            
            # Add source information
            token_data['source'] = 'on-chain'
            token_data['discovery_time'] = time.time()
            
            # Update metrics
            self.state_manager.update_component_metric(
                'trading_integration',
                'onchain_tokens_discovered',
                1,
                increment=True
            )
            
            # Basic filtering - use the same filter function as PumpPortal events
            if await self._should_skip_token(token_data):
                logger.info(f"Skipping on-chain token {token_symbol}: did not pass initial filters")
                return
            
            logger.info(f"On-chain token {token_symbol} passed initial filters, proceeding to full analysis")
            
            # For on-chain discovered tokens, we need to fetch some additional data
            # since we don't have all the metadata that PumpPortal provides
            enriched_token_data = await self._enrich_onchain_token_data(token_data)
            
            # Analyze token further and consider for trading
            if enriched_token_data:
                await self._analyze_token_for_trading(enriched_token_data)
            
        except Exception as e:
            logger.error(f"Error processing on-chain new token event: {str(e)}")
            logger.debug(f"Token data causing error: {token_data}")
    
    async def _handle_whale_activity(self, data: Dict[str, Any]):
        """
        Handle whale activity event from OnchainAnalyzer
        
        Args:
            data (Dict[str, Any]): Whale activity data
        """
        try:
            wallet = data.get('wallet')
            amount_sol = data.get('amount_sol', 0)
            direction = data.get('direction', 'unknown')
            
            if not wallet:
                logger.warning(f"Received invalid whale activity data without wallet address: {data}")
                return
            
            logger.info(f"Detected whale activity: {wallet} {direction} {amount_sol} SOL")
            
            # Check if this is a wallet we're monitoring for specific tokens
            if wallet in self.active_trades.values():
                logger.info(f"Whale activity detected on a wallet we're tracking for active trades")
                
                # Check all active trades to see if this whale is involved
                for token_address, trade in self.active_trades.items():
                    if trade.get('whale_wallets') and wallet in trade.get('whale_wallets', []):
                        token_symbol = trade.get('token_symbol', 'Unknown')
                        
                        # If a whale is selling a large amount, consider this a warning signal
                        if direction == 'sent' and amount_sol > 50:
                            logger.warning(f"Whale selling detected for {token_symbol}: {wallet} sold {amount_sol} SOL")
                            
                            # Send notification
                            if self.telegram_notifier:
                                await self.telegram_notifier.send_message(
                                    f"⚠️ Whale selling detected for {token_symbol}!\n"
                                    f"Wallet: {wallet}\n"
                                    f"Amount: {amount_sol} SOL",
                                    level="WARNING"
                                )
                            
                            # Update risk assessment for this token
                            trade['risk_level'] = trade.get('risk_level', 'medium')
                            if amount_sol > 100:
                                trade['risk_level'] = 'high'
                                
                                # Consider partial exit based on risk tolerance
                                if self.config_manager.get('strategy.exit_on_whale_selling', True):
                                    await self._partial_exit(token_address, "Whale selling detected", percentage=50)
            
            # Check if this wallet has positions in tokens we're interested in
            if amount_sol > 30 and direction == 'sent':
                # This may indicate a whale is selling, investigate which tokens
                holdings = await self.onchain_analyzer.analyze_wallet_holdings(wallet)
                
                for holding in holdings:
                    token_mint = holding.get('token_mint')
                    if token_mint in self.active_trades:
                        logger.warning(f"Whale {wallet} with {token_mint} in holdings is moving funds")
                        
                        # Add to monitored wallets for this token
                        trade = self.active_trades[token_mint]
                        if 'whale_wallets' not in trade:
                            trade['whale_wallets'] = []
                        
                        if wallet not in trade['whale_wallets']:
                            trade['whale_wallets'].append(wallet)
                            logger.info(f"Added {wallet} to monitored whale wallets for {token_mint}")
        
        except Exception as e:
            logger.error(f"Error processing whale activity event: {str(e)}")
            logger.debug(f"Data causing error: {data}")
    
    async def _get_token_data_from_multiple_sources(self, token_address: str) -> Dict[str, Any]:
        """
        Fetch token data from multiple sources and reconcile data for improved reliability
        
        Args:
            token_address (str): Token address
        
        Returns:
            Dict[str, Any]: Combined and validated token data
        """
        results = {}
        
        try:
            # Source 1: OnchainAnalyzer
            onchain_data = await self.onchain_analyzer.get_token_info(token_address)
            if onchain_data:
                results['onchain'] = onchain_data
                
            # Source 2: RaydiumClient
            raydium_data = await self.raydium_client.get_token_info(token_address)
            if raydium_data:
                results['raydium'] = raydium_data
            
            # Additional sources can be added here
            
            # Get confidence scores for each source
            onchain_confidence = self.onchain_analyzer.data_confidence_scores.get('onchain', 100)
            pumpportal_confidence = self.onchain_analyzer.data_confidence_scores.get('pump_portal', 80)
            
            # No data from any source
            if not results:
                return {}
            
            # Combine and reconcile data from different sources
            combined_data = {}
            
            # Start with symbol and name (prioritize Raydium data)
            for field in ['symbol', 'name', 'address']:
                if 'raydium' in results and field in results['raydium']:
                    combined_data[field] = results['raydium'][field]
                elif 'onchain' in results and field in results['onchain']:
                    combined_data[field] = results['onchain'][field]
                else:
                    # Default values for critical fields
                    if field == 'address':
                        combined_data[field] = token_address
                    elif field == 'symbol':
                        combined_data[field] = f"UNK:{token_address[:6]}"
                    elif field == 'name':
                        combined_data[field] = f"Unknown Token {token_address[:8]}"
            
            # Price data - compare across sources and use weighted average if they differ
            price_values = []
            price_weights = []
            
            if 'raydium' in results and 'price_usd' in results['raydium'] and results['raydium']['price_usd'] > 0:
                price_values.append(results['raydium']['price_usd'])
                # Use dynamic confidence score
                price_weights.append(100)  # Raydium data is direct from blockchain
            
            if 'onchain' in results and 'price_usd' in results['onchain'] and results['onchain']['price_usd'] > 0:
                price_values.append(results['onchain']['price_usd'])
                price_weights.append(onchain_confidence)
            
            if price_values:
                # Weighted average if we have multiple price sources
                if len(price_values) > 1:
                    combined_data['price_usd'] = sum(p * w for p, w in zip(price_values, price_weights)) / sum(price_weights)
                    # Flag if price discrepancy is significant (more than 5%)
                    max_price = max(price_values)
                    min_price = min(price_values)
                    if max_price > 0 and (max_price - min_price) / max_price > 0.05:
                        combined_data['price_discrepancy'] = True
                else:
                    combined_data['price_usd'] = price_values[0]
            
            # Similar approach for liquidity data
            liquidity_values = []
            liquidity_weights = []
            
            if 'raydium' in results and 'liquidity_usd' in results['raydium'] and results['raydium']['liquidity_usd'] > 0:
                liquidity_values.append(results['raydium']['liquidity_usd'])
                liquidity_weights.append(100)
            
            if 'onchain' in results and 'liquidity_usd' in results['onchain'] and results['onchain']['liquidity_usd'] > 0:
                liquidity_values.append(results['onchain']['liquidity_usd'])
                liquidity_weights.append(onchain_confidence)
            
            if liquidity_values:
                if len(liquidity_values) > 1:
                    combined_data['liquidity_usd'] = sum(l * w for l, w in zip(liquidity_values, liquidity_weights)) / sum(liquidity_weights)
                else:
                    combined_data['liquidity_usd'] = liquidity_values[0]
            
            # Data source metadata
            combined_data['data_sources'] = list(results.keys())
            combined_data['validation_level'] = len(results)
            combined_data['last_validated'] = time.time()
            
            return combined_data
            
        except Exception as e:
            logger.error(f"Error getting token data from multiple sources: {str(e)}")
            return {}

    async def _enrich_onchain_token_data(self, token_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Enrich token data discovered on-chain with additional information
        
        Args:
            token_data (Dict[str, Any]): Basic token data from on-chain discovery
            
        Returns:
            Optional[Dict[str, Any]]: Enriched token data or None if validation fails
        """
        try:
            token_address = token_data.get('address')
            
            if not token_address:
                return None
            
            # Use our new multi-source validation function
            validated_data = await self._get_token_data_from_multiple_sources(token_address)
            
            if validated_data:
                # Merge the validated data with our original data
                # but preserve the source information
                source = token_data.get('source')
                discovery_time = token_data.get('discovery_time')
                
                # Update with validated data
                token_data.update(validated_data)
                
                # Preserve original metadata
                if source:
                    token_data['source'] = source
                if discovery_time:
                    token_data['discovery_time'] = discovery_time
            else:
                # If multi-source validation failed, fall back to direct method
                # Try to get more information from Raydium
                raydium_data = await self.raydium_client.get_token_info(token_address)
                
                if raydium_data:
                    # We got more detailed information from Raydium
                    token_data.update(raydium_data)
                
                # For very new tokens, we might need to set some default values
                # if they're missing but required for analysis
                if 'symbol' not in token_data or not token_data['symbol']:
                    token_data['symbol'] = f"UNK:{token_address[:6]}"
                    
                if 'name' not in token_data or not token_data['name']:
                    token_data['name'] = f"Unknown Token {token_address[:8]}"
                
                # Important metrics for analysis
                if 'liquidity_usd' not in token_data or token_data['liquidity_usd'] <= 0:
                    liquidity = await self.raydium_client.get_token_liquidity(token_address)
                    token_data['liquidity_usd'] = liquidity if liquidity else 0
                
                if 'price_usd' not in token_data or token_data['price_usd'] <= 0:
                    price = await self.raydium_client.get_token_price(token_address)
                    token_data['price_usd'] = price if price else 0
            
            return token_data
            
        except Exception as e:
            logger.error(f"Error enriching on-chain token data: {str(e)}")
            return None
    
    async def _partial_exit(self, token_address: str, reason: str, percentage: float = 50.0):
        """
        Perform a partial exit from a position
        
        Args:
            token_address (str): Token address
            reason (str): Reason for partial exit
            percentage (float): Percentage of position to exit
        """
        try:
            if token_address not in self.active_trades:
                return
                
            trade = self.active_trades[token_address]
            token_symbol = trade.get('token_symbol', 'Unknown')
            
            logger.warning(f"Initiating {percentage}% partial exit for {token_symbol}: {reason}")
            
            # Send a partial sell order to the DEX
            sell_result = await self.raydium_client.sell_token(
                token_address,
                percent_of_holdings=percentage,
                max_slippage=2.0
            )
            
            if sell_result and sell_result.get('success'):
                logger.info(f"Partial exit successful for {token_symbol}: {sell_result}")
                
                # Update trade data
                trade['partial_exits'] = trade.get('partial_exits', []) + [{
                    'timestamp': time.time(),
                    'percentage': percentage,
                    'reason': reason,
                    'tx_signature': sell_result.get('signature'),
                    'sell_price': sell_result.get('price'),
                    'amount_sold': sell_result.get('amount')
                }]
                
                # Notify about partial exit
                if self.telegram_notifier:
                    await self.telegram_notifier.send_message(
                        f"🔄 Partial exit ({percentage}%) for {token_symbol}\n"
                        f"Reason: {reason}\n"
                        f"Sell price: ${sell_result.get('price', 0):.6f}",
                        level="INFO"
                    )
            else:
                logger.error(f"Failed to execute partial exit for {token_symbol}: {sell_result}")
                
        except Exception as e:
            logger.error(f"Error in partial exit: {str(e)}")
    
    async def _handle_price_change(self, data: Dict[str, Any]):
        """
        Handle price change event from PumpPortal
        
        Args:
            data (Dict[str, Any]): Price change data
        """
        try:
            token_address = data.get('token_address')
            old_price = data.get('old_price_usd', 0)
            new_price = data.get('new_price_usd', 0)
            percentage_change = data.get('percentage_change', 0)
            
            # Validate data
            if not token_address:
                logger.warning(f"Received invalid price change data without token address: {data}")
                return
                
            if new_price <= 0:
                logger.warning(f"Received invalid price (zero or negative) for {token_address}: {new_price}")
                return
                
            logger.debug(f"Trading Integration received price change for {token_address}: {old_price:.6f} -> {new_price:.6f} USD ({percentage_change:.2f}%)")
            
            # Check if this token is in our active trades
            if token_address in self.active_trades:
                trade = self.active_trades[token_address]
                token_symbol = trade.get('token_symbol', 'Unknown')
                entry_price = trade.get('entry_price', 0)
                
                # Calculate profit/loss
                if entry_price > 0:
                    pnl_percentage = ((new_price - entry_price) / entry_price) * 100
                    
                    # Update the trade data
                    trade['current_price'] = new_price
                    trade['current_pnl_percentage'] = pnl_percentage
                    trade['last_updated'] = time.time()
                    
                    logger.info(f"Position update for {token_symbol} ({token_address}): PnL: {pnl_percentage:.2f}%")
                    
                    # Update metrics in state manager
                    self.state_manager.update_component_metric(
                        'trading_integration',
                        f'trade_{token_address}_pnl',
                        pnl_percentage
                    )
                    
                    # Check significant price movements
                    if abs(percentage_change) > 10:
                        logger.warning(f"Significant price movement for {token_symbol}: {percentage_change:.2f}%")
                    
                    # Check stop loss / take profit conditions
                    await self._check_exit_conditions(token_address, trade)
                else:
                    logger.warning(f"Trade for {token_symbol} has invalid entry price: {entry_price}")
            else:
                # Token not in active trades, but still log for analytics
                logger.debug(f"Price change for non-active token {token_address}: {percentage_change:.2f}%")
        
        except Exception as e:
            logger.error(f"Error processing price change event: {str(e)}")
            logger.debug(f"Data causing error: {data}")
    
    async def _should_skip_token(self, token_data: Dict[str, Any]) -> bool:
        """
        Determine if a token should be skipped based on initial filters
        
        Args:
            token_data (Dict[str, Any]): Token data
            
        Returns:
            bool: True if token should be skipped, False otherwise
        """
        token_address = token_data.get('address')
        token_symbol = token_data.get('symbol', '')
        token_name = token_data.get('name', '')
        liquidity_usd = token_data.get('liquidity_usd', 0)
        
        # Check if already processed
        if token_address in [t.get('token_address') for t in self.trade_history]:
            return True
        
        # Check minimum liquidity
        if liquidity_usd < self.min_liquidity_usd:
            logger.debug(f"Skipping {token_symbol}: liquidity ${liquidity_usd} below minimum ${self.min_liquidity_usd}")
            return True
        
        # Check blacklisted keywords in name or symbol
        token_text = f"{token_name} {token_symbol}".lower()
        for keyword in self.blacklisted_keywords:
            if keyword.lower() in token_text:
                logger.debug(f"Skipping {token_symbol}: contains blacklisted keyword '{keyword}'")
                return True
        
        return False
    
    async def _analyze_token_for_trading(self, token_data: Dict[str, Any]):
        """
        Analyze a token for trading potential
        
        Args:
            token_data (Dict[str, Any]): Token data
        """
        token_address = token_data.get('address')
        token_symbol = token_data.get('symbol', '')
        
        logger.info(f"Analyzing {token_symbol} ({token_address}) for trading")
        
        try:
            # Skip processing if token_address is None
            if token_address is None:
                logger.warning(f"Skipping token analysis: token address is None for {token_symbol}")
                return
            
            # Get validated data from multiple sources
            validated_data = await self._get_token_data_from_multiple_sources(token_address)
            
            if not validated_data:
                logger.warning(f"Could not get validated data for {token_symbol}")
                return
            
            # Update token data with validated information
            token_data.update(validated_data)
            
            # Get validated price from our multi-source data
            price_usd = token_data.get('price_usd')
            
            # Log any data discrepancies detected during validation
            if token_data.get('price_discrepancy'):
                logger.warning(f"Price discrepancy detected for {token_symbol} - data sources disagree")
            
            if not price_usd:
                logger.warning(f"Could not get price for {token_symbol}")
                return
            
            # Calculate volatility (ATR)
            atr = await self.raydium_client.calculate_atr(token_address)
            
            # Get market data
            market_data = await self.pump_portal_client.get_market_data(token_address)
            
            # Combine data for strategy evaluation
            combined_data = {
                **token_data,
                'price_usd': price_usd,
                'atr': atr,
                'market_data': market_data,
                'data_sources': token_data.get('data_sources', []),
                'validation_level': token_data.get('validation_level', 0)
            }
            
            # Check risk profile
            risk_assessment = await self.risk_manager.assess_token_risk(combined_data)
            
            if not risk_assessment['is_acceptable']:
                logger.info(f"Skipping {token_symbol}: failed risk assessment - {risk_assessment['reason']}")
                return
            
            # Apply strategy to determine if we should trade
            strategy_result = await self.strategy_manager.evaluate_token(combined_data)
            
            if not strategy_result['should_trade']:
                logger.info(f"Strategy decided not to trade {token_symbol}: {strategy_result['reason']}")
                return
            
            # Determine position size
            position_size = await self.risk_manager.calculate_position_size(combined_data)
            
            if position_size <= 0:
                logger.info(f"Position size for {token_symbol} is zero - not trading")
                return
            
            # Create trade and add to queue
            trade = {
                'token_address': token_address,
                'token_symbol': token_symbol,
                'token_name': token_data.get('name', ''),
                'amount_usd': position_size,
                'strategy': strategy_result['strategy_name'],
                'signals': strategy_result['signals'],
                'creation_time': time.time(),
                'status': 'queued',
                'risk_assessment': risk_assessment
            }
            
            logger.info(f"Adding trade to queue: {token_symbol} for ${position_size}")
            await self.trade_queue.put(trade)
            
        except Exception as e:
            logger.error(f"Error analyzing token {token_symbol}: {str(e)}")
    
    async def _connection_status_monitor(self):
        """
        Monitor connection status of external services and adapt system behavior
        - Increases on-chain analysis frequency when PumpPortal is unstable
        - Adjusts confidence scoring for data sources based on connection stability
        """
        logger.info("Connection status monitor started")
        
        while self.is_running:
            try:
                # Only check status periodically
                current_time = time.time()
                if current_time - self.last_pumpportal_status_check < self.status_check_interval:
                    await asyncio.sleep(1)
                    continue
                
                self.last_pumpportal_status_check = current_time
                
                # Check PumpPortal connection status from state manager
                component_state = self.state_manager.get_component_state('pump_portal_client')
                if not component_state:
                    logger.warning("Could not get PumpPortal component state")
                    await asyncio.sleep(10)
                    continue
                
                pumpportal_status = component_state.get('status')
                is_connected = pumpportal_status == 'running'
                
                # Track status change
                status_changed = self.last_pumpportal_status != pumpportal_status
                if status_changed:
                    if is_connected:
                        logger.info("PumpPortal connection restored")
                    else:
                        logger.warning(f"PumpPortal connection status changed to {pumpportal_status}")
                    
                    self.last_pumpportal_status = pumpportal_status
                
                # Get disconnection metrics to adjust confidence
                pumpportal_metrics = self.state_manager.get_component_metrics('pump_portal_client')
                disconnection_count = pumpportal_metrics.get('disconnection_count', 0) if pumpportal_metrics else 0
                
                # Update OnchainAnalyzer scan mode based on PumpPortal status
                if self.enable_connection_coordination:
                    await self.onchain_analyzer.update_scan_mode_for_pump_portal_status(
                        is_connected=is_connected,
                        disconnection_count=disconnection_count
                    )
                
                # If PumpPortal is disconnected, log periodic warnings
                if not is_connected:
                    logger.warning("PumpPortal connection is down, relying on direct on-chain analysis")
                    
                    # Notify about disconnection via Telegram if available
                    if self.telegram_notifier and status_changed:
                        await self.telegram_notifier.send_message(
                            "⚠️ PumpPortal connection is down. System is now relying on direct on-chain analysis.",
                            level="WARNING"
                        )
                
                # Check for reconnection
                elif status_changed and is_connected:
                    logger.info("PumpPortal connection has been restored")
                    
                    # Notify about reconnection via Telegram if available
                    if self.telegram_notifier:
                        await self.telegram_notifier.send_message(
                            "✅ PumpPortal connection has been restored.",
                            level="INFO"
                        )
                
                # Wait before next check
                await asyncio.sleep(10)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in connection monitor: {str(e)}")
                await asyncio.sleep(30)  # Longer delay on error
    
    async def _trade_worker(self):
        """Worker task to process the trade queue"""
        logger.info("Trade worker started")
        
        while self.is_running:
            try:
                # Check if we can open more positions
                if len(self.active_trades) >= self.max_active_positions:
                    # Wait and check again later
                    await asyncio.sleep(5)
                    continue
                
                # Get next trade from queue with timeout
                try:
                    trade = await asyncio.wait_for(self.trade_queue.get(), timeout=5)
                except asyncio.TimeoutError:
                    continue
                
                logger.info(f"Processing trade for {trade['token_symbol']}")
                
                # Update trade status
                trade['status'] = 'processing'
                
                # Final checks before execution
                token_address = trade['token_address']
                
                # Recheck token data with multi-source validation before execution
                validated_data = await self._get_token_data_from_multiple_sources(token_address)
                
                if not validated_data or 'price_usd' not in validated_data or not validated_data['price_usd']:
                    logger.warning(f"Could not get validated price data for {trade['token_symbol']} - cancelling trade")
                    trade['status'] = 'cancelled'
                    trade['end_time'] = time.time()
                    trade['reason'] = 'Could not get validated price data'
                    self.trade_history.append(trade)
                    self.trade_queue.task_done()
                    continue
                
                # Update with most recent price
                current_price = validated_data['price_usd']
                
                # Log data validation level
                validation_level = validated_data.get('validation_level', 0)
                data_sources = validated_data.get('data_sources', [])
                logger.info(f"Price data for {trade['token_symbol']} validated across {validation_level} sources: {', '.join(data_sources)}")
                
                # Price discrepancy warning
                if validated_data.get('price_discrepancy'):
                    logger.warning(f"Price discrepancy detected for {trade['token_symbol']} between data sources")
                    # Record this as a risk factor
                    trade['price_discrepancy_detected'] = True
                
                # Get optimal gas price
                gas_price = await self.gas_predictor.predict_gas_price()
                logger.debug(f"Using gas price: {gas_price}")
                
                # Check for MEV protection
                mev_protection = await self.mev_protection.protect_transaction(token_address, trade['amount_usd'])
                
                # Execute the trade
                try:
                    # Use the updated amount with MEV protection
                    buy_result = await self.raydium_client.buy_token(
                        token_address,
                        trade['amount_usd'],
                        max_slippage=self.config_manager.get('risk.max_slippage_percentage', 2.0)
                    )
                    
                    if buy_result['success']:
                        logger.info(f"Successfully bought {trade['token_symbol']} for ${trade['amount_usd']}")
                        
                        # Update trade data
                        trade['status'] = 'active'
                        trade['entry_time'] = time.time()
                        trade['entry_price'] = buy_result['price_usd']
                        trade['token_amount'] = buy_result['token_amount']
                        trade['transaction_hash'] = buy_result['transaction_hash']
                        trade['current_price'] = buy_result['price_usd']
                        trade['current_pnl_percentage'] = 0
                        
                        # Set stop loss and take profit targets
                        stop_loss_pct = self.config_manager.get('risk.stop_loss_percentage', 5)
                        take_profit_pct = self.config_manager.get('risk.take_profit_percentage', 10)
                        
                        trade['stop_loss_price'] = buy_result['price_usd'] * (1 - (stop_loss_pct / 100))
                        trade['take_profit_price'] = buy_result['price_usd'] * (1 + (take_profit_pct / 100))
                        
                        # Add to active trades
                        self.active_trades[token_address] = trade
                        
                        # Update metrics
                        self.state_manager.update_component_metric(
                            'trading_integration', 
                            'active_trades_count', 
                            len(self.active_trades)
                        )
                    else:
                        logger.warning(f"Failed to buy {trade['token_symbol']}: {buy_result.get('error', 'Unknown error')}")
                        trade['status'] = 'failed'
                        trade['end_time'] = time.time()
                        trade['reason'] = buy_result.get('error', 'Unknown error')
                        self.trade_history.append(trade)
                
                except Exception as e:
                    logger.error(f"Error executing trade for {trade['token_symbol']}: {str(e)}")
                    trade['status'] = 'error'
                    trade['end_time'] = time.time()
                    trade['reason'] = str(e)
                    self.trade_history.append(trade)
                
                # Mark task as done
                self.trade_queue.task_done()
            
            except asyncio.CancelledError:
                break
            
            except Exception as e:
                logger.error(f"Error in trade worker: {str(e)}")
                await asyncio.sleep(5)  # Wait before retrying
        
        logger.info("Trade worker stopped")
    
    async def _check_exit_conditions(self, token_address: str, trade: Dict[str, Any]):
        """
        Check if exit conditions are met for a trade
        
        Args:
            token_address (str): Token address
            trade (Dict[str, Any]): Trade data
        """
        current_price = trade['current_price']
        entry_price = trade['entry_price']
        pnl_percentage = trade['current_pnl_percentage']
        
        # Use dynamic profit manager for exit signals
        exit_signal = await self.dynamic_profit_manager.update_price(token_address, current_price)
        
        # If we received an exit signal, process it
        if exit_signal and exit_signal.get('should_exit', False):
            exit_type = exit_signal.get('exit_type', 'unknown')
            exit_percentage = exit_signal.get('exit_percentage', 100)
            reason = exit_signal.get('reason', 'Dynamic exit triggered')
            
            logger.info(f"{exit_type.title()} triggered for {trade['token_symbol']}: {pnl_percentage:.2f}% - {reason}")
            await self._exit_position(token_address, reason, exit_percentage)
            
            # For tiered exit, we don't exit the full position
            if exit_percentage < 100 and token_address in self.active_trades:
                # Re-evaluate and adjust remaining position
                await self.dynamic_profit_manager.adjust_profit_levels(token_address)
                logger.info(f"Adjusted profit levels for remaining {100-exit_percentage}% of {trade['token_symbol']} position")
        else:
            # Periodically check if market conditions have changed and adjust profit targets
            last_adjustment = trade.get('last_profit_adjustment', 0)
            if time.time() - last_adjustment > 3600:  # Adjust every hour
                await self.dynamic_profit_manager.adjust_profit_levels(token_address)
                if token_address in self.active_trades:
                    self.active_trades[token_address]['last_profit_adjustment'] = time.time()
                    logger.debug(f"Periodically adjusted profit levels for {trade['token_symbol']}")
    
    async def _exit_position(self, token_address: str, reason: str, percent_of_position: float):
        """
        Exit a trading position
        
        Args:
            token_address (str): Token address
            reason (str): Reason for exiting
            percent_of_position (float): Percentage of position to exit (0-100)
        """
        if token_address not in self.active_trades:
            logger.warning(f"Attempted to exit non-existent position: {token_address}")
            return
        
        trade = self.active_trades[token_address]
        logger.info(f"Exiting {percent_of_position}% of position for {trade['token_symbol']}: {reason}")
        
        try:
            # Sell the token
            sell_result = await self.raydium_client.sell_token(
                token_address,
                percent_of_holdings=percent_of_position,
                max_slippage=self.config_manager.get('risk.max_slippage_percentage', 2.0)
            )
            
            if sell_result['success']:
                logger.info(f"Successfully sold {percent_of_position}% of {trade['token_symbol']}")
                
                # Update trade data
                if percent_of_position >= 100:
                    # Complete exit
                    trade['status'] = 'completed'
                    trade['exit_time'] = time.time()
                    trade['exit_price'] = sell_result['price_usd']
                    trade['exit_reason'] = reason
                    trade['final_pnl_percentage'] = trade['current_pnl_percentage']
                    trade['transaction_hash_exit'] = sell_result['transaction_hash']
                    
                    # Move from active to completed
                    del self.active_trades[token_address]
                    self.completed_trades.append(trade)
                    self.trade_history.append(trade)
                    
                    # Update metrics
                    self.state_manager.update_component_metric(
                        'trading_integration', 
                        'active_trades_count', 
                        len(self.active_trades)
                    )
                    self.state_manager.update_component_metric(
                        'trading_integration', 
                        'completed_trades_count', 
                        len(self.completed_trades)
                    )
                else:
                    # Partial exit
                    # Update remaining position size
                    remaining_percent = 100 - percent_of_position
                    trade['amount_usd'] = trade['amount_usd'] * (remaining_percent / 100)
                    trade['token_amount'] = trade['token_amount'] * (remaining_percent / 100)
                    
                    # Add partial exit to history
                    partial_exit = {
                        'token_address': token_address,
                        'token_symbol': trade['token_symbol'],
                        'partial_exit': True,
                        'percent_exited': percent_of_position,
                        'exit_time': time.time(),
                        'exit_price': sell_result['price_usd'],
                        'exit_reason': reason,
                        'pnl_percentage': trade['current_pnl_percentage'],
                        'transaction_hash': sell_result['transaction_hash']
                    }
                    
                    self.trade_history.append(partial_exit)
            else:
                logger.warning(f"Failed to sell {trade['token_symbol']}: {sell_result.get('error', 'Unknown error')}")
        
        except Exception as e:
            logger.error(f"Error exiting position for {trade['token_symbol']}: {str(e)}")
    
    async def _emergency_exit(self, token_address: str, reason: str):
        """
        Emergency exit from a position (sell at market price)
        
        Args:
            token_address (str): Token address
            reason (str): Reason for emergency exit
        """
        trade = None
        token_symbol = "Unknown"
        
        try:
            # Validate token address
            if not token_address or not isinstance(token_address, str):
                logger.critical("Emergency exit called with invalid token address")
                return False
                
            # Log emergency exit
            logger.critical(f"EMERGENCY EXIT triggered for token {token_address}: {reason}")
            
            # Phát cảnh báo ngay lập tức
            try:
                if self.telegram_notifier:
                    await self.telegram_notifier.send_message(f"🚨 EMERGENCY EXIT: {token_address} - {reason}")
                else:
                    logger.warning("No telegram_notifier instance available for sending message")
            except Exception as notify_error:
                logger.warning(f"Error sending emergency notification: {str(notify_error)}")
            
            # Tạo cảnh báo trong hệ thống giám sát
            try:
                self.state_manager.create_alert(
                    'trading_integration', 
                    'CRITICAL', 
                    f"Emergency exit for {token_address}: {reason}"
                )
            except Exception as alert_error:
                logger.warning(f"Error creating alert: {str(alert_error)}")
            
            # Kiểm tra xem token có trong danh sách giao dịch đang hoạt động không
            if token_address not in self.active_trades:
                logger.warning(f"Emergency exit called for non-active trade: {token_address}")
                return False
                
            # Lấy dữ liệu giao dịch
            trade = self.active_trades[token_address]
            if not trade:
                logger.warning(f"Retrieved empty trade data for {token_address}")
                return False
                
            token_symbol = trade.get('token_symbol', 'Unknown')
            
            logger.critical(f"Executing emergency exit for {token_symbol} ({token_address})")
            
            # Sử dụng slippage cao hơn cho thoát khẩn cấp
            emergency_slippage = self.config_manager.get('risk.emergency_slippage_percentage', 5.0)
            normal_slippage = self.config_manager.get('risk.max_slippage_percentage', 2.0)
            
            # Nếu emergency_slippage thấp hơn normal slippage, tăng nó lên
            if emergency_slippage <= normal_slippage:
                emergency_slippage = normal_slippage * 2.5
                
            logger.info(f"Using increased slippage tolerance for emergency exit: {emergency_slippage}%")
            
            # Thử nhiều lần với slippage tăng dần nếu cần
            max_attempts = 3
            for attempt in range(1, max_attempts + 1):
                try:
                    current_slippage = emergency_slippage * attempt
                    logger.info(f"Emergency exit attempt {attempt} with slippage {current_slippage}%")
                    
                    # Thực hiện bán
                    sell_result = None
                    try:
                        sell_result = await self.raydium_client.sell_token(
                            token_address,
                            percent_of_holdings=100,
                            max_slippage=current_slippage
                        )
                    except Exception as sell_error:
                        logger.error(f"Error calling sell_token in attempt {attempt}: {str(sell_error)}")
                        # Đợi ngắn rồi thử lại
                        await asyncio.sleep(2)
                        continue
                    
                    if not sell_result:
                        logger.warning(f"Received empty sell result in attempt {attempt}")
                        await asyncio.sleep(2)
                        continue
                    
                    # Kiểm tra kết quả bán
                    if sell_result.get('success', False):
                        logger.info(f"Emergency exit successful for {token_symbol} on attempt {attempt}")
                        
                        # Cập nhật dữ liệu giao dịch cho báo cáo
                        trade['status'] = 'emergency_exit'
                        trade['exit_time'] = time.time()
                        trade['exit_price'] = sell_result.get('price_usd', 0)
                        trade['exit_reason'] = f"EMERGENCY: {reason}"
                        trade['final_pnl_percentage'] = trade.get('current_pnl_percentage', 0)
                        trade['transaction_hash_exit'] = sell_result.get('transaction_hash', '')
                        
                        try:
                            # Chuyển từ giao dịch hoạt động sang giao dịch hoàn thành
                            # Sao chép trước khi xóa để tránh lỗi tham chiếu
                            completed_trade = trade.copy()
                            if token_address in self.active_trades:
                                del self.active_trades[token_address]
                            self.completed_trades.append(completed_trade)
                            self.trade_history.append(completed_trade)
                        except Exception as move_error:
                            logger.error(f"Error moving trade to completed list: {str(move_error)}")
                        
                        try:
                            # Cập nhật metrics
                            self.state_manager.update_component_metric(
                                'trading_integration', 
                                'active_trades_count', 
                                len(self.active_trades)
                            )
                            # Tăng bộ đếm thoát khẩn cấp
                            metrics = self.state_manager.get_component_metrics('trading_integration') or {}
                            current_count = metrics.get('emergency_exits_count', 0)
                            self.state_manager.update_component_metric(
                                'trading_integration', 
                                'emergency_exits_count', 
                                current_count + 1
                            )
                        except Exception as metric_error:
                            logger.error(f"Error updating metrics: {str(metric_error)}")
                        
                        # Gửi thông báo
                        try:
                            result_pnl = trade.get('final_pnl_percentage', 0)
                            logger.info(f"Emergency exit complete for {token_symbol} - Result: {result_pnl:.2f}% PnL")
                            if self.telegram_notifier:
                                await self.telegram_notifier.send_message(
                                    f"✅ Emergency exit completed for {token_symbol}\n"
                                    f"Result: {result_pnl:.2f}% PnL\n"
                                    f"Reason: {reason}",
                                    level="TRADE"
                                )
                            else:
                                logger.warning("No telegram_notifier instance available for sending message")
                        except Exception as notify_error:
                            logger.warning(f"Error sending completion notification: {str(notify_error)}")
                            
                        return True
                    else:
                        error = sell_result.get('error', 'Unknown error')
                        logger.warning(f"Emergency exit attempt {attempt} failed for {token_symbol}: {error}")
                        
                        # Nếu đây là lần thử cuối cùng, ghi lại thất bại
                        if attempt == max_attempts:
                            logger.critical(f"All emergency exit attempts failed for {token_symbol}")
                            
                            # Ghi lại lần thoát thất bại
                            trade['last_emergency_exit_attempt'] = time.time()
                            trade['emergency_exit_attempts'] = trade.get('emergency_exit_attempts', 0) + 1
                        
                        # Đợi một chút trước khi thử lại
                        await asyncio.sleep(attempt * 2)  # Tăng thời gian đợi theo số lần thử
                        
                except Exception as attempt_error:
                    logger.error(f"Error during emergency exit attempt {attempt} for {token_symbol}: {str(attempt_error)}")
                    # Đợi trước khi thử lại
                    await asyncio.sleep(attempt * 2)
            
            # Nếu tất cả các nỗ lực thất bại, thử cách khác
            logger.critical(f"All emergency exit attempts failed for {token_symbol}, marking position as emergency_failed")
            if token_address in self.active_trades:
                self.active_trades[token_address]['status'] = 'emergency_failed'
                self.active_trades[token_address]['emergency_attempts'] = max_attempts
                self.active_trades[token_address]['last_emergency_attempt'] = time.time()
            
            try:
                # Gửi thông báo thất bại
                if self.telegram_notifier:
                    await self.telegram_notifier.send_message(
                        f"❌ Emergency exit FAILED for {token_symbol} ({token_address})\n"
                        f"Reason for exit: {reason}\n"
                        f"All {max_attempts} attempts failed!",
                        level="ERROR"
                    )
                else:
                    logger.warning("No telegram_notifier instance available for sending message")
            except Exception as notify_error:
                logger.warning(f"Error sending failure notification: {str(notify_error)}")
                
            return False
            
        except Exception as e:
            logger.critical(f"Critical error during emergency exit for {token_address}: {str(e)}")
            # Ghi chi tiết lỗi
            import traceback
            traceback.print_exc()
            
            try:
                # Gửi thông báo về lỗi nghiêm trọng
                token_info = f"{token_symbol} ({token_address})" if token_symbol else token_address
                if self.telegram_notifier:
                    await self.telegram_notifier.send_message(
                        f"❌ CRITICAL ERROR during emergency exit for {token_info}:\n"
                        f"{str(e)}\n"
                        f"Manual intervention required!",
                        level="CRITICAL"
                    )
                else:
                    logger.warning("No telegram_notifier instance available for sending message")
            except Exception as notify_error:
                logger.warning(f"Error sending critical error notification: {str(notify_error)}")
                
            return False
    
    def get_active_trades(self) -> List[Dict[str, Any]]:
        """
        Get list of active trades
        
        Returns:
            List[Dict[str, Any]]: List of active trades
        """
        return list(self.active_trades.values())
    
    def get_completed_trades(self) -> List[Dict[str, Any]]:
        """
        Get list of completed trades
        
        Returns:
            List[Dict[str, Any]]: List of completed trades
        """
        return self.completed_trades
    
    def get_trade_history(self) -> List[Dict[str, Any]]:
        """
        Get full trade history
        
        Returns:
            List[Dict[str, Any]]: Full trade history
        """
        return self.trade_history
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get performance summary
        
        Returns:
            Dict[str, Any]: Performance summary
        """
        completed = [t for t in self.trade_history if t.get('status') == 'completed']
        
        if not completed:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0,
                'avg_profit': 0,
                'avg_loss': 0,
                'profit_factor': 0,
                'total_pnl_percentage': 0
            }
        
        # Calculate statistics
        winning = [t for t in completed if t.get('final_pnl_percentage', 0) > 0]
        losing = [t for t in completed if t.get('final_pnl_percentage', 0) <= 0]
        
        total_profit = sum(t.get('final_pnl_percentage', 0) for t in winning)
        total_loss = abs(sum(t.get('final_pnl_percentage', 0) for t in losing))
        
        win_rate = len(winning) / len(completed) if completed else 0
        avg_profit = total_profit / len(winning) if winning else 0
        avg_loss = total_loss / len(losing) if losing else 0
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        
        return {
            'total_trades': len(completed),
            'winning_trades': len(winning),
            'losing_trades': len(losing),
            'win_rate': win_rate,
            'avg_profit': avg_profit,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'total_pnl_percentage': total_profit - total_loss
        }
