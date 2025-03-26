"""
OnchainAnalyzer Component
Responsible for direct blockchain analysis to detect potential trading opportunities
without relying solely on external services like PumpPortal.
"""

import asyncio
import base64
import json
import logging
import time
import os
from typing import Dict, Any, List, Optional, Tuple, Set, Callable, Awaitable
from datetime import datetime, timedelta

# For safe imports
try:
    import solders
    from solders.pubkey import Pubkey as PublicKey
    from solana.rpc.types import TokenAccountOpts
    from solana.rpc.commitment import Commitment
except ImportError:
    pass  # Server has these packages installed, LSP may not detect them

from core.state_manager import StateManager
from core.config_manager import ConfigManager
from core.security_manager import SecurityManager
from core.base_component import BaseComponent
from utils.api_resilience import with_retry, with_timeout
from utils.connection_pool import ConnectionPool

logger = logging.getLogger(__name__)

class OnchainAnalyzer(BaseComponent):
    """
    OnchainAnalyzer analyzes Solana blockchain data directly to:
    - Detect new token launches
    - Monitor liquidity pools
    - Track whale wallet activity
    - Analyze memecoin metrics independent of PumpPortal
    """
    
    def __init__(self, 
                config_manager: ConfigManager, 
                state_manager: StateManager,
                security_manager: SecurityManager):
        """
        Initialize the OnchainAnalyzer
        
        Args:
            config_manager (ConfigManager): Configuration manager instance
            state_manager (StateManager): State manager instance
            security_manager (SecurityManager): Security manager instance
        """
        # Initialize BaseComponent
        super().__init__(state_manager, 'onchain_analyzer')
        
        self.config_manager = config_manager
        self.security_manager = security_manager
        
        # Data confidence scores for different data sources
        # These scores are used to weight data from different sources during validation
        # Higher score means more reliable data source (0-100)
        self.data_confidence_scores = {
            'onchain': 100,  # Direct blockchain data has highest confidence
            'raydium': 95,   # DEX data is very reliable but can have small delays
            'pump_portal': 80,  # Third-party service has lower confidence
            # This can be adjusted dynamically based on connection reliability
        }
        
        # RPC connection settings
        rpc_endpoints = self.config_manager.get('solana.rpc_endpoints', [
            {
                "url": "https://api.mainnet-beta.solana.com",
                "weight": 1
            }
        ])
        
        # Create connection pool with the configured endpoints
        self.connection_pool = ConnectionPool(
            endpoints=rpc_endpoints,
            max_connections=5,
            cache_enabled=True,
            cache_ttl=15.0
        )
        
        # Setup state variables
        self.known_tokens = set()
        self.known_pools = {}  # pool_address -> pool_data
        self.whale_wallets = set()
        self.monitored_wallets = set()
        self.transactions_cache = {}
        
        # Important Solana program IDs
        self.token_program_id = "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"
        self.raydium_amm_program_id = "675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8"
        self.jupiter_program_id = "JUP4Fb2cqiRUcaTHdrPC8h2gNsA2ETXiPDD33WcGuJB"
        
        # Event callbacks
        self.new_token_callbacks = set()
        self.whale_activity_callbacks = set()
        self.liquidity_change_callbacks = set()
        
        # Analysis settings
        self.min_liquidity_usd = self.config_manager.get('onchain_analyzer.min_liquidity_usd', 10000)
        self.min_holders = self.config_manager.get('onchain_analyzer.min_holders', 20)
        self.whale_min_size_sol = self.config_manager.get('onchain_analyzer.whale_min_size_sol', 50)
        self.normal_scan_interval = self.config_manager.get('onchain_analyzer.scan_interval', 60)  # seconds
        self.accelerated_scan_interval = self.config_manager.get('onchain_analyzer.accelerated_scan_interval', 20)  # seconds
        self.scan_interval = self.normal_scan_interval  # current interval, will adapt based on PumpPortal status
        
        # Dynamic scan settings
        self.enable_adaptive_scanning = self.config_manager.get('onchain_analyzer.enable_adaptive_scanning', True)
        self.adaptive_mode_active = False
        self.last_scan_mode_change = time.time()
        self.max_blocks_normal = self.config_manager.get('onchain_analyzer.max_blocks_normal', 10)
        self.max_blocks_accelerated = self.config_manager.get('onchain_analyzer.max_blocks_accelerated', 25)
        self.current_max_blocks = self.max_blocks_normal
        
        # Data confidence scoring
        self.data_confidence_scores = {
            'onchain': 100,  # Direct blockchain data is always 100% reliable
            'pump_portal': 80,  # Default confidence in PumpPortal data
        }
        
        # Tasks
        self.scanning_task = None
        self.whale_monitoring_task = None
        
        logger.info("OnchainAnalyzer initialized")
    
    async def start(self):
        """Start the OnchainAnalyzer"""
        # Call base class start method
        await super().start()
        
        # Start connection pool
        await self.connection_pool.start()
        
        # Load known data
        await self._load_known_data()
        
        # Start scanning tasks
        self.scanning_task = asyncio.create_task(self._scanning_loop())
        self.whale_monitoring_task = asyncio.create_task(self._whale_monitoring_loop())
        
        logger.info("OnchainAnalyzer started")
    
    async def stop(self):
        """Stop the OnchainAnalyzer safely"""
        # Stop ongoing tasks
        if self.scanning_task:
            self.scanning_task.cancel()
            try:
                await self.scanning_task
            except asyncio.CancelledError:
                pass
            
        if self.whale_monitoring_task:
            self.whale_monitoring_task.cancel()
            try:
                await self.whale_monitoring_task
            except asyncio.CancelledError:
                pass
        
        # Save known data
        await self._save_known_data()
        
        # Stop connection pool
        await self.connection_pool.stop()
        
        # Call base class stop method
        await super().stop()
        
        logger.info("OnchainAnalyzer stopped")
    
    def register_new_token_callback(self, callback: Callable[[Dict[str, Any]], Awaitable[None]]):
        """
        Register callback for new token events
        
        Args:
            callback (Callable): Callback function
        """
        self.new_token_callbacks.add(callback)
    
    def register_whale_activity_callback(self, callback: Callable[[Dict[str, Any]], Awaitable[None]]):
        """
        Register callback for whale activity events
        
        Args:
            callback (Callable): Callback function
        """
        self.whale_activity_callbacks.add(callback)
    
    def register_liquidity_change_callback(self, callback: Callable[[Dict[str, Any]], Awaitable[None]]):
        """
        Register callback for liquidity change events
        
        Args:
            callback (Callable): Callback function
        """
        self.liquidity_change_callbacks.add(callback)
    
    async def _load_known_data(self):
        """Load previously saved tokens and pools"""
        try:
            # Load known tokens
            tokens_file = "data/onchain/known_tokens.json"
            if os.path.exists(tokens_file):
                with open(tokens_file, "r") as f:
                    tokens_data = json.load(f)
                    self.known_tokens = set(tokens_data)
                    logger.info(f"Loaded {len(self.known_tokens)} known tokens")
            
            # Load known pools
            pools_file = "data/onchain/known_pools.json"
            if os.path.exists(pools_file):
                with open(pools_file, "r") as f:
                    self.known_pools = json.load(f)
                    logger.info(f"Loaded {len(self.known_pools)} known pools")
            
            # Load whale wallets
            whales_file = "data/onchain/whale_wallets.json"
            if os.path.exists(whales_file):
                with open(whales_file, "r") as f:
                    whale_data = json.load(f)
                    self.whale_wallets = set(whale_data)
                    logger.info(f"Loaded {len(self.whale_wallets)} whale wallets")
            
        except Exception as e:
            logger.error(f"Error loading known data: {str(e)}")
    
    async def _save_known_data(self):
        """Save current tokens and pools data"""
        try:
            # Ensure directory exists
            os.makedirs("data/onchain", exist_ok=True)
            
            # Save known tokens
            tokens_file = "data/onchain/known_tokens.json"
            with open(tokens_file, "w") as f:
                json.dump(list(self.known_tokens), f)
            
            # Save known pools
            pools_file = "data/onchain/known_pools.json"
            with open(pools_file, "w") as f:
                json.dump(self.known_pools, f)
            
            # Save whale wallets
            whales_file = "data/onchain/whale_wallets.json"
            with open(whales_file, "w") as f:
                json.dump(list(self.whale_wallets), f)
            
            logger.info("Saved on-chain analysis data")
            
        except Exception as e:
            logger.error(f"Error saving known data: {str(e)}")
    
    async def _scanning_loop(self):
        """Main scanning loop for on-chain analysis"""
        while self.is_running:
            try:
                # Check for new tokens from recent blocks
                await self._scan_for_new_tokens()
                
                # Update liquidity pool data
                await self._update_liquidity_pools()
                
                # Clean up old transaction data
                self._cleanup_transaction_cache()
                
                # Store data periodically
                await self._save_known_data()
                
                # Update component metrics
                await self._update_metrics()
                
                # Wait for next scan
                await asyncio.sleep(self.scan_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in scanning loop: {str(e)}")
                await asyncio.sleep(30)  # Wait before retry
    
    async def _whale_monitoring_loop(self):
        """Monitor whale wallet activity"""
        while self.is_running:
            try:
                # Monitor whale wallet activities
                for wallet in self.monitored_wallets:
                    await self._check_wallet_activity(wallet)
                
                # Discover new potential whale wallets
                await self._discover_new_whales()
                
                # Wait for next monitoring cycle (different from main scan to spread load)
                await asyncio.sleep(self.scan_interval * 1.5)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in whale monitoring loop: {str(e)}")
                await asyncio.sleep(30)  # Wait before retry
    
    async def _scan_for_new_tokens(self):
        """Scan recent Solana blocks for new token creations"""
        try:
            # Track scan start time for performance metrics
            scan_start_time = time.time()
            
            # Get recent block signatures with more reliable error handling
            max_retries = 3
            retry_count = 0
            latest_blockhash_response = None
            
            while retry_count < max_retries:
                try:
                    # Use enhanced call with retry parameters
                    latest_blockhash_response = await self.connection_pool.call(
                        "getLatestBlockhash",
                        {"commitment": "confirmed"},
                        max_retries=2,
                        retry_delay=0.5
                    )
                    
                    # Break if successful
                    if latest_blockhash_response and "result" in latest_blockhash_response:
                        break
                        
                except Exception as e:
                    retry_count += 1
                    wait_time = 2 ** retry_count * 0.5  # Exponential backoff
                    logger.warning(f"Error getting latest blockhash (attempt {retry_count}/{max_retries}): {str(e)}, retrying in {wait_time}s")
                    await asyncio.sleep(wait_time)
            
            # Check if we succeeded after retries
            if not latest_blockhash_response or "result" not in latest_blockhash_response:
                logger.error("Failed to get latest blockhash after multiple retries")
                # Report health issue to state manager
                self.state_manager.update_component_status(
                    'onchain_analyzer',
                    'degraded',
                    'Failed to get latest blockhash from Solana RPC'
                )
                return
            
            # Get recent blocks (based on current scanning mode)
            current_slot = latest_blockhash_response["result"]["context"]["slot"]
            blocks_to_scan = self.current_max_blocks
            
            logger.debug(f"Scanning {blocks_to_scan} recent blocks for new tokens (interval: {self.scan_interval}s)")
            
            # Get list of blocks with timeout protection
            try:
                blocks_response = await asyncio.wait_for(
                    self.connection_pool.call(
                        "getBlocks",
                        [current_slot - blocks_to_scan, current_slot],
                        max_retries=2
                    ),
                    timeout=20.0  # Set a reasonable timeout
                )
            except asyncio.TimeoutError:
                logger.error("Timeout getting blocks list")
                return
            except Exception as e:
                logger.error(f"Error getting blocks list: {str(e)}")
                return
            
            if "result" not in blocks_response or not blocks_response["result"]:
                logger.warning("No blocks returned in getBlocks response")
                return
            
            # Update health status - we're successfully retrieving data
            self.state_manager.update_component_status(
                'onchain_analyzer',
                'running',
                'Successfully retrieving blockchain data'
            )
            
            # Track number of tokens discovered
            tokens_discovered = 0
            blocks_processed = 0
            
            # Process each block with concurrent limit
            block_semaphore = asyncio.Semaphore(3)  # Limit concurrent block processing
            
            async def process_block(block_num):
                nonlocal tokens_discovered
                
                async with block_semaphore:
                    try:
                        # Get block details with transactions
                        block_response = await self.connection_pool.call(
                            "getBlock",
                            [block_num, {"encoding": "json", "transactionDetails": "full", "maxSupportedTransactionVersion": 0}],
                            max_retries=1
                        )
                        
                        if "result" not in block_response or not block_response["result"]:
                            return
                        
                        block = block_response["result"]
                        
                        # Skip if no transactions
                        if "transactions" not in block or not block["transactions"]:
                            return
                        
                        # Process each transaction in the block
                        for tx in block["transactions"]:
                            # Skip failed transactions
                            if "meta" in tx and tx["meta"].get("err") is not None:
                                continue
                            
                            # Check for token creation transactions
                            new_tokens = await self._analyze_transaction_for_tokens(tx)
                            if new_tokens:
                                tokens_discovered += new_tokens
                    
                    except Exception as e:
                        logger.error(f"Error processing block {block_num}: {str(e)}")
            
            # Create tasks for block processing
            tasks = []
            for block_num in blocks_response["result"]:
                task = asyncio.create_task(process_block(block_num))
                tasks.append(task)
            
            # Wait for all blocks to be processed
            await asyncio.gather(*tasks)
            blocks_processed = len(tasks)
            
            # Calculate scan time
            scan_time = time.time() - scan_start_time
            
            # Update metrics about scan performance
            self.state_manager.update_component_metric(
                'onchain_analyzer', 
                'blocks_scanned_per_cycle', 
                blocks_processed
            )
            
            self.state_manager.update_component_metric(
                'onchain_analyzer', 
                'tokens_discovered', 
                tokens_discovered
            )
            
            self.state_manager.update_component_metric(
                'onchain_analyzer', 
                'scan_duration_seconds', 
                scan_time
            )
            
            logger.debug(f"Scan completed: {blocks_processed} blocks processed in {scan_time:.2f}s, {tokens_discovered} new tokens discovered")
        
        except asyncio.CancelledError:
            # Handle cancellation gracefully
            logger.info("Token scanning task cancelled")
            raise
        except Exception as e:
            logger.error(f"Error scanning for new tokens: {str(e)}")
            self.state_manager.update_component_status(
                'onchain_analyzer',
                'error',
                f'Error in token scanning: {str(e)}'
            )
    
    async def _analyze_transaction_for_tokens(self, tx: Dict[str, Any]) -> int:
        """
        Analyze a transaction for potential token creation events
        
        Args:
            tx (Dict[str, Any]): Transaction data
            
        Returns:
            int: Number of new tokens discovered in this transaction
        """
        tokens_discovered = 0
        
        try:
            # Skip if no transaction details
            if "transaction" not in tx or "message" not in tx["transaction"]:
                return 0
            
            # Get program IDs from accountKeys
            message = tx["transaction"]["message"]
            account_keys = [account for account in message.get("accountKeys", [])]
            
            # Check if token program is involved
            token_program_index = None
            
            for i, key in enumerate(account_keys):
                if key == self.token_program_id:
                    token_program_index = i
                    break
            
            if token_program_index is None:
                return 0  # Token program not involved
            
            # Check instructions for potential token creation
            instructions = message.get("instructions", [])
            
            # Transaction signature for logging
            tx_signature = tx.get("transaction", {}).get("signatures", ["unknown"])[0]
            
            for instruction in instructions:
                # Skip if not for token program
                if instruction.get("programIdIndex") != token_program_index:
                    continue
                
                # Check instruction data - we're looking for token initialization
                # or mint instruction patterns
                data = instruction.get("data", "")
                
                # Simplified analysis: in a real implementation, we'd decode the instruction
                # to identify token creation more precisely, but that requires the full SPL
                # token program interface
                
                # For now, let's identify potential token accounts created and analyze them
                account_indices = instruction.get("accounts", [])
                for idx in account_indices:
                    if idx < len(account_keys):
                        potential_token = account_keys[idx]
                        
                        # Skip if already known
                        if potential_token in self.known_tokens:
                            continue
                        
                        # Check if it's a token mint account
                        is_token = await self._verify_token_mint(potential_token)
                        
                        if is_token:
                            # Add to known tokens
                            self.known_tokens.add(potential_token)
                            tokens_discovered += 1
                            
                            # Get token details
                            token_data = await self._get_token_details(potential_token)
                            
                            if token_data:
                                # Add transaction signature to token data
                                token_data["tx_signature"] = tx_signature
                                
                                # Notify callbacks of new token
                                notify_success = True
                                for callback in self.new_token_callbacks:
                                    try:
                                        await callback(token_data)
                                    except Exception as e:
                                        notify_success = False
                                        logger.error(f"Error in new token callback: {str(e)}")
                                
                                # Log based on whether callbacks succeeded
                                if notify_success:
                                    logger.info(f"Detected and notified new token: {potential_token} (tx: {tx_signature[:8]}...)")
                                else:
                                    logger.warning(f"Detected new token but notification failed: {potential_token} (tx: {tx_signature[:8]}...)")
                            else:
                                logger.warning(f"Detected new token mint but could not get details: {potential_token} (tx: {tx_signature[:8]}...)")
            
            return tokens_discovered
        
        except Exception as e:
            logger.error(f"Error analyzing transaction: {str(e)}")
            return 0
    
    async def _verify_token_mint(self, address: str) -> bool:
        """
        Verify if an address is a token mint account
        
        Args:
            address (str): Account address to check
            
        Returns:
            bool: True if token mint, False otherwise
        """
        try:
            # Get account info
            response = await self.connection_pool.call(
                "getAccountInfo",
                [address, {"encoding": "jsonParsed", "commitment": "confirmed"}]
            )
            
            if "result" not in response or not response["result"] or not response["result"]["value"]:
                return False
            
            account_info = response["result"]["value"]
            
            # Check if owned by token program
            if account_info.get("owner") != self.token_program_id:
                return False
            
            # Check for token mint data structure
            if "data" in account_info and account_info["data"].get("program") == "spl-token":
                parsed_data = account_info["data"].get("parsed", {})
                
                # Verify it's a mint account
                if parsed_data.get("type") == "mint":
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error verifying token mint: {str(e)}")
            return False
    
    async def _get_token_details(self, token_address: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a token
        
        Args:
            token_address (str): Token mint address
            
        Returns:
            Optional[Dict[str, Any]]: Token details or None if not available
        """
        try:
            # Get token supply
            supply_response = await self.connection_pool.call(
                "getTokenSupply",
                [token_address, {"commitment": "confirmed"}]
            )
            
            if "result" not in supply_response or not supply_response["result"]:
                return None
            
            supply_info = supply_response["result"]["value"]
            total_supply = float(supply_info["amount"]) / (10 ** supply_info["decimals"])
            
            # Get token accounts (holders)
            holders_response = await self.connection_pool.call(
                "getTokenLargestAccounts",
                [token_address, {"commitment": "confirmed"}]
            )
            
            holder_count = 0
            if "result" in holders_response and holders_response["result"]:
                holder_count = len(holders_response["result"]["value"])
            
            # Check for liquidity pool
            liquidity_data = await self._find_token_liquidity(token_address)
            
            # Combine token data
            token_data = {
                "address": token_address,
                "supply": total_supply,
                "decimal": supply_info["decimals"],
                "holder_count": holder_count,
                "creation_time": int(time.time()),
                "liquidity_usd": liquidity_data.get("liquidity_usd", 0),
                "price_usd": liquidity_data.get("price_usd", 0),
                "market_cap_usd": total_supply * liquidity_data.get("price_usd", 0),
                "is_verified": False,  # On-chain can't determine verification status
                "source": "on-chain"  # Mark as discovered on-chain vs from PumpPortal
            }
            
            return token_data
            
        except Exception as e:
            logger.error(f"Error getting token details: {str(e)}")
            return None
    
    async def _find_token_liquidity(self, token_address: str) -> Dict[str, Any]:
        """
        Find liquidity pools for a token and calculate metrics
        
        Args:
            token_address (str): Token mint address
            
        Returns:
            Dict[str, Any]: Liquidity data
        """
        result = {
            "liquidity_usd": 0,
            "price_usd": 0,
            "pools": []
        }
        
        try:
            # This requires detailed knowledge of Raydium and other DEX pool structures
            # For simplified implementation, we'll use a basic approach
            
            # Get current SOL price (needed for USD calculations)
            sol_price = await self._get_sol_price()
            if sol_price <= 0:
                sol_price = 100  # Fallback estimate if can't get real price
            
            # Check for Raydium pools
            # In a full implementation, we would need to query the Raydium program
            # accounts with specific filters to find pools containing this token
            
            # For the proof of concept, we can demonstrate the structure:
            # 1. We'd query all associated token accounts for the token mint
            # 2. Filter for those owned by AMM programs
            # 3. Calculate liquidity based on these pools
            
            # This is where integration with specific DEX knowledge is important
            
            # Placeholder for detailed implementation
            result["liquidity_usd"] = 0
            result["price_usd"] = 0
            
            return result
            
        except Exception as e:
            logger.error(f"Error finding token liquidity: {str(e)}")
            return result
    
    async def _get_sol_price(self) -> float:
        """
        Get current SOL price in USD
        
        Returns:
            float: SOL price in USD
        """
        try:
            # In a real-world scenario, you would query a price oracle or API
            # For this sample, we're using a simplified approach
            
            # Use Jupiter API to get SOL/USDC price
            # USDC mint on Solana
            usdc_mint = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
            
            # WSOL mint on Solana
            wsol_mint = "So11111111111111111111111111111111111111112"
            
            # Query Jupiter for price
            response = await self.connection_pool.call(
                "getTokenSupply",
                [wsol_mint, {"commitment": "confirmed"}]
            )
            
            # Placeholder for actual implementation
            return 150.0  # Example SOL price
            
        except Exception as e:
            logger.error(f"Error getting SOL price: {str(e)}")
            return 0
    
    async def _update_liquidity_pools(self):
        """Update liquidity data for known pools"""
        # In a real implementation, this would update all known liquidity pools
        # For tokens that we're tracking
        pass
    
    async def _check_wallet_activity(self, wallet_address: str):
        """
        Check activity for a specific wallet
        
        Args:
            wallet_address (str): Wallet address to check
        """
        try:
            # Get recent transactions for the wallet
            response = await self.connection_pool.call(
                "getSignaturesForAddress",
                [wallet_address, {"limit": 10}]
            )
            
            if "result" not in response or not response["result"]:
                return
            
            # Get the transaction signatures
            signatures = [tx["signature"] for tx in response["result"]]
            
            # Process each transaction
            for sig in signatures:
                # Skip if already processed
                if sig in self.transactions_cache:
                    continue
                
                # Get transaction details
                tx_response = await self.connection_pool.call(
                    "getTransaction",
                    [sig, {"encoding": "jsonParsed", "maxSupportedTransactionVersion": 0}]
                )
                
                if "result" not in tx_response or not tx_response["result"]:
                    continue
                
                # Process the transaction
                tx_data = tx_response["result"]
                
                # Add to cache with timestamp
                self.transactions_cache[sig] = {
                    "timestamp": time.time(),
                    "processed": True
                }
                
                # For whale activity, we're interested in large transfers or swaps
                await self._analyze_whale_transaction(wallet_address, tx_data)
                
        except Exception as e:
            logger.error(f"Error checking wallet activity: {str(e)}")
    
    async def _analyze_whale_transaction(self, wallet_address: str, tx_data: Dict[str, Any]):
        """
        Analyze a transaction for whale activity
        
        Args:
            wallet_address (str): Wallet address
            tx_data (Dict[str, Any]): Transaction data
        """
        try:
            # Check if transaction was successful
            if "meta" in tx_data and tx_data["meta"].get("err") is not None:
                return
            
            # Check for token transfers or swaps
            # This is a simplified analysis; a complete implementation would
            # decode all instructions and track value movements
            
            # For now, focus on post-balance changes as an indicator
            pre_balances = tx_data["meta"].get("preBalances", [])
            post_balances = tx_data["meta"].get("postBalances", [])
            
            # Simple heuristic for significant balance changes
            if len(pre_balances) == len(post_balances) and len(pre_balances) > 0:
                for i, (pre, post) in enumerate(zip(pre_balances, post_balances)):
                    # Calculate change in SOL
                    change_sol = abs(post - pre) / 1e9  # Convert lamports to SOL
                    
                    # Check if change is significant
                    if change_sol >= self.whale_min_size_sol:
                        # Get the account associated with this change
                        if "transaction" in tx_data and "message" in tx_data["transaction"]:
                            account_keys = tx_data["transaction"]["message"].get("accountKeys", [])
                            if i < len(account_keys):
                                affected_account = account_keys[i]
                                
                                # Prepare whale activity event
                                event_data = {
                                    "wallet": wallet_address,
                                    "related_account": affected_account,
                                    "amount_sol": change_sol,
                                    "timestamp": time.time(),
                                    "tx_signature": tx_data.get("transaction", {}).get("signatures", [""])[0],
                                    "direction": "received" if post > pre else "sent"
                                }
                                
                                # Notify callbacks
                                for callback in self.whale_activity_callbacks:
                                    try:
                                        await callback(event_data)
                                    except Exception as e:
                                        logger.error(f"Error in whale activity callback: {str(e)}")
                                
                                logger.info(f"Detected whale activity: {wallet_address} {event_data['direction']} {change_sol} SOL")
            
        except Exception as e:
            logger.error(f"Error analyzing whale transaction: {str(e)}")
    
    async def _discover_new_whales(self):
        """Discover new potential whale wallets to monitor"""
        try:
            # In a real implementation, this would scan for large holders of tokens
            # or significant liquidity providers
            
            # For now, just ensure we're monitoring current whales
            self.monitored_wallets = self.whale_wallets.copy()
            
        except Exception as e:
            logger.error(f"Error discovering new whales: {str(e)}")
    
    def _cleanup_transaction_cache(self):
        """Clean up old transactions from the cache"""
        current_time = time.time()
        to_remove = []
        
        # Find old transactions (older than 1 hour)
        for sig, tx_data in self.transactions_cache.items():
            if current_time - tx_data["timestamp"] > 3600:  # 1 hour
                to_remove.append(sig)
        
        # Remove old transactions
        for sig in to_remove:
            del self.transactions_cache[sig]
    
    async def update_scan_mode_for_pump_portal_status(self, is_connected: bool, disconnection_count: int = 0):
        """
        Adjust scanning mode based on PumpPortal connection status
        
        Args:
            is_connected (bool): Whether PumpPortal is connected
            disconnection_count (int): Number of recent disconnections
        """
        if not self.enable_adaptive_scanning:
            logger.info("Adaptive scanning disabled, maintaining normal scan interval")
            return
            
        old_interval = self.scan_interval
        old_blocks = self.current_max_blocks
        
        # If PumpPortal disconnected, increase scanning frequency
        if not is_connected:
            if self.scan_interval != self.accelerated_scan_interval:
                self.scan_interval = self.accelerated_scan_interval
                self.current_max_blocks = self.max_blocks_accelerated
                
                # Reduce confidence in PumpPortal data based on disconnection count
                # More disconnections = lower confidence score
                confidence_reduction = min(50, disconnection_count * 5)  # Max 50% reduction
                self.data_confidence_scores['pump_portal'] = max(30, 80 - confidence_reduction)
                
                logger.info(f"PumpPortal disconnected - Accelerating on-chain scanning: {old_interval}s → {self.scan_interval}s, "
                          f"blocks: {old_blocks} → {self.current_max_blocks}, "
                          f"PumpPortal confidence: {self.data_confidence_scores['pump_portal']}%")
                
                # Update state manager with component status
                self.state_manager.update_component_status(
                    'onchain_analyzer',
                    'accelerated',
                    'Accelerated scanning due to PumpPortal disconnection'
                )
        # If PumpPortal connected, return to normal scanning
        elif self.scan_interval != self.normal_scan_interval:
            self.scan_interval = self.normal_scan_interval
            self.current_max_blocks = self.max_blocks_normal
            
            # Gradually increase confidence in PumpPortal data
            self.data_confidence_scores['pump_portal'] = min(80, self.data_confidence_scores['pump_portal'] + 5)
            
            logger.info(f"PumpPortal connected - Returning to normal on-chain scanning: {old_interval}s → {self.scan_interval}s, "
                      f"blocks: {old_blocks} → {self.current_max_blocks}, "
                      f"PumpPortal confidence: {self.data_confidence_scores['pump_portal']}%")
            
            # Update state manager with component status
            self.state_manager.update_component_status(
                'onchain_analyzer',
                'running',
                'Normal scanning mode'
            )
            
    async def _update_metrics(self):
        """Update component metrics"""
        metrics = {
            "known_tokens": len(self.known_tokens),
            "known_pools": len(self.known_pools),
            "whale_wallets": len(self.whale_wallets),
            "monitored_wallets": len(self.monitored_wallets),
            "transaction_cache_size": len(self.transactions_cache),
            "scan_interval": self.scan_interval,
            "max_blocks_per_scan": self.current_max_blocks,
            "pump_portal_confidence": self.data_confidence_scores['pump_portal'],
            "onchain_confidence": self.data_confidence_scores['onchain']
        }
        
        # Update state manager with metrics
        self.state_manager.update_component_metrics('onchain_analyzer', metrics)
    
    async def update_scan_mode_for_pump_portal_status(self, is_connected: bool, disconnection_count: int = 0) -> Dict[str, Any]:
        """
        Update scanning mode based on PumpPortal connection status
        
        This method is called by the TradingIntegration component to coordinate 
        scanning behavior based on external service availability.
        
        Args:
            is_connected (bool): Whether PumpPortal is currently connected
            disconnection_count (int): Number of disconnections detected
            
        Returns:
            Dict[str, Any]: Updated scan settings
        """
        # Check if adaptive scanning is enabled
        if not self.enable_adaptive_scanning:
            return {
                "adaptive_mode": False,
                "scan_interval": self.scan_interval,
                "max_blocks": self.current_max_blocks
            }
        
        # Calculate PumpPortal data confidence based on disconnection frequency
        # More disconnections = less confidence in the data
        if disconnection_count > 0:
            # Reduce confidence by 5% for each disconnection, minimum 50%
            new_confidence = max(50, 80 - (disconnection_count * 5))
            self.data_confidence_scores['pump_portal'] = new_confidence
        
        current_time = time.time()
        mode_change_cooling_period = 60  # Don't change modes more often than every 60 seconds
        
        # Determine if we should change scan mode
        should_change_mode = (
            (not is_connected and not self.adaptive_mode_active) or  # PumpPortal disconnected, need to accelerate
            (is_connected and self.adaptive_mode_active and disconnection_count < 3) or  # PumpPortal restored, can return to normal
            (current_time - self.last_scan_mode_change > mode_change_cooling_period)  # Cooling period passed
        )
        
        if not should_change_mode:
            return {
                "adaptive_mode": self.adaptive_mode_active,
                "scan_interval": self.scan_interval,
                "max_blocks": self.current_max_blocks
            }
        
        # Update scan mode based on connection status
        if not is_connected:
            # Accelerate scanning if PumpPortal is disconnected
            if not self.adaptive_mode_active:
                logger.info("PumpPortal disconnected, activating accelerated scanning mode")
                self.scan_interval = self.accelerated_scan_interval
                self.current_max_blocks = self.max_blocks_accelerated
                self.adaptive_mode_active = True
                self.last_scan_mode_change = current_time
                
                # Update component status
                self.state_manager.update_component_status(
                    'onchain_analyzer',
                    'running',
                    'Accelerated scanning mode due to PumpPortal disconnection'
                )
        else:
            # Return to normal mode if PumpPortal is connected and stable (low disconnection count)
            if self.adaptive_mode_active and disconnection_count < 3:
                logger.info("PumpPortal restored, returning to normal scanning mode")
                self.scan_interval = self.normal_scan_interval
                self.current_max_blocks = self.max_blocks_normal
                self.adaptive_mode_active = False
                self.last_scan_mode_change = current_time
                
                # Update component status
                self.state_manager.update_component_status(
                    'onchain_analyzer',
                    'running',
                    'Normal scanning mode'
                )
        
        # Return current settings
        return {
            "adaptive_mode": self.adaptive_mode_active,
            "scan_interval": self.scan_interval,
            "max_blocks": self.current_max_blocks,
            "pump_portal_confidence": self.data_confidence_scores['pump_portal']
        }
    
    async def get_token_info(self, address: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a token, available as public API
        
        Args:
            address (str): Token address
            
        Returns:
            Optional[Dict[str, Any]]: Token information or None if not found
        """
        # Check if we already know this token
        if address in self.known_tokens:
            return await self._get_token_details(address)
        
        # Verify if it's actually a token
        is_token = await self._verify_token_mint(address)
        if is_token:
            # Add to known tokens
            self.known_tokens.add(address)
            # Get and return token details
            return await self._get_token_details(address)
        
        return None
    
    async def get_wallet_info(self, address: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a wallet
        
        Args:
            address (str): Wallet address
            
        Returns:
            Optional[Dict[str, Any]]: Wallet information or None if not found
        """
        try:
            # Get account balance
            balance_response = await self.connection_pool.call(
                "getBalance",
                [address, {"commitment": "confirmed"}]
            )
            
            if "result" not in balance_response or not balance_response["result"]:
                return None
            
            balance_lamports = balance_response["result"]["value"]
            balance_sol = balance_lamports / 1e9
            
            # Check if it's a whale
            is_whale = address in self.whale_wallets
            
            # Get recent transactions count
            tx_response = await self.connection_pool.call(
                "getSignaturesForAddress",
                [address, {"limit": 10}]
            )
            
            recent_tx_count = 0
            if "result" in tx_response and tx_response["result"]:
                recent_tx_count = len(tx_response["result"])
            
            # Combine wallet data
            wallet_data = {
                "address": address,
                "balance_sol": balance_sol,
                "is_whale": is_whale,
                "recent_tx_count": recent_tx_count,
                "last_activity": time.time()
            }
            
            return wallet_data
            
        except Exception as e:
            logger.error(f"Error getting wallet info: {str(e)}")
            return None
    
    async def analyze_wallet_holdings(self, address: str) -> List[Dict[str, Any]]:
        """
        Analyze token holdings of a wallet
        
        Args:
            address (str): Wallet address
            
        Returns:
            List[Dict[str, Any]]: List of token holdings
        """
        holdings = []
        
        try:
            # Get token accounts owned by the wallet
            response = await self.connection_pool.call(
                "getTokenAccountsByOwner",
                [
                    address,
                    {"programId": self.token_program_id},
                    {"encoding": "jsonParsed", "commitment": "confirmed"}
                ]
            )
            
            if "result" not in response or not response["result"] or not response["result"]["value"]:
                return holdings
            
            # Process each token account
            for account in response["result"]["value"]:
                if "account" not in account or "data" not in account["account"]:
                    continue
                
                parsed_data = account["account"]["data"].get("parsed", {})
                info = parsed_data.get("info", {})
                
                # Get token mint and amount
                token_mint = info.get("mint")
                amount = info.get("tokenAmount", {})
                
                if not token_mint or not amount:
                    continue
                
                # Only include non-zero balances
                if float(amount.get("amount", "0")) > 0:
                    # Calculate token value
                    token_amount = float(amount.get("amount", "0")) / (10 ** int(amount.get("decimals", "0")))
                    
                    # Get token details if known
                    token_info = await self.get_token_info(token_mint)
                    price_usd = token_info.get("price_usd", 0) if token_info else 0
                    value_usd = token_amount * price_usd
                    
                    holdings.append({
                        "token_mint": token_mint,
                        "amount": token_amount,
                        "decimals": int(amount.get("decimals", "0")),
                        "price_usd": price_usd,
                        "value_usd": value_usd
                    })
            
            # Sort by value (descending)
            holdings.sort(key=lambda x: x["value_usd"], reverse=True)
            
            return holdings
            
        except Exception as e:
            logger.error(f"Error analyzing wallet holdings: {str(e)}")
            return holdings


# Usage example:
# onchain_analyzer = OnchainAnalyzer(config_manager, state_manager, security_manager)
# await onchain_analyzer.start()
# 
# # Register callbacks
# onchain_analyzer.register_new_token_callback(handle_new_token)
# onchain_analyzer.register_whale_activity_callback(handle_whale_activity)

class RestartableOnchainAnalyzer(OnchainAnalyzer):
    """Extended OnchainAnalyzer with restart capability for self-healing system"""
    
    async def restart(self):
        """
        Restart the OnchainAnalyzer component
        
        This method is designed to be called by the self-healing system.
        It will stop and then restart the component.
        """
        logger.info("Restarting OnchainAnalyzer component")
        
        try:
            # First, stop the component if it's running
            if self.is_running:
                logger.info("Stopping OnchainAnalyzer before restart")
                await self.stop()
                
                # Short pause to ensure clean shutdown
                await asyncio.sleep(2)
            
            # Reset state
            self.blocks_scanned = 0
            self.last_scanned_block = 0
            self.new_tokens_detected = 0
            self._error_count = 0
            
            # Clear token caches to force fresh data
            self.token_info_cache.clear()
            self.known_token_addresses.clear()
            
            # Start the component again
            logger.info("Starting OnchainAnalyzer after restart")
            await self.start()
            
            # Log success
            if self.is_running:
                logger.info("OnchainAnalyzer successfully restarted")
                return True
            else:
                logger.error("OnchainAnalyzer failed to restart properly")
                return False
                
        except Exception as e:
            logger.error(f"Error during OnchainAnalyzer restart: {str(e)}")
            # Log traceback for debugging
            import traceback
            logger.error(traceback.format_exc())
            return False

# Monkey patch OnchainAnalyzer with restart method
OnchainAnalyzer.restart = RestartableOnchainAnalyzer.restart