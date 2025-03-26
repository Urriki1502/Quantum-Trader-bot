"""
Mempool Scanner Component
Analyzes the Solana mempool for early signals of trading opportunities
and transaction front-running protection.
"""

import asyncio
import logging
import time
import random
from typing import Dict, List, Any, Optional, Tuple

from core.state_manager import StateManager
from core.config_manager import ConfigManager

logger = logging.getLogger(__name__)

class TransactionAnalyzer:
    """Analyzes blockchain transactions for trading opportunities"""
    
    def __init__(self, config_manager: ConfigManager):
        """
        Initialize the TransactionAnalyzer
        
        Args:
            config_manager (ConfigManager): Configuration manager instance
        """
        self.config_manager = config_manager
        
        # Analysis settings
        self.min_transaction_size = self.config_manager.get(
            'mempool.min_transaction_size', 1000)
        self.opportunity_threshold = self.config_manager.get(
            'mempool.opportunity_threshold', 70)
        
        # Initialize pattern matchers
        self.patterns = {
            'whale_accumulation': self._match_whale_accumulation,
            'token_listing': self._match_token_listing,
            'liquidity_addition': self._match_liquidity_addition,
            'multiple_buys': self._match_multiple_buys,
            'large_swap': self._match_large_swap
        }
    
    async def analyze(self, transaction: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a transaction for potential opportunities
        
        Args:
            transaction (Dict[str, Any]): Transaction data
            
        Returns:
            Dict[str, Any]: Analysis results
        """
        result = {
            'transaction_id': transaction.get('signature', 'unknown'),
            'timestamp': time.time(),
            'patterns_matched': [],
            'opportunity_score': 0,
            'token_addresses': [],
            'is_opportunity': False
        }
        
        # Skip small transactions
        if transaction.get('size', 0) < self.min_transaction_size:
            return result
        
        # Apply each pattern matcher
        pattern_scores = []
        
        for pattern_name, pattern_func in self.patterns.items():
            pattern_result = await pattern_func(transaction)
            
            if pattern_result['matched']:
                result['patterns_matched'].append(pattern_name)
                pattern_scores.append(pattern_result['score'])
                
                # Add token addresses
                if pattern_result.get('token_addresses'):
                    result['token_addresses'].extend(pattern_result['token_addresses'])
        
        # Calculate opportunity score as weighted average of pattern scores
        if pattern_scores:
            result['opportunity_score'] = sum(pattern_scores) / len(pattern_scores)
            result['is_opportunity'] = result['opportunity_score'] >= self.opportunity_threshold
        
        # Deduplicate token addresses
        result['token_addresses'] = list(set(result['token_addresses']))
        
        return result
    
    async def _match_whale_accumulation(self, transaction: Dict[str, Any]) -> Dict[str, Any]:
        """
        Match whale accumulation pattern
        
        Args:
            transaction (Dict[str, Any]): Transaction data
            
        Returns:
            Dict[str, Any]: Pattern match result
        """
        result = {
            'matched': False,
            'score': 0,
            'token_addresses': []
        }
        
        # Check for whale accumulation patterns
        # In real implementation, would look for:
        # - Large buy transactions from known whale addresses
        # - Specific instruction patterns for token purchases
        # - Multiple smaller buys from related addresses
        
        # For demonstration, use a simplified check
        instructions = transaction.get('instructions', [])
        for instruction in instructions:
            # Look for token purchase instructions
            if instruction.get('program') == 'spl-token' and instruction.get('type') == 'transfer':
                # Check for large amount
                amount = instruction.get('amount', 0)
                if amount > 10000:  # Simplified threshold
                    result['matched'] = True
                    result['score'] = min(100, 50 + amount / 1000)  # Score based on size
                    
                    # Add token address
                    token_address = instruction.get('token')
                    if token_address:
                        result['token_addresses'].append(token_address)
                    
                    break
        
        return result
    
    async def _match_token_listing(self, transaction: Dict[str, Any]) -> Dict[str, Any]:
        """
        Match token listing pattern
        
        Args:
            transaction (Dict[str, Any]): Transaction data
            
        Returns:
            Dict[str, Any]: Pattern match result
        """
        result = {
            'matched': False,
            'score': 0,
            'token_addresses': []
        }
        
        # Check for token listing patterns
        # In real implementation, would look for:
        # - Token mint/create instructions
        # - Initial liquidity pool creation
        # - Raydium pool setup instructions
        
        # For demonstration, use a simplified check
        instructions = transaction.get('instructions', [])
        
        # Check for mint instruction followed by pool creation
        has_mint = False
        has_pool_create = False
        token_address = None
        
        for instruction in instructions:
            if instruction.get('program') == 'token-program' and instruction.get('type') == 'create-mint':
                has_mint = True
                token_address = instruction.get('mint')
            
            if instruction.get('program') == 'amm-program' and instruction.get('type') == 'initialize-pool':
                has_pool_create = True
        
        if has_mint and has_pool_create and token_address:
            result['matched'] = True
            result['score'] = 90  # High score for new token listings
            result['token_addresses'].append(token_address)
        
        return result
    
    async def _match_liquidity_addition(self, transaction: Dict[str, Any]) -> Dict[str, Any]:
        """
        Match liquidity addition pattern
        
        Args:
            transaction (Dict[str, Any]): Transaction data
            
        Returns:
            Dict[str, Any]: Pattern match result
        """
        result = {
            'matched': False,
            'score': 0,
            'token_addresses': []
        }
        
        # Check for liquidity addition patterns
        # In real implementation, would look for:
        # - Add liquidity instructions to DEX pools
        # - Significant liquidity changes (e.g., > 20% of existing pool)
        
        # For demonstration, use a simplified check
        instructions = transaction.get('instructions', [])
        for instruction in instructions:
            if (instruction.get('program') in ['amm-program', 'raydium-program'] and 
                instruction.get('type') in ['add-liquidity', 'deposit-liquidity']):
                
                # Check liquidity amount
                amount_a = instruction.get('amount_a', 0)
                amount_b = instruction.get('amount_b', 0)
                total_amount = amount_a + amount_b
                
                if total_amount > 5000:  # Simplified threshold
                    result['matched'] = True
                    result['score'] = min(85, 40 + total_amount / 1000)  # Score based on size
                    
                    # Add token addresses
                    token_a = instruction.get('token_a')
                    token_b = instruction.get('token_b')
                    
                    if token_a:
                        result['token_addresses'].append(token_a)
                    if token_b:
                        result['token_addresses'].append(token_b)
                    
                    break
        
        return result
    
    async def _match_multiple_buys(self, transaction: Dict[str, Any]) -> Dict[str, Any]:
        """
        Match multiple buys pattern
        
        Args:
            transaction (Dict[str, Any]): Transaction data
            
        Returns:
            Dict[str, Any]: Pattern match result
        """
        result = {
            'matched': False,
            'score': 0,
            'token_addresses': []
        }
        
        # Check for multiple buys pattern
        # In real implementation, would track:
        # - Multiple buy transactions for the same token in recent timeframe
        # - Pattern of increasing buy sizes
        
        # For demonstration, this would require transaction history context
        # that's not available in a single transaction
        # So we'll use a simplified approach
        
        # Check if transaction has multiple swap instructions for the same token
        instructions = transaction.get('instructions', [])
        token_counts = {}
        
        for instruction in instructions:
            if instruction.get('type') == 'swap':
                # Count swap instructions by output token
                output_token = instruction.get('output_token')
                if output_token:
                    token_counts[output_token] = token_counts.get(output_token, 0) + 1
        
        # Find tokens with multiple swaps
        for token, count in token_counts.items():
            if count >= 2:
                result['matched'] = True
                result['score'] = min(80, 50 + count * 10)  # Score based on count
                result['token_addresses'].append(token)
        
        return result
    
    async def _match_large_swap(self, transaction: Dict[str, Any]) -> Dict[str, Any]:
        """
        Match large swap pattern
        
        Args:
            transaction (Dict[str, Any]): Transaction data
            
        Returns:
            Dict[str, Any]: Pattern match result
        """
        result = {
            'matched': False,
            'score': 0,
            'token_addresses': []
        }
        
        # Check for large swap patterns
        # In real implementation, would look for:
        # - Large swap instructions relative to pool size
        # - Swaps from known trading addresses
        
        # For demonstration, use a simplified check
        instructions = transaction.get('instructions', [])
        for instruction in instructions:
            if instruction.get('type') == 'swap':
                # Check swap amount
                amount_in = instruction.get('amount_in', 0)
                
                if amount_in > 10000:  # Simplified threshold
                    result['matched'] = True
                    result['score'] = min(75, 40 + amount_in / 1000)  # Score based on size
                    
                    # Add token addresses
                    input_token = instruction.get('input_token')
                    output_token = instruction.get('output_token')
                    
                    if input_token:
                        result['token_addresses'].append(input_token)
                    if output_token:
                        result['token_addresses'].append(output_token)
                    
                    break
        
        return result


class MempoolScanner:
    """
    Scans the Solana mempool for early signals and opportunities
    
    This component analyzes pending transactions to detect potential
    trading opportunities before they affect the market, giving the bot
    a timing advantage.
    """
    
    def __init__(self, 
                config_manager: ConfigManager,
                state_manager: StateManager,
                rpc_client):
        """
        Initialize the MempoolScanner
        
        Args:
            config_manager (ConfigManager): Configuration manager instance
            state_manager (StateManager): State manager instance
            rpc_client: RPC client instance
        """
        self.config_manager = config_manager
        self.state_manager = state_manager
        self.rpc_client = rpc_client
        
        # Initialize transaction analyzer
        self.transaction_analyzer = TransactionAnalyzer(config_manager)
        
        # Scanning settings
        self.scan_interval = self.config_manager.get('mempool.scan_interval', 2.0)
        self.max_transactions = self.config_manager.get('mempool.max_transactions', 100)
        self.min_priority = self.config_manager.get('mempool.min_priority', 0)
        
        # State tracking
        self.is_running = False
        self.scanning_task = None
        self.processed_txs = set()  # Keep track of processed transaction IDs
        self.max_processed_txs = 10000  # Prevent memory growth
        
        # Opportunity tracking
        self.detected_opportunities = []
        self.max_opportunities = 100
        
        # Callbacks for new opportunities
        self.opportunity_callbacks = set()
        
        logger.info("MempoolScanner initialized")
    
    async def start(self):
        """Start the mempool scanner"""
        if self.is_running:
            logger.warning("MempoolScanner is already running")
            return
        
        self.is_running = True
        self.scanning_task = asyncio.create_task(self._scanning_loop())
        
        logger.info("MempoolScanner started")
        
        # Update component status
        self.state_manager.update_component_status(
            'mempool_scanner', 
            'running',
            "Scanning mempool for opportunities"
        )
    
    async def stop(self):
        """Stop the mempool scanner"""
        if not self.is_running:
            return
            
        self.is_running = False
        
        if self.scanning_task:
            self.scanning_task.cancel()
            try:
                await self.scanning_task
            except asyncio.CancelledError:
                pass
            self.scanning_task = None
            
        logger.info("MempoolScanner stopped")
        
        # Update component status
        self.state_manager.update_component_status(
            'mempool_scanner', 
            'stopped'
        )
    
    async def _scanning_loop(self):
        """Main scanning loop"""
        logger.info(f"Starting mempool scanning loop (interval: {self.scan_interval}s)")
        
        scan_count = 0
        
        while self.is_running:
            try:
                start_time = time.time()
                
                # Get pending transactions
                pending_txs = await self._get_pending_transactions()
                
                # Process transactions
                if pending_txs:
                    opportunities = await self._process_transactions(pending_txs)
                    
                    # Notify about opportunities
                    if opportunities:
                        for callback in self.opportunity_callbacks:
                            try:
                                await callback(opportunities)
                            except Exception as e:
                                logger.error(f"Error in opportunity callback: {str(e)}")
                
                # Update scan count and metrics
                scan_count += 1
                if scan_count % 10 == 0:  # Update metrics every 10 scans
                    self._update_metrics(scan_count)
                
                # Sleep until next scan, accounting for processing time
                elapsed = time.time() - start_time
                sleep_time = max(0.1, self.scan_interval - elapsed)
                await asyncio.sleep(sleep_time)
                
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error(f"Error in mempool scanning loop: {str(e)}")
                await asyncio.sleep(self.scan_interval)
    
    async def _get_pending_transactions(self) -> List[Dict[str, Any]]:
        """
        Get pending transactions from the mempool
        
        Returns:
            List[Dict[str, Any]]: Pending transactions
        """
        try:
            # In a real implementation, this would:
            # 1. Call RPC method getRecentPrioritizationFees or similar
            # 2. Get transaction details for recent high-priority transactions
            # 3. Parse and format transaction data
            
            result = await self.rpc_client._call("getRecentPrioritizationFees")
            
            if not result or 'result' not in result:
                return []
            
            # Filter by minimum priority fee
            priority_fees = [
                item for item in result['result']
                if item.get('prioritizationFee', 0) >= self.min_priority
            ]
            
            # Limit to max transactions
            priority_fees = priority_fees[:self.max_transactions]
            
            # In a real implementation, we would now fetch transaction details
            # This is simplified for demonstration
            pending_txs = []
            
            for fee_item in priority_fees:
                # Skip already processed transactions
                if fee_item.get('slot') in self.processed_txs:
                    continue
                
                # Mark as processed
                self.processed_txs.add(fee_item.get('slot'))
                
                # Create simplified transaction object
                # In reality, would call getTransaction and parse result
                tx = {
                    'signature': f"simulated_{fee_item.get('slot')}",
                    'slot': fee_item.get('slot'),
                    'priority_fee': fee_item.get('prioritizationFee', 0),
                    'size': random.randint(1000, 5000),
                    'instructions': self._generate_sample_instructions()
                }
                
                pending_txs.append(tx)
            
            # Trim processed tx set to prevent memory growth
            if len(self.processed_txs) > self.max_processed_txs:
                self.processed_txs = set(list(self.processed_txs)[-self.max_processed_txs:])
            
            return pending_txs
            
        except Exception as e:
            logger.error(f"Error getting pending transactions: {str(e)}")
            return []
    
    def _generate_sample_instructions(self) -> List[Dict[str, Any]]:
        """
        Generate sample instructions for demonstration
        
        In a real implementation, this would be replaced with actual
        parsed transaction instructions from the blockchain.
        
        Returns:
            List[Dict[str, Any]]: Sample instructions
        """
        # This is only for demonstration - in a real system,
        # we would parse actual transaction data
        num_instructions = random.randint(1, 5)
        instructions = []
        
        for _ in range(num_instructions):
            # Choose a random instruction type
            instruction_type = random.choice([
                'transfer', 'swap', 'create-mint', 'initialize-pool',
                'add-liquidity', 'deposit-liquidity'
            ])
            
            # Choose a random program
            program = random.choice([
                'spl-token', 'token-program', 'amm-program', 
                'raydium-program', 'system-program'
            ])
            
            # Generate a sample instruction
            instruction = {
                'type': instruction_type,
                'program': program
            }
            
            # Add instruction-specific fields
            if instruction_type == 'transfer':
                instruction['amount'] = random.randint(1000, 100000)
                instruction['token'] = f"token{random.randint(1, 100)}"
                
            elif instruction_type == 'swap':
                instruction['amount_in'] = random.randint(1000, 100000)
                instruction['input_token'] = f"token{random.randint(1, 100)}"
                instruction['output_token'] = f"token{random.randint(1, 100)}"
                
            elif instruction_type == 'create-mint':
                instruction['mint'] = f"token{random.randint(1, 100)}"
                
            elif instruction_type in ['add-liquidity', 'deposit-liquidity']:
                instruction['amount_a'] = random.randint(1000, 100000)
                instruction['amount_b'] = random.randint(1000, 100000)
                instruction['token_a'] = f"token{random.randint(1, 100)}"
                instruction['token_b'] = f"token{random.randint(1, 100)}"
            
            instructions.append(instruction)
        
        return instructions
    
    async def _process_transactions(self, 
                                  transactions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process transactions to find opportunities
        
        Args:
            transactions (List[Dict[str, Any]]): Transactions to process
            
        Returns:
            List[Dict[str, Any]]: Detected opportunities
        """
        opportunities = []
        
        for tx in transactions:
            # Analyze transaction
            analysis = await self.transaction_analyzer.analyze(tx)
            
            # Check if it's an opportunity
            if analysis['is_opportunity']:
                # Add to detected opportunities
                opportunity = {
                    'transaction_id': tx.get('signature', 'unknown'),
                    'token_addresses': analysis['token_addresses'],
                    'opportunity_score': analysis['opportunity_score'],
                    'patterns_matched': analysis['patterns_matched'],
                    'timestamp': time.time(),
                    'priority_fee': tx.get('priority_fee', 0)
                }
                
                opportunities.append(opportunity)
                self.detected_opportunities.append(opportunity)
                
                # Log the opportunity
                logger.info(f"Detected opportunity in mempool (score: {analysis['opportunity_score']:.1f}): {', '.join(analysis['patterns_matched'])}")
                
                # Trim opportunities list if needed
                if len(self.detected_opportunities) > self.max_opportunities:
                    self.detected_opportunities = self.detected_opportunities[-self.max_opportunities:]
        
        return opportunities
    
    def _update_metrics(self, scan_count: int):
        """
        Update metrics in state manager
        
        Args:
            scan_count (int): Number of scans performed
        """
        metrics = {
            'scan_count': scan_count,
            'processed_tx_count': len(self.processed_txs),
            'opportunity_count': len(self.detected_opportunities),
            'last_scan': time.time()
        }
        
        # Add recent opportunity metrics if available
        if self.detected_opportunities:
            recent = self.detected_opportunities[-10:]  # Last 10 opportunities
            avg_score = sum(o['opportunity_score'] for o in recent) / len(recent)
            metrics['avg_opportunity_score'] = avg_score
            metrics['latest_opportunity_time'] = recent[-1]['timestamp']
        
        # Update state manager
        self.state_manager.update_component_metrics('mempool_scanner', metrics)
    
    def register_opportunity_callback(self, callback: Callable):
        """
        Register a callback for new opportunities
        
        Args:
            callback (Callable): Callback function that takes a list of opportunities
        """
        self.opportunity_callbacks.add(callback)
    
    def unregister_opportunity_callback(self, callback: Callable):
        """
        Unregister an opportunity callback
        
        Args:
            callback (Callable): Callback function to unregister
        """
        self.opportunity_callbacks.discard(callback)
    
    async def scan_for_opportunities(self) -> List[Dict[str, Any]]:
        """
        Perform an immediate scan for opportunities
        
        Returns:
            List[Dict[str, Any]]: Detected opportunities
        """
        # Get pending transactions
        pending_txs = await self._get_pending_transactions()
        
        # Process transactions
        if pending_txs:
            return await self._process_transactions(pending_txs)
        
        return []
    
    def get_recent_opportunities(self, 
                               max_count: int = 10, 
                               min_score: float = 0) -> List[Dict[str, Any]]:
        """
        Get recent opportunities
        
        Args:
            max_count (int): Maximum number of opportunities to return
            min_score (float): Minimum opportunity score
            
        Returns:
            List[Dict[str, Any]]: Recent opportunities
        """
        # Filter by minimum score
        filtered = [o for o in self.detected_opportunities if o['opportunity_score'] >= min_score]
        
        # Sort by timestamp (newest first)
        sorted_opps = sorted(filtered, key=lambda o: o['timestamp'], reverse=True)
        
        # Limit to max count
        return sorted_opps[:max_count]