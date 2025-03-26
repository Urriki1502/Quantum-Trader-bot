"""
Transaction Analysis Service Component
Analyzes Solana transactions to extract insights, monitor execution,
and improve future trading performance.
"""

import asyncio
import logging
import time
import base64
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field

from core.state_manager import StateManager
from core.config_manager import ConfigManager
from utils.performance_optimizer import PerformanceOptimizer
from utils.connection_pool import ConnectionPool

logger = logging.getLogger(__name__)

@dataclass
class TransactionInsights:
    """Extracted insights from transaction analysis"""
    
    # Transaction metadata
    transaction_id: str
    timestamp: float
    block_number: Optional[int] = None
    block_timestamp: Optional[float] = None
    
    # Transaction details
    success: bool = False
    num_instructions: int = 0
    fee_paid: float = 0
    execution_time_ms: Optional[float] = None
    
    # Extracted data
    token_addresses: List[str] = field(default_factory=list)
    token_amounts: Dict[str, float] = field(default_factory=dict)
    token_prices: Dict[str, float] = field(default_factory=dict)
    
    # Trading metrics
    slippage: Optional[float] = None
    price_impact: Optional[float] = None
    effective_amount: Optional[float] = None
    
    # Performance metrics
    confirmation_time: Optional[float] = None
    priority_fee: Optional[float] = None
    gas_used: Optional[float] = None
    
    # Warnings and issues
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary
        
        Returns:
            Dict[str, Any]: Dictionary representation
        """
        return {
            'transaction_id': self.transaction_id,
            'timestamp': self.timestamp,
            'block_number': self.block_number,
            'block_timestamp': self.block_timestamp,
            'success': self.success,
            'num_instructions': self.num_instructions,
            'fee_paid': self.fee_paid,
            'execution_time_ms': self.execution_time_ms,
            'token_addresses': self.token_addresses,
            'token_amounts': self.token_amounts,
            'token_prices': self.token_prices,
            'slippage': self.slippage,
            'price_impact': self.price_impact,
            'effective_amount': self.effective_amount,
            'confirmation_time': self.confirmation_time,
            'priority_fee': self.priority_fee,
            'gas_used': self.gas_used,
            'warnings': self.warnings,
            'errors': self.errors
        }


class InstructionParser:
    """Parses transaction instructions to extract detailed information"""
    
    def __init__(self, config_manager: ConfigManager):
        """
        Initialize instruction parser
        
        Args:
            config_manager (ConfigManager): Configuration manager instance
        """
        self.config_manager = config_manager
        
        # Initialize instruction handlers
        self.program_handlers = {
            'TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA': self._handle_token_program,
            '675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8': self._handle_raydium_program,
            '11111111111111111111111111111111': self._handle_system_program
        }
    
    async def parse_instructions(self, 
                               instructions: List[Dict[str, Any]],
                               accounts: List[str]) -> Dict[str, Any]:
        """
        Parse transaction instructions
        
        Args:
            instructions (List[Dict[str, Any]]): Transaction instructions
            accounts (List[str]): Transaction accounts
            
        Returns:
            Dict[str, Any]: Parsed instruction data
        """
        parsed_data = {
            'num_instructions': len(instructions),
            'token_addresses': set(),
            'token_amounts': {},
            'token_prices': {},
            'operations': [],
            'unknown_programs': set()
        }
        
        # Parse each instruction
        for idx, instruction in enumerate(instructions):
            program_id = instruction.get('programId')
            
            if program_id in self.program_handlers:
                # Use specific handler
                handler = self.program_handlers[program_id]
                instruction_data = await handler(instruction, accounts, idx)
                
                # Add to operations
                if instruction_data:
                    parsed_data['operations'].append(instruction_data)
                    
                    # Extract token addresses
                    if 'token_address' in instruction_data:
                        parsed_data['token_addresses'].add(instruction_data['token_address'])
                    
                    # Extract token amounts
                    if 'token_address' in instruction_data and 'amount' in instruction_data:
                        token = instruction_data['token_address']
                        amount = instruction_data['amount']
                        
                        if token in parsed_data['token_amounts']:
                            parsed_data['token_amounts'][token] += amount
                        else:
                            parsed_data['token_amounts'][token] = amount
                    
                    # Extract token prices
                    if 'token_address' in instruction_data and 'price' in instruction_data:
                        token = instruction_data['token_address']
                        price = instruction_data['price']
                        parsed_data['token_prices'][token] = price
            else:
                # Unknown program
                parsed_data['unknown_programs'].add(program_id)
        
        # Convert sets to lists for serialization
        parsed_data['token_addresses'] = list(parsed_data['token_addresses'])
        parsed_data['unknown_programs'] = list(parsed_data['unknown_programs'])
        
        return parsed_data
    
    async def _handle_token_program(self, 
                                  instruction: Dict[str, Any],
                                  accounts: List[str],
                                  idx: int) -> Optional[Dict[str, Any]]:
        """
        Handle token program instruction
        
        Args:
            instruction (Dict[str, Any]): Instruction data
            accounts (List[str]): Transaction accounts
            idx (int): Instruction index
            
        Returns:
            Optional[Dict[str, Any]]: Parsed instruction data
        """
        data = instruction.get('data')
        account_indices = instruction.get('accounts', [])
        
        # Convert account indices to addresses
        account_addresses = [accounts[i] for i in account_indices if i < len(accounts)]
        
        if not data:
            return None
        
        # Decode instruction data
        try:
            data_bytes = base64.b64decode(data)
            instruction_type = data_bytes[0] if data_bytes else None
            
            # Parse based on instruction type
            if instruction_type == 3:  # Transfer
                return {
                    'type': 'token_transfer',
                    'program': 'token_program',
                    'token_address': account_addresses[0] if len(account_addresses) > 0 else None,
                    'from_address': account_addresses[1] if len(account_addresses) > 1 else None,
                    'to_address': account_addresses[2] if len(account_addresses) > 2 else None,
                    'amount': int.from_bytes(data_bytes[1:9], byteorder='little'),
                    'instruction_index': idx
                }
            elif instruction_type == 7:  # MintTo
                return {
                    'type': 'mint_to',
                    'program': 'token_program',
                    'token_address': account_addresses[0] if len(account_addresses) > 0 else None,
                    'to_address': account_addresses[1] if len(account_addresses) > 1 else None,
                    'amount': int.from_bytes(data_bytes[1:9], byteorder='little'),
                    'instruction_index': idx
                }
            elif instruction_type == 8:  # Burn
                return {
                    'type': 'burn',
                    'program': 'token_program',
                    'token_address': account_addresses[0] if len(account_addresses) > 0 else None,
                    'from_address': account_addresses[1] if len(account_addresses) > 1 else None,
                    'amount': int.from_bytes(data_bytes[1:9], byteorder='little'),
                    'instruction_index': idx
                }
            else:
                return {
                    'type': f'token_instruction_{instruction_type}',
                    'program': 'token_program',
                    'data': data,
                    'accounts': account_addresses,
                    'instruction_index': idx
                }
                
        except Exception as e:
            logger.error(f"Error parsing token program instruction: {str(e)}")
            return {
                'type': 'unknown_token_instruction',
                'program': 'token_program',
                'data': data,
                'accounts': account_addresses,
                'instruction_index': idx,
                'error': str(e)
            }
    
    async def _handle_raydium_program(self, 
                                    instruction: Dict[str, Any],
                                    accounts: List[str],
                                    idx: int) -> Optional[Dict[str, Any]]:
        """
        Handle Raydium program instruction
        
        Args:
            instruction (Dict[str, Any]): Instruction data
            accounts (List[str]): Transaction accounts
            idx (int): Instruction index
            
        Returns:
            Optional[Dict[str, Any]]: Parsed instruction data
        """
        data = instruction.get('data')
        account_indices = instruction.get('accounts', [])
        
        # Convert account indices to addresses
        account_addresses = [accounts[i] for i in account_indices if i < len(accounts)]
        
        if not data:
            return None
        
        # Decode instruction data
        try:
            data_bytes = base64.b64decode(data)
            instruction_type = data_bytes[0] if data_bytes else None
            
            # Parse based on instruction type
            if instruction_type == 1:  # Swap
                # Extract token addresses
                token_a = account_addresses[4] if len(account_addresses) > 4 else None
                token_b = account_addresses[7] if len(account_addresses) > 7 else None
                
                # Attempt to extract amounts from data
                amount_in = int.from_bytes(data_bytes[1:9], byteorder='little') if len(data_bytes) >= 9 else 0
                
                return {
                    'type': 'swap',
                    'program': 'raydium_program',
                    'token_address_in': token_a,
                    'token_address_out': token_b,
                    'amount': amount_in,
                    'accounts': account_addresses,
                    'instruction_index': idx
                }
            elif instruction_type == 2:  # Add Liquidity
                return {
                    'type': 'add_liquidity',
                    'program': 'raydium_program',
                    'pool': account_addresses[2] if len(account_addresses) > 2 else None,
                    'token_a': account_addresses[5] if len(account_addresses) > 5 else None,
                    'token_b': account_addresses[8] if len(account_addresses) > 8 else None,
                    'instruction_index': idx
                }
            elif instruction_type == 3:  # Remove Liquidity
                return {
                    'type': 'remove_liquidity',
                    'program': 'raydium_program',
                    'pool': account_addresses[2] if len(account_addresses) > 2 else None,
                    'token_a': account_addresses[5] if len(account_addresses) > 5 else None,
                    'token_b': account_addresses[8] if len(account_addresses) > 8 else None,
                    'instruction_index': idx
                }
            else:
                return {
                    'type': f'raydium_instruction_{instruction_type}',
                    'program': 'raydium_program',
                    'data': data,
                    'accounts': account_addresses,
                    'instruction_index': idx
                }
                
        except Exception as e:
            logger.error(f"Error parsing Raydium program instruction: {str(e)}")
            return {
                'type': 'unknown_raydium_instruction',
                'program': 'raydium_program',
                'data': data,
                'accounts': account_addresses,
                'instruction_index': idx,
                'error': str(e)
            }
    
    async def _handle_system_program(self, 
                                   instruction: Dict[str, Any],
                                   accounts: List[str],
                                   idx: int) -> Optional[Dict[str, Any]]:
        """
        Handle system program instruction
        
        Args:
            instruction (Dict[str, Any]): Instruction data
            accounts (List[str]): Transaction accounts
            idx (int): Instruction index
            
        Returns:
            Optional[Dict[str, Any]]: Parsed instruction data
        """
        data = instruction.get('data')
        account_indices = instruction.get('accounts', [])
        
        # Convert account indices to addresses
        account_addresses = [accounts[i] for i in account_indices if i < len(accounts)]
        
        if not data:
            return None
        
        # Decode instruction data
        try:
            data_bytes = base64.b64decode(data)
            instruction_type = int.from_bytes(data_bytes[:4], byteorder='little') if len(data_bytes) >= 4 else None
            
            # Parse based on instruction type
            if instruction_type == 2:  # Transfer
                amount = int.from_bytes(data_bytes[4:12], byteorder='little') if len(data_bytes) >= 12 else 0
                
                return {
                    'type': 'sol_transfer',
                    'program': 'system_program',
                    'from_address': account_addresses[0] if len(account_addresses) > 0 else None,
                    'to_address': account_addresses[1] if len(account_addresses) > 1 else None,
                    'amount': amount,
                    'instruction_index': idx
                }
            else:
                return {
                    'type': f'system_instruction_{instruction_type}',
                    'program': 'system_program',
                    'data': data,
                    'accounts': account_addresses,
                    'instruction_index': idx
                }
                
        except Exception as e:
            logger.error(f"Error parsing system program instruction: {str(e)}")
            return {
                'type': 'unknown_system_instruction',
                'program': 'system_program',
                'data': data,
                'accounts': account_addresses,
                'instruction_index': idx,
                'error': str(e)
            }


class TransactionAnalysisService:
    """
    Analyzes Solana transactions to extract insights and improve trading
    
    This component helps optimize trading by:
    - Analyzing transaction execution details
    - Tracking slippage and price impact
    - Monitoring gas costs and confirmation times
    - Detecting transaction failures and reasons
    """
    
    def __init__(self, 
                config_manager: ConfigManager,
                state_manager: StateManager,
                connection_pool: ConnectionPool,
                performance_optimizer: PerformanceOptimizer):
        """
        Initialize transaction analysis service
        
        Args:
            config_manager (ConfigManager): Configuration manager instance
            state_manager (StateManager): State manager instance
            connection_pool (ConnectionPool): Connection pool instance
            performance_optimizer (PerformanceOptimizer): Performance optimizer instance
        """
        self.config_manager = config_manager
        self.state_manager = state_manager
        self.connection_pool = connection_pool
        self.performance_optimizer = performance_optimizer
        
        # Service configuration
        self.max_transaction_age = self.config_manager.get(
            'transaction_analysis.max_transaction_age', 86400)  # 24 hours
        self.max_cached_insights = self.config_manager.get(
            'transaction_analysis.max_cached_insights', 1000)
        
        # Initialize instruction parser
        self.instruction_parser = InstructionParser(config_manager)
        
        # Cached insights
        self.transaction_insights = {}
        
        # Tracked metrics
        self.metrics = {
            'transactions_analyzed': 0,
            'successful_transactions': 0,
            'failed_transactions': 0,
            'avg_confirmation_time': 0.0,
            'avg_slippage': 0.0,
            'avg_price_impact': 0.0,
            'avg_gas_cost': 0.0,
            'avg_priority_fee': 0.0
        }
        
        # Cache for transaction data
        self.transaction_cache = {}
        self.max_cache_size = 100
        
        logger.info("TransactionAnalysisService initialized")
    
    async def analyze_transaction(self, 
                                transaction_id: str,
                                expected_data: Optional[Dict[str, Any]] = None) -> TransactionInsights:
        """
        Analyze a transaction
        
        Args:
            transaction_id (str): Transaction ID (signature)
            expected_data (Dict[str, Any], optional): Expected transaction data
            
        Returns:
            TransactionInsights: Transaction insights
        """
        # Check cache
        if transaction_id in self.transaction_insights:
            return self.transaction_insights[transaction_id]
        
        # Initialize insights
        insights = TransactionInsights(
            transaction_id=transaction_id,
            timestamp=time.time()
        )
        
        try:
            # Fetch transaction data
            transaction_data = await self._fetch_transaction_data(transaction_id)
            
            if not transaction_data:
                insights.errors.append("Failed to fetch transaction data")
                return insights
            
            # Check if transaction was successful
            status = self._extract_transaction_status(transaction_data)
            insights.success = status.get('success', False)
            
            if not insights.success:
                error_msg = status.get('error', 'Unknown error')
                insights.errors.append(f"Transaction failed: {error_msg}")
            
            # Extract basic transaction data
            metadata = self._extract_transaction_metadata(transaction_data)
            insights.block_number = metadata.get('block_number')
            insights.block_timestamp = metadata.get('block_timestamp')
            insights.fee_paid = metadata.get('fee', 0)
            insights.execution_time_ms = metadata.get('execution_time_ms')
            
            # Extract performance metrics
            performance = self._extract_performance_metrics(transaction_data)
            insights.confirmation_time = performance.get('confirmation_time')
            insights.priority_fee = performance.get('priority_fee')
            insights.gas_used = performance.get('gas_used')
            
            # Parse transaction instructions
            if 'transaction' in transaction_data and 'message' in transaction_data['transaction']:
                message = transaction_data['transaction']['message']
                
                # Get accounts and instructions
                accounts = message.get('accountKeys', [])
                instructions = message.get('instructions', [])
                
                # Set number of instructions
                insights.num_instructions = len(instructions)
                
                # Parse instructions
                parsed_data = await self.instruction_parser.parse_instructions(
                    instructions, accounts)
                
                # Extract token addresses
                insights.token_addresses = parsed_data.get('token_addresses', [])
                
                # Extract token amounts
                insights.token_amounts = parsed_data.get('token_amounts', {})
                
                # Extract token prices
                insights.token_prices = parsed_data.get('token_prices', {})
            
            # Compare with expected data if provided
            if expected_data:
                self._compare_with_expected(insights, expected_data)
                
            # Calculate trading metrics
            self._calculate_trading_metrics(insights, expected_data)
            
            # Update metrics
            self._update_metrics(insights)
            
            # Store in cache
            self.transaction_insights[transaction_id] = insights
            
            # Trim cache if needed
            if len(self.transaction_insights) > self.max_cached_insights:
                # Remove oldest entries
                oldest_ids = sorted(
                    self.transaction_insights.keys(),
                    key=lambda tx_id: self.transaction_insights[tx_id].timestamp
                )[:len(self.transaction_insights) - self.max_cached_insights]
                
                for old_id in oldest_ids:
                    del self.transaction_insights[old_id]
            
            logger.info(f"Transaction analysis completed for {transaction_id}")
            return insights
            
        except Exception as e:
            logger.error(f"Error analyzing transaction {transaction_id}: {str(e)}")
            insights.errors.append(f"Analysis error: {str(e)}")
            return insights
    
    async def _fetch_transaction_data(self, transaction_id: str) -> Optional[Dict[str, Any]]:
        """
        Fetch transaction data from RPC
        
        Args:
            transaction_id (str): Transaction ID (signature)
            
        Returns:
            Optional[Dict[str, Any]]: Transaction data or None if not found
        """
        # Check cache
        if transaction_id in self.transaction_cache:
            return self.transaction_cache[transaction_id]
        
        try:
            # Fetch transaction
            response = await self.connection_pool.call(
                "getTransaction",
                [transaction_id, {"encoding": "json", "maxSupportedTransactionVersion": 0}]
            )
            
            if not response or 'result' not in response:
                return None
            
            transaction_data = response['result']
            
            # Store in cache
            self.transaction_cache[transaction_id] = transaction_data
            
            # Trim cache if needed
            if len(self.transaction_cache) > self.max_cache_size:
                # Get oldest key
                oldest_key = next(iter(self.transaction_cache))
                del self.transaction_cache[oldest_key]
            
            return transaction_data
            
        except Exception as e:
            logger.error(f"Error fetching transaction {transaction_id}: {str(e)}")
            return None
    
    def _extract_transaction_status(self, transaction_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract transaction status
        
        Args:
            transaction_data (Dict[str, Any]): Transaction data
            
        Returns:
            Dict[str, Any]: Transaction status
        """
        status = {
            'success': False,
            'error': None
        }
        
        # Check meta field
        if 'meta' in transaction_data:
            meta = transaction_data['meta']
            
            # Check status
            if 'err' not in meta or meta['err'] is None:
                status['success'] = True
            else:
                status['success'] = False
                status['error'] = str(meta['err'])
        
        return status
    
    def _extract_transaction_metadata(self, transaction_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract transaction metadata
        
        Args:
            transaction_data (Dict[str, Any]): Transaction data
            
        Returns:
            Dict[str, Any]: Transaction metadata
        """
        metadata = {
            'block_number': None,
            'block_timestamp': None,
            'fee': 0,
            'execution_time_ms': None
        }
        
        # Extract block information
        metadata['block_number'] = transaction_data.get('slot')
        
        # Extract block time
        if 'blockTime' in transaction_data:
            metadata['block_timestamp'] = transaction_data['blockTime']
        
        # Extract fee
        if 'meta' in transaction_data:
            meta = transaction_data['meta']
            
            # Get fee
            metadata['fee'] = meta.get('fee', 0)
            
            # Get execution time if available
            if 'executionTime' in meta:
                # Convert to milliseconds
                metadata['execution_time_ms'] = meta['executionTime']
        
        return metadata
    
    def _extract_performance_metrics(self, transaction_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract performance metrics
        
        Args:
            transaction_data (Dict[str, Any]): Transaction data
            
        Returns:
            Dict[str, Any]: Performance metrics
        """
        metrics = {
            'confirmation_time': None,
            'priority_fee': None,
            'gas_used': None
        }
        
        # Calculate confirmation time
        if ('blockTime' in transaction_data and 
            'transaction' in transaction_data and 
            'signatures' in transaction_data['transaction']):
            
            # Get block timestamp
            block_time = transaction_data['blockTime']
            
            # Estimate submission time from first signature
            # This is an approximation, as we don't know the exact submission time
            # In a real system, we would track the submission time when sending the transaction
            signatures = transaction_data['transaction']['signatures']
            if signatures and len(signatures) > 0:
                # Estimate submission from signature
                # In practice, we would record this when sending the transaction
                # This is just a placeholder for demonstration
                submission_time = block_time - 2  # Assume 2 seconds confirmation time
                
                # Calculate confirmation time
                metrics['confirmation_time'] = block_time - submission_time
        
        # Extract priority fee
        if 'meta' in transaction_data:
            meta = transaction_data['meta']
            
            # Priority fee may be stored in different ways depending on Solana version
            if 'priorityFee' in meta:
                metrics['priority_fee'] = meta['priorityFee']
            elif 'prioritizationFee' in meta:
                metrics['priority_fee'] = meta['prioritizationFee']
                
            # Extract gas usage if available
            if 'computeUnitsConsumed' in meta:
                metrics['gas_used'] = meta['computeUnitsConsumed']
        
        return metrics
    
    def _compare_with_expected(self, 
                             insights: TransactionInsights, 
                             expected_data: Dict[str, Any]):
        """
        Compare transaction with expected data
        
        Args:
            insights (TransactionInsights): Transaction insights
            expected_data (Dict[str, Any]): Expected transaction data
        """
        # Check if transaction was successful
        if not insights.success:
            insights.warnings.append("Transaction failed but success was expected")
        
        # Check token address
        expected_token = expected_data.get('token_address')
        if expected_token and expected_token not in insights.token_addresses:
            insights.warnings.append(f"Expected token {expected_token} not found in transaction")
        
        # Check amounts
        expected_amount = expected_data.get('amount')
        if expected_amount and expected_token:
            actual_amount = insights.token_amounts.get(expected_token, 0)
            # Calculate difference percentage
            if actual_amount > 0:
                difference_pct = abs(actual_amount - expected_amount) / expected_amount * 100
                
                # Warn if difference is significant
                if difference_pct > 5:
                    insights.warnings.append(
                        f"Amount difference of {difference_pct:.2f}% detected "
                        f"(expected: {expected_amount}, actual: {actual_amount})"
                    )
                    
                # Calculate slippage
                insights.slippage = difference_pct
    
    def _calculate_trading_metrics(self, 
                                 insights: TransactionInsights, 
                                 expected_data: Optional[Dict[str, Any]]):
        """
        Calculate trading metrics
        
        Args:
            insights (TransactionInsights): Transaction insights
            expected_data (Dict[str, Any], optional): Expected transaction data
        """
        # Slippage already calculated in _compare_with_expected
        
        # Estimate price impact (simplified)
        if expected_data and 'price_impact' in expected_data:
            # Use provided price impact
            insights.price_impact = expected_data['price_impact']
        else:
            # Estimate from transaction data (simplified)
            # In a real implementation, would need more context about the pool and trade
            token_prices = insights.token_prices
            if len(token_prices) > 0:
                # Simple placeholder - in practice needs more context
                insights.price_impact = 0.5  # Placeholder value
        
        # Calculate effective amount
        if expected_data and 'amount_usd' in expected_data:
            expected_amount_usd = expected_data['amount_usd']
            
            # If slippage is calculated
            if insights.slippage is not None:
                # Apply slippage to get effective amount
                slippage_factor = insights.slippage / 100
                insights.effective_amount = expected_amount_usd * (1 - slippage_factor)
            else:
                insights.effective_amount = expected_amount_usd
    
    def _update_metrics(self, insights: TransactionInsights):
        """
        Update service metrics
        
        Args:
            insights (TransactionInsights): Transaction insights
        """
        # Update transaction counts
        self.metrics['transactions_analyzed'] += 1
        
        if insights.success:
            self.metrics['successful_transactions'] += 1
        else:
            self.metrics['failed_transactions'] += 1
        
        # Update averages
        self._update_average('avg_confirmation_time', insights.confirmation_time)
        self._update_average('avg_slippage', insights.slippage)
        self._update_average('avg_price_impact', insights.price_impact)
        self._update_average('avg_gas_cost', insights.gas_used)
        self._update_average('avg_priority_fee', insights.priority_fee)
        
        # Update component metrics
        self.state_manager.update_component_metrics(
            'transaction_analysis_service', self.metrics)
    
    def _update_average(self, metric_name: str, value: Optional[float]):
        """
        Update average metric
        
        Args:
            metric_name (str): Metric name
            value (float, optional): New value
        """
        if value is None:
            return
            
        current_avg = self.metrics[metric_name]
        current_count = self.metrics['transactions_analyzed']
        
        if current_count == 1:
            # First value
            self.metrics[metric_name] = value
        else:
            # Weighted average
            self.metrics[metric_name] = (current_avg * 0.95) + (value * 0.05)
    
    async def get_transaction_insights(self, 
                                     transaction_id: str,
                                     force_refresh: bool = False) -> TransactionInsights:
        """
        Get transaction insights
        
        Args:
            transaction_id (str): Transaction ID (signature)
            force_refresh (bool): Whether to force refresh
            
        Returns:
            TransactionInsights: Transaction insights
        """
        if transaction_id in self.transaction_insights and not force_refresh:
            return self.transaction_insights[transaction_id]
        
        return await self.analyze_transaction(transaction_id)
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get service metrics
        
        Returns:
            Dict[str, Any]: Service metrics
        """
        return self.metrics.copy()
    
    def clear_cache(self):
        """Clear transaction cache"""
        self.transaction_cache = {}
        logger.info("Transaction cache cleared")
    
    async def batch_analyze(self, 
                          transaction_ids: List[str]) -> Dict[str, TransactionInsights]:
        """
        Analyze multiple transactions in batch
        
        Args:
            transaction_ids (List[str]): Transaction IDs
            
        Returns:
            Dict[str, TransactionInsights]: Transaction insights by ID
        """
        results = {}
        
        # Process in parallel using performance optimizer
        tasks = []
        
        for tx_id in transaction_ids:
            future = await self.performance_optimizer.enqueue_task(
                task_id=f"analyze_tx_{tx_id}",
                task_func=self.analyze_transaction,
                args=(tx_id,),
                priority='normal'
            )
            tasks.append((tx_id, future))
        
        # Wait for all tasks to complete
        for tx_id, future in tasks:
            try:
                insights = await future
                results[tx_id] = insights
            except Exception as e:
                logger.error(f"Error analyzing transaction {tx_id}: {str(e)}")
                # Create error insights
                insights = TransactionInsights(
                    transaction_id=tx_id,
                    timestamp=time.time()
                )
                insights.errors.append(f"Analysis error: {str(e)}")
                results[tx_id] = insights
        
        return results