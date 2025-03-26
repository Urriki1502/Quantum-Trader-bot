"""
RiskManager Component
Responsible for managing trading risk, position sizing,
and monitoring risk exposure across the portfolio.
"""

import asyncio
import logging
import math
import time
import numpy as np
from typing import Dict, Any, List, Optional, Tuple

from core.state_manager import StateManager
from core.config_manager import ConfigManager

logger = logging.getLogger(__name__)

class RiskManager:
    """
    RiskManager handles:
    - Risk assessment for tokens
    - Position sizing calculations
    - Risk exposure monitoring
    - Stop loss and take profit management
    """
    
    def __init__(self, 
                config_manager: ConfigManager,
                state_manager: StateManager):
        """
        Initialize the RiskManager
        
        Args:
            config_manager (ConfigManager): Configuration manager instance
            state_manager (StateManager): State manager instance
        """
        self.config_manager = config_manager
        self.state_manager = state_manager
        
        # Risk settings
        self.max_position_size_usd = self.config_manager.get('risk.max_position_size_usd', 1000)
        self.max_exposure_percentage = self.config_manager.get('risk.max_exposure_percentage', 5)
        self.stop_loss_percentage = self.config_manager.get('risk.stop_loss_percentage', 5)
        self.take_profit_percentage = self.config_manager.get('risk.take_profit_percentage', 10)
        self.max_slippage_percentage = self.config_manager.get('risk.max_slippage_percentage', 2)
        self.var_confidence_level = self.config_manager.get('risk.var_confidence_level', 0.95)
        
        # Portfolio state
        self.portfolio_value_usd = 0
        self.current_exposure = {}  # token_address -> exposure data
        self.risk_assessments = {}  # token_address -> risk assessment data
        
        # Historical volatility data
        self.volatility_data = {}  # token_address -> historical volatility
        
        logger.info("RiskManager initialized")
    
    async def assess_token_risk(self, token_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Assess the risk of trading a specific token
        
        Args:
            token_data (Dict[str, Any]): Token data
            
        Returns:
            Dict[str, Any]: Risk assessment result
        """
        token_address = token_data.get('address')
        token_symbol = token_data.get('symbol', '')
        
        logger.debug(f"Assessing risk for {token_symbol} ({token_address})")
        
        # Get cached assessment if available and recent (within 5 minutes)
        if token_address in self.risk_assessments:
            cached = self.risk_assessments[token_address]
            if time.time() - cached.get('timestamp', 0) < 300:
                logger.debug(f"Using cached risk assessment for {token_symbol}")
                return cached
        
        # Extract relevant metrics
        liquidity_usd = token_data.get('liquidity_usd', 0)
        contract_verified = token_data.get('contract_verified', False)
        contract_audit = token_data.get('contract_audit', {})
        age_hours = (time.time() - token_data.get('discovery_time', time.time())) / 3600
        
        # Check if we have liquidity data
        if liquidity_usd <= 0:
            assessment = {
                'is_acceptable': False,
                'risk_score': 100,
                'reason': 'No liquidity data available',
                'timestamp': time.time()
            }
            self.risk_assessments[token_address] = assessment
            return assessment
        
        # Check minimum liquidity requirement - updated for live trading
        min_liquidity = self.config_manager.get('strategy.min_liquidity_usd', 10000)
        if liquidity_usd < min_liquidity:
            assessment = {
                'is_acceptable': False,
                'risk_score': 90,
                'reason': f'Insufficient liquidity: ${liquidity_usd:,.2f} < ${min_liquidity:,.2f}',
                'liquidity_usd': liquidity_usd,
                'timestamp': time.time()
            }
            self.risk_assessments[token_address] = assessment
            return assessment
        
        # Calculate base risk score (0-100, higher is riskier)
        risk_score = self._calculate_base_risk_score(token_data)
        
        # Initialize full assessment with all data
        risk_assessment = {
            'is_acceptable': False,  # Default to false, will update if passes all checks
            'risk_score': risk_score,
            'liquidity_usd': liquidity_usd,
            'volatility': token_data.get('atr', 0),
            'contract_verified': contract_verified,
            'age_hours': age_hours,
            'max_position_size': self._calculate_max_position_size(token_data),
            'price_impact_10k': self._estimate_price_impact(liquidity_usd, 10000),
            'price_impact_50k': self._estimate_price_impact(liquidity_usd, 50000),
            'timestamp': time.time(),
            'buy_tax': token_data.get('buy_tax', 0),
            'sell_tax': token_data.get('sell_tax', 0),
            'audited': bool(contract_audit),
            'token_name': token_data.get('name', ''),
            'token_symbol': token_symbol
        }
        
        # Run enhanced risk checks for live trading
        
        # 1. Check risk score against maximum
        max_risk_score = self.config_manager.get('risk.max_risk_score', 70)
        if risk_score > max_risk_score:
            risk_assessment['reason'] = f'Risk score too high: {risk_score} > {max_risk_score}'
            self.risk_assessments[token_address] = risk_assessment
            return risk_assessment
        
        # 2. Check token age - avoid very new tokens
        min_age_hours = self.config_manager.get('risk.min_token_age_hours', 1)
        if age_hours < min_age_hours:
            risk_assessment['reason'] = f'Token too new: {age_hours:.2f}h < {min_age_hours}h minimum'
            self.risk_assessments[token_address] = risk_assessment
            return risk_assessment
        
        # 3. Check for high taxes/fees
        max_tax = self.config_manager.get('risk.max_tax_percentage', 15)
        if token_data.get('sell_tax', 0) > max_tax:
            risk_assessment['reason'] = f'Sell tax too high: {token_data.get("sell_tax")}% > {max_tax}%'
            self.risk_assessments[token_address] = risk_assessment
            return risk_assessment
        
        # 4. Check for honeypot indicators
        if self._check_honeypot_indicators(token_data):
            risk_assessment['reason'] = 'Potential honeypot detected'
            risk_assessment['has_honeypot_indicators'] = True
            self.risk_assessments[token_address] = risk_assessment
            return risk_assessment
        
        # 5. Check for suspicious token naming patterns
        if self._check_suspicious_naming(token_data):
            risk_assessment['reason'] = 'Suspicious token name or symbol'
            self.risk_assessments[token_address] = risk_assessment
            return risk_assessment
        
        # 6. Additional checks for higher value tokens
        if liquidity_usd > 50000:
            # For higher liquidity tokens, require verified contract
            if not contract_verified:
                risk_assessment['reason'] = 'High-value token with unverified contract'
                self.risk_assessments[token_address] = risk_assessment
                return risk_assessment
        
        # All checks passed - token is acceptable
        risk_assessment['is_acceptable'] = True
        
        # Store assessment
        self.risk_assessments[token_address] = risk_assessment
        
        logger.info(f"Token {token_symbol} risk assessment: score={risk_score}, acceptable={risk_assessment['is_acceptable']}")
        return risk_assessment
    
    def _check_honeypot_indicators(self, token_data: Dict[str, Any]) -> bool:
        """
        Check for common honeypot indicators in token data
        
        Args:
            token_data (Dict[str, Any]): Token data
            
        Returns:
            bool: True if honeypot indicators detected
        """
        # Check sell tax
        sell_tax = token_data.get('sell_tax', 0)
        if sell_tax > 20:  # Very suspicious if sell tax > 20%
            logger.warning(f"Token {token_data.get('symbol')} has very high sell tax: {sell_tax}%")
            return True
            
        # Check for suspiciously high buy:sell ratio
        buy_count = token_data.get('buy_count_24h', 0)
        sell_count = token_data.get('sell_count_24h', 0)
        if buy_count > 30 and sell_count < 3 and buy_count/max(1, sell_count) > 10:
            # Many buys but almost no sells is suspicious
            logger.warning(f"Token {token_data.get('symbol')} has suspicious buy/sell ratio: {buy_count} buys, {sell_count} sells")
            return True
            
        # Check contract code for suspicious patterns
        contract_analysis = token_data.get('contract_analysis', {})
        if contract_analysis.get('has_blacklist', False):
            # Blacklist function can be used to prevent certain addresses from selling
            logger.warning(f"Token {token_data.get('symbol')} has blacklist function")
            return True
            
        if contract_analysis.get('has_lockable_transfers', False):
            # Ability to disable transfers is suspicious
            logger.warning(f"Token {token_data.get('symbol')} has lockable transfers")
            return True
            
        if contract_analysis.get('has_hidden_owner', False):
            # Hidden owner functions are suspicious
            logger.warning(f"Token {token_data.get('symbol')} has hidden owner functions")
            return True
            
        if contract_analysis.get('has_proxy_contract', False) and not token_data.get('contract_verified', False):
            # Unverified proxy contracts can hide malicious code
            logger.warning(f"Token {token_data.get('symbol')} has unverified proxy contract")
            return True
            
        # No honeypot indicators found
        return False
        
    def _check_suspicious_naming(self, token_data: Dict[str, Any]) -> bool:
        """
        Check for suspicious patterns in token name and symbol
        
        Args:
            token_data (Dict[str, Any]): Token data
            
        Returns:
            bool: True if suspicious naming detected
        """
        name = token_data.get('name', '').lower()
        symbol = token_data.get('symbol', '').lower()
        
        # List of suspicious words that often appear in scam tokens
        suspicious_words = [
            'elon', 'musk', 'trump', 'biden', 'bezos', 'zuck', 
            'presale', 'pre-sale', 'rugpull', 'squid', 'ponzi',
            'scam', 'binance', 'pump', 'airdrop', 'tesla', 'spacex',
            'free', 'money', 'giveaway', '1000x', 'moon',
        ]
        
        for word in suspicious_words:
            if word in name or word in symbol:
                logger.warning(f"Token name/symbol contains suspicious word '{word}': {token_data.get('name')} ({token_data.get('symbol')})")
                return True
                
        # Check for excessive use of emoji or special characters in name
        emoji_count = sum(1 for c in name if not c.isalnum() and not c.isspace())
        if emoji_count > 3:
            logger.warning(f"Token name has excessive special characters: {token_data.get('name')}")
            return True
            
        # No suspicious patterns found
        return False
    
    def _calculate_base_risk_score(self, token_data: Dict[str, Any]) -> float:
        """
        Calculate base risk score for a token
        
        Args:
            token_data (Dict[str, Any]): Token data
            
        Returns:
            float: Risk score (0-100, higher is riskier)
        """
        # Extract relevant metrics
        liquidity_usd = token_data.get('liquidity_usd', 0)
        volume_24h = token_data.get('volume_24h_usd', 0)
        age_hours = (time.time() - token_data.get('discovery_time', time.time())) / 3600
        atr = token_data.get('atr', 0)  # Volatility
        
        # Calculate risk based on liquidity (higher liquidity = lower risk)
        liquidity_risk = max(0, min(40, 40 * (1 - min(liquidity_usd / 1000000, 1))))
        
        # Calculate risk based on volume (higher volume = lower risk)
        volume_risk = max(0, min(20, 20 * (1 - min(volume_24h / 500000, 1))))
        
        # Calculate risk based on age (newer = higher risk)
        age_risk = max(0, min(15, 15 * (1 - min(age_hours / 168, 1))))  # 168 hours = 1 week
        
        # Calculate risk based on volatility (higher volatility = higher risk)
        volatility_risk = min(25, max(0, 25 * (min(atr * 5, 1) if atr else 0.5)))
        
        # Combine risk factors
        total_risk = liquidity_risk + volume_risk + age_risk + volatility_risk
        
        logger.debug(f"Risk breakdown - Liquidity: {liquidity_risk:.1f}, Volume: {volume_risk:.1f}, Age: {age_risk:.1f}, Volatility: {volatility_risk:.1f}")
        
        return total_risk
    
    def _estimate_price_impact(self, liquidity_usd: float, trade_amount_usd: float) -> float:
        """
        Estimate price impact for a given trade amount and liquidity
        
        Args:
            liquidity_usd (float): Token liquidity in USD
            trade_amount_usd (float): Trade amount in USD
            
        Returns:
            float: Estimated price impact as percentage
        """
        if liquidity_usd <= 0:
            return 100.0
        
        # Simple price impact model based on constant product formula
        # In reality, DEXes use more complex calculations
        impact = (trade_amount_usd / liquidity_usd) * 100
        
        # Cap at 100%
        return min(impact, 100.0)
    
    def _calculate_max_position_size(self, token_data: Dict[str, Any]) -> float:
        """
        Calculate maximum position size for a token based on liquidity
        
        Args:
            token_data (Dict[str, Any]): Token data
            
        Returns:
            float: Maximum position size in USD
        """
        # Extract relevant metrics
        liquidity_usd = token_data.get('liquidity_usd', 0)
        age_hours = (time.time() - token_data.get('discovery_time', time.time())) / 3600
        market_cap = token_data.get('market_cap_usd', 0)
        
        if liquidity_usd <= 0:
            return 0
        
        # Base position size calculation based on liquidity and max slippage
        max_slippage = self.max_slippage_percentage / 100
        base_size_by_liquidity = liquidity_usd * max_slippage
        
        # Additional safety factors for live trading
        
        # 1. Apply liquidity scaling factor (more conservative with lower liquidity)
        # Logarithmic scale: ln(liquidity/1000) ranges from ~0 to ~9 for 1K to 1M liquidity
        liquidity_factor = min(1.0, max(0.1, 0.2 + 0.1 * math.log(max(1000, liquidity_usd) / 1000)))
        
        # 2. Apply token age factor (more conservative with newer tokens)
        # Scale from 0.3 (brand new) to 1.0 (>= 7 days old)
        age_factor = min(1.0, max(0.3, 0.3 + 0.7 * (min(age_hours, 168) / 168)))
        
        # 3. Apply global max position limit
        position_limit = self.config_manager.get('risk.max_position_size_usd', 1000)
        
        # 4. Market cap limit - don't take more than 0.5% of market cap for small caps
        market_cap_limit = float('inf')
        if market_cap > 0:
            market_cap_limit = market_cap * 0.005  # 0.5% of market cap
        
        # Calculate final position size with all factors
        max_size = min(
            base_size_by_liquidity * liquidity_factor * age_factor,
            position_limit,
            market_cap_limit
        )
        
        # Log detailed calculation for debugging
        logger.debug(f"Position size calculation: "
                    f"Base={base_size_by_liquidity:.2f}, "
                    f"LiquidityFactor={liquidity_factor:.2f}, "
                    f"AgeFactor={age_factor:.2f}, "
                    f"Final=${max_size:.2f}")
        
        return max_size
    
    async def calculate_position_size(self, token_data: Dict[str, Any]) -> float:
        """
        Calculate optimal position size for a token
        
        Args:
            token_data (Dict[str, Any]): Token data
            
        Returns:
            float: Position size in USD
        """
        token_address = token_data.get('address')
        token_symbol = token_data.get('symbol', '')
        
        logger.debug(f"Calculating position size for {token_symbol} ({token_address})")
        
        # Get or calculate risk assessment
        if token_address in self.risk_assessments:
            risk_assessment = self.risk_assessments[token_address]
        else:
            risk_assessment = await self.assess_token_risk(token_data)
        
        if not risk_assessment['is_acceptable']:
            logger.debug(f"Token {token_symbol} failed risk assessment, position size = 0")
            return 0
        
        # Get max position size
        max_position_size = risk_assessment.get('max_position_size', self.max_position_size_usd)
        
        # Calculate optimal position size based on risk metrics
        risk_score = risk_assessment.get('risk_score', 50)
        
        # Adjust position size based on risk score
        # Higher risk = smaller position
        risk_factor = 1 - (risk_score / 100)
        optimal_size = max_position_size * risk_factor
        
        # Check portfolio exposure
        portfolio_value = await self.get_portfolio_value()
        
        # Calculate maximum allowed exposure based on portfolio value
        max_exposure = portfolio_value * (self.max_exposure_percentage / 100)
        
        # Limit position to max exposure
        if optimal_size > max_exposure:
            logger.debug(f"Position size limited by max exposure: ${optimal_size} -> ${max_exposure}")
            optimal_size = max_exposure
        
        # Apply minimum position size
        min_position_size = self.config_manager.get('risk.min_position_size_usd', 50)
        if optimal_size < min_position_size:
            logger.debug(f"Position size below minimum (${min_position_size}), not trading")
            return 0
        
        logger.info(f"Calculated position size for {token_symbol}: ${optimal_size:.2f}")
        return optimal_size
    
    async def get_portfolio_value(self) -> float:
        """
        Get current portfolio value
        
        Returns:
            float: Portfolio value in USD
        """
        # In a real implementation, this would query wallet balances
        # For now, use a fixed value from config or default
        portfolio_value = self.config_manager.get('portfolio.initial_value_usd', 10000)
        
        # Add active positions
        active_positions_value = sum(self.current_exposure.values())
        
        self.portfolio_value_usd = portfolio_value + active_positions_value
        return self.portfolio_value_usd
    
    async def update_exposure(self, token_address: str, exposure_usd: float):
        """
        Update current exposure for a token
        
        Args:
            token_address (str): Token address
            exposure_usd (float): Exposure in USD
        """
        logger.debug(f"Updating exposure for {token_address}: ${exposure_usd}")
        
        if exposure_usd <= 0:
            # Remove exposure if zero or negative
            if token_address in self.current_exposure:
                del self.current_exposure[token_address]
        else:
            # Update exposure
            self.current_exposure[token_address] = exposure_usd
        
        # Update total exposure metric
        total_exposure = sum(self.current_exposure.values())
        self.state_manager.update_component_metric(
            'risk_manager', 
            'total_exposure_usd', 
            total_exposure
        )
        
        # Update portfolio value
        await self.get_portfolio_value()
    
    async def calculate_var(self, 
                          token_data: Dict[str, Any], 
                          position_size: float,
                          horizon_days: int = 1) -> float:
        """
        Calculate Value at Risk (VaR) for a position
        
        Args:
            token_data (Dict[str, Any]): Token data
            position_size (float): Position size in USD
            horizon_days (int): Time horizon in days
            
        Returns:
            float: Value at Risk in USD
        """
        token_address = token_data.get('address')
        
        # Get historical volatility (annualized)
        volatility = token_data.get('atr', 0)
        
        if not volatility or volatility <= 0:
            # Default to high volatility if unknown
            volatility = 2.0
        
        # Convert to daily volatility
        daily_volatility = volatility / math.sqrt(365)
        
        # Adjust for time horizon
        period_volatility = daily_volatility * math.sqrt(horizon_days)
        
        # Get confidence factor (z-score)
        # 95% confidence = 1.645, 99% confidence = 2.326
        z_score = 1.645  # For 95% confidence
        if self.var_confidence_level > 0.95:
            z_score = 2.326  # For 99% confidence
        
        # Calculate VaR
        var = position_size * z_score * period_volatility
        
        logger.debug(f"VaR for {token_address}: ${var:.2f} (${position_size} position, {horizon_days} day horizon, {self.var_confidence_level*100}% confidence)")
        
        return var
    
    async def check_circuit_breaker(self, token_data: Dict[str, Any]) -> bool:
        """
        Check if circuit breaker should be triggered for a token
        
        Args:
            token_data (Dict[str, Any]): Token data
            
        Returns:
            bool: True if circuit breaker should be triggered, False otherwise
        """
        token_address = token_data.get('address')
        token_symbol = token_data.get('symbol', '')
        
        # Check for extreme price movement
        price_change = token_data.get('price_change_24h', 0)
        
        # Get circuit breaker threshold from config
        threshold = self.config_manager.get('risk.circuit_breaker_threshold', 50)
        
        if abs(price_change) > threshold:
            logger.warning(f"Circuit breaker triggered for {token_symbol}: price change {price_change:.2f}% exceeds threshold {threshold}%")
            
            # Create alert
            self.state_manager.create_alert(
                'risk_manager', 
                'WARNING', 
                f"Circuit breaker triggered for {token_symbol} due to {price_change:.2f}% price movement"
            )
            
            return True
        
        return False
    
    def get_risk_assessment(self, token_address: str) -> Optional[Dict[str, Any]]:
        """
        Get stored risk assessment for a token
        
        Args:
            token_address (str): Token address
            
        Returns:
            Optional[Dict[str, Any]]: Risk assessment or None if not found
        """
        return self.risk_assessments.get(token_address)
    
    def get_all_risk_assessments(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all stored risk assessments
        
        Returns:
            Dict[str, Dict[str, Any]]: All risk assessments
        """
        return self.risk_assessments
