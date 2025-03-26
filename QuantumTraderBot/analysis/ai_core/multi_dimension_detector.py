"""
Multi-Dimension Token Detector Component
Responsible for comprehensive token analysis across multiple dimensions
to identify high-potential memecoin opportunities.
"""

import asyncio
import logging
import time
from typing import Dict, List, Any, Optional, Tuple

from core.state_manager import StateManager
from core.config_manager import ConfigManager
from analysis.ai_core.token_analyzer import AdvancedTokenAnalyzer

logger = logging.getLogger(__name__)

class MultiDimensionDetector:
    """
    MultiDimensionDetector analyzes tokens across multiple dimensions:
    - On-chain metrics
    - Social media signals
    - Market patterns
    - Whale activity
    - Liquidity dynamics
    """
    
    def __init__(self, 
                config_manager: ConfigManager,
                state_manager: StateManager,
                token_analyzer: AdvancedTokenAnalyzer):
        """
        Initialize the MultiDimensionDetector
        
        Args:
            config_manager (ConfigManager): Configuration manager instance
            state_manager (StateManager): State manager instance
            token_analyzer (AdvancedTokenAnalyzer): Advanced token analyzer instance
        """
        self.config_manager = config_manager
        self.state_manager = state_manager
        self.token_analyzer = token_analyzer
        
        # Detection thresholds
        self.opportunity_threshold = self.config_manager.get(
            'ai_core.opportunity_threshold', 70)
        
        # Initialize analyzers
        self.on_chain_analyzer = OnChainAnalyzer(config_manager)
        self.social_analyzer = SocialMediaAnalyzer(config_manager)
        self.market_analyzer = MarketPatternAnalyzer(config_manager)
        self.whale_tracker = WhaleTracker(config_manager)
        
        # Weighting for different dimensions
        self.dimension_weights = {
            'on_chain': 0.3,
            'social': 0.25,
            'market': 0.25,
            'whale': 0.2
        }
        
        # Cache for opportunity scores
        self.opportunity_cache = {}
        self.cache_ttl = 120  # 2 minutes cache for fast-moving markets
        
        logger.info("MultiDimensionDetector initialized")
    
    async def evaluate_token(self, token_address: str, token_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform multi-dimensional evaluation of a token
        
        Args:
            token_address (str): Token address
            token_data (Dict[str, Any]): Token data
            
        Returns:
            Dict[str, Any]: Comprehensive evaluation results
        """
        # Check cache first
        if token_address in self.opportunity_cache:
            cache_entry = self.opportunity_cache[token_address]
            if time.time() - cache_entry['timestamp'] < self.cache_ttl:
                logger.debug(f"Using cached evaluation for {token_address}")
                return cache_entry['result']
        
        try:
            # Start all analysis tasks concurrently
            on_chain_task = asyncio.create_task(
                self.on_chain_analyzer.analyze(token_address, token_data))
            
            social_task = asyncio.create_task(
                self.social_analyzer.analyze(token_address, token_data))
            
            market_task = asyncio.create_task(
                self.market_analyzer.analyze(token_address, token_data))
            
            whale_task = asyncio.create_task(
                self.whale_tracker.analyze(token_address, token_data))
            
            # Wait for all tasks to complete
            on_chain_result = await on_chain_task
            social_result = await social_task
            market_result = await market_task
            whale_result = await whale_task
            
            # Also get ML-based token analysis
            ml_analysis = await self.token_analyzer.analyze_token(token_data)
            
            # Combine results
            results = {
                'token_address': token_address,
                'timestamp': time.time(),
                'dimensions': {
                    'on_chain': on_chain_result,
                    'social': social_result,
                    'market': market_result,
                    'whale': whale_result,
                    'ml_analysis': ml_analysis
                }
            }
            
            # Calculate composite score
            composite_score = self._calculate_composite_score(results)
            results['composite_score'] = composite_score
            
            # Add opportunity classification
            results['is_opportunity'] = composite_score >= self.opportunity_threshold
            
            # Generate actionable insights
            results['insights'] = self._generate_insights(results)
            
            # Cache the results
            self.opportunity_cache[token_address] = {
                'timestamp': time.time(),
                'result': results
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Error evaluating token {token_address}: {str(e)}")
            return {
                'token_address': token_address,
                'error': str(e),
                'is_opportunity': False,
                'composite_score': 0
            }
    
    def _calculate_composite_score(self, results: Dict[str, Any]) -> float:
        """
        Calculate composite opportunity score from all dimensions
        
        Args:
            results (Dict[str, Any]): Multi-dimensional analysis results
            
        Returns:
            float: Composite opportunity score (0-100)
        """
        dimensions = results['dimensions']
        
        # Extract scores from each dimension
        scores = {}
        for dim_name, dim_result in dimensions.items():
            if dim_name == 'ml_analysis':
                # ML analysis has its own composite score
                scores[dim_name] = dim_result.get('composite_score', 0)
            else:
                # Other dimensions provide a score directly
                scores[dim_name] = dim_result.get('score', 0)
        
        # Calculate weighted average
        weighted_sum = 0
        weight_sum = 0
        
        for dim_name, score in scores.items():
            if dim_name in self.dimension_weights:
                weight = self.dimension_weights[dim_name]
                weighted_sum += score * weight
                weight_sum += weight
        
        # Add ML analysis with a higher weight for sophisticated predictions
        ml_weight = 0.4  # Higher weight for ML analysis
        weighted_sum += scores.get('ml_analysis', 0) * ml_weight
        weight_sum += ml_weight
        
        # Normalize
        if weight_sum > 0:
            composite_score = weighted_sum / weight_sum
        else:
            composite_score = 0
            
        return min(100, max(0, composite_score))
    
    def _generate_insights(self, results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate actionable insights from analysis results
        
        Args:
            results (Dict[str, Any]): Analysis results
            
        Returns:
            List[Dict[str, Any]]: Actionable insights
        """
        insights = []
        dimensions = results['dimensions']
        composite_score = results['composite_score']
        
        # High opportunity insight
        if composite_score >= self.opportunity_threshold:
            insights.append({
                'type': 'opportunity',
                'priority': 'high',
                'message': f"High-potential token detected with score {composite_score:.1f}",
                'details': f"This token shows strong signals across multiple dimensions"
            })
        
        # On-chain insights
        on_chain = dimensions.get('on_chain', {})
        if on_chain.get('contract_risk', 100) > 70:
            insights.append({
                'type': 'risk',
                'priority': 'high',
                'message': "High contract risk detected",
                'details': on_chain.get('contract_risk_details', "Contract may have concerning code patterns")
            })
        
        # Social insights
        social = dimensions.get('social', {})
        if social.get('score', 0) > 80:
            insights.append({
                'type': 'trend',
                'priority': 'medium',
                'message': "Strong social media momentum",
                'details': social.get('trend_details', "Token is gaining significant attention on social platforms")
            })
        
        # Market insights
        market = dimensions.get('market', {})
        if market.get('volatility', 0) > 70:
            insights.append({
                'type': 'volatility',
                'priority': 'medium',
                'message': "High market volatility detected",
                'details': "Consider smaller position size and tighter stop loss"
            })
        
        # Whale insights
        whale = dimensions.get('whale', {})
        if whale.get('recent_activity', False):
            insights.append({
                'type': 'whale',
                'priority': 'high',
                'message': "Recent whale activity detected",
                'details': whale.get('activity_details', "Large addresses are accumulating or selling")
            })
        
        # ML-based insights
        ml = dimensions.get('ml_analysis', {})
        if 'recommendation' in ml:
            rec = ml['recommendation']
            if rec.get('action') in ['strong_buy', 'buy']:
                insights.append({
                    'type': 'ai_signal',
                    'priority': 'high',
                    'message': f"AI model recommends {rec.get('action')} with {rec.get('confidence', 0):.1f}% confidence",
                    'details': rec.get('reasoning', "Favorable risk-reward profile detected")
                })
        
        return insights


class OnChainAnalyzer:
    """Analyzer for on-chain metrics and patterns"""
    
    def __init__(self, config_manager: ConfigManager):
        """Initialize the OnChainAnalyzer"""
        self.config_manager = config_manager
    
    async def analyze(self, token_address: str, token_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze on-chain metrics for a token
        
        Args:
            token_address (str): Token address
            token_data (Dict[str, Any]): Token data
            
        Returns:
            Dict[str, Any]: On-chain analysis results
        """
        # In a real implementation, this would perform in-depth on-chain analysis
        # For now, extract relevant data from token_data
        
        result = {
            'token_address': token_address,
            'timestamp': time.time(),
            'score': 0,  # Default score
            'metrics': {}
        }
        
        # Contract verification status
        contract_verified = token_data.get('contract_verified', False)
        result['metrics']['contract_verified'] = contract_verified
        
        # Contract audit data
        contract_audit = token_data.get('contract_audit', {})
        audit_score = contract_audit.get('score', 0)
        result['metrics']['audit_score'] = audit_score
        
        # Calculate contract risk
        if not contract_verified:
            contract_risk = 90  # High risk for unverified contracts
            contract_risk_details = "Contract is not verified, proceed with extreme caution"
        elif audit_score < 30:
            contract_risk = 70  # Medium-high risk for low audit scores
            contract_risk_details = "Contract has a low audit score, review carefully"
        elif audit_score < 70:
            contract_risk = 40  # Medium risk for moderate audit scores
            contract_risk_details = "Contract has a moderate audit score"
        else:
            contract_risk = 20  # Lower risk for high audit scores
            contract_risk_details = "Contract has a favorable audit score"
        
        result['contract_risk'] = contract_risk
        result['contract_risk_details'] = contract_risk_details
        
        # Calculate overall score (higher is better)
        # Invert contract risk (higher contract risk = lower score)
        contract_score = 100 - contract_risk
        
        # Include liquidity and age in scoring
        liquidity = token_data.get('liquidity_usd', 0)
        liquidity_score = min(100, max(0, liquidity / 10000))  # Scale liquidity
        
        # Token age can indicate legitimacy
        age_days = 0
        if 'first_seen_at' in token_data:
            first_seen = token_data['first_seen_at']
            if isinstance(first_seen, (int, float)):
                age_days = (time.time() - first_seen) / (24 * 3600)
            
        age_score = min(100, max(0, age_days * 10))  # 10 points per day, max 100
        
        # Combine scores with weights
        result['score'] = (contract_score * 0.5 + 
                          liquidity_score * 0.3 + 
                          age_score * 0.2)
        
        return result


class SocialMediaAnalyzer:
    """Analyzer for social media signals related to tokens"""
    
    def __init__(self, config_manager: ConfigManager):
        """Initialize the SocialMediaAnalyzer"""
        self.config_manager = config_manager
    
    async def analyze(self, token_address: str, token_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze social media signals for a token
        
        Args:
            token_address (str): Token address
            token_data (Dict[str, Any]): Token data
            
        Returns:
            Dict[str, Any]: Social media analysis results
        """
        # In a real implementation, this would analyze social media data
        # For demonstration, use simulated data
        
        # Social metrics that might be in token_data in a real system
        social_data = token_data.get('social_metrics', {})
        
        # Extract or simulate metrics
        mentions = social_data.get('mentions_count', 0)
        sentiment = social_data.get('sentiment_score', 50)  # 0-100, higher is more positive
        trend_change = social_data.get('trend_change_24h', 0)  # Percentage change in mentions
        
        # If no social data, use token age as a proxy for a simulated score
        if not social_data and 'first_seen_at' in token_data:
            first_seen = token_data['first_seen_at']
            if isinstance(first_seen, (int, float)):
                age_hours = (time.time() - first_seen) / 3600
                # Newer tokens might have more social buzz
                simulated_score = max(0, 100 - age_hours)  # Higher score for newer tokens
            else:
                simulated_score = 50  # Default
        else:
            simulated_score = 50  # Default
        
        # Calculate social score
        if mentions > 0:
            # Use actual metrics if available
            social_score = (mentions * 0.4 + 
                           sentiment * 0.3 + 
                           max(0, trend_change) * 0.3)
            social_score = min(100, social_score)
        else:
            # Use simulated score
            social_score = simulated_score
        
        result = {
            'token_address': token_address,
            'timestamp': time.time(),
            'score': social_score,
            'metrics': {
                'mentions': mentions,
                'sentiment': sentiment,
                'trend_change': trend_change
            }
        }
        
        # Add trend details if significant
        if trend_change > 50:
            result['trend_details'] = f"Token mentions increased by {trend_change:.1f}% in 24h"
        elif social_score > 70:
            result['trend_details'] = "Token has strong social media presence"
        
        return result


class MarketPatternAnalyzer:
    """Analyzer for market patterns and technical indicators"""
    
    def __init__(self, config_manager: ConfigManager):
        """Initialize the MarketPatternAnalyzer"""
        self.config_manager = config_manager
    
    async def analyze(self, token_address: str, token_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze market patterns for a token
        
        Args:
            token_address (str): Token address
            token_data (Dict[str, Any]): Token data
            
        Returns:
            Dict[str, Any]: Market pattern analysis results
        """
        # Extract market data
        price = token_data.get('price_usd', 0)
        volume_24h = token_data.get('volume_24h_usd', 0)
        market_cap = token_data.get('market_cap_usd', 0)
        liquidity = token_data.get('liquidity_usd', 0)
        
        # Calculate derived metrics
        if market_cap > 0:
            volume_to_mcap = volume_24h / market_cap
        else:
            volume_to_mcap = 0
            
        if liquidity > 0:
            mcap_to_liquidity = market_cap / liquidity
        else:
            mcap_to_liquidity = 0
        
        # Volatility estimation
        price_change_24h = token_data.get('price_change_24h', 0)
        volatility = abs(price_change_24h)  # Simple volatility measure
        
        # Market trend detection
        price_change_1h = token_data.get('price_change_1h', 0)
        price_change_4h = token_data.get('price_change_4h', 0)
        
        if price_change_1h > 5 and price_change_4h > 10 and price_change_24h > 20:
            trend = "strong_uptrend"
        elif price_change_1h > 2 and price_change_4h > 5 and price_change_24h > 10:
            trend = "uptrend"
        elif price_change_1h < -5 and price_change_4h < -10 and price_change_24h < -20:
            trend = "strong_downtrend"
        elif price_change_1h < -2 and price_change_4h < -5 and price_change_24h < -10:
            trend = "downtrend"
        else:
            trend = "sideways"
        
        # Calculate market pattern score
        base_score = 50  # Neutral starting point
        
        # Volume component (higher volume to mcap is better)
        volume_score = min(50, volume_to_mcap * 1000)
        
        # Trend component
        if trend == "strong_uptrend":
            trend_score = 40
        elif trend == "uptrend":
            trend_score = 20
        elif trend == "sideways":
            trend_score = 0
        elif trend == "downtrend":
            trend_score = -20
        else:  # strong_downtrend
            trend_score = -40
        
        # Combine scores
        market_score = base_score + volume_score + trend_score
        market_score = min(100, max(0, market_score))
        
        return {
            'token_address': token_address,
            'timestamp': time.time(),
            'score': market_score,
            'trend': trend,
            'volatility': volatility,
            'metrics': {
                'price_usd': price,
                'volume_24h_usd': volume_24h,
                'volume_to_mcap': volume_to_mcap,
                'mcap_to_liquidity': mcap_to_liquidity,
                'price_change_1h': price_change_1h,
                'price_change_4h': price_change_4h,
                'price_change_24h': price_change_24h
            }
        }


class WhaleTracker:
    """Tracker for whale activity related to tokens"""
    
    def __init__(self, config_manager: ConfigManager):
        """Initialize the WhaleTracker"""
        self.config_manager = config_manager
    
    async def analyze(self, token_address: str, token_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze whale activity for a token
        
        Args:
            token_address (str): Token address
            token_data (Dict[str, Any]): Token data
            
        Returns:
            Dict[str, Any]: Whale activity analysis results
        """
        # In a real implementation, this would analyze on-chain whale transactions
        # For demonstration, use holder data if available
        
        holders_data = token_data.get('holders', {})
        top10_percent = holders_data.get('top10_percent', 0)
        unique_holders = holders_data.get('count', 0)
        
        # Detect recent whale activity (in real system, would check recent large txs)
        # For demo, use a proxy based on token age and holder concentration
        recent_activity = False
        activity_details = ""
        
        if 'first_seen_at' in token_data:
            first_seen = token_data['first_seen_at']
            if isinstance(first_seen, (int, float)):
                age_hours = (time.time() - first_seen) / 3600
                # Newer tokens with high concentration might indicate whale activity
                if age_hours < 48 and top10_percent > 70:
                    recent_activity = True
                    activity_details = "New token with high holder concentration, possible whale accumulation"
        
        # Concentration risk
        if top10_percent > 90:
            concentration_risk = "extreme"
            concentration_details = "Extreme holder concentration, high manipulation risk"
        elif top10_percent > 80:
            concentration_risk = "high"
            concentration_details = "High holder concentration, potential manipulation risk"
        elif top10_percent > 60:
            concentration_risk = "medium"
            concentration_details = "Moderate holder concentration"
        else:
            concentration_risk = "low"
            concentration_details = "Well-distributed token ownership"
        
        # Calculate whale score (lower is better due to risks)
        whale_risk = top10_percent  # Higher concentration = higher risk
        
        # Adjust for holder count (more holders = lower risk)
        if unique_holders > 1000:
            holder_adjustment = 30
        elif unique_holders > 500:
            holder_adjustment = 20
        elif unique_holders > 100:
            holder_adjustment = 10
        else:
            holder_adjustment = 0
            
        # Final whale risk score
        whale_risk = max(0, whale_risk - holder_adjustment)
        
        # Convert to whale score (higher is better for opportunity)
        whale_score = 100 - whale_risk
        
        return {
            'token_address': token_address,
            'timestamp': time.time(),
            'score': whale_score,
            'recent_activity': recent_activity,
            'activity_details': activity_details,
            'concentration_risk': concentration_risk,
            'concentration_details': concentration_details,
            'metrics': {
                'top10_percent': top10_percent,
                'unique_holders': unique_holders
            }
        }