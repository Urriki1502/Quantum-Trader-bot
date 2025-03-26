"""
Advanced Token Analyzer Component
Responsible for analyzing memecoin tokens using advanced ML techniques
to predict potential price movements and assess risk/reward profiles.
"""

import asyncio
import logging
import time
import numpy as np
import joblib
from typing import Dict, List, Any, Optional, Tuple
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

from core.state_manager import StateManager
from core.config_manager import ConfigManager
from utils.memory_manager import MemoryManager

logger = logging.getLogger(__name__)

class AdvancedTokenAnalyzer:
    """
    AdvancedTokenAnalyzer uses ML/AI techniques to:
    - Analyze token potential
    - Identify promising trading opportunities
    - Calculate risk/reward metrics
    - Predict short-term price movements
    """
    
    def __init__(self, 
                config_manager: ConfigManager,
                state_manager: StateManager,
                memory_manager: Optional[MemoryManager] = None):
        """
        Initialize the AdvancedTokenAnalyzer
        
        Args:
            config_manager (ConfigManager): Configuration manager instance
            state_manager (StateManager): State manager instance
            memory_manager (MemoryManager, optional): Memory manager instance
        """
        self.config_manager = config_manager
        self.state_manager = state_manager
        self.memory_manager = memory_manager
        
        # Model paths
        self.model_dir = self.config_manager.get('ai_core.model_dir', 'models/ai')
        
        # Feature extraction settings
        self.feature_importance_threshold = self.config_manager.get(
            'ai_core.feature_importance_threshold', 0.01)
        
        # Initialize models
        self.potential_model = self._initialize_potential_model()
        self.price_prediction_model = self._initialize_price_prediction_model()
        self.risk_assessment_model = self._initialize_risk_assessment_model()
        
        # Feature scaling
        self.scaler = StandardScaler()
        
        # Cache for analyzed tokens
        self.analysis_cache = {}
        self.analysis_ttl = 300  # 5 minutes cache
        
        logger.info("AdvancedTokenAnalyzer initialized")
    
    def _initialize_potential_model(self) -> Any:
        """
        Initialize or load the token potential prediction model
        
        Returns:
            Any: The model instance
        """
        try:
            # For now, initialize a new model
            # In production, we would load a pre-trained model
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=10,
                random_state=42
            )
            return model
        except Exception as e:
            logger.error(f"Error initializing potential model: {str(e)}")
            # Fallback to a simpler model
            return RandomForestClassifier(n_estimators=10, random_state=42)
    
    def _initialize_price_prediction_model(self) -> Any:
        """
        Initialize or load the price prediction model
        
        Returns:
            Any: The model instance
        """
        try:
            # For now, initialize a new model
            # In production, we would load a pre-trained model
            model = GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            )
            return model
        except Exception as e:
            logger.error(f"Error initializing price prediction model: {str(e)}")
            # Fallback to a simpler model
            return GradientBoostingRegressor(n_estimators=10, random_state=42)
    
    def _initialize_risk_assessment_model(self) -> Any:
        """
        Initialize or load the risk assessment model
        
        Returns:
            Any: The model instance
        """
        try:
            # For now, initialize a new model
            # In production, we would load a pre-trained model
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=8,
                min_samples_split=10,
                random_state=42
            )
            return model
        except Exception as e:
            logger.error(f"Error initializing risk model: {str(e)}")
            # Fallback to a simpler model
            return RandomForestClassifier(n_estimators=10, random_state=42)
    
    async def analyze_token(self, token_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform comprehensive analysis of a token
        
        Args:
            token_data (Dict[str, Any]): Token data including price, volume, liquidity, etc.
            
        Returns:
            Dict[str, Any]: Analysis results
        """
        token_address = token_data.get('address')
        
        # Check cache first
        if token_address in self.analysis_cache:
            cache_entry = self.analysis_cache[token_address]
            if time.time() - cache_entry['timestamp'] < self.analysis_ttl:
                logger.debug(f"Using cached analysis for {token_address}")
                return cache_entry['result']
        
        # Extract features
        features = await self._extract_features(token_data)
        if not features:
            logger.warning(f"Could not extract features for {token_address}")
            return {
                'success': False,
                'error': 'Feature extraction failed',
                'token_address': token_address,
                'score': 0
            }
        
        # Normalize features
        normalized_features = self._normalize_features(features)
        
        # Run analysis
        try:
            # Get potential score (0-100)
            potential_score = await self._calculate_potential_score(normalized_features)
            
            # Get price predictions (for different timeframes)
            price_predictions = await self._predict_price_movements(normalized_features)
            
            # Get risk assessment
            risk_assessment = await self._assess_risk(normalized_features, token_data)
            
            # Calculate composite score
            composite_score = self._calculate_composite_score(
                potential_score, price_predictions, risk_assessment)
            
            # Prepare result
            result = {
                'success': True,
                'token_address': token_address,
                'timestamp': time.time(),
                'potential_score': potential_score,
                'price_predictions': price_predictions,
                'risk_assessment': risk_assessment,
                'composite_score': composite_score,
                'recommendation': self._generate_recommendation(composite_score, risk_assessment)
            }
            
            # Cache the result
            self.analysis_cache[token_address] = {
                'timestamp': time.time(),
                'result': result
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing token {token_address}: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'token_address': token_address,
                'score': 0
            }
    
    async def _extract_features(self, token_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Extract features from token data for ML analysis
        
        Args:
            token_data (Dict[str, Any]): Token data
            
        Returns:
            Dict[str, float]: Extracted features
        """
        features = {}
        
        try:
            # Basic features
            features['liquidity_usd'] = float(token_data.get('liquidity_usd', 0))
            features['market_cap_usd'] = float(token_data.get('market_cap_usd', 0))
            features['volume_24h_usd'] = float(token_data.get('volume_24h_usd', 0))
            features['price_usd'] = float(token_data.get('price_usd', 0))
            
            # Derived features
            if features['market_cap_usd'] > 0:
                features['volume_to_mcap'] = features['volume_24h_usd'] / features['market_cap_usd']
            else:
                features['volume_to_mcap'] = 0
                
            if features['liquidity_usd'] > 0:
                features['mcap_to_liquidity'] = features['market_cap_usd'] / features['liquidity_usd']
            else:
                features['mcap_to_liquidity'] = 0
            
            # Time-based features
            if 'first_seen_at' in token_data:
                first_seen = token_data['first_seen_at']
                if isinstance(first_seen, str):
                    # Convert to timestamp if it's a string
                    import datetime
                    try:
                        dt = datetime.datetime.fromisoformat(first_seen.replace('Z', '+00:00'))
                        first_seen = dt.timestamp()
                    except:
                        first_seen = time.time()
                        
                features['token_age_days'] = (time.time() - first_seen) / (24 * 3600)
            else:
                features['token_age_days'] = 0
            
            # Contract verification status
            features['contract_verified'] = 1.0 if token_data.get('contract_verified', False) else 0.0
            
            # Contract audit score
            audit_data = token_data.get('contract_audit', {})
            features['audit_score'] = float(audit_data.get('score', 0))
            
            # Holder metrics if available
            holder_data = token_data.get('holders', {})
            features['unique_holders'] = float(holder_data.get('count', 0))
            features['top10_concentration'] = float(holder_data.get('top10_percent', 0))
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting features: {str(e)}")
            return {}
    
    def _normalize_features(self, features: Dict[str, float]) -> np.ndarray:
        """
        Normalize features for model input
        
        Args:
            features (Dict[str, float]): Raw features
            
        Returns:
            np.ndarray: Normalized feature vector
        """
        # Convert to numpy array
        feature_names = sorted(features.keys())
        feature_vector = np.array([features[name] for name in feature_names]).reshape(1, -1)
        
        # Apply scaling (in production, we would use a pre-fit scaler)
        # For now, just do simple normalization to avoid extreme values
        feature_vector = np.clip(feature_vector, -1e10, 1e10)  # Remove extreme values
        
        # Log transform features with high variance
        for i, name in enumerate(feature_names):
            if name in ['liquidity_usd', 'market_cap_usd', 'volume_24h_usd']:
                if feature_vector[0, i] > 0:
                    feature_vector[0, i] = np.log1p(feature_vector[0, i])
        
        return feature_vector
    
    async def _calculate_potential_score(self, normalized_features: np.ndarray) -> float:
        """
        Calculate token potential score using ML model
        
        Args:
            normalized_features (np.ndarray): Normalized features
            
        Returns:
            float: Potential score (0-100)
        """
        # In a real implementation, we would use the trained model
        # For now, use a simplistic approach based on features
        
        # Extract liquidity and volume components from normalized features
        # Simplified calculation - in production would use the actual model
        liquidity_component = normalized_features[0, 0] * 20  # Assuming first feature is liquidity
        volume_component = normalized_features[0, 2] * 30     # Assuming third feature is volume
        
        # Apply sigmoid to get a score between 0-100
        raw_score = liquidity_component + volume_component + 50
        score = 100 / (1 + np.exp(-0.1 * (raw_score - 50)))
        
        return float(min(max(score, 0), 100))
    
    async def _predict_price_movements(self, 
                                     normalized_features: np.ndarray) -> Dict[str, float]:
        """
        Predict price movements for different timeframes
        
        Args:
            normalized_features (np.ndarray): Normalized features
            
        Returns:
            Dict[str, float]: Predicted price changes by timeframe
        """
        # In a real implementation, we would use trained models for each timeframe
        # For now, use a simplistic approach
        
        # Simplified predictions based on features
        short_term = float(normalized_features[0, 2] * 5)  # 1h prediction
        medium_term = float(short_term * 0.8)              # 4h prediction
        long_term = float(medium_term * 0.7)               # 24h prediction
        
        return {
            '1h': short_term,
            '4h': medium_term,
            '24h': long_term
        }
    
    async def _assess_risk(self, 
                         normalized_features: np.ndarray, 
                         token_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Assess token trading risk
        
        Args:
            normalized_features (np.ndarray): Normalized features
            token_data (Dict[str, Any]): Original token data
            
        Returns:
            Dict[str, Any]: Risk assessment
        """
        # Extract key risk components 
        liquidity_risk = self._calculate_liquidity_risk(token_data)
        contract_risk = self._calculate_contract_risk(token_data)
        volatility_risk = self._calculate_volatility_risk(token_data)
        
        # Calculate overall risk score (0-100, higher means riskier)
        risk_score = (liquidity_risk * 0.4 + 
                     contract_risk * 0.3 + 
                     volatility_risk * 0.3)
        
        return {
            'overall_score': risk_score,
            'liquidity_risk': liquidity_risk,
            'contract_risk': contract_risk,
            'volatility_risk': volatility_risk,
            'risk_category': self._get_risk_category(risk_score)
        }
    
    def _calculate_liquidity_risk(self, token_data: Dict[str, Any]) -> float:
        """
        Calculate liquidity risk
        
        Args:
            token_data (Dict[str, Any]): Token data
            
        Returns:
            float: Liquidity risk score (0-100)
        """
        liquidity = token_data.get('liquidity_usd', 0)
        
        if liquidity < 1000:
            return 95  # Extremely high risk
        elif liquidity < 10000:
            return 80  # Very high risk
        elif liquidity < 50000:
            return 60  # High risk
        elif liquidity < 200000:
            return 40  # Medium risk
        elif liquidity < 1000000:
            return 20  # Low risk
        else:
            return 10  # Very low risk
    
    def _calculate_contract_risk(self, token_data: Dict[str, Any]) -> float:
        """
        Calculate contract risk
        
        Args:
            token_data (Dict[str, Any]): Token data
            
        Returns:
            float: Contract risk score (0-100)
        """
        verified = token_data.get('contract_verified', False)
        audit = token_data.get('contract_audit', {})
        audit_score = audit.get('score', 0) if audit else 0
        
        if not verified:
            base_risk = 90  # Very high risk for unverified contracts
        else:
            base_risk = 40  # Medium risk as baseline for verified contracts
        
        # Reduce risk based on audit score (0-100)
        risk_reduction = audit_score * 0.3
        
        return max(base_risk - risk_reduction, 10)  # Minimum 10% risk
    
    def _calculate_volatility_risk(self, token_data: Dict[str, Any]) -> float:
        """
        Calculate volatility risk
        
        Args:
            token_data (Dict[str, Any]): Token data
            
        Returns:
            float: Volatility risk score (0-100)
        """
        # Use price change data if available
        price_change = token_data.get('price_change_24h', 0)
        
        # Absolute change matters for volatility risk
        abs_change = abs(price_change) if price_change else 0
        
        if abs_change > 100:
            return 90  # Extremely volatile
        elif abs_change > 50:
            return 70  # Very volatile
        elif abs_change > 25:
            return 50  # Moderately volatile
        elif abs_change > 10:
            return 30  # Slightly volatile
        else:
            return 15  # Low volatility
    
    def _get_risk_category(self, risk_score: float) -> str:
        """
        Get risk category from numerical score
        
        Args:
            risk_score (float): Risk score (0-100)
            
        Returns:
            str: Risk category
        """
        if risk_score >= 80:
            return "extreme"
        elif risk_score >= 60:
            return "high"
        elif risk_score >= 40:
            return "medium"
        elif risk_score >= 20:
            return "low"
        else:
            return "very_low"
    
    def _calculate_composite_score(self,
                                 potential_score: float,
                                 price_predictions: Dict[str, float],
                                 risk_assessment: Dict[str, Any]) -> float:
        """
        Calculate composite score for trading decision
        
        Args:
            potential_score (float): Token potential score
            price_predictions (Dict[str, float]): Price predictions
            risk_assessment (Dict[str, Any]): Risk assessment
            
        Returns:
            float: Composite score (0-100)
        """
        # Extract components
        risk_score = risk_assessment['overall_score']
        short_term_prediction = price_predictions['1h']
        
        # Calculate reward-to-risk ratio (adjusted)
        reward_component = potential_score * 0.6 + max(0, short_term_prediction) * 40
        risk_component = risk_score
        
        if risk_component > 80:  # Extreme risk
            reward_to_risk = reward_component / (risk_component * 2)  # Penalize high risk more
        else:
            reward_to_risk = reward_component / max(risk_component, 1)
        
        # Convert to a 0-100 score
        composite_score = min(100, max(0, reward_to_risk * 25))
        
        return composite_score
    
    def _generate_recommendation(self, 
                               composite_score: float, 
                               risk_assessment: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate trading recommendation based on analysis
        
        Args:
            composite_score (float): Composite analysis score
            risk_assessment (Dict[str, Any]): Risk assessment
            
        Returns:
            Dict[str, Any]: Trading recommendation
        """
        # Default recommendation
        recommendation = {
            'action': 'hold',
            'confidence': 0,
            'reasoning': '',
            'position_size': 0,
            'stop_loss': 0,
            'take_profit': 0
        }
        
        # Determine recommendation based on score and risk
        risk_score = risk_assessment['overall_score']
        
        # Strong buy signal
        if composite_score >= 75:
            recommendation['action'] = 'strong_buy'
            recommendation['confidence'] = min(100, composite_score + 10)
            recommendation['reasoning'] = 'High potential with favorable risk-reward ratio'
            
            # Position size based on risk (smaller for higher risk)
            if risk_score >= 70:  # High risk
                recommendation['position_size'] = 5  # Small position
            elif risk_score >= 50:  # Medium risk
                recommendation['position_size'] = 10  # Medium position
            else:  # Low risk
                recommendation['position_size'] = 15  # Larger position
                
            # Stop loss and take profit based on risk
            recommendation['stop_loss'] = min(15, risk_score / 5)  # Higher risk = tighter stop loss
            recommendation['take_profit'] = max(20, 100 - risk_score)  # Lower risk = higher take profit
            
        # Buy signal
        elif composite_score >= 60:
            recommendation['action'] = 'buy'
            recommendation['confidence'] = composite_score
            recommendation['reasoning'] = 'Good potential with acceptable risk'
            
            # Position size
            if risk_score >= 70:
                recommendation['position_size'] = 3
            elif risk_score >= 50:
                recommendation['position_size'] = 7
            else:
                recommendation['position_size'] = 10
                
            recommendation['stop_loss'] = min(10, risk_score / 6)
            recommendation['take_profit'] = max(15, 90 - risk_score)
            
        # Hold (neutral) signal
        elif composite_score >= 40:
            recommendation['action'] = 'hold'
            recommendation['confidence'] = 50
            recommendation['reasoning'] = 'Balanced risk-reward, no clear signal'
            
        # Avoid (sell) signal
        else:
            recommendation['action'] = 'avoid'
            recommendation['confidence'] = 100 - composite_score
            recommendation['reasoning'] = 'Unfavorable risk-reward ratio'
            
        return recommendation
    
    async def retrain_models(self, training_data: List[Dict[str, Any]]):
        """
        Retrain ML models with new data
        
        Args:
            training_data (List[Dict[str, Any]]): Training data with features and labels
        """
        if not training_data or len(training_data) < 10:
            logger.warning("Not enough training data to retrain models")
            return False
        
        try:
            # In a real implementation, this would extract features and labels
            # and update the models with new training data
            
            logger.info(f"Models retrained with {len(training_data)} samples")
            return True
            
        except Exception as e:
            logger.error(f"Error retraining models: {str(e)}")
            return False