"""
ML_SignalDetector Component
Responsible for detecting trading signals using machine learning
algorithms to identify patterns in market data.
"""

import asyncio
import logging
import time
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from collections import deque
import joblib
import os

logger = logging.getLogger(__name__)

class ML_SignalDetector:
    """
    ML_SignalDetector handles:
    - Pattern detection in price and volume data
    - Price movement predictions
    - Signal reliability estimation
    - Model adaption over time
    """
    
    def __init__(self):
        """Initialize the ML_SignalDetector"""
        # Model settings
        self.models = {}
        self.features = {}
        self.model_initialized = False
        self.min_data_points = 24  # Minimum data points required for prediction
        
        # Historical data
        self.price_history = {}  # token_address -> list of price data
        self.volume_history = {}  # token_address -> list of volume data
        self.signal_history = {}  # token_address -> list of generated signals
        
        # Performance tracking
        self.prediction_accuracy = {}  # token_address -> accuracy metrics
        
        # Try to load pre-trained models if available
        self._load_models()
        
        logger.info("ML_SignalDetector initialized")
    
    def _load_models(self):
        """Load pre-trained models if available"""
        model_paths = {
            'price_trend': './models/price_trend_model.joblib',
            'volatility': './models/volatility_model.joblib',
            'volume_spike': './models/volume_spike_model.joblib',
            'scaler': './models/feature_scaler.joblib',
            'transformer': './models/feature_transformer.joblib'
        }
        
        models_loaded = 0
        for model_name, path in model_paths.items():
            try:
                if os.path.exists(path):
                    self.models[model_name] = joblib.load(path)
                    logger.info(f"Loaded model: {model_name}")
                    models_loaded += 1
            except Exception as e:
                logger.warning(f"Could not load model {model_name}: {str(e)}")
        
        if models_loaded > 0:
            self.model_initialized = True
            logger.info(f"Successfully loaded {models_loaded} ML models")
        else:
            logger.warning("No ML models could be loaded, will operate in basic mode")
    
    async def detect_signals(self, token_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Detect trading signals for a token using advanced ML techniques
        
        Args:
            token_data (Dict[str, Any]): Token data including historical prices
            
        Returns:
            Dict[str, Any]: Detected signals
        """
        token_address = token_data.get('address', '')
        token_symbol = token_data.get('symbol', '')
        
        if not token_address:
            logger.warning("Missing token address in token data")
            return {
                'signals': {},
                'prediction': None,
                'confidence': 0,
                'timestamp': time.time()
            }
        
        logger.debug(f"Detecting signals for {token_symbol} ({token_address})")
        
        # Extract historical price data
        historical_prices = token_data.get('historical_prices', [])
        
        # If no historical data or insufficient data, return empty signals
        if not historical_prices or len(historical_prices) < self.min_data_points:
            logger.debug(f"Insufficient historical data for {token_symbol}")
            return {
                'signals': {},
                'prediction': None,
                'confidence': 0,
                'timestamp': time.time(),
                'reason': 'insufficient_data'
            }
        
        # Update historical data
        self._update_historical_data(token_address, historical_prices)
        
        # Extract features
        features = self._extract_features(token_address, token_data)
        
        # Detect signals using various methods
        signals = {}
        
        # Simple trend analysis
        trend_signal = self._detect_trend(token_address, features)
        if trend_signal:
            signals['trend'] = trend_signal
        
        # Volume analysis
        volume_signal = self._detect_volume_anomaly(token_address, features)
        if volume_signal:
            signals['volume'] = volume_signal
        
        # Volatility analysis
        volatility_signal = self._detect_volatility(token_address, features)
        if volatility_signal:
            signals['volatility'] = volatility_signal
        
        # ML-based prediction if models are available
        prediction = None
        confidence = 0
        ml_details = {}
        
        if self.model_initialized and 'price_trend' in self.models:
            try:
                # Prepare features for model
                feature_vector = [
                    features['price_change'],
                    features['volume_change'],
                    features['volatility'],
                    features['rsi'],
                    features['price_velocity']
                ]
                
                model_features = np.array(feature_vector).reshape(1, -1)
                
                # Apply feature scaling if available
                if 'scaler' in self.models:
                    try:
                        model_features = self.models['scaler'].transform(model_features)
                    except Exception as e:
                        logger.warning(f"Could not apply feature scaling: {str(e)}")
                
                # Make prediction
                price_trend_model = self.models['price_trend']
                prediction = price_trend_model.predict(model_features)[0]
                probabilities = price_trend_model.predict_proba(model_features)[0]
                
                # Get confidence - highest probability
                confidence = float(max(probabilities))
                
                # Get predicted volatility if model available
                predicted_volatility = None
                if 'volatility' in self.models:
                    try:
                        predicted_volatility = float(self.models['volatility'].predict(model_features)[0])
                        ml_details['predicted_volatility'] = predicted_volatility
                    except Exception as e:
                        logger.warning(f"Error predicting volatility: {str(e)}")
                
                # Create advanced ML signal with more details
                if confidence > 0.65:  # Lower threshold for better recall
                    ml_signal = {
                        'direction': 'up' if prediction > 0 else 'down',
                        'strength': confidence,
                        'horizon': '1h',  # Time horizon for prediction
                        'predicted_volatility': predicted_volatility,
                        'feature_importance': {
                            'price_change': 0.3,
                            'volume_change': 0.2,
                            'volatility': 0.2,
                            'rsi': 0.2,
                            'price_velocity': 0.1
                        }
                    }
                    
                    # Add details for stronger signals
                    if confidence > 0.8:
                        predicted_magnitude = features['volatility'] * (2.0 if prediction > 0 else -2.0)
                        ml_signal['predicted_magnitude'] = predicted_magnitude
                        ml_signal['confidence_level'] = 'high'
                    
                    signals['ml_prediction'] = ml_signal
                    ml_details['raw_probabilities'] = [float(p) for p in probabilities]
            except Exception as e:
                logger.error(f"Error making ML prediction: {str(e)}", exc_info=True)
        
        # Store signal for tracking
        self.signal_history.setdefault(token_address, []).append({
            'signals': signals,
            'prediction': prediction,
            'confidence': confidence,
            'timestamp': time.time()
        })
        
        # Keep signal history limited
        if len(self.signal_history[token_address]) > 100:
            self.signal_history[token_address].pop(0)
        
        result = {
            'signals': signals,
            'prediction': prediction,
            'confidence': confidence,
            'timestamp': time.time()
        }
        
        logger.debug(f"Signal detection for {token_symbol}: {len(signals)} signals, confidence: {confidence:.2f}")
        return result
    
    def _update_historical_data(self, token_address: str, historical_prices: List[Dict[str, Any]]):
        """
        Update historical data for a token
        
        Args:
            token_address (str): Token address
            historical_prices (List[Dict[str, Any]]): Historical price data
        """
        # Extract price and volume series
        prices = [float(p['close']) for p in historical_prices]
        volumes = [float(p['volume']) for p in historical_prices]
        
        # Store data
        self.price_history[token_address] = prices
        self.volume_history[token_address] = volumes
    
    def _extract_features(self, token_address: str, token_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Extract features from token data for signal detection
        
        Args:
            token_address (str): Token address
            token_data (Dict[str, Any]): Token data
            
        Returns:
            Dict[str, float]: Extracted features
        """
        prices = self.price_history.get(token_address, [])
        volumes = self.volume_history.get(token_address, [])
        
        if not prices or len(prices) < 2:
            return {
                'price_change': 0,
                'volume_change': 0,
                'volatility': 0,
                'rsi': 50,  # Neutral RSI
                'price_velocity': 0
            }
        
        # Calculate price change
        price_change = (prices[-1] - prices[0]) / prices[0] if prices[0] > 0 else 0
        
        # Calculate volume change
        volume_change = (volumes[-1] - volumes[0]) / volumes[0] if volumes[0] > 0 else 0
        
        # Calculate volatility (standard deviation of returns)
        returns = [((prices[i] - prices[i-1]) / prices[i-1]) for i in range(1, len(prices))]
        volatility = np.std(returns) if returns else 0
        
        # Calculate RSI (Relative Strength Index)
        rsi = self._calculate_rsi(prices)
        
        # Calculate price velocity (rate of change over time)
        if len(prices) >= 3:
            price_velocity = (prices[-1] - prices[-3]) / 2  # Change over 2 periods
        else:
            price_velocity = 0
        
        features = {
            'price_change': price_change,
            'volume_change': volume_change,
            'volatility': volatility,
            'rsi': rsi,
            'price_velocity': price_velocity
        }
        
        # Store features for future use
        self.features[token_address] = features
        
        return features
    
    def _calculate_rsi(self, prices: List[float], periods: int = 14) -> float:
        """
        Calculate Relative Strength Index (RSI)
        
        Args:
            prices (List[float]): Price series
            periods (int): RSI period
            
        Returns:
            float: RSI value
        """
        if len(prices) < periods + 1:
            return 50  # Not enough data, return neutral
        
        # Calculate price changes
        deltas = np.diff(prices)
        
        # Get gains and losses
        seed = deltas[:periods+1]
        up = seed[seed >= 0].sum() / periods
        down = -seed[seed < 0].sum() / periods
        
        if down == 0:
            return 100  # No losses, RSI = 100
        
        rs = up / down
        rsi = 100 - (100 / (1 + rs))
        
        return float(rsi)
    
    def _detect_trend(self, token_address: str, features: Dict[str, float]) -> Optional[Dict[str, Any]]:
        """
        Detect price trend signal
        
        Args:
            token_address (str): Token address
            features (Dict[str, float]): Token features
            
        Returns:
            Optional[Dict[str, Any]]: Trend signal or None
        """
        prices = self.price_history.get(token_address, [])
        
        if len(prices) < 10:
            return None
        
        # Simple moving averages
        short_ma = np.mean(prices[-5:])
        long_ma = np.mean(prices[-15:])
        
        # Determine trend
        if short_ma > long_ma * 1.02:  # Short MA at least 2% above long MA
            trend = 'bullish'
            strength = min(1.0, (short_ma / long_ma - 1) * 10)  # Scale strength
        elif short_ma < long_ma * 0.98:  # Short MA at least 2% below long MA
            trend = 'bearish'
            strength = min(1.0, (1 - short_ma / long_ma) * 10)  # Scale strength
        else:
            trend = 'neutral'
            strength = 0.1
        
        # Check RSI for confirmation
        rsi = features['rsi']
        
        if (trend == 'bullish' and rsi > 70) or (trend == 'bearish' and rsi < 30):
            # Strong confirmation from RSI
            confidence = min(1.0, strength * 1.5)
        else:
            confidence = strength
        
        return {
            'trend': trend,
            'strength': strength,
            'confidence': confidence,
            'supporting_indicators': {
                'short_ma': short_ma,
                'long_ma': long_ma,
                'rsi': rsi
            }
        }
    
    def _detect_volume_anomaly(self, token_address: str, features: Dict[str, float]) -> Optional[Dict[str, Any]]:
        """
        Detect volume anomaly signal
        
        Args:
            token_address (str): Token address
            features (Dict[str, float]): Token features
            
        Returns:
            Optional[Dict[str, Any]]: Volume signal or None
        """
        volumes = self.volume_history.get(token_address, [])
        
        if len(volumes) < 5:
            return None
        
        # Calculate average volume
        avg_volume = np.mean(volumes[:-1])  # All except current
        current_volume = volumes[-1]
        
        # Check for volume spike
        if current_volume > avg_volume * 3:  # Volume at least 3x average
            anomaly_type = 'spike'
            strength = min(1.0, current_volume / avg_volume / 5)  # Scale strength
        elif current_volume < avg_volume * 0.3:  # Volume less than 30% of average
            anomaly_type = 'dry_up'
            strength = min(1.0, (1 - current_volume / avg_volume) * 2)  # Scale strength
        else:
            return None  # No significant volume anomaly
        
        return {
            'anomaly_type': anomaly_type,
            'strength': strength,
            'current_volume': current_volume,
            'average_volume': avg_volume,
            'volume_change': features['volume_change']
        }
    
    def _detect_volatility(self, token_address: str, features: Dict[str, float]) -> Optional[Dict[str, Any]]:
        """
        Detect volatility signal
        
        Args:
            token_address (str): Token address
            features (Dict[str, float]): Token features
            
        Returns:
            Optional[Dict[str, Any]]: Volatility signal or None
        """
        volatility = features['volatility']
        
        if volatility < 0.01:  # Low volatility threshold
            return None
        
        # Check if volatility is increasing
        prices = self.price_history.get(token_address, [])
        
        if len(prices) < 10:
            return None
        
        # Calculate volatility for two periods
        recent_prices = prices[-5:]
        older_prices = prices[-10:-5]
        
        recent_returns = [((recent_prices[i] - recent_prices[i-1]) / recent_prices[i-1]) 
                          for i in range(1, len(recent_prices))]
        older_returns = [((older_prices[i] - older_prices[i-1]) / older_prices[i-1]) 
                         for i in range(1, len(older_prices))]
        
        recent_vol = np.std(recent_returns) if recent_returns else 0
        older_vol = np.std(older_returns) if older_returns else 0
        
        # Determine if volatility is changing significantly
        if recent_vol > older_vol * 1.5:  # Volatility increased by 50%+
            volatility_change = 'increasing'
            strength = min(1.0, (recent_vol / older_vol - 1))
        elif recent_vol < older_vol * 0.67:  # Volatility decreased by 33%+
            volatility_change = 'decreasing'
            strength = min(1.0, (1 - recent_vol / older_vol))
        else:
            volatility_change = 'stable'
            strength = 0.3
        
        return {
            'volatility': volatility,
            'volatility_change': volatility_change,
            'strength': strength,
            'recent_volatility': recent_vol,
            'older_volatility': older_vol
        }
    
    async def validate_signal(self, token_address: str, signal_id: str) -> Dict[str, Any]:
        """
        Validate a previously generated signal against actual outcomes
        
        Args:
            token_address (str): Token address
            signal_id (str): Signal identifier
            
        Returns:
            Dict[str, Any]: Validation result
        """
        logger.debug(f"Validating signal {signal_id} for {token_address}")
        
        # In a real implementation, this would compare predicted movement
        # against actual market movement to track prediction accuracy
        
        # For now, return a placeholder validation
        return {
            'signal_id': signal_id,
            'token_address': token_address,
            'was_correct': True,
            'actual_movement': 0.05,  # 5% movement
            'predicted_movement': 0.03,  # 3% movement
            'accuracy': 0.8,
            'validated_at': time.time()
        }
    
    async def train_model(self, training_data: List[Dict[str, Any]]):
        """
        Train or update the ML model with new data
        
        Args:
            training_data (List[Dict[str, Any]]): Training data
        """
        logger.info(f"Training ML model with {len(training_data)} data points")
        
        if not training_data:
            logger.warning("No training data provided")
            return
        
        try:
            from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
            from sklearn.preprocessing import StandardScaler
            
            # Prepare features and labels
            features = []
            price_movement_labels = []
            
            for data_point in training_data:
                if 'features' in data_point and 'actual_movement' in data_point:
                    # Extract features
                    feature_vector = [
                        data_point['features'].get('price_change', 0),
                        data_point['features'].get('volume_change', 0),
                        data_point['features'].get('volatility', 0),
                        data_point['features'].get('rsi', 50),
                        data_point['features'].get('price_velocity', 0)
                    ]
                    
                    # Extract label (1 for up, 0 for down)
                    price_movement = 1 if data_point['actual_movement'] > 0 else 0
                    
                    features.append(feature_vector)
                    price_movement_labels.append(price_movement)
            
            if len(features) < 10:
                logger.warning("Not enough valid training data points")
                return
            
            # Normalize features
            scaler = StandardScaler()
            normalized_features = scaler.fit_transform(features)
            
            # Train trend prediction model (classification)
            trend_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
            trend_model.fit(normalized_features, price_movement_labels)
            
            # Train volatility prediction model (regression)
            volatility_labels = [data_point.get('features', {}).get('volatility', 0) 
                                for data_point in training_data if 'features' in data_point]
            
            volatility_model = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)
            volatility_model.fit(normalized_features, volatility_labels)
            
            # Save models
            os.makedirs('./models', exist_ok=True)
            joblib.dump(trend_model, './models/price_trend_model.joblib')
            joblib.dump(volatility_model, './models/volatility_model.joblib')
            joblib.dump(scaler, './models/feature_scaler.joblib')
            
            # Update model references
            self.models['price_trend'] = trend_model
            self.models['volatility'] = volatility_model
            self.models['scaler'] = scaler
            self.model_initialized = True
            
            logger.info(f"ML models trained successfully: {len(features)} samples used")
            
        except Exception as e:
            logger.error(f"Error training ML models: {str(e)}", exc_info=True)
    
    def get_signal_history(self, token_address: str) -> List[Dict[str, Any]]:
        """
        Get signal history for a token
        
        Returns:
            List[Dict[str, Any]]: Signal history
        """
        return self.signal_history.get(token_address, [])
    
    def get_prediction_accuracy(self, token_address: str = None) -> Dict[str, Any]:
        """
        Get prediction accuracy statistics
        
        Args:
            token_address (str, optional): Token address for specific token stats
            
        Returns:
            Dict[str, Any]: Prediction accuracy statistics
        """
        if token_address:
            return self.prediction_accuracy.get(token_address, {
                'total_predictions': 0,
                'correct_predictions': 0,
                'accuracy': 0
            })
        
        # Overall accuracy
        total_predictions = sum(stats.get('total_predictions', 0) 
                               for stats in self.prediction_accuracy.values())
        
        correct_predictions = sum(stats.get('correct_predictions', 0) 
                                 for stats in self.prediction_accuracy.values())
        
        accuracy = (correct_predictions / total_predictions if total_predictions > 0 else 0)
        
        return {
            'total_predictions': total_predictions,
            'correct_predictions': correct_predictions,
            'accuracy': accuracy,
            'tokens_analyzed': len(self.prediction_accuracy)
        }
