"""
DRL Strategy Component
Uses Deep Reinforcement Learning to dynamically optimize trading decisions
based on real-time market feedback and self-adaptive learning with real market data.
"""

import asyncio
import logging
import time
import numpy as np
import random
import os
import joblib
from typing import Dict, List, Any, Optional, Tuple, Union
from collections import deque
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor

from core.state_manager import StateManager
from core.config_manager import ConfigManager

logger = logging.getLogger(__name__)

class StateEncoder:
    """Encodes market state into a format suitable for DRL models"""
    
    def __init__(self):
        """Initialize the StateEncoder"""
        # Core market features
        self.base_features = [
            'price_usd', 
            'price_change_1h', 
            'price_change_24h', 
            'volume_24h_usd', 
            'liquidity_usd',
            'market_cap_usd',
            'volatility',
            'risk_score'
        ]
        
        # Advanced technical indicators
        self.tech_features = [
            'rsi_14',           # Relative Strength Index
            'macd',             # MACD signal
            'bollinger_band',   # Bollinger band position
            'avg_trade_size',   # Average trade size
            'buy_sell_ratio',   # Buy vs sell ratio
            'whale_activity',   # Large wallet transaction activity
            'txn_frequency',    # Transaction frequency
            'price_momentum'    # Price momentum indicator
        ]
        
        # Market context features
        self.market_features = [
            'market_trend',      # Overall market trend (bull/bear/sideways)
            'solana_price',      # SOL price as reference
            'memecoin_index',    # Memecoin market sentiment index
            'time_since_launch', # Time since token launched (normalized)
            'holder_growth',     # Growth rate of holders
            'social_sentiment',  # Social media sentiment score
            'similar_token_perf' # Performance of similar tokens
        ]
        
        # Combine all features
        self.feature_list = self.base_features + self.tech_features + self.market_features
        
        # Initialize scaler for standardization
        self.scaler = None
        self.is_fit = False
        
        # Feature mean values for missing data imputation
        self.feature_means = {}
        
        # Create directory for model artifacts if it doesn't exist
        os.makedirs('models/drl', exist_ok=True)
        
        # Path for the scaler
        self.scaler_path = 'models/drl/state_scaler.joblib'
        
        # Try to load existing scaler
        self._load_scaler()
    
    def _load_scaler(self):
        """Load saved scaler if it exists"""
        try:
            if os.path.exists(self.scaler_path):
                self.scaler = joblib.load(self.scaler_path)
                self.is_fit = True
                logger.info("Loaded existing state scaler")
            else:
                self.scaler = StandardScaler()
                logger.info("Created new state scaler")
        except Exception as e:
            logger.warning(f"Error loading state scaler: {str(e)}, creating new one")
            self.scaler = StandardScaler()
    
    def _save_scaler(self):
        """Save scaler for future use"""
        try:
            joblib.dump(self.scaler, self.scaler_path)
            logger.info("Saved state scaler")
        except Exception as e:
            logger.warning(f"Error saving state scaler: {str(e)}")
    
    def encode(self, market_state: Dict[str, Any]) -> np.ndarray:
        """
        Encode market state into a feature vector
        
        Args:
            market_state (Dict[str, Any]): Market state data
            
        Returns:
            np.ndarray: Encoded state vector
        """
        # Extract raw features and handle missing values
        features = self._extract_features(market_state)
        
        # Apply feature transformations (log, clip, etc.)
        transformed_features = self._transform_features(features)
        
        # Convert to numpy array
        feature_array = np.array(transformed_features, dtype=np.float32).reshape(1, -1)
        
        # Apply scaling if scaler is fit
        if self.is_fit:
            scaled_features = self.scaler.transform(feature_array).flatten()
        else:
            # If first time, just return normalized features
            scaled_features = feature_array.flatten()
            
            # Update mean values for future reference
            for i, feature in enumerate(self.feature_list):
                self.feature_means[feature] = scaled_features[i]
        
        return scaled_features
    
    def update_scaler(self, batch_states: List[Dict[str, Any]]):
        """
        Update scaler with batch of market states
        
        Args:
            batch_states (List[Dict[str, Any]]): Batch of market states
        """
        if not batch_states:
            return
            
        # Extract and transform features from all states
        all_features = []
        for state in batch_states:
            raw_features = self._extract_features(state)
            transformed = self._transform_features(raw_features)
            all_features.append(transformed)
        
        # Convert to numpy array
        feature_array = np.array(all_features, dtype=np.float32)
        
        # Fit or partial fit the scaler
        if self.is_fit:
            self.scaler.partial_fit(feature_array)
        else:
            self.scaler.fit(feature_array)
            self.is_fit = True
            
        # Save updated scaler
        self._save_scaler()
        
        # Update mean values
        mean_values = np.mean(feature_array, axis=0)
        for i, feature in enumerate(self.feature_list):
            self.feature_means[feature] = mean_values[i]
            
        logger.info(f"Updated state scaler with {len(batch_states)} samples")
    
    def _extract_features(self, market_state: Dict[str, Any]) -> List[float]:
        """
        Extract features from market state
        
        Args:
            market_state (Dict[str, Any]): Market state data
            
        Returns:
            List[float]: Extracted feature values
        """
        features = []
        
        for feature in self.feature_list:
            # Get feature value or use stored mean or default
            if feature in market_state:
                value = float(market_state[feature])
            elif feature in self.feature_means:
                # Use stored mean value for missing features
                value = self.feature_means[feature]
            else:
                # Default to 0 if no information available
                value = 0.0
                
            features.append(value)
            
        return features
    
    def _transform_features(self, features: List[float]) -> List[float]:
        """
        Apply transformations to features
        
        Args:
            features (List[float]): Raw feature list
            
        Returns:
            List[float]: Transformed features
        """
        transformed = []
        
        for i, value in enumerate(features):
            feature_name = self.feature_list[i]
            
            # Price-related features - log transform
            if feature_name in ['price_usd', 'solana_price']:
                transformed.append(np.log1p(max(0, value)))
            
            # Percentage change features - clip to reasonable range
            elif feature_name in ['price_change_1h', 'price_change_24h', 'holder_growth']:
                transformed.append(max(-5, min(5, value / 20)))  # Normalize and limit extreme values
            
            # Volume, liquidity, market cap - log transform
            elif feature_name in ['volume_24h_usd', 'liquidity_usd', 'market_cap_usd']:
                transformed.append(np.log1p(max(0, value)))
            
            # Indicator features already in -1 to 1 range
            elif feature_name in ['rsi_14', 'macd', 'bollinger_band', 'social_sentiment']:
                transformed.append(max(-1, min(1, value)))
            
            # Features in 0-100 range - normalize to 0-1
            elif feature_name in ['risk_score', 'market_trend', 'memecoin_index']:
                transformed.append(value / 100.0)
            
            # Time-based features - apply decay function
            elif feature_name == 'time_since_launch':
                # Convert to days and apply diminishing returns
                days = max(0, min(365, value / (24 * 3600)))
                transformed.append(1.0 - np.exp(-days / 30))  # Higher for older tokens
            
            # Volatility - normalize with sigmoid-like function
            elif feature_name == 'volatility':
                transformed.append(2.0 / (1.0 + np.exp(-value / 25)) - 1.0)  # Maps to -1 to 1
                
            # Ratio features - normalize to -1 to 1
            elif feature_name == 'buy_sell_ratio':
                transformed.append(max(-1, min(1, (value - 1) * 2)))  # 0=all sells, 2=all buys, 1=equal
                
            # Activity indicators - log transform
            elif feature_name in ['whale_activity', 'txn_frequency', 'avg_trade_size']:
                transformed.append(np.log1p(max(0, value)) / 5)  # Dampened scale
                
            # Performance comparisons - clip to reasonable range
            elif feature_name == 'similar_token_perf':
                transformed.append(max(-1, min(1, value / 2)))  # -1 to 1 range
                
            # Momentum - already normalized
            elif feature_name == 'price_momentum':
                transformed.append(max(-1, min(1, value)))
                
            # Default transformation - pass through
            else:
                transformed.append(value)
        
        return transformed


class RewardCalculator:
    """Calculates rewards for reinforcement learning using real market data"""
    
    def __init__(self):
        """Initialize the RewardCalculator"""
        self.reward_weights = {
            'pnl': 1.0,                # Profit and loss weight
            'risk': 0.3,               # Risk adjustment weight
            'target_reached': 0.5,     # Additional reward for reaching targets
            'loss_penalty': 1.2,       # Penalty multiplier for losses
            'market_alignment': 0.4,   # Reward for trading with market trend
            'volatility_alignment': 0.3, # Reward for appropriate action based on volatility
            'liquidity_efficiency': 0.2, # Reward for efficient use of liquidity
            'timing_precision': 0.5,   # Reward for good entry/exit timing
            'slippage_penalty': 0.3,   # Penalty for high slippage
            'whale_alignment': 0.4     # Reward for aligning with whale activity
        }
        
        # Initialize tracking for cumulative metrics
        self.total_trades = 0
        self.profitable_trades = 0
        self.cumulative_pnl = 0.0
        
        # Create directory for storing reward statistics
        os.makedirs('data/reward_stats', exist_ok=True)
        self.stats_file = 'data/reward_stats/reward_history.joblib'
        
        # Load previous stats if available
        self._load_stats()
    
    def _load_stats(self):
        """Load previous reward statistics if available"""
        try:
            if os.path.exists(self.stats_file):
                stats = joblib.load(self.stats_file)
                self.total_trades = stats.get('total_trades', 0)
                self.profitable_trades = stats.get('profitable_trades', 0)
                self.cumulative_pnl = stats.get('cumulative_pnl', 0.0)
                logger.info(f"Loaded reward statistics: {self.total_trades} trades, "
                          f"{self.profitable_trades} profitable, "
                          f"cumulative PnL: {self.cumulative_pnl:.2f}%")
        except Exception as e:
            logger.warning(f"Error loading reward statistics: {str(e)}")
    
    def _save_stats(self):
        """Save reward statistics"""
        try:
            stats = {
                'total_trades': self.total_trades,
                'profitable_trades': self.profitable_trades,
                'cumulative_pnl': self.cumulative_pnl,
                'win_rate': self.profitable_trades / max(1, self.total_trades),
                'timestamp': datetime.now().isoformat()
            }
            joblib.dump(stats, self.stats_file)
        except Exception as e:
            logger.warning(f"Error saving reward statistics: {str(e)}")
    
    def calculate(self, experience: Dict[str, Any]) -> float:
        """
        Calculate reward from trading experience using real market data
        
        Args:
            experience (Dict[str, Any]): Trading experience data
            
        Returns:
            float: Calculated reward
        """
        # Extract basic metrics
        action = experience.get('action', 'hold')
        profit_pct = experience.get('profit_pct', 0)
        risk_score = experience.get('risk_score', 50)
        trade_duration = experience.get('duration_hours', 0)
        target_reached = experience.get('target_reached', False)
        
        # Extract advanced metrics
        market_trend = experience.get('market_trend', 0)  # -100 to 100
        volatility = experience.get('volatility', 10)
        entry_slippage_pct = experience.get('entry_slippage_pct', 0)
        exit_slippage_pct = experience.get('exit_slippage_pct', 0)
        liquidity_usd = experience.get('liquidity_usd', 0)
        position_size_usd = experience.get('position_size_usd', 0)
        whale_activity = experience.get('whale_activity', 0)  # -1 (selling) to 1 (buying)
        buy_sell_ratio = experience.get('buy_sell_ratio', 1)  # Ratio of buys to sells (1 = equal)
        price_momentum = experience.get('price_momentum', 0)  # -1 to 1
        
        # On-chain metrics
        solana_price = experience.get('solana_price', 0)
        holder_growth = experience.get('holder_growth', 0)
        time_since_launch = experience.get('time_since_launch', 0)
        memecoin_index = experience.get('memecoin_index', 50)  # 0-100 scale
        social_sentiment = experience.get('social_sentiment', 0)  # -1 to 1 scale
        
        # Base reward from PnL
        if action == 'hold':
            # Hold actions have smaller rewards/penalties
            base_reward = profit_pct * 0.3
        else:
            base_reward = profit_pct
        
        # Apply loss penalty for negative outcomes
        if base_reward < 0:
            base_reward *= self.reward_weights['loss_penalty']
        
        # Risk adjustment - higher risk should reduce reward
        risk_adjustment = -1 * (risk_score / 100) * self.reward_weights['risk']
        
        # Target reached bonus
        target_bonus = self.reward_weights['target_reached'] if target_reached else 0
        
        # Duration efficiency adjustment
        # Reward quicker profitable trades, penalize holding too long for losses
        duration_factor = 0
        if profit_pct > 0:
            # For profits, quicker is better
            duration_factor = max(0, 1 - (trade_duration / 24))  # 1 for instant, 0 for 24h+
        elif profit_pct < 0:
            # For losses, quicker cuts are better
            duration_factor = max(0, 1 - (trade_duration / 6))  # 1 for quick cut, 0 for 6h+
        
        # Market alignment - reward for aligning with market trend
        market_alignment = 0
        if action == 'buy' and market_trend > 0:
            # Buying in uptrend is good
            market_alignment = (market_trend / 100) * self.reward_weights['market_alignment']
        elif action == 'sell' and market_trend < 0:
            # Selling in downtrend is good
            market_alignment = (-market_trend / 100) * self.reward_weights['market_alignment']
        elif action == 'hold' and abs(market_trend) < 20:
            # Holding in sideways market is good
            market_alignment = (1 - abs(market_trend) / 100) * self.reward_weights['market_alignment'] * 0.5
        
        # Volatility alignment
        volatility_alignment = 0
        normalized_volatility = min(1, volatility / 50)  # Scale to 0-1
        if action == 'buy' and normalized_volatility < 0.3:
            # Buying in low volatility can be good for accumulation
            volatility_alignment = (0.3 - normalized_volatility) * self.reward_weights['volatility_alignment']
        elif action == 'sell' and normalized_volatility > 0.7:
            # Selling in high volatility can protect from crashes
            volatility_alignment = (normalized_volatility - 0.7) * self.reward_weights['volatility_alignment']
        
        # Liquidity efficiency
        liquidity_efficiency = 0
        if liquidity_usd > 0 and position_size_usd > 0:
            # Penalize if position size is too large relative to liquidity
            position_to_liquidity_ratio = position_size_usd / liquidity_usd
            if position_to_liquidity_ratio > 0.1:  # Using more than 10% of liquidity
                liquidity_efficiency = -position_to_liquidity_ratio * self.reward_weights['liquidity_efficiency']
            else:
                # Reward for appropriate position sizing
                liquidity_efficiency = (0.1 - position_to_liquidity_ratio) * self.reward_weights['liquidity_efficiency']
        
        # Slippage penalty
        slippage_penalty = 0
        total_slippage = entry_slippage_pct + exit_slippage_pct
        if total_slippage > 0:
            slippage_penalty = -total_slippage * self.reward_weights['slippage_penalty']
        
        # Whale alignment - reward for aligning with whale activity
        whale_alignment = 0
        if whale_activity != 0:  # If there's significant whale activity
            if (action == 'buy' and whale_activity > 0) or (action == 'sell' and whale_activity < 0):
                # Aligned with whales
                whale_alignment = abs(whale_activity) * self.reward_weights['whale_alignment']
        
        # Timing precision - reward for buying on dips and selling on peaks
        timing_precision = 0
        if action == 'buy' and price_momentum < -0.3:
            # Buying when momentum is negative (price dipping)
            timing_precision = abs(price_momentum) * self.reward_weights['timing_precision']
        elif action == 'sell' and price_momentum > 0.3:
            # Selling when momentum is positive (price rising)
            timing_precision = price_momentum * self.reward_weights['timing_precision']
        
        # Combine all components
        total_reward = (
            base_reward + 
            risk_adjustment + 
            target_bonus + 
            duration_factor +
            market_alignment +
            volatility_alignment +
            liquidity_efficiency +
            slippage_penalty +
            whale_alignment +
            timing_precision
        )
        
        # Update statistics
        if action != 'hold':  # Only count actual trades
            self.total_trades += 1
            if profit_pct > 0:
                self.profitable_trades += 1
            self.cumulative_pnl += profit_pct
            
            # Save stats periodically
            if self.total_trades % 10 == 0:
                self._save_stats()
        
        # Log significant rewards
        if abs(total_reward) > 5:
            logger.info(f"Significant reward calculated: {total_reward:.2f} for action={action}, "
                        f"profit={profit_pct:.2f}%, risk={risk_score}")
        
        return total_reward


class DRLModel:
    """
    Enhanced DRL model implementation using neural networks
    
    This implementation leverages scikit-learn's MLPRegressor as a Q-network
    for improved performance with real market data
    """
    
    def __init__(self, state_size: int, action_size: int, config: Dict[str, Any]):
        """
        Initialize the DRL model
        
        Args:
            state_size (int): Size of state vector
            action_size (int): Size of action space
            config (Dict[str, Any]): Model configuration
        """
        self.state_size = state_size
        self.action_size = action_size
        self.config = config
        
        # Learning parameters
        self.learning_rate = config.get('learning_rate', 0.001)
        self.gamma = config.get('gamma', 0.95)  # discount factor
        self.epsilon = config.get('epsilon', 1.0)  # exploration rate
        self.epsilon_decay = config.get('epsilon_decay', 0.995)
        self.epsilon_min = config.get('epsilon_min', 0.05)
        self.batch_size = config.get('batch_size', 32)
        
        # Initialize memory buffer for experience replay
        self.memory_size = config.get('memory_size', 10000)
        self.memory = deque(maxlen=self.memory_size)
        
        # Priority memory for important experiences
        self.priority_memory = deque(maxlen=config.get('priority_memory_size', 1000))
        
        # Statistics for monitoring
        self.training_iterations = 0
        self.total_reward = 0
        self.recent_rewards = deque(maxlen=100)  # Store last 100 rewards
        
        # Model path
        self.model_dir = 'models/drl'
        os.makedirs(self.model_dir, exist_ok=True)
        self.model_path = os.path.join(self.model_dir, 'drl_model.joblib')
        
        # Create neural network model
        self._create_model()
        
        # Load existing model if available
        self._load_model()
        
        logger.info(f"Enhanced DRLModel initialized with {state_size} states and {action_size} actions")
    
    def _create_model(self):
        """Create the neural network model"""
        # Get hidden layer sizes from config or use defaults
        hidden_layer_sizes = self.config.get('hidden_layer_sizes', (64, 32))
        
        # Create Q-network using scikit-learn's MLPRegressor
        self.model = MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            activation='relu',
            solver='adam',
            alpha=0.0001,  # L2 regularization
            batch_size='auto',
            learning_rate='constant',
            learning_rate_init=self.learning_rate,
            max_iter=1,  # We'll handle iterations manually
            warm_start=True,  # Enable incremental learning
            random_state=42,
        )
        
        # Create initial fake data to initialize the model
        # This is needed because scikit-learn requires an initial fit
        X_init = np.random.normal(0, 0.1, (10, self.state_size))
        y_init = np.random.normal(0, 0.1, (10, self.action_size))
        
        try:
            # Initialize the model
            self.model.fit(X_init, y_init)
            logger.info("Neural network model initialized")
        except Exception as e:
            logger.error(f"Error initializing neural network: {str(e)}")
            # Fallback to simpler model if needed
            self._create_fallback_model()
    
    def _create_fallback_model(self):
        """Create a fallback linear model if neural network initialization fails"""
        logger.warning("Using fallback linear model instead of neural network")
        # Simple linear model as fallback
        self.weights = np.random.rand(self.state_size, self.action_size) * 0.1 - 0.05
        self.using_fallback = True
    
    def predict(self, state: np.ndarray) -> np.ndarray:
        """
        Predict action probabilities from state
        
        Args:
            state (np.ndarray): State vector
            
        Returns:
            np.ndarray: Action probabilities
        """
        # Ensure state is correctly shaped
        if state.ndim == 1:
            state_reshaped = state.reshape(1, -1)
        else:
            state_reshaped = state
            
        try:
            # Use neural network for predictions
            if hasattr(self, 'using_fallback') and self.using_fallback:
                # Fallback linear model
                q_values = np.dot(state, self.weights)
            else:
                # Neural network prediction
                q_values = self.model.predict(state_reshaped)[0]
                
            # Convert to probabilities using softmax
            q_values = np.asarray(q_values, dtype=np.float64)
            exp_values = np.exp(q_values - np.max(q_values))
            probabilities = exp_values / np.sum(exp_values)
            
            return probabilities
            
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            # Return equal probabilities as fallback
            return np.ones(self.action_size) / self.action_size
    
    def act(self, state: np.ndarray) -> int:
        """
        Choose action based on epsilon-greedy policy
        
        Args:
            state (np.ndarray): State vector
            
        Returns:
            int: Selected action index
        """
        # Exploration with annealing epsilon
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        # Exploitation: choose best action
        action_probs = self.predict(state)
        return np.argmax(action_probs)
    
    def remember(self, state: np.ndarray, action: int, reward: float, 
                next_state: np.ndarray, done: bool, priority: bool = False):
        """
        Store experience in memory
        
        Args:
            state (np.ndarray): Current state
            action (int): Action taken
            reward (float): Reward received
            next_state (np.ndarray): Next state
            done (bool): Whether episode is done
            priority (bool): Whether this is a priority experience
        """
        experience = (state, action, reward, next_state, done)
        
        # Store in appropriate memory
        if priority:
            self.priority_memory.append(experience)
        else:
            self.memory.append(experience)
        
        # Track rewards for monitoring
        self.total_reward += reward
        self.recent_rewards.append(reward)
    
    def replay(self, batch_size: int = None):
        """
        Train model on batch of experiences
        
        Args:
            batch_size (int, optional): Batch size for training
        """
        if batch_size is None:
            batch_size = self.batch_size
            
        # Need enough samples in memory
        if len(self.memory) < batch_size // 2:
            return
        
        # Create a combined batch from regular and priority memories
        regular_batch_size = min(batch_size * 3 // 4, len(self.memory))
        priority_batch_size = min(batch_size - regular_batch_size, len(self.priority_memory))
        
        # Sample from both memories
        regular_batch = random.sample(list(self.memory), regular_batch_size)
        priority_batch = random.sample(list(self.priority_memory), priority_batch_size) if priority_batch_size > 0 else []
        
        # Combine batches
        minibatch = regular_batch + priority_batch
        
        # Prepare training data
        states = np.vstack([exp[0].reshape(1, -1) for exp in minibatch])
        next_states = np.vstack([exp[3].reshape(1, -1) for exp in minibatch])
        
        # Get current Q values
        if hasattr(self, 'using_fallback') and self.using_fallback:
            # Fallback linear model
            current_q = np.dot(states, self.weights)
            next_q = np.dot(next_states, self.weights)
        else:
            # Neural network prediction
            try:
                current_q = self.model.predict(states)
                next_q = self.model.predict(next_states)
            except Exception as e:
                logger.error(f"Error in Q prediction during replay: {str(e)}")
                return
        
        # Prepare training arrays
        X_train = []
        y_train = []
        
        # Update Q values with bellman equation
        for idx, (state, action, reward, next_state, done) in enumerate(minibatch):
            target = current_q[idx].copy()
            
            if done:
                target[action] = reward
            else:
                # Q-learning update with discounted future reward
                target[action] = reward + self.gamma * np.max(next_q[idx])
            
            X_train.append(state)
            y_train.append(target)
        
        # Convert to numpy arrays
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        
        # Train the model
        if hasattr(self, 'using_fallback') and self.using_fallback:
            # Update weights for fallback linear model
            for i in range(len(X_train)):
                state = X_train[i]
                target = y_train[i]
                
                # Simplified gradient descent update
                current = np.dot(state, self.weights)
                error = target - current
                
                for s in range(self.state_size):
                    for a in range(self.action_size):
                        self.weights[s, a] += self.learning_rate * error[a] * state[s]
        else:
            # Train neural network
            try:
                self.model.fit(X_train, y_train)
            except Exception as e:
                logger.error(f"Error training model: {str(e)}")
        
        # Track training progress
        self.training_iterations += 1
        
        # Decay exploration rate
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        # Periodically log progress
        if self.training_iterations % 100 == 0:
            avg_reward = sum(self.recent_rewards) / len(self.recent_rewards) if self.recent_rewards else 0
            logger.info(f"DRL training progress: iterations={self.training_iterations}, "
                      f"epsilon={self.epsilon:.4f}, avg_reward={avg_reward:.4f}")
    
    def update(self, experience: Dict[str, Any], reward: float):
        """
        Update model with new experience
        
        Args:
            experience (Dict[str, Any]): Experience data
            reward (float): Calculated reward
        """
        # Extract data from experience
        state = experience.get('state')
        action = experience.get('action_index', 0)
        next_state = experience.get('next_state')
        done = experience.get('done', False)
        
        # Determine if this is a priority experience (high reward or high loss)
        is_priority = abs(reward) > 5.0  # High impact experiences
        
        if state is not None and next_state is not None:
            # Store in memory
            self.remember(state, action, reward, next_state, done, priority=is_priority)
            
            # Train on batch
            self.replay()
            
            # Save periodically
            if self.training_iterations % 500 == 0:
                self.save(self.model_path)
    
    def save(self, filepath: str = None):
        """
        Save model to file
        
        Args:
            filepath (str, optional): File path to save model
        """
        if filepath is None:
            filepath = self.model_path
            
        try:
            if hasattr(self, 'using_fallback') and self.using_fallback:
                # Save weights for fallback model
                np.save(filepath, self.weights)
            else:
                # Save neural network model
                joblib.dump(self.model, filepath)
                
            # Save state
            state_path = os.path.join(self.model_dir, 'drl_state.joblib')
            state = {
                'epsilon': self.epsilon,
                'training_iterations': self.training_iterations,
                'total_reward': self.total_reward,
                'recent_rewards': list(self.recent_rewards),
                'timestamp': datetime.now().isoformat()
            }
            joblib.dump(state, state_path)
            
            logger.info(f"DRL model saved to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error saving DRL model: {str(e)}")
            return False
    
    def _load_model(self):
        """Load saved model if it exists"""
        try:
            # Try to load neural network model
            if os.path.exists(self.model_path):
                if self.model_path.endswith('.npy'):
                    # Load weights for fallback model
                    self.weights = np.load(self.model_path)
                    self.using_fallback = True
                    logger.info(f"Loaded fallback DRL model from {self.model_path}")
                else:
                    # Load neural network model
                    self.model = joblib.load(self.model_path)
                    logger.info(f"Loaded neural network DRL model from {self.model_path}")
                
                # Try to load state
                state_path = os.path.join(self.model_dir, 'drl_state.joblib')
                if os.path.exists(state_path):
                    state = joblib.load(state_path)
                    self.epsilon = state.get('epsilon', self.epsilon)
                    self.training_iterations = state.get('training_iterations', 0)
                    self.total_reward = state.get('total_reward', 0)
                    if 'recent_rewards' in state:
                        self.recent_rewards = deque(state['recent_rewards'], maxlen=100)
                    
                    logger.info(f"Loaded DRL model state: iterations={self.training_iterations}, "
                              f"epsilon={self.epsilon:.4f}")
                
                return True
        except Exception as e:
            logger.error(f"Error loading DRL model: {str(e)}")
            # If there's an error, we'll initialize a new model
            self._create_model()
            return False
        
        return False
    
    def load(self, filepath: str):
        """
        Load model from file
        
        Args:
            filepath (str): File path to load model from
        """
        try:
            if filepath.endswith('.npy'):
                # Load weights for fallback model
                self.weights = np.load(filepath)
                self.using_fallback = True
                logger.info(f"Loaded fallback DRL model from {filepath}")
            else:
                # Load neural network model
                self.model = joblib.load(filepath)
                logger.info(f"Loaded neural network DRL model from {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error loading DRL model: {str(e)}")
            return False
            
    def get_stats(self) -> Dict[str, Any]:
        """
        Get model statistics
        
        Returns:
            Dict[str, Any]: Model statistics
        """
        avg_reward = sum(self.recent_rewards) / len(self.recent_rewards) if self.recent_rewards else 0
        
        return {
            'model_type': 'fallback_linear' if (hasattr(self, 'using_fallback') and self.using_fallback) else 'neural_network',
            'training_iterations': self.training_iterations,
            'epsilon': self.epsilon,
            'memory_size': len(self.memory),
            'priority_memory_size': len(self.priority_memory),
            'total_reward': self.total_reward,
            'average_reward': avg_reward,
            'learning_rate': self.learning_rate,
        }


class DRLStrategy:
    """
    Advanced trading strategy using Deep Reinforcement Learning with real market data
    
    This strategy dynamically learns optimal trading decisions through
    experience and continuous reinforcement learning using real market data
    and technical/on-chain analysis.
    """
    
    def __init__(self, 
                config_manager: ConfigManager,
                state_manager: StateManager):
        """
        Initialize the DRLStrategy
        
        Args:
            config_manager (ConfigManager): Configuration manager instance
            state_manager (StateManager): State manager instance
        """
        self.config_manager = config_manager
        self.state_manager = state_manager
        
        # Load DRL config
        drl_config = self.config_manager.get('strategy.drl', {})
        
        # Initialize components
        self.state_encoder = StateEncoder()
        self.reward_calculator = RewardCalculator()
        
        # Action mapping
        self.actions = ['hold', 'buy', 'sell']
        self.action_size = len(self.actions)
        
        # Feature list length is the state size
        self.state_size = len(self.state_encoder.feature_list)
        
        # Initialize model directory
        self.model_dir = 'models/drl'
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Initialize data collection for real market data
        self.market_data_buffer = deque(maxlen=drl_config.get('market_data_buffer_size', 1000))
        self.training_data_file = os.path.join(self.model_dir, 'market_data.joblib')
        
        # Initialize DRL model
        self.model = DRLModel(self.state_size, self.action_size, drl_config)
        
        # Load model if specified
        model_path = drl_config.get('model_path', os.path.join(self.model_dir, 'drl_model.joblib'))
        if os.path.exists(model_path):
            self.model.load(model_path)
        
        # Training parameters
        self.training_enabled = drl_config.get('training_enabled', True)
        self.exploration_enabled = drl_config.get('exploration_enabled', True)
        self.save_interval = drl_config.get('save_interval', 100)  # Save every 100 trades
        self.batch_training_size = drl_config.get('batch_training_size', 64)
        self.update_scaler_interval = drl_config.get('update_scaler_interval', 50)
        
        # Strategy state
        self.trade_count = 0
        self.experiences = []
        self.performance_metrics = {
            'win_rate': 0.0,
            'avg_profit': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0,
            'model_confidence': 0.0
        }
        
        # Load market data if available
        self._load_market_data()
        
        logger.info(f"Enhanced DRLStrategy initialized with {self.state_size} state features")
    
    def _load_market_data(self):
        """Load previously saved market data"""
        try:
            if os.path.exists(self.training_data_file):
                data = joblib.load(self.training_data_file)
                if isinstance(data, list) and len(data) > 0:
                    self.market_data_buffer.extend(data)
                    
                    # Use the data to update the state encoder
                    if len(self.market_data_buffer) >= 10:  # Need enough samples
                        self.state_encoder.update_scaler(list(self.market_data_buffer))
                        
                    logger.info(f"Loaded {len(self.market_data_buffer)} market data samples for DRL training")
        except Exception as e:
            logger.warning(f"Error loading market data: {str(e)}")
    
    def _save_market_data(self):
        """Save collected market data for future training"""
        try:
            if len(self.market_data_buffer) > 0:
                joblib.dump(list(self.market_data_buffer), self.training_data_file)
                logger.info(f"Saved {len(self.market_data_buffer)} market data samples")
        except Exception as e:
            logger.warning(f"Error saving market data: {str(e)}")
    
    def collect_market_data(self, market_state: Dict[str, Any]):
        """
        Collect real market data for training
        
        Args:
            market_state (Dict[str, Any]): Current market state
        """
        # Only store states with sufficient data
        required_fields = ['price_usd', 'liquidity_usd', 'volume_24h_usd']
        if all(field in market_state for field in required_fields):
            # Add timestamp if not present
            if 'timestamp' not in market_state:
                market_state['timestamp'] = time.time()
                
            # Store in buffer
            self.market_data_buffer.append(market_state)
            
            # Periodically update the scaler with new data
            if len(self.market_data_buffer) % self.update_scaler_interval == 0:
                self.state_encoder.update_scaler(list(self.market_data_buffer)[-self.update_scaler_interval:])
                
            # Periodically save the data
            if len(self.market_data_buffer) % 100 == 0:
                self._save_market_data()
                
            return True
        return False
    
    async def decide_action(self, market_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Decide trading action based on current market state
        
        Args:
            market_state (Dict[str, Any]): Current market state
            
        Returns:
            Dict[str, Any]: Action decision with parameters
        """
        # Encode market state
        encoded_state = self.state_encoder.encode(market_state)
        
        # Get action probabilities
        action_probs = self.model.predict(encoded_state)
        
        # During training with exploration, use epsilon-greedy
        if self.training_enabled and self.exploration_enabled:
            action_index = self.model.act(encoded_state)
        else:
            # In production, use the highest probability action
            action_index = np.argmax(action_probs)
        
        # Map to action string
        action = self.actions[action_index]
        
        # Calculate confidence based on probability
        confidence = float(action_probs[action_index] * 100)
        
        # Position sizing based on confidence and risk
        position_size = self._calculate_position_size(
            confidence, market_state.get('risk_score', 50))
        
        # Calculate take profit and stop loss
        take_profit, stop_loss = self._calculate_profit_levels(
            confidence, market_state.get('volatility', 10))
        
        # Build decision
        decision = {
            'action': action,
            'confidence': confidence,
            'position_size': position_size,
            'take_profit': take_profit,
            'stop_loss': stop_loss,
            'params': {
                'action_probs': action_probs.tolist(),
                'action_index': action_index,
                'state': encoded_state,
                'timestamp': time.time()
            }
        }
        
        return decision
    
    def _calculate_position_size(self, confidence: float, risk_score: float) -> float:
        """
        Calculate position size based on confidence and risk
        
        Args:
            confidence (float): Action confidence (0-100)
            risk_score (float): Risk score (0-100)
            
        Returns:
            float: Position size percentage (0-100)
        """
        # Base size based on confidence
        base_size = confidence / 2  # 0-50% range
        
        # Risk adjustment
        risk_factor = 1 - (risk_score / 200)  # 0.5-1.0 range
        
        # Final size
        position_size = base_size * risk_factor
        
        return max(1, min(50, position_size))  # Cap at 1-50%
    
    def _calculate_profit_levels(self, confidence: float, volatility: float) -> Tuple[float, float]:
        """
        Calculate take profit and stop loss levels
        
        Args:
            confidence (float): Action confidence (0-100)
            volatility (float): Market volatility
            
        Returns:
            Tuple[float, float]: Take profit and stop loss percentages
        """
        # Base take profit and stop loss
        base_tp = 15  # 15% default take profit
        base_sl = 7   # 7% default stop loss
        
        # Adjust based on volatility
        volatility_factor = max(0.5, min(3, volatility / 10))
        
        # Adjust based on confidence
        confidence_factor = max(0.8, min(1.5, confidence / 50))
        
        # Calculate final values
        take_profit = base_tp * volatility_factor * confidence_factor
        stop_loss = base_sl * volatility_factor * (2 - confidence_factor)
        
        return take_profit, stop_loss
    
    async def learn_from_experience(self, 
                                   experience: Dict[str, Any], 
                                   outcome: Dict[str, Any]):
        """
        Learn from trading experience
        
        Args:
            experience (Dict[str, Any]): Original trading decision data
            outcome (Dict[str, Any]): Trade outcome data
        """
        if not self.training_enabled:
            return
        
        try:
            # Extract original decision params
            params = experience.get('params', {})
            if not params:
                logger.warning("No params in experience, cannot learn")
                return
            
            # Extract state and action
            state = params.get('state')
            action_index = params.get('action_index')
            
            if state is None or action_index is None:
                logger.warning("Missing state or action in experience")
                return
            
            # Extract outcome data
            profit_pct = outcome.get('profit_pct', 0)
            risk_score = outcome.get('risk_score', 50)
            duration_hours = outcome.get('duration_hours', 0)
            target_reached = outcome.get('target_reached', False)
            
            # Encode new state if available
            next_state = None
            if 'current_state' in outcome:
                next_state = self.state_encoder.encode(outcome['current_state'])
            else:
                # If no new state, use the original but with small random changes
                next_state = state + np.random.normal(0, 0.01, self.state_size)
            
            # Prepare experience for learning
            learning_exp = {
                'state': state,
                'action_index': action_index,
                'next_state': next_state,
                'done': outcome.get('done', False),
                'profit_pct': profit_pct,
                'risk_score': risk_score,
                'duration_hours': duration_hours,
                'target_reached': target_reached
            }
            
            # Calculate reward
            reward = self.reward_calculator.calculate(learning_exp)
            
            # Update the model
            self.model.update(learning_exp, reward)
            
            # Save periodically
            self.trade_count += 1
            if self.trade_count % self.save_interval == 0:
                model_path = self.config_manager.get('strategy.drl.model_path', 'models/drl/model.npy')
                self.model.save(model_path)
                
                # Log performance metrics
                self._log_performance_metrics()
            
        except Exception as e:
            logger.error(f"Error learning from experience: {str(e)}")
    
    def _log_performance_metrics(self):
        """Log DRL performance metrics"""
        metrics = {
            'trade_count': self.trade_count,
            'exploration_rate': self.model.epsilon
        }
        
        # Update state manager with metrics
        self.state_manager.update_component_metrics('drl_strategy', metrics)
        
        logger.info(f"DRL metrics updated: trade_count={self.trade_count}, "
                  f"exploration_rate={self.model.epsilon:.4f}")