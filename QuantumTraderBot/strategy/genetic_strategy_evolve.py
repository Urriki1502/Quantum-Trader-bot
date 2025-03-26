"""
Genetic Strategy Evolver Component
Responsible for evolving trading strategies using genetic algorithms,
enabling the system to continuously adapt to changing market conditions.
"""

import asyncio
import logging
import time
import random
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from copy import deepcopy

from core.state_manager import StateManager
from core.config_manager import ConfigManager

logger = logging.getLogger(__name__)

class StrategyGene:
    """Represents a single strategy gene with parameters"""
    
    def __init__(self, 
                name: str, 
                parameters: Dict[str, Any],
                metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize a strategy gene
        
        Args:
            name (str): Strategy name
            parameters (Dict[str, Any]): Strategy parameters
            metadata (Dict[str, Any], optional): Strategy metadata
        """
        self.name = name
        self.parameters = parameters
        self.metadata = metadata or {}
        self.fitness = 0.0
        self.generation = 0
        self.last_updated = time.time()
        self.id = f"{name}_{int(time.time() * 1000)}_{random.randint(1000, 9999)}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert gene to dictionary"""
        return {
            'id': self.id,
            'name': self.name,
            'parameters': self.parameters,
            'metadata': self.metadata,
            'fitness': self.fitness,
            'generation': self.generation,
            'last_updated': self.last_updated
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StrategyGene':
        """Create gene from dictionary"""
        gene = cls(
            name=data['name'],
            parameters=data['parameters'],
            metadata=data.get('metadata', {})
        )
        gene.id = data.get('id', gene.id)
        gene.fitness = data.get('fitness', 0.0)
        gene.generation = data.get('generation', 0)
        gene.last_updated = data.get('last_updated', time.time())
        return gene


class FitnessEvaluator:
    """Evaluates the fitness of strategy genes"""
    
    def __init__(self, config_manager: ConfigManager):
        """
        Initialize the FitnessEvaluator
        
        Args:
            config_manager (ConfigManager): Configuration manager instance
        """
        self.config_manager = config_manager
        
        # Fitness metrics weights
        self.fitness_weights = {
            'profit_loss': 0.5,       # PnL is the most important
            'win_rate': 0.2,          # Win rate is important but secondary
            'risk_adjusted_return': 0.15,  # Risk-adjusted returns
            'consistency': 0.1,       # Consistency across market conditions
            'trade_frequency': 0.05   # Lower importance for trade frequency
        }
    
    async def evaluate(self, gene: StrategyGene, performance_data: Dict[str, Any]) -> float:
        """
        Evaluate the fitness of a strategy gene
        
        Args:
            gene (StrategyGene): Strategy gene to evaluate
            performance_data (Dict[str, Any]): Performance data for the strategy
            
        Returns:
            float: Fitness score (0-100)
        """
        # Extract performance metrics
        profit_loss = performance_data.get('profit_loss_pct', 0)
        win_rate = performance_data.get('win_rate', 0)
        risk_adjusted_return = performance_data.get('sharpe_ratio', 0)
        consistency = performance_data.get('consistency', 0)
        trade_count = performance_data.get('trade_count', 0)
        
        # Normalize metrics
        normalized_pl = self._normalize_profit_loss(profit_loss)
        normalized_win_rate = win_rate  # Already 0-100
        normalized_rar = self._normalize_risk_adjusted_return(risk_adjusted_return)
        normalized_consistency = consistency  # Already 0-100
        normalized_trade_freq = self._normalize_trade_frequency(trade_count)
        
        # Calculate weighted fitness
        fitness = (
            normalized_pl * self.fitness_weights['profit_loss'] +
            normalized_win_rate * self.fitness_weights['win_rate'] +
            normalized_rar * self.fitness_weights['risk_adjusted_return'] +
            normalized_consistency * self.fitness_weights['consistency'] +
            normalized_trade_freq * self.fitness_weights['trade_frequency']
        )
        
        return fitness
    
    def _normalize_profit_loss(self, profit_loss: float) -> float:
        """
        Normalize profit/loss to 0-100 scale
        
        Args:
            profit_loss (float): Profit/loss percentage
            
        Returns:
            float: Normalized score (0-100)
        """
        # Use sigmoid function centered at 0 with scaling
        if profit_loss >= 0:
            # Positive returns: 50-100 range
            return 50 + min(50, profit_loss * 5)
        else:
            # Negative returns: 0-50 range, more severe penalty
            return max(0, 50 + profit_loss * 10)
    
    def _normalize_risk_adjusted_return(self, sharpe_ratio: float) -> float:
        """
        Normalize risk-adjusted return to 0-100 scale
        
        Args:
            sharpe_ratio (float): Sharpe ratio
            
        Returns:
            float: Normalized score (0-100)
        """
        if sharpe_ratio <= 0:
            return max(0, 40 + sharpe_ratio * 20)  # 0-40 range for negative
        else:
            return min(100, 40 + sharpe_ratio * 20)  # 40-100 range for positive
    
    def _normalize_trade_frequency(self, trade_count: int) -> float:
        """
        Normalize trade frequency to 0-100 scale
        
        Args:
            trade_count (int): Number of trades
            
        Returns:
            float: Normalized score (0-100)
        """
        # Too few trades is bad (not enough data)
        # Too many trades is also suboptimal (excessive fees)
        # Ideal range is 20-100 trades
        if trade_count < 5:
            return trade_count * 10  # 0-40 range
        elif trade_count < 20:
            return 40 + (trade_count - 4) * 3  # 40-88 range
        elif trade_count <= 100:
            return 88 + (trade_count - 19) * 0.15  # 88-100 range
        else:
            return max(50, 100 - (trade_count - 100) * 0.1)  # Decreasing after 100


class MutationEngine:
    """Performs mutations on strategy genes"""
    
    def __init__(self, config_manager: ConfigManager):
        """
        Initialize the MutationEngine
        
        Args:
            config_manager (ConfigManager): Configuration manager instance
        """
        self.config_manager = config_manager
        
        # Mutation settings
        self.mutation_rate = self.config_manager.get(
            'genetic.mutation_rate', 0.3)
        self.mutation_strength = self.config_manager.get(
            'genetic.mutation_strength', 0.2)
        self.parameter_mutation_prob = self.config_manager.get(
            'genetic.parameter_mutation_prob', 0.5)
    
    def mutate(self, gene: StrategyGene) -> StrategyGene:
        """
        Mutate a strategy gene
        
        Args:
            gene (StrategyGene): Strategy gene to mutate
            
        Returns:
            StrategyGene: Mutated gene
        """
        # Clone the gene to avoid modifying the original
        mutated_gene = deepcopy(gene)
        
        # Only mutate with certain probability
        if random.random() > self.mutation_rate:
            return mutated_gene
        
        # Mutate parameters
        mutated_params = {}
        for param_name, param_value in gene.parameters.items():
            # Skip metadata parameters
            if param_name.startswith('_'):
                mutated_params[param_name] = param_value
                continue
                
            # Only mutate with certain probability per parameter
            if random.random() > self.parameter_mutation_prob:
                mutated_params[param_name] = param_value
                continue
            
            # Mutate based on type
            if isinstance(param_value, (int, float)):
                mutated_params[param_name] = self._mutate_numeric(param_name, param_value)
            elif isinstance(param_value, bool):
                mutated_params[param_name] = random.random() < 0.5
            elif isinstance(param_value, str) and param_name.endswith('_type'):
                # For type parameters, randomly select from predefined options
                options = self._get_type_options(param_name)
                mutated_params[param_name] = random.choice(options)
            else:
                # For complex types, keep original
                mutated_params[param_name] = param_value
        
        # Update the gene
        mutated_gene.parameters = mutated_params
        mutated_gene.generation = gene.generation + 1
        mutated_gene.last_updated = time.time()
        mutated_gene.id = f"{gene.name}_{int(time.time() * 1000)}_{random.randint(1000, 9999)}"
        
        return mutated_gene
    
    def _mutate_numeric(self, param_name: str, value: float) -> float:
        """
        Mutate a numeric parameter
        
        Args:
            param_name (str): Parameter name
            value (float): Current value
            
        Returns:
            float: Mutated value
        """
        # Different parameters have different scales and constraints
        if param_name.endswith('_pct') or param_name.endswith('_percentage'):
            # Percentage parameters (0-100)
            min_val, max_val = 0, 100
            mutation_amt = self.mutation_strength * 20  # Higher mutation for percentages
        elif param_name.endswith('_ratio'):
            # Ratio parameters (usually 0-1 or 0-10)
            if value < 2:
                min_val, max_val = 0, 2
            else:
                min_val, max_val = 0, 10
            mutation_amt = self.mutation_strength * value * 0.5
        elif param_name.endswith('_threshold'):
            # Threshold parameters
            min_val = max(0, value * 0.5)
            max_val = value * 2
            mutation_amt = self.mutation_strength * value * 0.3
        elif param_name.endswith('_period'):
            # Period parameters (usually integers)
            is_int = isinstance(value, int)
            min_val = max(1, value * 0.5)
            max_val = value * 2
            mutation_amt = self.mutation_strength * value * 0.3
        else:
            # Default numeric parameters
            min_val = value * 0.5
            max_val = value * 2
            mutation_amt = self.mutation_strength * value * 0.2
        
        # Apply mutation
        delta = random.uniform(-mutation_amt, mutation_amt)
        mutated_value = value + delta
        
        # Ensure value is within bounds
        mutated_value = max(min_val, min(max_val, mutated_value))
        
        # Convert back to int if original was int
        if isinstance(value, int):
            mutated_value = int(round(mutated_value))
            
        return mutated_value
    
    def _get_type_options(self, param_name: str) -> List[str]:
        """
        Get options for type parameters
        
        Args:
            param_name (str): Parameter name
            
        Returns:
            List[str]: Available options
        """
        if param_name == 'indicator_type':
            return ['macd', 'rsi', 'bollinger', 'ema', 'sma', 'atr', 'obv']
        elif param_name == 'signal_type':
            return ['crossover', 'threshold', 'divergence', 'trend']
        elif param_name == 'exit_type':
            return ['fixed', 'trailing', 'indicator', 'hybrid']
        else:
            # Default options
            return ['default', 'alternative', 'aggressive', 'conservative']


class GeneticStrategyEvolver:
    """
    Evolves trading strategies using genetic algorithms
    
    This system allows strategies to adapt to changing market conditions
    through evolution, mutation, and natural selection.
    """
    
    def __init__(self, 
                config_manager: ConfigManager, 
                state_manager: StateManager):
        """
        Initialize the GeneticStrategyEvolver
        
        Args:
            config_manager (ConfigManager): Configuration manager instance
            state_manager (StateManager): State manager instance
        """
        self.config_manager = config_manager
        self.state_manager = state_manager
        
        # Initialize components
        self.fitness_evaluator = FitnessEvaluator(config_manager)
        self.mutation_engine = MutationEngine(config_manager)
        
        # Evolution settings
        self.population_size = self.config_manager.get('genetic.population_size', 20)
        self.elite_size = self.config_manager.get('genetic.elite_size', 4)
        self.generation_limit = self.config_manager.get('genetic.generation_limit', 100)
        self.crossover_rate = self.config_manager.get('genetic.crossover_rate', 0.7)
        
        # Population
        self.population = []
        self.generation = 0
        self.best_gene = None
        
        # Strategy templates
        self.strategy_templates = self._load_strategy_templates()
        
        logger.info("GeneticStrategyEvolver initialized")
    
    def _load_strategy_templates(self) -> Dict[str, Dict[str, Any]]:
        """
        Load strategy templates from configuration
        
        Returns:
            Dict[str, Dict[str, Any]]: Strategy templates
        """
        default_templates = {
            'momentum': {
                'parameters': {
                    'indicator_type': 'macd',
                    'fast_period': 12,
                    'slow_period': 26,
                    'signal_period': 9,
                    'entry_threshold': 0.0,
                    'exit_threshold': 0.0,
                    'stop_loss_pct': 7.0,
                    'take_profit_pct': 15.0,
                    'position_size_pct': 10.0,
                    'max_holding_period_hours': 24,
                    'use_trailing_stop': True,
                    'trailing_stop_distance_pct': 3.0,
                    'min_volume_usd': 10000,
                    'min_liquidity_usd': 20000
                },
                'metadata': {
                    'description': 'Momentum-based strategy using MACD',
                    'version': 1.0,
                    'market_condition': 'trending'
                }
            },
            'breakout': {
                'parameters': {
                    'indicator_type': 'bollinger',
                    'period': 20,
                    'std_dev': 2.0,
                    'breakout_threshold_pct': 2.0,
                    'confirmation_period': 3,
                    'stop_loss_pct': 8.0,
                    'take_profit_pct': 20.0,
                    'position_size_pct': 10.0,
                    'max_holding_period_hours': 48,
                    'use_trailing_stop': True,
                    'trailing_stop_distance_pct': 5.0,
                    'min_volume_usd': 15000,
                    'min_liquidity_usd': 30000
                },
                'metadata': {
                    'description': 'Breakout strategy using Bollinger Bands',
                    'version': 1.0,
                    'market_condition': 'volatile'
                }
            },
            'reversal': {
                'parameters': {
                    'indicator_type': 'rsi',
                    'period': 14,
                    'oversold_threshold': 30,
                    'overbought_threshold': 70,
                    'confirmation_period': 2,
                    'stop_loss_pct': 6.0,
                    'take_profit_pct': 12.0,
                    'position_size_pct': 8.0,
                    'max_holding_period_hours': 24,
                    'use_trailing_stop': False,
                    'min_volume_usd': 10000,
                    'min_liquidity_usd': 25000
                },
                'metadata': {
                    'description': 'Reversal strategy using RSI',
                    'version': 1.0,
                    'market_condition': 'ranging'
                }
            },
            'new_listing': {
                'parameters': {
                    'max_age_hours': 24,
                    'min_initial_liquidity_usd': 15000,
                    'min_liquidity_growth_pct': 20,
                    'entry_delay_minutes': 10,
                    'stop_loss_pct': 10.0,
                    'take_profit_pct': 25.0,
                    'position_size_pct': 5.0,
                    'max_holding_period_hours': 12,
                    'use_trailing_stop': True,
                    'trailing_stop_distance_pct': 7.0
                },
                'metadata': {
                    'description': 'New listing strategy for newly created tokens',
                    'version': 1.0,
                    'market_condition': 'any'
                }
            },
            'volume_spike': {
                'parameters': {
                    'volume_increase_threshold_pct': 200,
                    'price_change_min_pct': 3.0,
                    'confirmation_period': 2,
                    'stop_loss_pct': 8.0,
                    'take_profit_pct': 15.0,
                    'position_size_pct': 7.0,
                    'max_holding_period_hours': 24,
                    'use_trailing_stop': True,
                    'trailing_stop_distance_pct': 4.0,
                    'min_base_volume_usd': 5000
                },
                'metadata': {
                    'description': 'Volume spike strategy for sudden interest in tokens',
                    'version': 1.0,
                    'market_condition': 'any'
                }
            }
        }
        
        # Load from config (fallback to defaults)
        templates = self.config_manager.get('strategy.templates', default_templates)
        
        return templates
    
    async def initialize_population(self):
        """Initialize the population with strategy templates and variations"""
        self.population = []
        
        # Start with the base templates
        for name, template in self.strategy_templates.items():
            gene = StrategyGene(
                name=name,
                parameters=template['parameters'].copy(),
                metadata=template['metadata'].copy()
            )
            gene.metadata['type'] = 'template'
            self.population.append(gene)
        
        # Add variations of each template
        for name, template in self.strategy_templates.items():
            for i in range(3):  # 3 variations per template
                gene = StrategyGene(
                    name=f"{name}_var{i+1}",
                    parameters=template['parameters'].copy(),
                    metadata=template['metadata'].copy()
                )
                gene.metadata['type'] = 'variation'
                gene.metadata['parent'] = name
                
                # Apply random variations
                for param_name, param_value in gene.parameters.items():
                    if isinstance(param_value, (int, float)) and not param_name.startswith('_'):
                        # Random variation within ±20%
                        variation_factor = 1.0 + random.uniform(-0.2, 0.2)
                        gene.parameters[param_name] = param_value * variation_factor
                        
                        # Round integers
                        if isinstance(param_value, int):
                            gene.parameters[param_name] = int(round(gene.parameters[param_name]))
                
                self.population.append(gene)
        
        # Fill remaining population with random strategies
        while len(self.population) < self.population_size:
            # Choose a random template as base
            template_name = random.choice(list(self.strategy_templates.keys()))
            template = self.strategy_templates[template_name]
            
            gene = StrategyGene(
                name=f"{template_name}_rand{len(self.population)}",
                parameters=template['parameters'].copy(),
                metadata=template['metadata'].copy()
            )
            gene.metadata['type'] = 'random'
            
            # Apply stronger random variations to all parameters
            for param_name, param_value in gene.parameters.items():
                if isinstance(param_value, (int, float)) and not param_name.startswith('_'):
                    # Stronger random variation within ±50%
                    variation_factor = 1.0 + random.uniform(-0.5, 0.5)
                    gene.parameters[param_name] = param_value * variation_factor
                    
                    # Round integers
                    if isinstance(param_value, int):
                        gene.parameters[param_name] = int(round(gene.parameters[param_name]))
                elif isinstance(param_value, bool):
                    # Randomly flip booleans
                    gene.parameters[param_name] = random.random() < 0.5
            
            self.population.append(gene)
        
        self.generation = 0
        logger.info(f"Population initialized with {len(self.population)} genes")
    
    async def evolve_strategies(self, performance_data: Dict[str, Dict[str, Any]]) -> List[StrategyGene]:
        """
        Evolve strategies based on performance data
        
        Args:
            performance_data (Dict[str, Dict[str, Any]]): Performance data by strategy name
            
        Returns:
            List[StrategyGene]: New population of strategy genes
        """
        # Initialize population if empty
        if not self.population:
            await self.initialize_population()
            
        # Evaluate fitness of current population
        fitness_scores = await self._evaluate_fitness(performance_data)
        
        # Sort population by fitness
        self.population.sort(key=lambda g: g.fitness, reverse=True)
        
        # Store best gene
        if self.population:
            self.best_gene = deepcopy(self.population[0])
            logger.info(f"Best gene: {self.best_gene.name}, fitness: {self.best_gene.fitness:.2f}")
        
        # Select parents
        parents = self._select_parents()
        
        # Keep elite individuals
        elite = self.population[:self.elite_size]
        
        # Create new generation through crossover and mutation
        new_generation = elite.copy()  # Keep elite unchanged
        
        # Create offspring
        while len(new_generation) < self.population_size:
            if random.random() < self.crossover_rate:
                # Select two random parents
                parent1 = random.choice(parents)
                parent2 = random.choice(parents)
                
                # Create child through crossover
                child = self._crossover(parent1, parent2)
            else:
                # Clone a parent
                child = deepcopy(random.choice(parents))
            
            # Apply mutation
            child = self.mutation_engine.mutate(child)
            new_generation.append(child)
        
        # Update population
        self.population = new_generation
        self.generation += 1
        
        # Update state manager with evolution metrics
        self._update_evolution_metrics()
        
        logger.info(f"Evolved to generation {self.generation}")
        return self.population
    
    async def _evaluate_fitness(self, performance_data: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
        """
        Evaluate fitness of all genes in the population
        
        Args:
            performance_data (Dict[str, Dict[str, Any]]): Performance data by strategy name
            
        Returns:
            Dict[str, float]: Fitness scores by gene ID
        """
        fitness_scores = {}
        
        for gene in self.population:
            # Get performance data for this strategy
            strategy_data = performance_data.get(gene.name, {})
            
            if not strategy_data:
                # If no performance data, use a simplified fitness estimate
                gene.fitness = self._estimate_fitness(gene)
            else:
                # Evaluate fitness based on performance
                gene.fitness = await self.fitness_evaluator.evaluate(gene, strategy_data)
            
            fitness_scores[gene.id] = gene.fitness
        
        return fitness_scores
    
    def _estimate_fitness(self, gene: StrategyGene) -> float:
        """
        Estimate fitness for genes without performance data
        
        Args:
            gene (StrategyGene): Strategy gene
            
        Returns:
            float: Estimated fitness (0-100)
        """
        # Base fitness score
        base_fitness = 50.0
        
        # For template strategies, give higher base fitness
        if gene.metadata.get('type') == 'template':
            base_fitness = 60.0
        
        # Apply some randomness (±10)
        random_factor = random.uniform(-10, 10)
        
        # Estimate based on parameter reasonableness
        params = gene.parameters
        parameter_score = 0
        
        # Check if stop loss exists and is reasonable
        if 'stop_loss_pct' in params:
            sl = params['stop_loss_pct']
            if 3 <= sl <= 15:
                parameter_score += 5
            elif sl > 15:
                parameter_score -= 5
        
        # Check if take profit exists and is reasonable
        if 'take_profit_pct' in params:
            tp = params['take_profit_pct']
            if 10 <= tp <= 30:
                parameter_score += 5
            elif tp < 10:
                parameter_score -= 5
        
        # Check position sizing
        if 'position_size_pct' in params:
            ps = params['position_size_pct']
            if 5 <= ps <= 15:
                parameter_score += 5
            elif ps > 20:
                parameter_score -= 10
        
        return min(100, max(0, base_fitness + parameter_score + random_factor))
    
    def _select_parents(self) -> List[StrategyGene]:
        """
        Select parents for the next generation using tournament selection
        
        Returns:
            List[StrategyGene]: Selected parents
        """
        # Tournament selection
        tournament_size = 3
        parents = []
        
        # Select parents proportional to fitness
        for _ in range(self.population_size):
            # Select random candidates for tournament
            candidates = random.sample(self.population, min(tournament_size, len(self.population)))
            
            # Select the best candidate
            winner = max(candidates, key=lambda g: g.fitness)
            parents.append(winner)
        
        return parents
    
    def _crossover(self, parent1: StrategyGene, parent2: StrategyGene) -> StrategyGene:
        """
        Create child through crossover of two parents
        
        Args:
            parent1 (StrategyGene): First parent
            parent2 (StrategyGene): Second parent
            
        Returns:
            StrategyGene: Child gene
        """
        # Child inherits name from higher fitness parent
        if parent1.fitness > parent2.fitness:
            base_parent, other_parent = parent1, parent2
        else:
            base_parent, other_parent = parent2, parent1
        
        # Create child name
        child_name = f"{base_parent.name}_{self.generation}"
        
        # Create empty parameter set
        child_params = {}
        
        # Uniform crossover
        for param_name in set(base_parent.parameters.keys()) | set(other_parent.parameters.keys()):
            # If parameter exists in both parents, select randomly
            if param_name in base_parent.parameters and param_name in other_parent.parameters:
                if random.random() < 0.5:
                    child_params[param_name] = base_parent.parameters[param_name]
                else:
                    child_params[param_name] = other_parent.parameters[param_name]
            # If only in base parent
            elif param_name in base_parent.parameters:
                child_params[param_name] = base_parent.parameters[param_name]
            # If only in other parent
            else:
                child_params[param_name] = other_parent.parameters[param_name]
        
        # Create metadata
        child_metadata = base_parent.metadata.copy()
        child_metadata['parents'] = [parent1.name, parent2.name]
        child_metadata['type'] = 'crossover'
        
        # Create child gene
        child = StrategyGene(
            name=child_name,
            parameters=child_params,
            metadata=child_metadata
        )
        
        # Set generation
        child.generation = self.generation
        
        return child
    
    def get_top_strategies(self, count: int = 5) -> List[StrategyGene]:
        """
        Get top performing strategies
        
        Args:
            count (int): Number of strategies to return
            
        Returns:
            List[StrategyGene]: Top strategies
        """
        # Sort by fitness and return top
        sorted_population = sorted(self.population, key=lambda g: g.fitness, reverse=True)
        return sorted_population[:min(count, len(sorted_population))]
    
    def _update_evolution_metrics(self):
        """Update evolution metrics in state manager"""
        # Calculate statistics
        if not self.population:
            return
            
        fitness_values = [g.fitness for g in self.population]
        avg_fitness = sum(fitness_values) / len(fitness_values)
        max_fitness = max(fitness_values)
        min_fitness = min(fitness_values)
        
        # Get fitness of template strategies
        template_fitness = [g.fitness for g in self.population 
                         if g.metadata.get('type') == 'template']
        avg_template_fitness = sum(template_fitness) / max(1, len(template_fitness))
        
        # Calculate diversity (standard deviation of fitness)
        diversity = np.std(fitness_values) if len(fitness_values) > 1 else 0
        
        # Update metrics
        metrics = {
            'generation': self.generation,
            'population_size': len(self.population),
            'avg_fitness': avg_fitness,
            'max_fitness': max_fitness,
            'min_fitness': min_fitness,
            'avg_template_fitness': avg_template_fitness,
            'diversity': diversity,
            'best_strategy': self.best_gene.name if self.best_gene else 'None'
        }
        
        self.state_manager.update_component_metrics('genetic_strategy_evolver', metrics)


class PerformanceTracker:
    """Tracks strategy performance for fitness evaluation"""
    
    def __init__(self, state_manager: StateManager):
        """
        Initialize the PerformanceTracker
        
        Args:
            state_manager (StateManager): State manager instance
        """
        self.state_manager = state_manager
        
        # Performance data storage
        self.strategy_performance = {}
        
        # Tracking periods
        self.tracking_periods = [
            {'name': 'last_day', 'hours': 24},
            {'name': 'last_week', 'hours': 168},  # 7 days
            {'name': 'last_month', 'hours': 720}  # 30 days
        ]
    
    def record_trade_result(self, 
                           strategy_name: str, 
                           result: Dict[str, Any]):
        """
        Record a trade result for a strategy
        
        Args:
            strategy_name (str): Strategy name
            result (Dict[str, Any]): Trade result
        """
        # Create entry if not exists
        if strategy_name not in self.strategy_performance:
            self.strategy_performance[strategy_name] = {
                'trades': [],
                'stats': {},
                'last_updated': time.time()
            }
        
        # Add trade to performance data
        self.strategy_performance[strategy_name]['trades'].append({
            'timestamp': time.time(),
            'profit_pct': result.get('profit_pct', 0),
            'is_win': result.get('profit_pct', 0) > 0,
            'token_address': result.get('token_address', ''),
            'duration_hours': result.get('duration_hours', 0),
            'entry_price': result.get('entry_price', 0),
            'exit_price': result.get('exit_price', 0)
        })
        
        # Update stats
        self._update_strategy_stats(strategy_name)
    
    def _update_strategy_stats(self, strategy_name: str):
        """
        Update performance statistics for a strategy
        
        Args:
            strategy_name (str): Strategy name
        """
        if strategy_name not in self.strategy_performance:
            return
            
        performance = self.strategy_performance[strategy_name]
        trades = performance['trades']
        
        if not trades:
            performance['stats'] = {
                'trade_count': 0,
                'win_count': 0,
                'loss_count': 0,
                'win_rate': 0,
                'profit_loss_pct': 0,
                'avg_win_pct': 0,
                'avg_loss_pct': 0,
                'largest_win_pct': 0,
                'largest_loss_pct': 0,
                'sharpe_ratio': 0,
                'consistency': 0
            }
            return
        
        # Basic stats
        trade_count = len(trades)
        win_trades = [t for t in trades if t['is_win']]
        loss_trades = [t for t in trades if not t['is_win']]
        
        win_count = len(win_trades)
        loss_count = len(loss_trades)
        
        win_rate = (win_count / trade_count) * 100 if trade_count > 0 else 0
        
        # Profit/loss
        profit_loss_pct = sum(t['profit_pct'] for t in trades)
        
        # Average win/loss
        avg_win_pct = sum(t['profit_pct'] for t in win_trades) / win_count if win_count > 0 else 0
        avg_loss_pct = sum(t['profit_pct'] for t in loss_trades) / loss_count if loss_count > 0 else 0
        
        # Largest win/loss
        largest_win_pct = max([t['profit_pct'] for t in win_trades]) if win_trades else 0
        largest_loss_pct = min([t['profit_pct'] for t in loss_trades]) if loss_trades else 0
        
        # Calculate Sharpe ratio (simplified)
        returns = [t['profit_pct'] for t in trades]
        mean_return = sum(returns) / len(returns)
        std_dev = np.std(returns) if len(returns) > 1 else 1
        sharpe_ratio = mean_return / std_dev if std_dev > 0 else 0
        
        # Calculate consistency (percentage of trades in same direction as overall PnL)
        if profit_loss_pct > 0:
            consistency = (win_count / trade_count) * 100 if trade_count > 0 else 0
        else:
            consistency = (loss_count / trade_count) * 100 if trade_count > 0 else 0
        
        # Update stats
        performance['stats'] = {
            'trade_count': trade_count,
            'win_count': win_count,
            'loss_count': loss_count,
            'win_rate': win_rate,
            'profit_loss_pct': profit_loss_pct,
            'avg_win_pct': avg_win_pct,
            'avg_loss_pct': avg_loss_pct,
            'largest_win_pct': largest_win_pct,
            'largest_loss_pct': largest_loss_pct,
            'sharpe_ratio': sharpe_ratio,
            'consistency': consistency
        }
        
        # Update period stats
        for period in self.tracking_periods:
            self._update_period_stats(strategy_name, period['name'], period['hours'])
        
        # Update in state manager
        self.state_manager.update_component_metrics(
            f"strategy_{strategy_name}", performance['stats'])
        
        # Update last updated timestamp
        performance['last_updated'] = time.time()
    
    def _update_period_stats(self, strategy_name: str, period_name: str, hours: int):
        """
        Update statistics for a specific time period
        
        Args:
            strategy_name (str): Strategy name
            period_name (str): Period name
            hours (int): Period duration in hours
        """
        if strategy_name not in self.strategy_performance:
            return
            
        performance = self.strategy_performance[strategy_name]
        all_trades = performance['trades']
        
        # Filter trades for this period
        cutoff_time = time.time() - (hours * 3600)
        period_trades = [t for t in all_trades if t['timestamp'] >= cutoff_time]
        
        if not period_trades:
            performance['stats'][f"{period_name}_trade_count"] = 0
            performance['stats'][f"{period_name}_win_rate"] = 0
            performance['stats'][f"{period_name}_profit_loss_pct"] = 0
            return
        
        # Calculate period stats
        trade_count = len(period_trades)
        win_trades = [t for t in period_trades if t['is_win']]
        win_count = len(win_trades)
        
        win_rate = (win_count / trade_count) * 100 if trade_count > 0 else 0
        profit_loss_pct = sum(t['profit_pct'] for t in period_trades)
        
        # Update stats
        performance['stats'][f"{period_name}_trade_count"] = trade_count
        performance['stats'][f"{period_name}_win_rate"] = win_rate
        performance['stats'][f"{period_name}_profit_loss_pct"] = profit_loss_pct
    
    def get_performance_data(self) -> Dict[str, Dict[str, Any]]:
        """
        Get performance data for all strategies
        
        Returns:
            Dict[str, Dict[str, Any]]: Performance data by strategy name
        """
        # Return only the stats part of performance data
        return {name: data['stats'] for name, data in self.strategy_performance.items()}