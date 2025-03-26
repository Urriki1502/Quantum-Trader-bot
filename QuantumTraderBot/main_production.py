"""
Quantum Memecoin Trading Bot - Production Mode
Main entry point for the trading bot in production mode with real funds.
"""

import os
import sys
import time
import logging
import asyncio
import argparse
from datetime import datetime

from core.config_manager import ConfigManager
from core.state_manager import StateManager
from core.log_manager import LogManager
from core.memory_manager import MemoryManager
from core.portfolio_manager import PortfolioManager
from core.mainnet_validator import MainnetValidator
from core.self_healing_system import SelfHealingSystem

from network.pump_portal_client import PumpPortalClient
from network.onchain_analyzer import OnchainAnalyzer

from trading.raydium_client import RaydiumClient
from trading.trading_integration import TradingIntegration
from trading.risk_manager import RiskManager
from trading.gas_predictor import GasPredictor
from trading.mev_protection import MEVProtection
from trading.flash_executor import FlashExecutor
from trading.parallel_executor import ParallelExecutor
from trading.dynamic_profit_manager import DynamicProfitManager

from strategy.strategy_manager import StrategyManager
from strategy.adaptive_strategy import AdaptiveStrategy

from security.wallet_security import WalletSecurityManager
from security.quantum_resistant_security import PostQuantumCryptoEngine

from utils.connection_pool import ConnectionPool
from utils.performance_optimizer import PerformanceOptimizer
from utils.api_resilience import reset_all_circuit_breakers

from analysis.token_contract_analyzer import TokenContractAnalyzer

from monitoring.health_monitor import HealthMonitor

# Logger setup
logger = logging.getLogger(__name__)

class ProductionQuantumMememcoinTradingBot:
    """Main bot class that orchestrates all components in production mode"""
    
    def __init__(self, config_path=None):
        """
        Initialize the trading bot and all its components
        
        Args:
            config_path (str, optional): Path to configuration file
        """
        # Logo
        self._print_logo()
        
        logger.info("Initializing Quantum Memecoin Trading Bot in PRODUCTION mode")
        logger.warning("IMPORTANT: This is a PRODUCTION build with REAL FUNDS enabled")
        
        # Configuration
        self.config_manager = ConfigManager(config_path)
        self.state_manager = StateManager()
        
        # Validate environment
        self._validate_production_environment()
        
        # Core infrastructure
        self.log_manager = LogManager()
        self.memory_manager = MemoryManager()
        self.performance_optimizer = PerformanceOptimizer()
        
        # Create Solana RPC connection pool
        rpc_endpoints = self.config_manager.get('network.solana_rpc_endpoints', [])
        if not rpc_endpoints:
            logger.error("No Solana RPC endpoints configured, using defaults (not recommended for production)")
            rpc_endpoints = [
                {"url": "https://api.mainnet-beta.solana.com", "weight": 1}
            ]
        
        self.connection_pool = ConnectionPool(rpc_endpoints)
        
        # Security components
        self.post_quantum_crypto = PostQuantumCryptoEngine(self.config_manager)
        self.wallet_security = WalletSecurityManager(self.config_manager, None)  # Will set security manager later
        
        # Network components
        self.pump_portal_client = PumpPortalClient(
            self.config_manager,
            self.state_manager
        )
        
        self.onchain_analyzer = OnchainAnalyzer(
            self.config_manager,
            self.state_manager,
            self.connection_pool
        )
        
        # Trading components
        self.gas_predictor = GasPredictor(self.config_manager)
        self.mev_protection = MEVProtection(self.config_manager)
        
        # Initialize DEX client
        self.raydium_client = RaydiumClient(
            self.config_manager,
            self.state_manager,
            None  # Will set security manager later
        )
        
        self.risk_manager = RiskManager(
            self.config_manager,
            self.state_manager
        )
        
        self.flash_executor = FlashExecutor(
            self.config_manager,
            self.state_manager,
            self.raydium_client,
            None,  # RPC client will be set later
            self.mev_protection,
            self.gas_predictor
        )
        
        self.parallel_executor = ParallelExecutor(
            self.config_manager,
            self.state_manager,
            self.flash_executor,
            self.performance_optimizer
        )
        
        self.dynamic_profit_manager = DynamicProfitManager(
            self.config_manager,
            self.state_manager,
            self.risk_manager
        )
        
        # Portfolio management
        self.portfolio_manager = PortfolioManager(
            self.config_manager,
            self.state_manager,
            self.raydium_client
        )
        
        # Strategy components
        self.strategy_manager = StrategyManager(
            self.config_manager,
            self.state_manager
        )
        
        self.adaptive_strategy = AdaptiveStrategy()
        
        # Analysis components
        self.token_contract_analyzer = TokenContractAnalyzer(
            self.config_manager,
            self.state_manager
        )
        
        # Production safeguards
        self.mainnet_validator = MainnetValidator(
            self.config_manager,
            self.state_manager
        )
        
        # Trading integration
        self.trading_integration = TradingIntegration(
            self.pump_portal_client,
            self.onchain_analyzer,
            self.raydium_client,
            self.risk_manager,
            self.strategy_manager,
            self.state_manager,
            self.config_manager
        )
        
        # Monitoring
        self.health_monitor = HealthMonitor(
            self.state_manager,
            self.config_manager
        )
        
        # Self-healing system
        self.self_healing_system = SelfHealingSystem(
            self.config_manager,
            self.state_manager
        )
        
        # Register components for self-healing
        self._register_components()
        
        # Register restart handlers
        self._register_restart_handlers()
        
        # Component initialization is complete
        logger.info("Component initialization complete")
    
    def _print_logo(self):
        """Print the bot logo"""
        logo = """
        ╔═══════════════════════════════════════════════════════════════════════╗
        ║  ___                  _                 __  __                         ║
        ║ / _ \ _   _  __ _ _ __ | |_ _   _ _ __ ___ |  \/  | ___ _ __ ___   ___ ║
        ║| | | | | | |/ _` | '_ \| __| | | | '_ ` _ \| |\/| |/ _ \ '_ ` _ \ / _ \║
        ║| |_| | |_| | (_| | | | | |_| |_| | | | | | | |  | |  __/ | | | | |  __/║
        ║ \__\_\\\\__,_|\__,_|_| |_|\__|\__,_|_| |_| |_|_|  |_|\___|_| |_| |_|\___|║
        ║                                                                       ║
        ║ Trading Bot - PRODUCTION MODE                                         ║
        ╚═══════════════════════════════════════════════════════════════════════╝
        """
        print(logo)
    
    def _validate_production_environment(self):
        """Validate that environment is properly set up for production"""
        # Check required environment variables
        required_env_vars = [
            'SOLANA_WALLET_PRIVATE_KEY',
            'PUMP_PORTAL_API_KEY'
        ]
        
        missing_vars = [var for var in required_env_vars if not os.environ.get(var)]
        
        if missing_vars:
            logger.error(f"Missing required environment variables for production: {', '.join(missing_vars)}")
            logger.error("Production mode requires all security credentials to be properly set")
            sys.exit(1)
        
        # Check RPC endpoints
        rpc_endpoints = self.config_manager.get('network.solana_rpc_endpoints', [])
        if not rpc_endpoints:
            logger.error("No Solana RPC endpoints configured in production mode")
            sys.exit(1)
        
        # Make sure we're not in test mode
        test_mode = self.config_manager.get('general.test_mode', False)
        if test_mode:
            logger.error("Test mode is enabled in production configuration")
            sys.exit(1)
    
    def _register_components(self):
        """Register components with state manager and self-healing system"""
        # Register key components with state manager
        components = [
            ('config_manager', self.config_manager),
            ('state_manager', self.state_manager),
            ('log_manager', self.log_manager),
            ('memory_manager', self.memory_manager),
            ('connection_pool', self.connection_pool),
            ('pump_portal_client', self.pump_portal_client),
            ('onchain_analyzer', self.onchain_analyzer),
            ('raydium_client', self.raydium_client),
            ('risk_manager', self.risk_manager),
            ('strategy_manager', self.strategy_manager),
            ('trading_integration', self.trading_integration),
            ('health_monitor', self.health_monitor),
            ('portfolio_manager', self.portfolio_manager),
            ('mainnet_validator', self.mainnet_validator)
        ]
        
        for name, component in components:
            self.state_manager.register_component(name, component)
            self.self_healing_system.register_component(name, component)
        
        # Register health monitoring
        for name, _ in components:
            self.health_monitor.register_component(name)
    
    def _register_restart_handlers(self):
        """Register component restart handlers with the health monitor"""
        # PumpPortalClient restart
        self.health_monitor.register_restartable_component(
            'pump_portal_client',
            self.pump_portal_client.restart
        )
        
        # OnchainAnalyzer restart
        self.health_monitor.register_restartable_component(
            'onchain_analyzer',
            self.onchain_analyzer.restart
        )
    
    def _setup_signal_handlers(self):
        """Set up signal handlers for graceful shutdown"""
        import signal
        
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating shutdown")
            asyncio.create_task(self.shutdown())
        
        # Register signal handlers
        signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
        signal.signal(signal.SIGTERM, signal_handler)  # Termination
    
    async def start(self):
        """Start the trading bot and all its components"""
        start_time = time.time()
        logger.info(f"Starting Quantum Memecoin Trading Bot in PRODUCTION mode at {datetime.now().isoformat()}")
        
        try:
            # Set up signal handlers
            self._setup_signal_handlers()
            
            # Start core infrastructure
            logger.info("Starting core infrastructure...")
            await self.connection_pool.start()
            await self.performance_optimizer.start()
            
            # Start monitoring
            logger.info("Starting monitoring systems...")
            await self.health_monitor.start()
            await self.self_healing_system.start()
            
            # Start network components with timeout protection
            logger.info("Starting network components...")
            network_tasks = [
                asyncio.create_task(self.pump_portal_client.start()),
                asyncio.create_task(self.onchain_analyzer.start())
            ]
            
            try:
                # Wait for network components with timeout
                await asyncio.wait_for(asyncio.gather(*network_tasks), timeout=30)
            except asyncio.TimeoutError:
                logger.warning("Network component startup timed out, continuing with partial initialization")
            
            # Start trading components
            logger.info("Starting trading components...")
            await self.raydium_client.start()
            await self.dynamic_profit_manager.start()
            await self.portfolio_manager.start()
            
            # Reset circuit breakers before trading starts
            reset_all_circuit_breakers()
            
            # Start trading integration (main trading logic)
            logger.info("Starting trading integration...")
            await self.trading_integration.start()
            
            # Log successful startup
            elapsed = time.time() - start_time
            logger.info(f"Quantum Memecoin Trading Bot started successfully in {elapsed:.2f} seconds")
            logger.warning("PRODUCTION MODE ACTIVE - Trading with real funds enabled")
            
            # Update state
            self.state_manager.update_component_state(
                'main', 
                {
                    'status': 'running',
                    'mode': 'production',
                    'start_time': start_time,
                    'startup_duration': elapsed
                }
            )
            
            # Run health check
            asyncio.create_task(self._run_initial_health_check())
            
            # Keep the main task alive
            while True:
                await asyncio.sleep(1)
                
        except Exception as e:
            logger.critical(f"Error during startup: {e}", exc_info=True)
            await self.shutdown()
            sys.exit(1)
    
    async def _run_initial_health_check(self):
        """Run initial health check after startup"""
        # Wait a moment for components to initialize
        await asyncio.sleep(5)
        
        try:
            from health_test import run_health_check
            health_result = await run_health_check()
            
            if not health_result.get('healthy', False):
                logger.error(f"Initial health check failed: {health_result}")
                
                # Create alert
                self.state_manager.create_alert(
                    'main',
                    'ERROR',
                    f"Initial health check failed: {health_result.get('reason', 'Unknown reason')}"
                )
            else:
                logger.info(f"Initial health check passed: {len(health_result.get('healthy_components', []))} components healthy")
        except Exception as e:
            logger.error(f"Error running initial health check: {e}")
    
    async def shutdown(self):
        """Shutdown the trading bot gracefully"""
        logger.info("Shutting down Quantum Memecoin Trading Bot...")
        
        # Update state
        self.state_manager.update_component_state(
            'main', 
            {
                'status': 'shutting_down',
                'shutdown_time': time.time()
            }
        )
        
        # Shutdown components in reverse order of dependencies
        shutdown_groups = [
            # Group 1: High-level components
            [
                ('Trading Integration', self.trading_integration.stop()),
                ('Portfolio Manager', self.portfolio_manager.stop())
            ],
            # Group 2: Mid-level components
            [
                ('Raydium Client', self.raydium_client.stop()),
                ('PumpPortal Client', self.pump_portal_client.stop()),
                ('Onchain Analyzer', self.onchain_analyzer.stop())
            ],
            # Group 3: Core infrastructure
            [
                ('Connection Pool', self.connection_pool.stop()),
                ('Performance Optimizer', self.performance_optimizer.stop()),
                ('Self-Healing System', self.self_healing_system.stop()),
                ('Health Monitor', None)  # Health monitor doesn't have a stop method
            ]
        ]
        
        # Shutdown each group in sequence
        for group_idx, group in enumerate(shutdown_groups):
            group_tasks = []
            
            for name, stop_coro in group:
                if stop_coro:
                    logger.debug(f"Shutting down {name}...")
                    task = asyncio.create_task(stop_coro)
                    group_tasks.append(task)
            
            if group_tasks:
                try:
                    await asyncio.wait_for(asyncio.gather(*group_tasks, return_exceptions=True), timeout=10)
                except asyncio.TimeoutError:
                    logger.warning(f"Timeout shutting down group {group_idx+1}")
        
        logger.info("Quantum Memecoin Trading Bot shutdown complete")
        
        # Update final state
        self.state_manager.update_component_state(
            'main', 
            {
                'status': 'stopped',
                'shutdown_completed': time.time()
            }
        )

async def main():
    """Main entry point for the application"""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Quantum Memecoin Trading Bot (Production Mode)')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    args = parser.parse_args()
    
    # Create bot instance
    bot = ProductionQuantumMememcoinTradingBot(args.config)
    
    # Start bot
    await bot.start()

if __name__ == "__main__":
    # Run the bot
    asyncio.run(main())