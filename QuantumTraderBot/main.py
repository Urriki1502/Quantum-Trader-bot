#!/usr/bin/env python3
"""
Quantum Memecoin Trading Bot
Main entry point for the trading bot application
"""
# Create an app object for wsgi server to use
from webapp_launcher import app

import asyncio
import logging
import os
import signal
import sys
from typing import Dict, Any

from core.state_manager import StateManager
from core.config_manager import ConfigManager
from core.security_manager import SecurityManager
from core.adapter import Adapter
from network.pump_portal_client import PumpPortalClient
from network.onchain_analyzer import OnchainAnalyzer
from trading.raydium_client import RaydiumClient
from trading.trading_integration import TradingIntegration
from trading.risk_manager import RiskManager
from monitoring.monitoring_system import MonitoringSystem
from monitoring.telegram_notifier import TelegramNotifier
from strategy.strategy_manager import StrategyManager
from utils.log_manager import LogManager
from utils.memory_manager import MemoryManager

# Initialize logger
logger = logging.getLogger(__name__)

class QuantumMememcoinTradingBot:
    """Main bot class that orchestrates all components"""
    
    def __init__(self):
        """Initialize the trading bot and all its components"""
        try:
            # Initialize log manager first for proper logging
            self.log_manager = LogManager()
            self.log_manager.setup()
            
            logger.info("Initializing Quantum Memecoin Trading Bot...")
            
            # Initialize core components
            self.config_manager = ConfigManager()
            self.state_manager = StateManager()
            self.security_manager = SecurityManager(self.config_manager)
            
            # Initialize utility components
            self.memory_manager = MemoryManager()
            
            # Initialize monitoring
            self.monitoring_system = MonitoringSystem(self.state_manager)
            self.telegram_notifier = TelegramNotifier(self.config_manager)
            
            # Initialize network components
            self.pump_portal_client = PumpPortalClient(
                self.config_manager, 
                self.state_manager,
                self.security_manager
            )
            
            # Initialize on-chain analyzer
            self.onchain_analyzer = OnchainAnalyzer(
                self.config_manager,
                self.state_manager,
                self.security_manager
            )
            
            # Initialize trading components
            self.raydium_client = RaydiumClient(
                self.config_manager,
                self.state_manager,
                self.security_manager
            )
            
            self.risk_manager = RiskManager(
                self.config_manager,
                self.state_manager
            )
            
            # Initialize strategy components
            self.strategy_manager = StrategyManager(
                self.config_manager,
                self.state_manager
            )
            
            # Initialize trading integration last as it depends on other components
            self.trading_integration = TradingIntegration(
                self.pump_portal_client,
                self.onchain_analyzer,
                self.raydium_client,
                self.risk_manager,
                self.strategy_manager,
                self.state_manager,
                self.config_manager
            )
            
            # Initialize adapter
            self.adapter = Adapter(
                self.pump_portal_client,
                self.raydium_client
            )
            
            # Register components with state manager
            self._register_components()
            
            logger.info("Bot initialized successfully")
            
            # Set up signal handlers
            self._setup_signal_handlers()
            
        except Exception as e:
            logger.critical(f"Failed to initialize bot: {str(e)}", exc_info=True)
            raise

    def _register_components(self):
        """Register all components with the state manager"""
        components = {
            'config_manager': self.config_manager,
            'security_manager': self.security_manager,
            'pump_portal_client': self.pump_portal_client,
            'onchain_analyzer': self.onchain_analyzer,
            'raydium_client': self.raydium_client,
            'trading_integration': self.trading_integration,
            'risk_manager': self.risk_manager,
            'strategy_manager': self.strategy_manager,
            'monitoring_system': self.monitoring_system,
            'telegram_notifier': self.telegram_notifier,
            'memory_manager': self.memory_manager,
            'log_manager': self.log_manager,
            'adapter': self.adapter
        }
        
        # Đăng ký component state để theo dõi
        for name, component in components.items():
            self.state_manager.register_component(name, component)
            
        # Đăng ký component instance để các thành phần khác có thể truy cập
        for name, component in components.items():
            self.state_manager.register_component_instance(name, component)
    
    def _setup_signal_handlers(self):
        """Set up signal handlers for graceful shutdown"""
        # Only set up signal handlers in the main thread to avoid errors in web threads
        try:
            import threading
            if threading.current_thread() == threading.main_thread():
                signal.signal(signal.SIGINT, self._handle_shutdown)
                signal.signal(signal.SIGTERM, self._handle_shutdown)
                logger.info("Signal handlers registered for graceful shutdown")
            else:
                logger.warning("Skipping signal handlers in non-main thread")
        except Exception as e:
            logger.warning(f"Could not set up signal handlers: {str(e)}")
    
    def _handle_shutdown(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.shutdown()
        sys.exit(0)
    
    async def start(self):
        """Start the trading bot and all its components"""
        try:
            logger.info("Starting Quantum Memecoin Trading Bot...")
            
            # Start monitoring system first
            await self.monitoring_system.start()
            
            # Start network components
            await self.pump_portal_client.start()
            await self.onchain_analyzer.start()
            
            # Start trading components
            await self.raydium_client.start()
            
            # Start trading integration
            await self.trading_integration.start()
            
            # Send startup notification
            await self.telegram_notifier.send_message("🚀 Quantum Memecoin Trading Bot started successfully!", level="SYSTEM")
            
            logger.info("Bot started successfully")
            
            # Keep the bot running
            while True:
                try:
                    # Monitor system resources
                    memory_stats = await self.memory_manager.monitor_memory()
                    memory_percent = memory_stats.get('percent', 0)
                    
                    # Perform memory management with adaptive frequency
                    if memory_percent > 85:
                        # Critical memory situation - aggressive cleanup
                        logger.warning(f"Critical memory usage detected: {memory_percent:.1f}%, performing emergency cleanup")
                        await self.memory_manager.perform_gc(force=True)
                        await self.memory_manager.cleanup()
                        await asyncio.sleep(10)  # Short interval to check progress
                    elif memory_percent > 75:
                        # High memory situation - regular cleanup
                        logger.info(f"High memory usage: {memory_percent:.1f}%, performing cleanup")
                        await self.memory_manager.cleanup()
                        await self.state_manager.check_components_health()
                        await asyncio.sleep(30)  # Medium interval
                    else:
                        # Normal operation
                        await self.memory_manager.cleanup()
                        await self.state_manager.check_components_health()
                        await asyncio.sleep(60)  # Standard interval
                except Exception as loop_error:
                    logger.error(f"Error in main loop: {str(loop_error)}")
                    await asyncio.sleep(30)  # Continue loop even if error occurs
                
        except Exception as e:
            logger.critical(f"Critical error in bot operation: {str(e)}", exc_info=True)
            try:
                await self.telegram_notifier.send_message(f"❌ CRITICAL ERROR: {str(e)}", level="CRITICAL")
            except Exception as notify_error:
                logger.error(f"Failed to send critical error notification: {str(notify_error)}")
            self.shutdown()
            sys.exit(1)
    
    def shutdown(self):
        """Shutdown the trading bot gracefully"""
        logger.info("Shutting down Quantum Memecoin Trading Bot...")
        
        # Cải thiện cơ chế shutdown để tránh "cannot reuse already awaited coroutine"
        # Sử dụng hàm bọc lấy các coroutine để không gọi trực tiếp
        async def wrap_shutdown():
            # Đóng các thành phần theo thứ tự: trading -> network -> monitoring
            try:
                logger.info("Stopping trading components...")
                await self.trading_integration.stop()
            except Exception as e:
                logger.error(f"Error stopping trading integration: {str(e)}", exc_info=True)
                
            try:
                logger.info("Stopping network components...")
                await self.pump_portal_client.stop()
                await self.onchain_analyzer.stop()
                await self.raydium_client.stop()
            except Exception as e:
                logger.error(f"Error stopping network components: {str(e)}", exc_info=True)
                
            try:
                logger.info("Stopping monitoring components...")
                await self.monitoring_system.stop()
            except Exception as e:
                logger.error(f"Error stopping monitoring system: {str(e)}", exc_info=True)
                
            try:
                await self.telegram_notifier.send_message("⚠️ Quantum Memecoin Trading Bot shutting down...", level="SYSTEM")
            except Exception as e:
                logger.error(f"Error sending shutdown notification: {str(e)}")
                
            # Các đối tượng ClientSession từ aiohttp đôi khi không được giải phóng
            # Thêm xử lý để đóng tất cả các session đang mở
            try:
                for task in asyncio.all_tasks():
                    task_name = task.get_name()
                    if not task.done() and 'client_session' in str(task):
                        logger.info(f"Canceling task: {task_name}")
                        task.cancel()
            except Exception as e:
                logger.error(f"Error canceling remaining tasks: {str(e)}")
        
        # Chạy shutdown trong một event loop mới nếu event loop hiện tại đang chạy
        try:
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # Nếu event loop đang chạy, tạo một event loop mới
                    logger.info("Using new event loop for shutdown tasks")
                    new_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(new_loop)
                    new_loop.run_until_complete(wrap_shutdown())
                    
                    # Đóng tất cả các task còn lại trước khi đóng loop
                    remaining_tasks = asyncio.all_tasks(new_loop)
                    if remaining_tasks:
                        logger.info(f"Canceling {len(remaining_tasks)} remaining tasks")
                        for task in remaining_tasks:
                            task.cancel()
                        # Chờ các task hủy bỏ
                        new_loop.run_until_complete(asyncio.gather(*remaining_tasks, return_exceptions=True))
                    
                    new_loop.close()
                else:
                    # Nếu event loop không chạy, sử dụng nó
                    loop.run_until_complete(wrap_shutdown())
            except RuntimeError:
                # Nếu không có event loop, tạo một event loop mới
                logger.info("No event loop found, creating new one for shutdown")
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                new_loop.run_until_complete(wrap_shutdown())
                new_loop.close()
        except Exception as e:
            logger.error(f"Error during shutdown: {str(e)}", exc_info=True)
        
        logger.info("Bot shutdown complete")


async def main():
    """Main entry point for the application"""
    try:
        # Create and start the bot
        bot = QuantumMememcoinTradingBot()
        await bot.start()
    except Exception as e:
        logging.critical(f"Fatal error: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nBot terminated by user")
        sys.exit(0)
