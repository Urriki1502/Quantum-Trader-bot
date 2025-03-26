"""
TelegramNotifier Component
Responsible for sending notifications and alerts to a Telegram channel or chat.
"""

import asyncio
import logging
import time
import aiohttp
from typing import Dict, Any, List, Optional, Union
import json

from core.config_manager import ConfigManager

logger = logging.getLogger(__name__)

class TelegramNotifier:
    """
    TelegramNotifier handles:
    - Sending notifications via Telegram
    - Formatting different types of messages
    - Managing notification priorities
    - Throttling excessive messages
    """
    
    def __init__(self, config_manager: ConfigManager):
        """
        Initialize the TelegramNotifier
        
        Args:
            config_manager (ConfigManager): Configuration manager instance
        """
        self.config_manager = config_manager
        
        # Telegram settings
        self.enabled = self.config_manager.get('telegram.enabled', True)
        self.bot_token = self.config_manager.get('telegram.bot_token', '')
        self.chat_id = self.config_manager.get('telegram.chat_id', '')
        self.api_url = f"https://api.telegram.org/bot{self.bot_token}"
        
        # Notification settings
        self.notification_levels = self.config_manager.get(
            'telegram.notification_levels', 
            ['ERROR', 'CRITICAL']
        )
        self.trade_notifications = self.config_manager.get('telegram.trade_notifications', True)
        self.system_notifications = self.config_manager.get('telegram.system_notifications', True)
        self.max_messages_per_minute = self.config_manager.get('telegram.max_messages_per_minute', 10)
        
        # Message history for throttling
        self.message_history = []
        self.last_throttle_warning = 0
        
        # HTTP session
        self.session = None
        
        # Check if properly configured
        if not self.bot_token or not self.chat_id:
            self.enabled = False
            logger.warning("Telegram notifier not properly configured (missing bot_token or chat_id)")
        
        logger.info(f"TelegramNotifier initialized (enabled: {self.enabled})")
    
    async def start(self):
        """Initialize the HTTP session"""
        if self.session is None:
            self.session = aiohttp.ClientSession()
    
    async def stop(self):
        """Close the HTTP session"""
        if self.session:
            await self.session.close()
            self.session = None
    
    async def send_message(self, 
                         message: str, 
                         level: str = "INFO", 
                         parse_mode: str = "HTML") -> bool:
        """
        Send a message to the Telegram chat
        
        Args:
            message (str): Message text
            level (str): Message level (INFO, WARNING, ERROR, CRITICAL)
            parse_mode (str): Message parse mode (HTML, Markdown)
            
        Returns:
            bool: True if message was sent successfully, False otherwise
        """
        # Check if enabled
        if not self.enabled:
            logger.debug(f"Telegram notifications disabled, not sending: {message}")
            return False
        
        # Check notification level
        if level not in self.notification_levels and level != "TRADE" and level != "SYSTEM":
            logger.debug(f"Notification level {level} not enabled, not sending: {message}")
            return False
        
        # Check specific notification types
        if level == "TRADE" and not self.trade_notifications:
            logger.debug(f"Trade notifications disabled, not sending: {message}")
            return False
        
        if level == "SYSTEM" and not self.system_notifications:
            logger.debug(f"System notifications disabled, not sending: {message}")
            return False
        
        # Check for throttling
        current_time = time.time()
        self.message_history = [t for t in self.message_history if current_time - t < 60]
        
        if len(self.message_history) >= self.max_messages_per_minute:
            # Throttling active
            if current_time - self.last_throttle_warning > 60:
                # Send throttle warning once per minute
                await self._send_raw_message(
                    f"⚠️ Message throttling active: {len(self.message_history)} messages in the last minute. Some messages will be skipped.",
                    parse_mode
                )
                self.last_throttle_warning = current_time
            
            logger.warning(f"Telegram message throttled: {message}")
            return False
        
        # Add to message history
        self.message_history.append(current_time)
        
        # Format message with level indicator
        formatted_message = self._format_message(message, level)
        
        # Send message
        return await self._send_raw_message(formatted_message, parse_mode)
    
    def _format_message(self, message: str, level: str) -> str:
        """
        Format message with level indicator
        
        Args:
            message (str): Message text
            level (str): Message level
            
        Returns:
            str: Formatted message
        """
        level_indicators = {
            "INFO": "ℹ️",
            "WARNING": "⚠️",
            "ERROR": "❌",
            "CRITICAL": "🚨",
            "TRADE": "💰",
            "SYSTEM": "🖥️"
        }
        
        indicator = level_indicators.get(level, "")
        
        if indicator:
            return f"{indicator} {message}"
        else:
            return message
    
    async def _send_raw_message(self, message: str, parse_mode: str) -> bool:
        """
        Send raw message to Telegram API
        
        Args:
            message (str): Message text
            parse_mode (str): Message parse mode
            
        Returns:
            bool: True if message was sent successfully, False otherwise
        """
        # Ensure session is initialized
        if self.session is None:
            await self.start()
        
        try:
            # Call Telegram API
            endpoint = f"{self.api_url}/sendMessage"
            data = {
                "chat_id": self.chat_id,
                "text": message,
                "parse_mode": parse_mode
            }
            
            async with self.session.post(endpoint, json=data) as response:
                if response.status != 200:
                    response_text = await response.text()
                    logger.error(f"Error sending Telegram message: {response.status} {response_text}")
                    return False
                
                logger.debug("Telegram message sent successfully")
                return True
            
        except Exception as e:
            logger.error(f"Error sending Telegram message: {str(e)}")
            return False
    
    async def send_trade_notification(self, trade_data: Dict[str, Any]) -> bool:
        """
        Send a trade notification
        
        Args:
            trade_data (Dict[str, Any]): Trade data
            
        Returns:
            bool: True if message was sent successfully, False otherwise
        """
        # Check if trade notifications are enabled
        if not self.trade_notifications:
            return False
        
        token_symbol = trade_data.get('token_symbol', '')
        token_address = trade_data.get('token_address', '')
        action = trade_data.get('action', 'traded')  # buy, sell, traded
        amount_usd = trade_data.get('amount_usd', 0)
        price = trade_data.get('price_usd', 0)
        pnl = trade_data.get('pnl_percentage', 0)
        
        # Format the message
        message = f"<b>Trade {action.upper()}</b>\n"
        
        if action == 'buy':
            message += f"Bought {token_symbol} for ${amount_usd:.2f}\n"
        elif action == 'sell':
            message += f"Sold {token_symbol} for ${amount_usd:.2f}\n"
            if pnl != 0:
                emoji = "🟢" if pnl > 0 else "🔴"
                message += f"PnL: {emoji} {pnl:.2f}%\n"
        else:
            message += f"{token_symbol} trade executed: ${amount_usd:.2f}\n"
        
        if price > 0:
            message += f"Price: ${price:.8f}\n"
        
        # Add token address (shortened)
        short_address = f"{token_address[:6]}...{token_address[-4:]}" if len(token_address) > 10 else token_address
        message += f"Token: {short_address}"
        
        # Send the notification
        return await self.send_message(message, level="TRADE")
    
    async def send_system_notification(self, title: str, data: Dict[str, Any]) -> bool:
        """
        Send a system notification
        
        Args:
            title (str): Notification title
            data (Dict[str, Any]): Additional data
            
        Returns:
            bool: True if message was sent successfully, False otherwise
        """
        # Check if system notifications are enabled
        if not self.system_notifications:
            return False
        
        # Format the message
        message = f"<b>{title}</b>\n"
        
        # Add data items
        for key, value in data.items():
            if isinstance(value, (int, float)):
                message += f"{key}: {value}\n"
            elif isinstance(value, bool):
                emoji = "✅" if value else "❌"
                message += f"{key}: {emoji}\n"
            else:
                message += f"{key}: {value}\n"
        
        # Send the notification
        return await self.send_message(message, level="SYSTEM")
    
    async def send_alert_notification(self, alert: Dict[str, Any]) -> bool:
        """
        Send an alert notification
        
        Args:
            alert (Dict[str, Any]): Alert data
            
        Returns:
            bool: True if message was sent successfully, False otherwise
        """
        # Extract alert data
        level = alert.get('level', 'INFO')
        component = alert.get('component', 'System')
        message = alert.get('message', 'Unknown alert')
        
        # Format the message
        formatted_message = f"<b>Alert from {component}</b>\n{message}"
        
        # Send the notification
        return await self.send_message(formatted_message, level=level)
    
    async def send_performance_report(self, performance_data: Dict[str, Any]) -> bool:
        """
        Send a performance report
        
        Args:
            performance_data (Dict[str, Any]): Performance data
            
        Returns:
            bool: True if message was sent successfully, False otherwise
        """
        # Format the message
        message = "<b>Performance Report</b>\n"
        
        # Trading summary
        total_trades = performance_data.get('total_trades', 0)
        win_rate = performance_data.get('win_rate', 0) * 100
        profit_factor = performance_data.get('profit_factor', 0)
        net_profit = performance_data.get('net_profit_usd', 0)
        
        message += f"Trades: {total_trades}\n"
        message += f"Win Rate: {win_rate:.1f}%\n"
        message += f"Profit Factor: {profit_factor:.2f}\n"
        message += f"Net Profit: ${net_profit:.2f}\n\n"
        
        # Add portfolio value
        initial_value = performance_data.get('initial_portfolio_value', 0)
        current_value = performance_data.get('current_portfolio_value', 0)
        roi = ((current_value - initial_value) / initial_value * 100) if initial_value > 0 else 0
        
        message += f"Portfolio: ${current_value:.2f}\n"
        message += f"ROI: {roi:.2f}%"
        
        # Send the notification
        return await self.send_message(message, level="SYSTEM")
    
    async def test_notification(self) -> bool:
        """
        Send a test notification
        
        Returns:
            bool: True if message was sent successfully, False otherwise
        """
        logger.info("Sending test Telegram notification")
        
        message = "🧪 <b>Test Notification</b>\n"
        message += "If you're seeing this, your Telegram notifications are working correctly."
        
        return await self.send_message(message, level="INFO")
