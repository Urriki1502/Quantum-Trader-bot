"""
LogManager Component
Responsible for configuring and managing logging,
handling log rotation, and providing logging utilities.
"""

import asyncio
import logging
import os
import time
import sys
import gzip
import shutil
from typing import Dict, Any, List, Optional, Union
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
import re

logger = logging.getLogger(__name__)

class GzipRotatingFileHandler(RotatingFileHandler):
    """
    Extended RotatingFileHandler that compresses rotated logs with gzip
    """
    
    def __init__(self, filename, mode='a', maxBytes=0, backupCount=0, encoding=None, delay=False):
        """Initialize the handler with file rotation settings"""
        super().__init__(filename, mode, maxBytes, backupCount, encoding, delay)
    
    def doRollover(self):
        """Compress the old log file after rotation"""
        super().doRollover()
        
        # Compress the rotated file
        if self.backupCount > 0:
            for i in range(1, self.backupCount + 1):
                source = f"{self.baseFilename}.{i}"
                target = f"{source}.gz"
                
                if os.path.exists(source):
                    with open(source, 'rb') as f_in:
                        with gzip.open(target, 'wb') as f_out:
                            shutil.copyfileobj(f_in, f_out)
                    
                    # Remove original file
                    os.remove(source)


class LogManager:
    """
    LogManager handles:
    - Log configuration and formatting
    - Log file rotation and compression
    - Log level management
    - Log filtering and formatting
    """
    
    def __init__(self):
        """Initialize the LogManager"""
        # Default settings
        self.log_dir = './logs'
        self.max_file_size = 10 * 1024 * 1024  # 10 MB
        self.backup_count = 5
        self.log_level = logging.INFO
        self.log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        self.date_format = '%Y-%m-%d %H:%M:%S'
        
        # Track handlers
        self.handlers = {}
        
        # Configure root logger
        self.root_logger = logging.getLogger()
        
        # Track message counts by level
        self.message_counts = {
            'DEBUG': 0,
            'INFO': 0,
            'WARNING': 0,
            'ERROR': 0,
            'CRITICAL': 0
        }
        
        # Custom logger for trading events
        self.trade_logger = logging.getLogger('trading')
        
        logger.info("LogManager initialized")
    
    def setup(self):
        """Set up logging configuration"""
        # Create log directory if it doesn't exist
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        
        # Configure formatter
        formatter = logging.Formatter(self.log_format, self.date_format)
        
        # Configure root logger
        self.root_logger.setLevel(self.log_level)
        
        # Clear existing handlers
        for handler in list(self.root_logger.handlers):
            self.root_logger.removeHandler(handler)
        
        # Add console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        self.root_logger.addHandler(console_handler)
        self.handlers['console'] = console_handler
        
        # Add file handler for general logs
        general_log_path = os.path.join(self.log_dir, 'quantum_bot.log')
        file_handler = GzipRotatingFileHandler(
            general_log_path,
            maxBytes=self.max_file_size,
            backupCount=self.backup_count
        )
        file_handler.setFormatter(formatter)
        self.root_logger.addHandler(file_handler)
        self.handlers['general'] = file_handler
        
        # Add file handler for error logs
        error_log_path = os.path.join(self.log_dir, 'error.log')
        error_handler = GzipRotatingFileHandler(
            error_log_path,
            maxBytes=self.max_file_size,
            backupCount=self.backup_count
        )
        error_handler.setFormatter(formatter)
        error_handler.setLevel(logging.ERROR)
        self.root_logger.addHandler(error_handler)
        self.handlers['error'] = error_handler
        
        # Add file handler for trade logs
        trade_log_path = os.path.join(self.log_dir, 'trades.log')
        trade_handler = GzipRotatingFileHandler(
            trade_log_path,
            maxBytes=self.max_file_size,
            backupCount=self.backup_count
        )
        trade_handler.setFormatter(formatter)
        self.trade_logger.addHandler(trade_handler)
        self.handlers['trade'] = trade_handler
        
        # Monkey patch logger to count messages
        self._patch_logger()
        
        logger.info(f"Logging setup complete (level: {logging.getLevelName(self.log_level)})")
    
    def _patch_logger(self):
        """Patch logger to count messages by level"""
        original_log = logging.Logger._log
        
        def _patched_log(self, level, msg, args, exc_info=None, extra=None, stack_info=False):
            # Call original method
            original_log(self, level, msg, args, exc_info, extra, stack_info)
            
            # Count the message
            level_name = logging.getLevelName(level)
            if level_name in LogManager.instance.message_counts:
                LogManager.instance.message_counts[level_name] += 1
        
        # Store instance for access in patched method
        LogManager.instance = self
        
        # Apply the patch
        logging.Logger._log = _patched_log
    
    def set_log_level(self, level: Union[str, int]):
        """
        Set the log level
        
        Args:
            level (Union[str, int]): Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL or corresponding int values)
        """
        # Convert string level to int if needed
        if isinstance(level, str):
            numeric_level = getattr(logging, level.upper(), None)
            if not isinstance(numeric_level, int):
                logger.warning(f"Invalid log level: {level}")
                return
            level = numeric_level
        
        # Set level for root logger
        self.root_logger.setLevel(level)
        self.log_level = level
        
        # Update console handler
        if 'console' in self.handlers:
            self.handlers['console'].setLevel(level)
        
        # Update general file handler
        if 'general' in self.handlers:
            self.handlers['general'].setLevel(level)
        
        # Error handler always stays at ERROR level
        
        logger.info(f"Log level set to {logging.getLevelName(level)}")
    
    def log_trade(self, message: str, trade_data: Optional[Dict[str, Any]] = None):
        """
        Log a trade event
        
        Args:
            message (str): Log message
            trade_data (Dict[str, Any], optional): Trade data to include in log
        """
        if trade_data:
            # Format trade data
            token_symbol = trade_data.get('token_symbol', '')
            token_address = trade_data.get('token_address', '')
            amount = trade_data.get('amount_usd', 0)
            action = trade_data.get('action', 'trade')
            
            log_msg = f"{message} - {action.upper()} {token_symbol} (${amount:.2f}) - {token_address}"
        else:
            log_msg = message
        
        # Log to trade logger
        self.trade_logger.info(log_msg)
    
    async def rotate_logs(self, force: bool = False) -> Dict[str, Any]:
        """
        Rotate log files
        
        Args:
            force (bool): Force rotation even if size limit not reached
            
        Returns:
            Dict[str, Any]: Rotation results
        """
        results = {}
        
        # Check each handler for rotation
        for name, handler in self.handlers.items():
            if isinstance(handler, (RotatingFileHandler, TimedRotatingFileHandler, GzipRotatingFileHandler)):
                try:
                    # Get current log file size
                    file_path = handler.baseFilename
                    if os.path.exists(file_path):
                        size = os.path.getsize(file_path)
                        size_mb = size / (1024 * 1024)
                        
                        # Rotate if forcing or size exceeds limit
                        if force or (isinstance(handler, RotatingFileHandler) and size >= handler.maxBytes):
                            logger.info(f"Rotating log file: {file_path} ({size_mb:.2f} MB)")
                            handler.doRollover()
                            results[name] = {
                                'rotated': True,
                                'size_mb': size_mb,
                                'file_path': file_path
                            }
                        else:
                            results[name] = {
                                'rotated': False,
                                'size_mb': size_mb,
                                'file_path': file_path
                            }
                except Exception as e:
                    logger.error(f"Error rotating log file for handler {name}: {str(e)}")
                    results[name] = {
                        'rotated': False,
                        'error': str(e)
                    }
        
        return results
    
    async def compress_old_logs(self) -> Dict[str, Any]:
        """
        Compress old log files that haven't been compressed
        
        Returns:
            Dict[str, Any]: Compression results
        """
        results = {
            'compressed': 0,
            'failed': 0,
            'skipped': 0,
            'details': []
        }
        
        try:
            # Find log files in log directory
            for filename in os.listdir(self.log_dir):
                file_path = os.path.join(self.log_dir, filename)
                
                # Skip non-files and already compressed files
                if not os.path.isfile(file_path) or filename.endswith('.gz'):
                    results['skipped'] += 1
                    continue
                
                # Check if it's a backup log file (contains a timestamp or number)
                if re.search(r'\.\d+$|\.\d{4}-\d{2}-\d{2}', filename):
                    try:
                        compressed_path = f"{file_path}.gz"
                        
                        # Compress the file
                        with open(file_path, 'rb') as f_in:
                            with gzip.open(compressed_path, 'wb') as f_out:
                                shutil.copyfileobj(f_in, f_out)
                        
                        # Remove original file
                        os.remove(file_path)
                        
                        logger.debug(f"Compressed log file: {file_path} -> {compressed_path}")
                        results['compressed'] += 1
                        results['details'].append({
                            'file': filename,
                            'status': 'compressed',
                            'compressed_path': compressed_path
                        })
                    except Exception as e:
                        logger.error(f"Error compressing log file {file_path}: {str(e)}")
                        results['failed'] += 1
                        results['details'].append({
                            'file': filename,
                            'status': 'failed',
                            'error': str(e)
                        })
                else:
                    results['skipped'] += 1
            
            logger.info(f"Compressed {results['compressed']} old log files")
            
        except Exception as e:
            logger.error(f"Error in compress_old_logs: {str(e)}")
            results['error'] = str(e)
        
        return results
    
    async def cleanup_old_logs(self, days: int = 30) -> Dict[str, Any]:
        """
        Delete old log files
        
        Args:
            days (int): Delete logs older than this many days
            
        Returns:
            Dict[str, Any]: Cleanup results
        """
        results = {
            'deleted': 0,
            'failed': 0,
            'skipped': 0,
            'details': []
        }
        
        try:
            # Calculate cutoff time
            cutoff_time = time.time() - (days * 24 * 60 * 60)
            
            # Find log files in log directory
            for filename in os.listdir(self.log_dir):
                file_path = os.path.join(self.log_dir, filename)
                
                # Skip non-files
                if not os.path.isfile(file_path):
                    results['skipped'] += 1
                    continue
                
                # Check file age
                file_time = os.path.getmtime(file_path)
                if file_time < cutoff_time:
                    try:
                        # Delete the file
                        os.remove(file_path)
                        
                        logger.debug(f"Deleted old log file: {file_path}")
                        results['deleted'] += 1
                        results['details'].append({
                            'file': filename,
                            'status': 'deleted',
                            'age_days': (time.time() - file_time) / (24 * 60 * 60)
                        })
                    except Exception as e:
                        logger.error(f"Error deleting log file {file_path}: {str(e)}")
                        results['failed'] += 1
                        results['details'].append({
                            'file': filename,
                            'status': 'failed',
                            'error': str(e)
                        })
                else:
                    results['skipped'] += 1
            
            logger.info(f"Cleaned up {results['deleted']} log files older than {days} days")
            
        except Exception as e:
            logger.error(f"Error in cleanup_old_logs: {str(e)}")
            results['error'] = str(e)
        
        return results
    
    def get_log_stats(self) -> Dict[str, Any]:
        """
        Get logging statistics
        
        Returns:
            Dict[str, Any]: Logging statistics
        """
        stats = {
            'message_counts': self.message_counts.copy(),
            'total_messages': sum(self.message_counts.values()),
            'log_level': logging.getLevelName(self.log_level),
            'log_files': {}
        }
        
        # Get log file sizes
        for name, handler in self.handlers.items():
            if isinstance(handler, (RotatingFileHandler, TimedRotatingFileHandler, GzipRotatingFileHandler)):
                file_path = handler.baseFilename
                if os.path.exists(file_path):
                    size = os.path.getsize(file_path)
                    stats['log_files'][name] = {
                        'path': file_path,
                        'size_bytes': size,
                        'size_mb': size / (1024 * 1024)
                    }
        
        # Get log directory size
        total_size = 0
        file_count = 0
        try:
            for filename in os.listdir(self.log_dir):
                file_path = os.path.join(self.log_dir, filename)
                if os.path.isfile(file_path):
                    total_size += os.path.getsize(file_path)
                    file_count += 1
        except Exception as e:
            logger.error(f"Error getting log directory size: {str(e)}")
        
        stats['log_directory'] = {
            'path': self.log_dir,
            'total_size_bytes': total_size,
            'total_size_mb': total_size / (1024 * 1024),
            'file_count': file_count
        }
        
        return stats
    
    def get_recent_logs(self, log_type: str = 'general', lines: int = 100) -> List[str]:
        """
        Get recent log entries
        
        Args:
            log_type (str): Log type ('general', 'error', 'trade')
            lines (int): Maximum number of lines to return
            
        Returns:
            List[str]: Recent log entries
        """
        log_files = {
            'general': os.path.join(self.log_dir, 'quantum_bot.log'),
            'error': os.path.join(self.log_dir, 'error.log'),
            'trade': os.path.join(self.log_dir, 'trades.log')
        }
        
        if log_type not in log_files:
            logger.warning(f"Unknown log type: {log_type}")
            return []
        
        file_path = log_files[log_type]
        if not os.path.exists(file_path):
            logger.warning(f"Log file not found: {file_path}")
            return []
        
        try:
            # Read last 'lines' lines from log file
            with open(file_path, 'r') as f:
                # Use a deque to efficiently keep only the last N lines
                from collections import deque
                recent_logs = deque(maxlen=lines)
                
                for line in f:
                    recent_logs.append(line.rstrip())
            
            return list(recent_logs)
            
        except Exception as e:
            logger.error(f"Error reading log file {file_path}: {str(e)}")
            return [f"Error reading log file: {str(e)}"]
    
    def search_logs(self, pattern: str, log_type: str = 'general', max_results: int = 100) -> List[str]:
        """
        Search log files for a pattern
        
        Args:
            pattern (str): Search pattern (regex)
            log_type (str): Log type ('general', 'error', 'trade', 'all')
            max_results (int): Maximum number of results to return
            
        Returns:
            List[str]: Matching log entries
        """
        log_files = {
            'general': [os.path.join(self.log_dir, 'quantum_bot.log')],
            'error': [os.path.join(self.log_dir, 'error.log')],
            'trade': [os.path.join(self.log_dir, 'trades.log')],
            'all': [
                os.path.join(self.log_dir, 'quantum_bot.log'),
                os.path.join(self.log_dir, 'error.log'),
                os.path.join(self.log_dir, 'trades.log')
            ]
        }
        
        if log_type not in log_files:
            logger.warning(f"Unknown log type: {log_type}")
            return []
        
        try:
            # Compile regex pattern
            regex = re.compile(pattern)
            
            results = []
            
            # Search in each file
            for file_path in log_files[log_type]:
                if not os.path.exists(file_path):
                    continue
                
                try:
                    with open(file_path, 'r') as f:
                        for line in f:
                            if regex.search(line):
                                results.append(line.rstrip())
                                if len(results) >= max_results:
                                    break
                    
                    if len(results) >= max_results:
                        break
                        
                except Exception as e:
                    logger.error(f"Error reading log file {file_path}: {str(e)}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching logs: {str(e)}")
            return [f"Error searching logs: {str(e)}"]
