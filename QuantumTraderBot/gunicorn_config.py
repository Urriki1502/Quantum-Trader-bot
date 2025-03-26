"""
Gunicorn configuration file for Quantum Memecoin Trading Bot web interface
"""
import os
import multiprocessing
import logging

# Customize logger to filter out winch signals
class NoWinchFilter(logging.Filter):
    def filter(self, record):
        return not (hasattr(record, 'message') and 'Handling signal: winch' in record.getMessage())

def post_worker_init(worker):
    # Add filter to logger
    logger = logging.getLogger('gunicorn.error')
    logger.addFilter(NoWinchFilter())
    
# Post worker initialization to set up logging filters
post_worker_init = post_worker_init

# Bind the server to 0.0.0.0:5000 (accessible from outside the container)
bind = '0.0.0.0:5000'

# Worker settings
workers = 1  # Start with a small number for development
threads = 2  # Threads per worker
worker_class = 'sync'

# Server settings
timeout = 120  # Increase timeout for long-running operations
keepalive = 5
max_requests = 1000
max_requests_jitter = 50

# Logging settings - even more reduced verbosity
accesslog = None  # Disable access logging completely
errorlog = None  # Disable error logging to eliminate winch messages
loglevel = 'critical'  # Only log critical errors

# Turn off console warnings for common events
capture_output = True
enable_stdio_inheritance = True

# Preload application to avoid loading modules multiple times
preload_app = True

# Clean up temporary files created by the worker
tmp_upload_dir = None

# Secure coding practices
limit_request_line = 4096
limit_request_fields = 100
limit_request_field_size = 8190