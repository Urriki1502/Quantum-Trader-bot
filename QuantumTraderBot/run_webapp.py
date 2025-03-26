#!/usr/bin/env python3
"""
Custom runner for Quantum Memecoin Trading Bot web interface
This script starts Gunicorn with a custom logger to filter out noisy logs
"""
import os
import sys
import logging
import subprocess
from logging.config import dictConfig

# Configure logging to filter out winch signals
class NoWinchFilter(logging.Filter):
    def filter(self, record):
        return not (
            hasattr(record, 'msg') and 
            isinstance(record.msg, str) and 
            'Handling signal: winch' in record.msg
        )

# Configure logging
dictConfig({
    'version': 1,
    'formatters': {
        'standard': {
            'format': '[%(asctime)s] [%(process)d] [%(levelname)s] %(message)s',
            'datefmt': '%Y-%m-%d %H:%M:%S %z',
        },
    },
    'filters': {
        'no_winch': {
            '()': NoWinchFilter,
        },
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'stream': 'ext://sys.stdout',
            'formatter': 'standard',
            'filters': ['no_winch'],
        },
    },
    'root': {
        'level': 'INFO',
        'handlers': ['console'],
    },
})

# Main function to run Gunicorn
def main():
    # Set up environment
    os.environ['PYTHONUNBUFFERED'] = '1'
    
    # Gunicorn command
    cmd = [
        'gunicorn',
        '-c', 'gunicorn_config.py',
        'wsgi:app'
    ]
    
    # Run Gunicorn as a subprocess, capturing output
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1  # Line buffered
        )
        
        # Process and filter log output
        for line in process.stdout:
            if 'Handling signal: winch' not in line:
                print(line, end='')
        
        # Wait for process to complete
        process.wait()
        return process.returncode
    
    except KeyboardInterrupt:
        print("Shutting down gracefully...")
        return 0
    except Exception as e:
        print(f"Error running Gunicorn: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())