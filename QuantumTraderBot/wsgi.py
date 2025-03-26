"""
WSGI entry point for the Quantum Memecoin Trading Bot web application
"""
import sys
import logging

# Suppress unrelated warnings and INFO messages
logging.basicConfig(level=logging.WARNING)

# Import the Flask app
from webapp_launcher import app

# Simple middleware to suppress unwanted log messages
class NoWinchFilter(logging.Filter):
    def filter(self, record):
        return "winch" not in record.getMessage().lower()

# Apply filter to root logger
root_logger = logging.getLogger()
root_logger.addFilter(NoWinchFilter())

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)