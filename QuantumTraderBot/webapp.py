"""
Quantum Memecoin Trading Bot Web Interface
Flask web application for monitoring and controlling the trading bot
"""
from app import app
import routes

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)