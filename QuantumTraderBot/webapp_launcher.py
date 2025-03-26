"""
Web Application Launcher for the Quantum Memecoin Trading Bot
"""
import os
import sys
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import DeclarativeBase

class Base(DeclarativeBase):
    pass

db = SQLAlchemy(model_class=Base)
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "quantum_memecoin_dev_secret")

# Configure database
app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get("DATABASE_URL", "sqlite:///trading_bot.db")
app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
    "pool_recycle": 300,
    "pool_pre_ping": True,
}

# Initialize app with SQLAlchemy
db.init_app(app)

# Import models and create db tables
with app.app_context():
    from models import User, TradingStrategy, Token, Trade
    db.create_all()

# Import and register routes with the app
from routes import *

# Run the app if executed directly
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)