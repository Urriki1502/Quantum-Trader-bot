from app import db
from flask_login import UserMixin


class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    # ensure password hash field has length of at least 256
    password_hash = db.Column(db.String(256))


class TradingStrategy(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(128), nullable=False)
    description = db.Column(db.Text)
    is_active = db.Column(db.Boolean, default=True)
    allocation_percentage = db.Column(db.Integer, default=0)  # 0-100
    parameters = db.Column(db.JSON)
    performance_stats = db.Column(db.JSON)


class Token(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    address = db.Column(db.String(64), unique=True, nullable=False)
    symbol = db.Column(db.String(16))
    name = db.Column(db.String(128))
    liquidity_usd = db.Column(db.Float, default=0.0)
    price_usd = db.Column(db.Float, default=0.0)
    market_cap_usd = db.Column(db.Float, default=0.0)
    volume_24h_usd = db.Column(db.Float, default=0.0)
    first_seen_at = db.Column(db.DateTime)
    risk_score = db.Column(db.Integer)  # 0-100
    contract_verified = db.Column(db.Boolean, default=False)
    contract_audit = db.Column(db.JSON)


class Trade(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    token_address = db.Column(db.String(64), nullable=False)
    strategy_id = db.Column(db.Integer, db.ForeignKey('trading_strategy.id'))
    entry_price_usd = db.Column(db.Float, nullable=False)
    exit_price_usd = db.Column(db.Float)
    amount_tokens = db.Column(db.Float, nullable=False)
    position_size_usd = db.Column(db.Float, nullable=False)
    exit_value_usd = db.Column(db.Float)
    profit_loss_usd = db.Column(db.Float)
    profit_loss_percent = db.Column(db.Float)
    entry_timestamp = db.Column(db.DateTime, nullable=False)
    exit_timestamp = db.Column(db.DateTime)
    status = db.Column(db.String(20), default='open')  # open, closed, cancelled
    exit_reason = db.Column(db.String(64))
    transaction_data = db.Column(db.JSON)