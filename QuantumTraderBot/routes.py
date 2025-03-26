from flask import render_template, request, redirect, url_for, flash, jsonify, session
from webapp_launcher import app, db
from models import User, TradingStrategy, Token, Trade
import logging
import threading
import asyncio

# Bot reference variables
bot_instance = None
bot_thread = None

# Import main bot class when needed
try:
    # Import the class but avoid circular imports by not initializing here
    import main
    HAS_BOT_MODULE = True
except ImportError:
    logging.debug("Main bot module not imported during webapp initialization")
    HAS_BOT_MODULE = False

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    # Get token data
    tokens = Token.query.all()
    
    # Get active trades
    active_trades = Trade.query.filter_by(status='open').all()
    
    # Get completed trades
    completed_trades = Trade.query.filter_by(status='closed').order_by(Trade.exit_timestamp.desc()).limit(10).all()
    
    # Get trading strategies
    strategies = TradingStrategy.query.all()
    
    return render_template(
        'dashboard.html',
        tokens=tokens,
        active_trades=active_trades,
        completed_trades=completed_trades,
        strategies=strategies
    )

@app.route('/trades')
def trades():
    # Get all trades
    trades_list = Trade.query.order_by(Trade.entry_timestamp.desc()).all()
    
    return render_template('trades.html', trades=trades_list)

@app.route('/tokens')
def tokens():
    # Get all tokens
    tokens_list = Token.query.order_by(Token.first_seen_at.desc()).all()
    
    return render_template('tokens.html', tokens=tokens_list)

@app.route('/strategies')
def strategies():
    # Get all strategies
    strategies_list = TradingStrategy.query.all()
    
    return render_template('strategies.html', strategies=strategies_list)

@app.route('/strategy/<int:strategy_id>')
def strategy_detail(strategy_id):
    # Get strategy
    strategy = TradingStrategy.query.get_or_404(strategy_id)
    
    # Get trades using this strategy
    strategy_trades = Trade.query.filter_by(strategy_id=strategy_id).order_by(Trade.entry_timestamp.desc()).all()
    
    return render_template(
        'strategy_detail.html',
        strategy=strategy,
        trades=strategy_trades
    )

@app.route('/token/<token_address>')
def token_detail(token_address):
    # Get token
    token = Token.query.filter_by(address=token_address).first_or_404()
    
    # Get trades for this token
    token_trades = Trade.query.filter_by(token_address=token_address).order_by(Trade.entry_timestamp.desc()).all()
    
    return render_template(
        'token_detail.html',
        token=token,
        trades=token_trades
    )

@app.route('/settings')
def settings():
    return render_template('settings.html')

@app.route('/api/bot/start', methods=['POST'])
def api_bot_start():
    global bot_instance, bot_thread
    
    if bot_instance is not None:
        return jsonify({'status': 'error', 'message': 'Bot already running'})
    
    if not HAS_BOT_MODULE:
        return jsonify({'status': 'error', 'message': 'Bot module not available'})
    
    try:
        def run_bot():
            global bot_instance
            try:
                import main  # Import inside function to avoid circular imports
                bot_instance = main.QuantumMememcoinTradingBot()
                # Use new event loop for each thread
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(bot_instance.start())
            except Exception as e:
                logging.error(f"Error in bot thread: {str(e)}", exc_info=True)
        
        bot_thread = threading.Thread(target=run_bot)
        bot_thread.daemon = True
        bot_thread.start()
        
        return jsonify({'status': 'success', 'message': 'Bot started successfully'})
    
    except Exception as e:
        logging.error(f"Error starting bot: {str(e)}")
        return jsonify({'status': 'error', 'message': f'Error starting bot: {str(e)}'})

@app.route('/api/bot/stop', methods=['POST'])
def api_bot_stop():
    global bot_instance, bot_thread
    
    if bot_instance is None:
        return jsonify({'status': 'error', 'message': 'Bot not running'})
    
    try:
        bot_instance.shutdown()
        bot_instance = None
        
        return jsonify({'status': 'success', 'message': 'Bot stopped successfully'})
    
    except Exception as e:
        logging.error(f"Error stopping bot: {str(e)}")
        return jsonify({'status': 'error', 'message': f'Error stopping bot: {str(e)}'})