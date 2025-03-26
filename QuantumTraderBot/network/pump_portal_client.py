"""
PumpPortalClient Component
Responsible for connecting to PumpPortal API and receiving real-time market data.
"""

import asyncio
import json
import logging
import time
from typing import Dict, Any, List, Optional, Callable, Set, Awaitable
try:
    import websockets
except ImportError:
    pass  # Server has websockets package installed, LSP may not detect it
try:
    import aiohttp
except ImportError:
    pass  # Server has aiohttp package installed, LSP may not detect it
import traceback
from core.state_manager import StateManager
from core.config_manager import ConfigManager
from core.security_manager import SecurityManager
from core.base_component import BaseComponent
from utils.api_resilience import with_retry, with_timeout

logger = logging.getLogger(__name__)

class PumpPortalClient(BaseComponent):
    """
    PumpPortalClient manages the connection to PumpPortal API for:
    - Receiving real-time market data
    - Detecting new token events
    - Monitoring liquidity changes
    """
    
    def __init__(self, 
                config_manager: ConfigManager, 
                state_manager: StateManager,
                security_manager: SecurityManager):
        """
        Initialize the PumpPortalClient
        
        Args:
            config_manager (ConfigManager): Configuration manager instance
            state_manager (StateManager): State manager instance
            security_manager (SecurityManager): Security manager instance
        """
        # Initialize BaseComponent
        super().__init__(state_manager, 'pump_portal_client')
        
        self.config_manager = config_manager
        self.security_manager = security_manager
        
        # Connection settings (URL chính thức từ tài liệu)
        self.websocket_url = self.config_manager.get('pump_portal.base_url', 'wss://pumpportal.fun/api/data')
        self.rest_url = self.config_manager.get('pump_portal.rest_url', 'https://pumpportal.fun/api')
        self.websocket = None
        self.session = None
        self._websocket = None  # Thêm alias cho websocket để tương thích với cơ chế khôi phục
        
        # API key from environment variables
        self.api_key = self.security_manager.get_api_key('pump_portal') or self.config_manager.get('pump_portal.api_key')
        
        # Connection state
        self.is_connected = False
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = self.config_manager.get('pump_portal.max_retries', 15)
        self.reconnect_interval = self.config_manager.get('pump_portal.reconnect_interval', 5)
        self.reconnect_backoff_factor = 1.5
        
        # Event callbacks
        self.new_token_callbacks = set()
        self.liquidity_change_callbacks = set()
        self.price_change_callbacks = set()
        self.trade_callbacks = set()
        
        # Tracked addresses
        self.tracked_accounts = self.config_manager.get('pump_portal.tracked_accounts', [])
        self.tracked_tokens = self.config_manager.get('pump_portal.tracked_tokens', [])
        
        # Data storage
        self.tokens = {}  # Token address -> token data
        self.last_ping_time = 0
        self.ping_interval = 30  # Send ping every 30 seconds
        
        # Connection metrics for monitoring
        self.connection_attempts = 0
        self.disconnection_count = 0
        self.uptime_start = None
        self.last_disconnect_time = None
        
        # Tasks
        self._message_handler_task = None
        self._heartbeat_task = None
        self._reconnect_task = None
        self._keep_alive_task = None
        
        logger.info("PumpPortalClient initialized")
    
    async def start(self):
        """Start the PumpPortalClient and connect to PumpPortal API"""
        # First call the BaseComponent start method which will:
        # - Set is_running to True
        # - Start the heartbeat task
        # - Update component status to "running"
        await super().start()
        
        logger.info("Starting PumpPortalClient")
        
        # Record uptime start time
        self.uptime_start = time.time()
        
        # Create HTTP session for REST API calls if needed
        if self.session is None:
            self.session = aiohttp.ClientSession()
        
        # Ghi log số lượng callback đã đăng ký trước khi kết nối
        logger.info(f"Before starting: {len(self.new_token_callbacks)} new token callbacks registered")
        
        # Connect to WebSocket API
        connected = await self._connect_websocket()
        
        if connected:
            # Wait for connection to complete
            try:
                # Ghi log số lượng callback sau khi kết nối
                logger.info(f"After connection: {len(self.new_token_callbacks)} new token callbacks registered")
                
                # Subscribe to events
                await self._subscribe_to_events()
                
                # Start message handler
                self._message_handler_task = asyncio.create_task(self._message_handler())
                self._tasks.append(self._message_handler_task)
                
                # Start keep-alive task (ping/pong)
                self._keep_alive_task = asyncio.create_task(self._keep_connection_alive())
                self._tasks.append(self._keep_alive_task)
                
                logger.info("PumpPortalClient started successfully")
            except Exception as e:
                logger.error(f"Error during PumpPortalClient startup: {str(e)}")
                self._record_error(e, "startup")
        else:
            error_msg = "Failed to connect to PumpPortal WebSocket"
            logger.error(error_msg)
            self._record_error(Exception(error_msg), "connect_websocket")
            
            # Try reconnecting in the background
            logger.info("Starting background reconnect task")
            self._reconnect_task = asyncio.create_task(self._reconnect_websocket_loop())
            self._tasks.append(self._reconnect_task)
        
        # Không cập nhật trạng thái ở đây vì đã được thực hiện trong super().start() hoặc các phương thức khác
    
    async def stop(self):
        """Stop the PumpPortalClient and disconnect from PumpPortal API"""
        if not self.is_running:
            logger.warning("PumpPortalClient not running")
            return
        
        logger.info("Stopping PumpPortalClient")
        
        # Đánh dấu component đang dừng lại - đã được xử lý trong BaseComponent
        # self.state_manager.update_component_status('pump_portal_client', 'stopping')
        
        # Set flags to prevent reconnection loops
        self.is_running = False
        self.is_connected = False
        
        # Hủy tất cả các background task
        try:
            await super().stop()  # Hàm stop() của BaseComponent sẽ hủy các task
        except Exception as e:
            logger.error(f"Error stopping background tasks: {str(e)}")
        
        # Đóng WebSocket connection an toàn
        if self.websocket:
            try:
                logger.info("Closing WebSocket connection")
                # Sử dụng timeout để tránh bị treo khi đóng
                await asyncio.wait_for(self.websocket.close(), timeout=3.0)
            except (asyncio.TimeoutError, Exception) as e:
                logger.warning(f"Error closing WebSocket connection: {str(e)}")
            finally:
                self.websocket = None
        
        # Đóng HTTP session an toàn
        if self.session:
            try:
                logger.info("Closing HTTP session")
                # Kiểm tra xem session có bị đóng chưa
                if not self.session.closed:
                    await self.session.close()
            except Exception as e:
                logger.warning(f"Error closing HTTP session: {str(e)}")
            finally:
                self.session = None
        
        # Đợi cho đến khi tất cả các task background đã dừng
        for task_name, task in self._get_pending_tasks().items():
            try:
                logger.info(f"Canceling task: {task_name}")
                task.cancel()
            except Exception as e:
                logger.warning(f"Error canceling task {task_name}: {str(e)}")
        
        logger.info("PumpPortalClient stopped")
        # Đã được xử lý trong hàm stop() của BaseComponent
        # self.state_manager.update_component_status('pump_portal_client', 'stopped')
    
    def _get_pending_tasks(self):
        """Get a dictionary of pending tasks related to this client"""
        pending_tasks = {}
        for task in asyncio.all_tasks():
            task_name = task.get_name()
            # Kiểm tra các task liên quan đến component này
            if ('pump_portal' in task_name or 'websocket' in task_name) and not task.done():
                pending_tasks[task_name] = task
        return pending_tasks
    
    async def _reconnect_websocket_loop(self):
        """Background task to handle reconnection attempts with exponential backoff"""
        logger.info("Starting reconnection loop with exponential backoff")
        
        # Lưu trữ số lượng callback trước khi kết nối lại
        stored_new_token_callbacks = self.new_token_callbacks.copy()
        stored_liquidity_change_callbacks = self.liquidity_change_callbacks.copy()
        stored_price_change_callbacks = self.price_change_callbacks.copy()
        stored_trade_callbacks = self.trade_callbacks.copy()
        
        # Ghi log số lượng callback đã lưu trữ
        logger.info(f"Stored callbacks before reconnection: {len(stored_new_token_callbacks)} new token callbacks")
        
        # Giới hạn số lần thử lại và thiết lập exponential backoff
        base_delay = 1.0  # Bắt đầu với 1 giây
        max_delay = 300.0  # Tối đa 5 phút
        backoff_factor = 1.5  # Hệ số nhân mỗi lần thử lại
        jitter = 0.1  # Thêm jitter để tránh reconnection storm
        attempt = 0
        
        # Đồng bộ hóa alias _websocket
        self._websocket = self.websocket
        
        while self.is_running and not self.is_connected:
            attempt += 1
            try:
                # Thông báo về lần thử kết nối
                logger.info(f"Reconnection attempt {attempt} (max attempts: {self.max_reconnect_attempts if self.max_reconnect_attempts > 0 else 'unlimited'})")
                
                # Nếu có giới hạn số lần thử và vượt quá, break
                if self.max_reconnect_attempts > 0 and attempt > self.max_reconnect_attempts:
                    logger.error(f"Exceeded maximum reconnection attempts ({self.max_reconnect_attempts}), giving up")
                    self.state_manager.update_component_status('pump_portal_client', 'error', "Max reconnection attempts reached")
                    break
                
                # Tính toán thời gian chờ với exponential backoff và jitter
                delay = min(base_delay * (backoff_factor ** (attempt - 1)), max_delay)
                # Thêm jitter ngẫu nhiên (±10%)
                import random
                jitter_amount = delay * jitter
                delay = delay + random.uniform(-jitter_amount, jitter_amount)
                
                # Thử kết nối lại
                success = await self._connect_websocket()
                
                # Nếu kết nối thành công, khôi phục các callback đã lưu
                if success and self.is_connected:
                    logger.info("Reconnection successful, restoring callbacks")
                    
                    # Khôi phục các callback đã lưu trước đó
                    self.new_token_callbacks.update(stored_new_token_callbacks)
                    self.liquidity_change_callbacks.update(stored_liquidity_change_callbacks)
                    self.price_change_callbacks.update(stored_price_change_callbacks)
                    self.trade_callbacks.update(stored_trade_callbacks)
                    
                    # Ghi log số lượng callback sau khi khôi phục
                    logger.info(f"Restored callbacks after reconnection: {len(self.new_token_callbacks)} new token callbacks")
                    
                    # Đăng ký lại các sự kiện
                    await self._subscribe_to_events()
                    logger.info("Re-subscribed to events after reconnection")
                    
                    # Cập nhật trạng thái
                    self.reconnect_attempts = 0  # Reset counter on success
                    self.state_manager.update_component_status('pump_portal_client', 'running')
                    break  # Exit the reconnection loop
                
                # Nếu kết nối không thành công, chờ và thử lại
                if not self.is_connected:
                    logger.info(f"Reconnection attempt {attempt} failed, waiting {delay:.2f} seconds before next attempt")
                    await asyncio.sleep(delay)
            
            except Exception as e:
                logger.error(f"Error in reconnection loop: {str(e)}")
                # Ghi lại traceback để debug chi tiết hơn
                logger.error(f"Traceback: {traceback.format_exc()}")
                
                # Chờ một khoảng thời gian ngắn trước khi thử lại khi có lỗi
                await asyncio.sleep(min(delay, 10))  # Tối đa 10 giây khi có lỗi
        
        logger.info("Reconnection loop ended")
    
    async def _connect_websocket(self):
        """Connect to PumpPortal WebSocket API"""
        # Check if we're in development/demo mode
        dev_mode = self.config_manager.get('environment', 'development') == 'development'
        
        # Ghi log số lượng callback đã đăng ký trước khi kết nối
        logger.info(f"Before connection: {len(self.new_token_callbacks)} new token callbacks registered")
        
        if dev_mode:
            # In development mode, simulate a successful connection but don't actually connect
            logger.info("Running in development mode, simulating PumpPortal connection")
            self.is_connected = True
            self.reconnect_attempts = 0
            self.state_manager.update_component_status('pump_portal_client', 'running')
            
            # Stay connected indefinitely
            while self.is_running:
                await asyncio.sleep(60)
            return
            
        # Normal connection flow for production
        while self.is_running and (self.reconnect_attempts < self.max_reconnect_attempts or self.max_reconnect_attempts <= 0):
            try:
                logger.info(f"Connecting to PumpPortal WebSocket API at {self.websocket_url}")
                
                # Connect to WebSocket server with API key in header
                # Using websockets 15.x syntax
                # Try multiple connection methods to handle different API requirements
                import urllib.parse
                
                # Mask API key when logging
                logger.info(f"Attempting connection with API key: {'*'*(len(self.api_key)-4) + self.api_key[-4:] if self.api_key else 'None'}")
                
                # PumpPortal API chỉ hỗ trợ một phương thức xác thực
                try:
                    logger.info("Kết nối với Pumpportal")
                    import websockets
                    
                    # Kiểm tra phiên bản websockets
                    ws_version = getattr(websockets, "__version__", "unknown")
                    logger.info(f"Phiên bản websockets: {ws_version}")
                    
                    # Phương thức kết nối đơn giản không sử dụng header
                    # Kết nối trực tiếp đến endpoint
                    self.websocket = await websockets.connect(self.websocket_url)
                    
                    # Sau khi kết nối, gửi API key thông qua tin nhắn xác thực
                    auth_message = {
                        "method": "auth", 
                        "key": self.api_key
                    }
                    
                    logger.info("Gửi thông điệp xác thực...")
                    await self.websocket.send(json.dumps(auth_message))
                    
                    logger.info("Kết nối WebSocket thành công")
                    
                except Exception as e:
                    logger.error(f"Lỗi kết nối: {str(e)}")
                    raise e
                
                self.is_connected = True
                self.reconnect_attempts = 0
                logger.info("Connected to PumpPortal WebSocket API")
                self.state_manager.update_component_status('pump_portal_client', 'running')
                
                # Ghi log số lượng callback sau khi kết nối
                logger.info(f"After connection: {len(self.new_token_callbacks)} new token callbacks registered")
                
                # Return success to indicate successful connection
                return True
                
            except (ConnectionError, 
                    OSError,
                    websockets.exceptions.ConnectionClosedError,
                    Exception) as e:
                
                self.is_connected = False
                self.reconnect_attempts += 1
                
                logger.error(f"WebSocket connection error: {str(e)}")
                self.state_manager.update_component_status(
                    'pump_portal_client', 
                    'error', 
                    f"WebSocket connection error: {str(e)}"
                )
                
                # Tính thời gian chờ với backoff
                reconnect_wait = min(self.reconnect_interval * (1.5 ** self.reconnect_attempts), 300)  # Max 5 minutes
                
                logger.info(f"Reconnecting in {reconnect_wait} seconds (attempt {self.reconnect_attempts})")
                await asyncio.sleep(reconnect_wait)
        
        logger.error("Max reconnection attempts reached, giving up")
        return False
    
    async def _subscribe_to_events(self):
        """Subscribe to events from PumpPortal"""
        if not self.is_connected or not self.websocket:
            logger.warning("Cannot subscribe to events: not connected")
            return
        
        try:
            # Sử dụng format chính xác từ tài liệu API của PumpPortal
            logger.info("Đăng ký sự kiện token mới với PumpPortal")
            
            # Đây là định dạng chính xác theo mẫu của API
            # "method": "subscribeNewToken" với trường "keys" bắt buộc
            # Tài liệu: https://pumpportal.fun/docs/#token-events
            new_token_msg = {
                "method": "subscribeNewToken",
                "keys": ["*"]  # Sử dụng "*" để subscribe tất cả token mới
            }
            
            # Ghi lại tin nhắn đầy đủ để debug
            logger.debug(f"Sending message: {json.dumps(new_token_msg)}")
            await self.websocket.send(json.dumps(new_token_msg))
            logger.info("Đã gửi yêu cầu đăng ký token mới")
            
            # Tạm dừng 1 giây giữa các yêu cầu để tránh quá tải
            await asyncio.sleep(1)
            
            # Subscribe to Raydium liquidity
            logger.info("Đăng ký sự kiện thay đổi thanh khoản với PumpPortal")
            
            # Format chuẩn cho subscribeRaydiumLiquidity
            liquidity_msg = {
                "method": "subscribeRaydiumLiquidity",
                "keys": ["*"]  # Sử dụng "*" để subscribe tất cả sự kiện thanh khoản
            }
            
            logger.debug(f"Sending message: {json.dumps(liquidity_msg)}")
            await self.websocket.send(json.dumps(liquidity_msg))
            logger.info("Đã gửi yêu cầu đăng ký sự kiện thanh khoản")
            logger.debug("Sent subscription request for Raydium liquidity events")
            
            # Subscribe to whale account trades if we have any configured
            if self.tracked_accounts and len(self.tracked_accounts) > 0:
                logger.info(f"Đăng ký theo dõi giao dịch từ {len(self.tracked_accounts)} tài khoản...")
                
                # Xử lý từng tài khoản một để tránh lỗi định dạng
                for account in self.tracked_accounts:
                    account_msg = {
                        "method": "subscribeAccountTrade",
                        "keys": [account]  # Đúng định dạng với keys là danh sách
                    }
                    logger.debug(f"Sending message: {json.dumps(account_msg)}")
                    await self.websocket.send(json.dumps(account_msg))
                    await asyncio.sleep(0.2)  # Tạm dừng giữa các yêu cầu
                
                logger.info(f"Đã đăng ký theo dõi tất cả các tài khoản")
            
            # Subscribe to specific token trades if we're tracking any
            if self.tracked_tokens and len(self.tracked_tokens) > 0:
                logger.info(f"Đăng ký theo dõi giao dịch cho {len(self.tracked_tokens)} token...")
                
                # Xử lý từng token một để tránh lỗi định dạng
                for token in self.tracked_tokens:
                    token_msg = {
                        "method": "subscribeTokenTrade",
                        "keys": [token]  # Đúng định dạng với keys là danh sách
                    }
                    logger.debug(f"Sending message: {json.dumps(token_msg)}")
                    await self.websocket.send(json.dumps(token_msg))
                    await asyncio.sleep(0.2)  # Tạm dừng giữa các yêu cầu
                
                logger.info(f"Đã đăng ký theo dõi tất cả các token")
            
            logger.info("Completed all PumpPortal subscription requests")
            
        except Exception as e:
            logger.error(f"Error subscribing to events: {str(e)}")
            self.state_manager.update_component_status(
                'pump_portal_client', 
                'error', 
                f"Error subscribing to events: {str(e)}"
            )
    
    async def _message_handler(self):
        """Handle incoming WebSocket messages"""
        if not self.is_connected or not self.websocket:
            logger.warning("Cannot handle messages: not connected")
            return
        
        try:
            async for message in self.websocket:
                try:
                    # Process the message
                    data = json.loads(message)
                    
                    # Update last activity time for heartbeat
                    self.last_ping_time = time.time()
                    
                    # Log the message for detailed debugging
                    logger.debug(f"Received PumpPortal message: {data}")
                    
                    # Identify message type according to PumpPortal docs format
                    
                    # Ghi log chi tiết hơn về tin nhắn nhận được
                    logger.debug(f"RAW MESSAGE DATA: {json.dumps(data)}")
                    
                    # Kiểm tra thông báo lỗi từ server
                    if 'errors' in data:
                        logger.error(f"PumpPortal API error: {data['errors']}")
                        # Nếu thông báo lỗi về format, ghi log chi tiết
                        if "Invalid message" in str(data['errors']):
                            logger.error(f"Invalid message format detected! Check subscription format")
                    
                    # Kiểm tra thông báo thành công
                    if 'message' in data:
                        logger.info(f"PumpPortal server message: {data['message']}")
                        # Nếu đã subscribe thành công, đánh dấu
                        if "Successfully subscribed" in str(data['message']) or "Subscribed to" in str(data['message']):
                            logger.info(f"✅ Subscription confirmed: {data['message']}")
                    
                    # Format chính thức theo tài liệu - Sự kiện token mới được gửi ở định dạng sau
                    # { 
                    #  "mint": "address", 
                    #  "symbol": "symbol", 
                    #  "name": "name", 
                    #  "txType": "create", 
                    #  "tx": "transaction signature" 
                    # }
                    if 'txType' in data and data.get('txType') == 'create' and 'mint' in data:
                        # Token mới được phát hiện!
                        token_address = data.get('mint')
                        token_name = data.get('name', 'Unknown')
                        token_symbol = data.get('symbol', 'Unknown')
                        
                        # Ghi log chi tiết với định dạng dễ nhìn
                        logger.info(f"✅ TOKEN MỚI PHÁT HIỆN!")
                        logger.info(f"   Địa chỉ: {token_address}")
                        logger.info(f"   Tên: {token_name}")
                        logger.info(f"   Ký hiệu: {token_symbol}")
                        logger.info(f"   Giao dịch: {data.get('tx', 'Unknown')}")
                        
                        # Chuyển đổi sang định dạng chuẩn để xử lý
                        token_data = {
                            'address': token_address,
                            'name': token_name,
                            'symbol': token_symbol,
                            'tx': data.get('tx'),  # Lưu lại mã giao dịch tạo token
                            'timestamp': time.time(),  # Thêm thời gian phát hiện
                            'source': 'pumpportal'  # Đánh dấu nguồn dữ liệu
                        }
                        
                        # Thông báo token mới cho các module khác
                        await self._handle_new_token_event(token_data)
                        return
                    
                    # Kiểm tra các loại sự kiện theo định dạng API chính xác của PumpPortal
                    method = data.get('method')
                    
                    if method == 'newToken':
                        # Sự kiện token mới - theo tài liệu PumpPortal API chính xác
                        logger.info(f"✅ Nhận được sự kiện token mới (newToken): {data.get('params', {}).get('symbol', 'Unknown')}")
                        
                        # Cấu trúc chuẩn từ PumpPortal có params chứa thông tin token
                        if 'params' in data:
                            # Chuyển đổi sang định dạng chuẩn để xử lý
                            token_data = data.get('params', {})
                            # Thêm thông tin nguồn và thời gian
                            token_data.update({
                                'timestamp': time.time(),
                                'source': 'pumpportal_newToken'
                            })
                            await self._handle_new_token_event(token_data)
                        else:
                            # Dữ liệu không đúng định dạng
                            logger.warning(f"Nhận được sự kiện newToken không có params: {data}")
                            await self._handle_new_token_event(data)
                    
                    elif method == 'tokenTrade':
                        # Token trade event
                        logger.info(f"Processing token trade event")
                        await self._handle_trade_event(data)
                    
                    elif method == 'accountTrade':
                        # Account trade event
                        logger.info(f"Processing account trade event")
                        await self._handle_trade_event(data)
                    
                    elif method == 'raydiumLiquidity':
                        # Raydium liquidity event
                        logger.info(f"Processing Raydium liquidity event")
                        await self._handle_liquidity_change_event(data)
                    
                    # Handle response to ping/authentication
                    elif method == 'pong' or method == 'auth':
                        logger.debug(f"Received {method} response from PumpPortal")
                    
                    # Fallback to legacy format handling for backward compatibility
                    elif 'event' in data:
                        event_type = data.get('event')
                        logger.info(f"Processing legacy event format: {event_type}")
                        
                        if event_type == 'new_token':
                            await self._handle_new_token_event(data)
                        elif event_type == 'liquidity_change':
                            await self._handle_liquidity_change_event(data)
                        elif event_type == 'price_change':
                            await self._handle_price_update_event(data)
                        elif event_type == 'trade':
                            await self._handle_trade_event(data)
                    
                    # Final fallback - try to determine the event type from message content
                    elif 'token' in data and 'address' in data.get('token', {}):
                        # Likely a token-related event
                        logger.info("Processing token event from message content")
                        await self._handle_new_token_event(data)
                    
                    elif 'liquidity' in data:
                        # Likely a liquidity change event
                        logger.info("Processing liquidity event from message content")
                        await self._handle_liquidity_change_event(data)
                    
                    elif 'price' in data:
                        # Likely a price update event
                        logger.info("Processing price event from message content")
                        await self._handle_price_update_event(data) 
                    
                    elif 'trade' in data:
                        # Likely a trade event
                        logger.info("Processing trade event from message content")
                        await self._handle_trade_event(data)
                    
                    else:
                        logger.warning(f"Received unknown message format: {data}")
                        # Log more details about the unrecognized message
                        logger.debug(f"Message keys: {list(data.keys())}")
                    
                except json.JSONDecodeError as e:
                    logger.error(f"Error decoding message: {str(e)}")
                
                except Exception as e:
                    logger.error(f"Error handling message: {str(e)}")
        
        except Exception as e:
            logger.warning(f"WebSocket connection closed: {str(e)}")
            self.is_connected = False
    
    async def _heartbeat(self):
        """Send periodic pings to keep the connection alive"""
        while self.is_running:
            try:
                if self.is_connected and self.websocket:
                    # If it's been more than ping_interval since last ping, send a ping
                    if time.time() - self.last_ping_time > self.ping_interval:
                        await self.websocket.send(json.dumps({"action": "ping"}))
                        self.last_ping_time = time.time()
                        logger.debug("Sent ping to PumpPortal")
            
            except Exception as e:
                logger.error(f"Error in heartbeat: {str(e)}")
            
            await asyncio.sleep(5)  # Check every 5 seconds
            
    async def _keep_connection_alive(self):
        """
        Dedicated task for ping/pong mechanism to keep WebSocket connection alive
        This helps prevent connection drops due to inactivity timeouts
        """
        ping_interval = 15  # Giảm thời gian ping xuống 15 giây
        pong_timeout = 10  # Timeout chờ pong response
        reconnect_on_failure = True
        max_missed_pongs = 2
        missed_pongs = 0
        
        while self.is_running:
            try:
                if self.is_connected and self.websocket:
                    # Đặt thời gian gửi ping gần đây nhất
                    ping_sent_time = time.time()
                    pong_received = False
                    
                    # Gửi ping frame để giữ kết nối
                    pong_waiter = await self.websocket.ping()
                    logger.debug("Sent WebSocket ping frame")
                    
                    # Đợi pong response với timeout
                    try:
                        await asyncio.wait_for(pong_waiter, timeout=pong_timeout)
                        # Pong nhận được
                        logger.debug("Received WebSocket pong response")
                        pong_received = True
                        missed_pongs = 0  # Reset missed pongs counter
                        self.last_ping_time = time.time()
                        
                        # Update metrics about connection status
                        uptime_hours = (time.time() - self.uptime_start) / 3600.0
                        reconnect_ratio = self.disconnection_count / max(1, uptime_hours)
                        self.state_manager.update_component_metric(
                            'pump_portal_client', 
                            'reconnect_ratio_per_hour', 
                            reconnect_ratio
                        )
                    except asyncio.TimeoutError:
                        # Không nhận được pong response trong thời gian timeout
                        missed_pongs += 1
                        logger.warning(f"No pong response received within {pong_timeout}s (missed: {missed_pongs}/{max_missed_pongs})")
                        
                        # Kiểm tra nếu quá nhiều pong bị miss
                        if missed_pongs >= max_missed_pongs and reconnect_on_failure:
                            logger.error(f"Too many missed pongs ({missed_pongs}), initiating reconnection")
                            # Đánh dấu kết nối đã đóng
                            self.is_connected = False
                            self.disconnection_count += 1
                            
                            # Cập nhật trạng thái component
                            self.state_manager.update_component_status(
                                'pump_portal_client',
                                'error',
                                'Connection lost - no pong responses'
                            )
                            self.last_disconnect_time = time.time()
                            
                            # Đóng kết nối hiện tại
                            try:
                                await self.websocket.close()
                            except:
                                pass  # Ignore errors during close
                            
                            # Khởi tạo task kết nối lại
                            self._reconnect_task = asyncio.create_task(self._reconnect_websocket_loop())
                            self._tasks.append(self._reconnect_task)
                    
                    # Gửi thêm keepalive message nếu cần
                    inactive_time = time.time() - self.last_ping_time
                    if inactive_time > ping_interval * 3:
                        logger.warning(f"Connection inactive for {inactive_time:.1f}s, sending additional keep-alive message")
                        # Gửi một JSON message để giữ kết nối
                        try:
                            await self.websocket.send(json.dumps({"method": "keepalive", "timestamp": int(time.time())}))
                            self.last_ping_time = time.time()
                        except Exception as e:
                            logger.error(f"Failed to send keepalive message: {str(e)}")
            except Exception as e:
                if "connection is closed" in str(e).lower() or "no close frame received or sent" in str(e).lower():
                    # Kết nối đã đóng, ghi nhận sự kiện ngắt kết nối
                    if self.is_connected:
                        self.is_connected = False
                        self.disconnection_count += 1
                        self.last_disconnect_time = time.time()
                        logger.warning(f"WebSocket connection closed during keep-alive (total disconnects: {self.disconnection_count})")
                        
                        # Khởi động lại kết nối
                        if reconnect_on_failure and self.is_running:
                            logger.info("Initiating WebSocket reconnection after connection closed")
                            self._reconnect_task = asyncio.create_task(self._reconnect_websocket_loop())
                            self._tasks.append(self._reconnect_task)
                else:
                    logger.error(f"Error in keep-alive task: {str(e)}")
                
            # Chờ đến chu kỳ ping tiếp theo
            await asyncio.sleep(ping_interval)
    
    async def _handle_new_token_event(self, data: Dict[str, Any]):
        """
        Handle new token event from PumpPortal
        
        Args:
            data (Dict[str, Any]): Event data
        """
        # Handle different data formats for new token events
        
        # Format 1: PumpPortal new format (method: newToken)
        if 'method' in data and data.get('method') == 'newToken' and 'params' in data:
            params = data.get('params', {})
            token_data = params
            token_address = params.get('address')
        
        # Format 2: Direct token data in 'token' field
        elif 'token' in data:
            token_data = data.get('token', {})
            token_address = token_data.get('address')
        
        # Format 3: Direct data fields
        else:
            token_data = data
            token_address = data.get('address')
        
        if not token_address:
            logger.warning(f"Received new token event without address: {data}")
            return
        
        logger.info(f"New token detected: {token_data.get('symbol', 'Unknown')} ({token_address})")
        
        # Store token data
        self.tokens[token_address] = token_data
        
        # Update metrics
        self.state_manager.update_component_metric(
            'pump_portal_client', 
            'new_tokens_detected', 
            len(self.tokens)
        )
        
        # Notify callbacks
        callback_count = len(self.new_token_callbacks)
        if callback_count > 0:
            token_symbol = token_data.get('symbol', 'Unknown')
            logger.info(f"Notifying {callback_count} callbacks about new token {token_symbol}")
            
            for callback in self.new_token_callbacks:
                try:
                    # Ghi log thông tin chi tiết về callback để debug
                    callback_name = getattr(callback, '__name__', str(callback))
                    callback_module = getattr(callback, '__module__', 'unknown')
                    logger.info(f"Executing callback: {callback_name} from {callback_module}")
                    
                    # Tạo và theo dõi task xử lý callback
                    task = asyncio.create_task(callback(token_data))
                    task.set_name(f"token_callback_{token_symbol}_{callback_name}")
                    logger.debug(f"Created task {task.get_name()} for new token callback")
                except Exception as e:
                    logger.error(f"Error in new token callback: {str(e)}")
                    logger.error(f"Callback details: {callback}")
                    # Lưu lại lỗi này trong metrics
                    self.state_manager.update_component_metric(
                        'pump_portal_client', 
                        'callback_errors', 
                        1, 
                        increment=True
                    )
        else:
            token_symbol = token_data.get('symbol', 'Unknown')
            logger.warning(f"No callbacks registered for new token events. Token {token_symbol} will not be processed.")
            logger.error(f"CALLBACK ERROR: Missing token callback registration for {token_symbol}")
            
            # Thông báo về vấn đề callback cho state manager
            self.state_manager.update_component_status(
                'pump_portal_client', 
                'warning', 
                f"No callbacks registered for token events - Check TradingIntegration"
            )
    
    async def _handle_liquidity_change_event(self, data: Dict[str, Any]):
        """
        Handle liquidity change event from PumpPortal
        
        Args:
            data (Dict[str, Any]): Event data
        """
        # Handle different data formats for liquidity change events
        
        # Format 1: PumpPortal new format (method: raydiumLiquidity)
        if 'method' in data and data.get('method') == 'raydiumLiquidity' and 'params' in data:
            params = data.get('params', {})
            token_address = params.get('address')
            new_liquidity = params.get('liquidity', 0)
            old_liquidity = params.get('old_liquidity', 0)
            
            # If old_liquidity not provided, try to get from our stored data
            if old_liquidity == 0 and token_address in self.tokens and 'liquidity_usd' in self.tokens[token_address]:
                old_liquidity = self.tokens[token_address]['liquidity_usd']
        
        # Format 2: Direct fields in data
        else:
            token_address = data.get('token_address')
            if not token_address and 'address' in data:
                token_address = data.get('address')
                
            old_liquidity = data.get('old_liquidity_usd', 0)
            if old_liquidity == 0:
                old_liquidity = data.get('old_liquidity', 0)
                
            new_liquidity = data.get('new_liquidity_usd', 0)
            if new_liquidity == 0:
                new_liquidity = data.get('liquidity', 0)
                if new_liquidity == 0:
                    new_liquidity = data.get('new_liquidity', 0)
        
        if not token_address:
            logger.warning(f"Received liquidity change event without token address: {data}")
            return
        
        # Calculate percentage change
        pct_change = 0
        if old_liquidity > 0:
            pct_change = ((new_liquidity - old_liquidity) / old_liquidity) * 100
        
        logger.debug(f"Liquidity change for {token_address}: {old_liquidity} -> {new_liquidity} USD ({pct_change:.2f}%)")
        
        # Update token data if we have it
        if token_address in self.tokens:
            self.tokens[token_address]['liquidity_usd'] = new_liquidity
        
        # Notify callbacks for significant changes (>5%)
        if abs(pct_change) >= 5:
            callback_count = len(self.liquidity_change_callbacks)
            if callback_count > 0:
                logger.info(f"Notifying {callback_count} callbacks about liquidity change of {pct_change:.2f}% for token {token_address}")
                
                change_data = {
                    'token_address': token_address,
                    'old_liquidity_usd': old_liquidity,
                    'new_liquidity_usd': new_liquidity,
                    'percentage_change': pct_change
                }
                
                for callback in self.liquidity_change_callbacks:
                    try:
                        task = asyncio.create_task(callback(change_data))
                        logger.debug(f"Created task {task.get_name()} for liquidity change callback")
                    except Exception as e:
                        logger.error(f"Error in liquidity change callback: {str(e)}")
            else:
                logger.debug(f"No callbacks registered for liquidity change events. Change of {pct_change:.2f}% for token {token_address} will not be processed.")
    
    async def _handle_price_update_event(self, data: Dict[str, Any]):
        """
        Handle price update event from PumpPortal
        
        Args:
            data (Dict[str, Any]): Event data
        """
        # Handle different data formats for price update events
        
        # Format 1: PumpPortal tokenPrice format
        if 'method' in data and data.get('method') == 'tokenPrice' and 'params' in data:
            params = data.get('params', {})
            token_address = params.get('address')
            new_price = params.get('price', 0)
            
            # Try to get old price from our stored data
            old_price = 0
            if token_address in self.tokens and 'price_usd' in self.tokens[token_address]:
                old_price = self.tokens[token_address]['price_usd']
        
        # Format 2: Direct fields in data - old format
        else:
            token_address = data.get('token_address')
            if not token_address and 'address' in data:
                token_address = data.get('address')
                
            old_price = data.get('old_price_usd', 0)
            if old_price == 0:
                old_price = data.get('old_price', 0)
                
            new_price = data.get('new_price_usd', 0)
            if new_price == 0:
                new_price = data.get('price_usd', 0)
                if new_price == 0:
                    new_price = data.get('price', 0)
                    if new_price == 0:
                        new_price = data.get('new_price', 0)
        
        if not token_address:
            logger.warning(f"Received price update event without token address: {data}")
            return
        
        # Calculate percentage change
        pct_change = 0
        if old_price > 0:
            pct_change = ((new_price - old_price) / old_price) * 100
        
        logger.debug(f"Price change for {token_address}: {old_price} -> {new_price} USD ({pct_change:.2f}%)")
        
        # Update token data if we have it
        if token_address in self.tokens:
            self.tokens[token_address]['price_usd'] = new_price
        
        # Notify callbacks for significant changes (>3%)
        if abs(pct_change) >= 3:
            callback_count = len(self.price_change_callbacks)
            if callback_count > 0:
                logger.info(f"Notifying {callback_count} callbacks about price change of {pct_change:.2f}% for token {token_address}")
                
                change_data = {
                    'token_address': token_address,
                    'old_price_usd': old_price,
                    'new_price_usd': new_price,
                    'percentage_change': pct_change
                }
                
                for callback in self.price_change_callbacks:
                    try:
                        task = asyncio.create_task(callback(change_data))
                        logger.debug(f"Created task {task.get_name()} for price change callback")
                    except Exception as e:
                        logger.error(f"Error in price change callback: {str(e)}")
            else:
                logger.debug(f"No callbacks registered for price change events. Change of {pct_change:.2f}% for token {token_address} will not be processed.")
    
    async def _handle_trade_event(self, data: Dict[str, Any]):
        """
        Handle trade event from PumpPortal
        
        Args:
            data (Dict[str, Any]): Event data
        """
        # Handle different data formats for trade events
        
        # Format 1: PumpPortal new format (method: tokenTrade or accountTrade)
        if 'method' in data and data.get('method') in ['tokenTrade', 'accountTrade'] and 'params' in data:
            params = data.get('params', {})
            token_address = params.get('address')
            amount = params.get('amount', 0)
            price = params.get('price', 0)
            side = params.get('side', '')  # 'buy' or 'sell'
            
            # Create structured trade data
            trade_data = {
                'token_address': token_address,
                'amount': amount,
                'price_usd': price,
                'side': side,
                'timestamp': time.time(),
                'value_usd': amount * price
            }
        
        # Format 2: Direct fields in data
        else:
            token_address = data.get('token_address')
            if not token_address and 'address' in data:
                token_address = data.get('address')
                
            amount = data.get('amount', 0)
            price = data.get('price_usd', 0)
            if price == 0:
                price = data.get('price', 0)
                
            side = data.get('side', '')  # 'buy' or 'sell'
            
            # Use the original data for callbacks
            trade_data = data
        
        if not token_address:
            logger.warning(f"Received trade event without token address: {data}")
            return
        
        value_usd = amount * price
        
        logger.debug(f"Trade for {token_address}: {side} {amount} at {price} USD (Total: ${value_usd:.2f})")
        
        # Notify callbacks for significant trades (worth at least $1000)
        if value_usd >= 1000:
            callback_count = len(self.trade_callbacks)
            if callback_count > 0:
                logger.info(f"Notifying {callback_count} callbacks about significant trade: {side} {amount} {token_address} at ${price} (Total: ${value_usd:.2f})")
                
                for callback in self.trade_callbacks:
                    try:
                        task = asyncio.create_task(callback(trade_data))
                        logger.debug(f"Created task {task.get_name()} for trade callback")
                    except Exception as e:
                        logger.error(f"Error in trade callback: {str(e)}")
            else:
                logger.debug(f"No callbacks registered for trade events. Significant trade of ${value_usd:.2f} for token {token_address} will not be processed.")
    
    def register_new_token_callback(self, callback: Callable):
        """
        Register a callback for new token events
        
        Args:
            callback: Callback function that takes token data as parameter
        """
        # Ghi log thông tin chi tiết về callback để debug
        callback_name = getattr(callback, '__name__', str(callback))
        callback_module = getattr(callback, '__module__', 'unknown')
        
        self.new_token_callbacks.add(callback)
        logger.info(f"Registered new token callback: {callback_name} from {callback_module}")
        logger.info(f"Total callbacks now: {len(self.new_token_callbacks)}")
    
    def register_liquidity_change_callback(self, callback: Callable):
        """
        Register a callback for liquidity change events
        
        Args:
            callback: Callback function that takes liquidity change data as parameter
        """
        self.liquidity_change_callbacks.add(callback)
        logger.debug("Registered liquidity change callback")
    
    def register_price_change_callback(self, callback: Callable):
        """
        Register a callback for price update events
        
        Args:
            callback: Callback function that takes price update data as parameter
        """
        self.price_change_callbacks.add(callback)
        logger.debug("Registered price change callback")
    
    def register_trade_callback(self, callback: Callable):
        """
        Register a callback for trade events
        
        Args:
            callback: Callback function that takes trade data as parameter
        """
        self.trade_callbacks.add(callback)
        logger.debug("Registered trade callback")
    
    async def restart(self):
        """
        Restart the PumpPortalClient component
        
        This method is designed to be called by the self-healing system.
        It will stop and then restart the component.
        """
        logger.info("Restarting PumpPortalClient component")
        
        try:
            # First, stop the component if it's running
            if self.is_running:
                logger.info("Stopping PumpPortalClient before restart")
                await self.stop()
                
                # Short pause to ensure clean shutdown
                await asyncio.sleep(2)
            
            # Reset connection state
            self.is_connected = False
            self.reconnect_attempts = 0
            self._websocket = None
            self.websocket = None
            
            # Reset error counters
            self._error_count = 0
            
            # Start the component again
            logger.info("Starting PumpPortalClient after restart")
            await self.start()
            
            # Log success
            if self.is_running:
                logger.info("PumpPortalClient successfully restarted")
                return True
            else:
                logger.error("PumpPortalClient failed to restart properly")
                return False
                
        except Exception as e:
            logger.error(f"Error during PumpPortalClient restart: {str(e)}")
            # Log detailed traceback for debugging
            logger.error(traceback.format_exc())
            return False
            
    async def call_api(self, method: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Call PumpPortal REST API
        
        Args:
            method (str): API method to call
            params (Dict[str, Any], optional): API parameters
            
        Returns:
            Any: API response
        """
        if not self.session:
            logger.warning("Cannot call API: session not initialized")
            return {}
        
        # Construct API URL
        url = f"{self.rest_url}/v1/{method}"
        
        # API params
        headers = {
            "X-API-KEY": self.api_key,
            "Content-Type": "application/json"
        }
        
        try:
            if method in ['get_token', 'get_tokens', 'get_market_data']:
                # Use GET for retrieval methods
                async with self.session.get(url, headers=headers, params=params) as response:
                    if response.status != 200:
                        logger.warning(f"API call failed: {response.status} {await response.text()}")
                        return {}
                    
                    return await response.json()
            else:
                # Use POST for other methods
                async with self.session.post(url, headers=headers, json=params) as response:
                    if response.status != 200:
                        logger.warning(f"API call failed: {response.status} {await response.text()}")
                        return {}
                    
                    return await response.json()
            
        except aiohttp.ClientError as e:
            logger.error(f"API call error: {str(e)}")
            self.state_manager.update_component_status(
                'pump_portal_client', 
                'error', 
                f"API call error: {str(e)}"
            )
            return {}
        
        except Exception as e:
            logger.error(f"Unexpected error in API call: {str(e)}")
            return {}
    
    async def get_token_info(self, token_address: str) -> Optional[Dict[str, Any]]:
        """
        Get token information
        
        Args:
            token_address (str): Token address
            
        Returns:
            Optional[Dict[str, Any]]: Token information or None if not found
        """
        logger.debug(f"Getting token info for {token_address}")
        
        # Check if we already have the token info
        if token_address in self.tokens:
            return self.tokens[token_address]
        
        # Call API to get token info
        response = await self.call_api('get_token', {'address': token_address})
        
        if response and 'token' in response:
            # Store token data
            self.tokens[token_address] = response['token']
            return response['token']
        
        return None
    
    async def get_new_tokens(self, 
                            limit: int = 20, 
                            min_liquidity_usd: float = 0) -> List[Dict[str, Any]]:
        """
        Get recently discovered tokens
        
        Args:
            limit (int): Maximum number of tokens to return
            min_liquidity_usd (float): Minimum liquidity in USD
            
        Returns:
            List[Dict[str, Any]]: List of token data
        """
        logger.debug(f"Getting new tokens (limit: {limit}, min liquidity: ${min_liquidity_usd})")
        
        response = await self.call_api('get_tokens', {
            'limit': limit,
            'sort': 'discovery_time',
            'direction': 'desc',
            'min_liquidity_usd': min_liquidity_usd
        })
        
        if response and 'tokens' in response:
            # Store tokens
            for token in response['tokens']:
                if 'address' in token:
                    self.tokens[token['address']] = token
            
            return response['tokens']
        
        return []
    
    @with_retry(max_retries=3, circuit_breaker_name="pump_portal_market_data", rate_limiter_name="pump_portal_api")
    async def get_market_data(self, token_address: str) -> Optional[Dict[str, Any]]:
        """
        Get market data for a token with enhanced error handling
        
        Args:
            token_address (str): Token address
            
        Returns:
            Optional[Dict[str, Any]]: Market data or None if not found
        """
        logger.debug(f"Getting market data for {token_address}")
        
        # Use timeout to prevent hanging
        try:
            response = await with_timeout(
                self.call_api('get_market_data', {'address': token_address}),
                timeout=10.0,  # 10 second timeout
                fallback_value=None
            )
            
            # Update component health status on success
            self.state_manager.update_component_status('pump_portal_client', 'running')
            
            if response and 'market_data' in response:
                return response['market_data']
            return None
            
        except Exception as e:
            # Log the error and update component status
            logger.error(f"Error getting market data for {token_address}: {str(e)}")
            self.state_manager.update_component_status(
                'pump_portal_client', 
                'error',
                f"API error: {str(e)}"
            )
            return None
