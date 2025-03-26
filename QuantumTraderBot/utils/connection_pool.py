"""
Connection Pool Component
Provides optimized connection pooling for high-throughput RPC requests,
improving performance and reliability when interfacing with the Solana blockchain.
"""

import asyncio
import logging
import time
import random
from typing import Dict, List, Any, Optional, Set, Tuple
import aiohttp

from utils.api_resilience import APICircuitBreaker, RateLimiter

logger = logging.getLogger(__name__)

class ConnectionPoolStats:
    """Statistics for connection pool"""
    
    def __init__(self):
        """Initialize connection pool statistics"""
        # Request statistics
        self.total_requests = 0
        self.failed_requests = 0
        self.successful_requests = 0
        
        # Performance statistics
        self.total_response_time = 0.0
        self.min_response_time = float('inf')
        self.max_response_time = 0.0
        
        # Concurrency statistics
        self.current_active_requests = 0
        self.max_active_requests = 0
        
        # Cache statistics
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Last update time
        self.last_update = time.time()
    
    def record_request(self, 
                     success: bool, 
                     response_time: float,
                     from_cache: bool = False):
        """
        Record request statistics
        
        Args:
            success (bool): Whether request was successful
            response_time (float): Response time in seconds
            from_cache (bool): Whether response was from cache
        """
        self.total_requests += 1
        
        if success:
            self.successful_requests += 1
        else:
            self.failed_requests += 1
        
        if from_cache:
            self.cache_hits += 1
        else:
            self.cache_misses += 1
            
            # Only count response time for non-cached requests
            self.total_response_time += response_time
            self.min_response_time = min(self.min_response_time, response_time)
            self.max_response_time = max(self.max_response_time, response_time)
        
        self.last_update = time.time()
    
    def get_average_response_time(self) -> float:
        """
        Get average response time
        
        Returns:
            float: Average response time in seconds
        """
        non_cached_requests = self.total_requests - self.cache_hits
        if non_cached_requests == 0:
            return 0.0
        return self.total_response_time / non_cached_requests
    
    def get_success_rate(self) -> float:
        """
        Get success rate
        
        Returns:
            float: Success rate (0-1)
        """
        if self.total_requests == 0:
            return 1.0
        return self.successful_requests / self.total_requests
    
    def get_cache_hit_rate(self) -> float:
        """
        Get cache hit rate
        
        Returns:
            float: Cache hit rate (0-1)
        """
        if self.total_requests == 0:
            return 0.0
        return self.cache_hits / self.total_requests
    
    def start_request(self):
        """Record start of request"""
        self.current_active_requests += 1
        self.max_active_requests = max(
            self.max_active_requests, self.current_active_requests)
    
    def end_request(self):
        """Record end of request"""
        self.current_active_requests = max(0, self.current_active_requests - 1)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary
        
        Returns:
            Dict[str, Any]: Statistics
        """
        return {
            'total_requests': self.total_requests,
            'successful_requests': self.successful_requests,
            'failed_requests': self.failed_requests,
            'success_rate': self.get_success_rate(),
            'avg_response_time': self.get_average_response_time(),
            'min_response_time': self.min_response_time if self.min_response_time != float('inf') else 0,
            'max_response_time': self.max_response_time,
            'current_active_requests': self.current_active_requests,
            'max_active_requests': self.max_active_requests,
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'cache_hit_rate': self.get_cache_hit_rate(),
            'last_update': self.last_update
        }


class Connection:
    """Connection to an RPC endpoint"""
    
    def __init__(self, 
                endpoint: str, 
                weight: int = 1,
                timeout: float = 30.0):
        """
        Initialize connection
        
        Args:
            endpoint (str): RPC endpoint URL
            weight (int): Endpoint weight for load balancing
            timeout (float): Request timeout in seconds
        """
        self.endpoint = endpoint
        self.weight = weight
        self.timeout = timeout
        
        # Connection state
        self.session = None
        self.is_active = False
        self.last_used = 0
        self.last_error = None
        self.consecutive_errors = 0
        
        # Performance metrics
        self.average_response_time = 0.0
        self.response_times = []
        self.max_response_times = 100
        
        # Circuit breaker
        self.circuit_breaker = APICircuitBreaker(endpoint)
        
        # Rate limiter
        self.rate_limiter = RateLimiter(endpoint, 20, 1.0)  # 20 requests per second
        
        # Statistics
        self.stats = ConnectionPoolStats()
    
    async def ensure_session(self):
        """Ensure HTTP session exists"""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession(
                connector=aiohttp.TCPConnector(ssl=False),
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            )
            self.is_active = True
    
    async def close(self):
        """Close connection"""
        if self.session and not self.session.closed:
            await self.session.close()
        self.session = None
        self.is_active = False
    
    async def call(self, 
                 method: str, 
                 params: Any = None) -> Tuple[Any, float]:
        """
        Make RPC call
        
        Args:
            method (str): RPC method
            params (Any, optional): Method parameters
            
        Returns:
            Tuple[Any, float]: Response and response time in seconds
        """
        # Check circuit breaker
        if not self.circuit_breaker.allow_request():
            self.last_error = "Circuit breaker open"
            raise Exception(f"Circuit breaker open for {self.endpoint}")
        
        # Check rate limit
        if not self.rate_limiter.allow_call():
            self.last_error = "Rate limit exceeded"
            raise Exception(f"Rate limit exceeded for {self.endpoint}")
        
        # Record rate limited call
        self.rate_limiter.record_call()
        
        # Update last used time
        self.last_used = time.time()
        
        # Ensure session exists
        await self.ensure_session()
        
        # Prepare request
        request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": method
        }
        
        if params is not None:
            request["params"] = params
        
        # Record start of request
        self.stats.start_request()
        start_time = time.time()
        
        try:
            # Make request
            async with self.session.post(
                self.endpoint, 
                json=request, 
                timeout=self.timeout
            ) as response:
                # Calculate response time
                response_time = time.time() - start_time
                
                # Parse response
                result = await response.json()
                
                # Update metrics
                self._update_metrics(True, response_time)
                
                # Record success
                self.circuit_breaker.record_success()
                self.consecutive_errors = 0
                
                return result, response_time
                
        except Exception as e:
            # Calculate response time
            response_time = time.time() - start_time
            
            # Update metrics
            self._update_metrics(False, response_time)
            
            # Record error
            self.last_error = str(e)
            self.consecutive_errors += 1
            self.circuit_breaker.record_failure()
            
            raise
            
        finally:
            # Record end of request
            self.stats.end_request()
    
    def _update_metrics(self, success: bool, response_time: float):
        """
        Update connection metrics
        
        Args:
            success (bool): Whether request was successful
            response_time (float): Response time in seconds
        """
        # Update statistics
        self.stats.record_request(success, response_time)
        
        # Update response times
        self.response_times.append(response_time)
        if len(self.response_times) > self.max_response_times:
            self.response_times = self.response_times[-self.max_response_times:]
        
        # Update average response time
        if self.response_times:
            self.average_response_time = sum(self.response_times) / len(self.response_times)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary
        
        Returns:
            Dict[str, Any]: Connection information
        """
        return {
            'endpoint': self.endpoint,
            'weight': self.weight,
            'is_active': self.is_active,
            'last_used': self.last_used,
            'last_error': self.last_error,
            'consecutive_errors': self.consecutive_errors,
            'average_response_time': self.average_response_time,
            'circuit_breaker_status': 'open' if not self.circuit_breaker.allow_request() else 'closed',
            'stats': self.stats.to_dict()
        }


class ConnectionPool:
    """
    Pool of connections to RPC endpoints
    
    This component manages multiple connections to RPC endpoints,
    providing load balancing, failover, and performance monitoring.
    """
    
    def __init__(self, 
                endpoints: List[Dict[str, Any]], 
                max_connections: int = 8,  # Reduced from 10 to decrease connection resources
                cache_enabled: bool = True,
                cache_ttl: float = 15.0):  # Increased from 5.0 to reduce request frequency
        """
        Initialize connection pool
        
        Args:
            endpoints (List[Dict[str, Any]]): List of endpoint configurations
            max_connections (int): Maximum number of connections
            cache_enabled (bool): Whether to enable response caching
            cache_ttl (float): Cache time-to-live in seconds
        """
        # Connection settings
        self.max_connections = max_connections
        self.connections = {}
        
        # Create connections
        for endpoint_config in endpoints:
            endpoint = endpoint_config['url']
            weight = endpoint_config.get('weight', 1)
            timeout = endpoint_config.get('timeout', 30.0)
            
            self.connections[endpoint] = Connection(
                endpoint=endpoint,
                weight=weight,
                timeout=timeout
            )
        
        # Active connections
        self.active_connections = set()
        
        # Response cache
        self.cache_enabled = cache_enabled
        self.cache_ttl = cache_ttl
        self.cache = {}
        
        # Statistics
        self.stats = ConnectionPoolStats()
        
        # Operational state
        self.is_running = False
        self.maintenance_task = None
        
        logger.info(f"ConnectionPool initialized with {len(endpoints)} endpoints")
    
    async def start(self):
        """Start connection pool"""
        if self.is_running:
            return
        
        self.is_running = True
        
        # Initialize connections
        for endpoint, connection in self.connections.items():
            try:
                # Initialize connection
                await connection.ensure_session()
                
                # Make test call to verify connection is working
                result, response_time = await connection.call("getHealth")
                
                # Check result
                if result.get('result') == 'ok':
                    # Add to active connections
                    self.active_connections.add(endpoint)
                    logger.info(f"Connection to {endpoint} is healthy (response time: {response_time:.3f}s)")
                else:
                    logger.warning(f"Connection to {endpoint} returned unexpected response: {result}")
            except Exception as e:
                logger.error(f"Failed to establish connection to {endpoint}: {str(e)}")
        
        # Start maintenance task
        self.maintenance_task = asyncio.create_task(self._maintenance_loop())
        
        # Log status
        active_count = len(self.active_connections)
        total_count = len(self.connections)
        if active_count == 0:
            logger.error(f"No active connections available (0/{total_count})")
            
            # Emergency recovery attempt: try all connections with more retries
            await self._emergency_connection_recovery()
            
            # Check if recovery was successful
            active_count = len(self.active_connections)
            if active_count > 0:
                logger.info(f"Emergency recovery successful: {active_count}/{total_count} connections active")
            else:
                logger.critical("Emergency recovery failed: No connections available")
        else:
            logger.info(f"Connection pool started with {active_count}/{total_count} active endpoints")
            
    async def _emergency_connection_recovery(self):
        """
        Emergency recovery for connection pool when no connections are available
        Uses extended timeouts and more aggressive retry strategy
        """
        logger.warning("Attempting emergency connection recovery")
        
        # Try all connections with extended timeout
        for endpoint, connection in self.connections.items():
            # Skip active connections
            if endpoint in self.active_connections:
                continue
                
            # Save original timeout
            original_timeout = connection.timeout
            
            try:
                # Set extended timeout for recovery
                connection.timeout = 45.0  # Extended timeout for emergency recovery
                
                # Close and recreate session if it exists
                if connection.session:
                    await connection.session.close()
                    connection.session = None
                
                # Ensure new session
                await connection.ensure_session()
                
                # Retry health check with multiple attempts
                for attempt in range(3):
                    try:
                        logger.debug(f"Emergency recovery attempt {attempt+1}/3 for {endpoint}")
                        
                        # Make test call with extended timeout
                        result, _ = await connection.call("getHealth")
                        
                        # Check result
                        if result.get('result') == 'ok':
                            # Add to active connections
                            self.active_connections.add(endpoint)
                            logger.info(f"Emergency recovery successful for {endpoint}")
                            break
                    except Exception as e:
                        logger.debug(f"Recovery attempt {attempt+1} failed for {endpoint}: {str(e)}")
                        await asyncio.sleep(2 * (attempt + 1))  # Increasing delay between attempts
            
            except Exception as e:
                logger.warning(f"Emergency recovery failed for {endpoint}: {str(e)}")
            
            finally:
                # Restore original timeout
                connection.timeout = original_timeout
    
    async def stop(self):
        """Stop connection pool"""
        if not self.is_running:
            return
        
        self.is_running = False
        
        # Stop maintenance task
        if self.maintenance_task:
            self.maintenance_task.cancel()
            try:
                await self.maintenance_task
            except asyncio.CancelledError:
                pass
            self.maintenance_task = None
        
        # Close all connections
        for endpoint, connection in self.connections.items():
            try:
                await connection.close()
            except Exception as e:
                logger.error(f"Error closing connection to {endpoint}: {str(e)}")
        
        self.active_connections = set()
        
        logger.info("ConnectionPool stopped")
    
    async def _maintenance_loop(self):
        """Maintenance loop for connection pool"""
        while self.is_running:
            try:
                # Get memory usage to optimize maintenance operations
                memory_usage_high = False
                try:
                    import psutil
                    process = psutil.Process()
                    memory_info = process.memory_info()
                    virtual_memory = psutil.virtual_memory()
                    memory_percent = (memory_info.rss / virtual_memory.total) * 100
                    if memory_percent > 80:
                        memory_usage_high = True
                        logger.warning(f"High memory usage during connection pool maintenance: {memory_percent:.1f}%")
                except Exception:
                    pass  # Memory check is optional

                # Check and repair connections (only if memory usage is not critical)
                if not memory_usage_high:
                    await self._check_connections()
                
                # Clean cache (always do this to free memory)
                cache_items = await self._clean_cache()
                
                # If memory is high, perform more aggressive cleanup
                if memory_usage_high and len(self.cache) > 10:
                    # Reduce cache size by half when memory pressure is high
                    keys_to_remove = list(self.cache.keys())[:len(self.cache)//2]
                    for key in keys_to_remove:
                        del self.cache[key]
                    logger.info(f"Memory pressure high: Removed {len(keys_to_remove)} additional cache items")
                
                # Wait before next check (longer interval if memory is high)
                wait_time = 60.0 if memory_usage_high else 30.0
                await asyncio.sleep(wait_time)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in maintenance loop: {str(e)}")
                await asyncio.sleep(30.0)
    
    async def _check_connections(self):
        """Check and repair connections"""
        for endpoint, connection in self.connections.items():
            # Skip active connections that are working
            if endpoint in self.active_connections and connection.consecutive_errors == 0:
                continue
            
            try:
                # Test connection
                await connection.ensure_session()
                
                # Make test call
                result, _ = await connection.call("getHealth")
                
                # Check result
                if result.get('result') == 'ok':
                    # Add to active connections
                    self.active_connections.add(endpoint)
                    logger.info(f"Connection to {endpoint} is healthy")
                
            except Exception as e:
                # Remove from active connections
                self.active_connections.discard(endpoint)
                logger.warning(f"Connection to {endpoint} is unhealthy: {str(e)}")
    
    async def _clean_cache(self):
        """Clean expired cache entries
        
        Returns:
            int: Number of removed cache items
        """
        if not self.cache_enabled:
            return 0
        
        current_time = time.time()
        expired_keys = []
        
        # Find expired entries
        for key, (timestamp, _) in self.cache.items():
            if current_time - timestamp > self.cache_ttl:
                expired_keys.append(key)
        
        # Remove expired entries
        for key in expired_keys:
            del self.cache[key]
        
        # If cache size is too large, remove oldest entries
        if len(self.cache) > 500:  # Set a reasonable upper limit for cache size
            # Sort by timestamp (oldest first)
            cache_items = sorted(self.cache.items(), key=lambda x: x[1][0])
            # Remove oldest 20% of entries
            oldest_count = len(cache_items) // 5
            oldest_keys = [item[0] for item in cache_items[:oldest_count]]
            for key in oldest_keys:
                del self.cache[key]
            expired_keys.extend(oldest_keys)
            logger.info(f"Cache size limit exceeded: Removed {len(oldest_keys)} oldest cache entries")
        
        if expired_keys:
            logger.debug(f"Cleaned {len(expired_keys)} cache entries")
            
        return len(expired_keys)
    
    async def call(self, 
                 method: str, 
                 params: Any = None,
                 use_cache: bool = True,
                 max_retries: int = 2,
                 retry_delay: float = 1.0) -> Any:
        """
        Make RPC call using connection pool with improved reliability
        
        Args:
            method (str): RPC method
            params (Any, optional): Method parameters
            use_cache (bool): Whether to use cache
            max_retries (int): Maximum number of retry attempts (per connection)
            retry_delay (float): Base delay between retries in seconds
            
        Returns:
            Any: Response
        """
        # Check cache if enabled and requested
        cache_key = None
        if self.cache_enabled and use_cache:
            cache_key = (method, str(params))
            cached_result = self._get_from_cache(cache_key)
            if cached_result is not None:
                # Update statistics
                self.stats.record_request(True, 0.0, from_cache=True)
                return cached_result
        
        # Record start of request
        self.stats.start_request()
        start_time = time.time()
        
        # Get weighted active connections
        weighted_connections = self._get_weighted_connections()
        
        # If no active connections, try to recover and get connections again
        if not weighted_connections:
            logger.warning("No active connections available, attempting recovery")
            await self._check_connections()
            weighted_connections = self._get_weighted_connections()
            
            # If still no connections, try all connections as a last resort
            if not weighted_connections:
                logger.warning("Recovery failed, trying all available connections")
                for endpoint, connection in self.connections.items():
                    weighted_connections.append(connection)
        
        # Try connections with advanced retry logic
        errors = []
        
        for connection in weighted_connections:
            # Apply circuit breaker pattern
            if not connection.circuit_breaker.allow_request():
                errors.append(f"Circuit breaker open for {connection.endpoint}")
                continue
                
            # Initialize retry counter for this connection
            retry_count = 0
            
            while retry_count <= max_retries:
                try:
                    # Make call with timeout protection
                    result, response_time = await connection.call(method, params)
                    
                    # Check for error in response
                    if 'error' in result:
                        error_msg = result['error'].get('message', str(result['error']))
                        error_code = result['error'].get('code', 0)
                        
                        # Some errors should be retried, others not
                        retryable_error = self._is_retryable_error(error_code, error_msg)
                        
                        if retryable_error and retry_count < max_retries:
                            # Retryable error, will retry
                            retry_count += 1
                            wait_time = retry_delay * (2 ** retry_count) * (0.5 + random.random())
                            logger.debug(f"Retryable error from {connection.endpoint}, retrying in {wait_time:.2f}s: {error_msg}")
                            await asyncio.sleep(wait_time)
                            continue
                        else:
                            # Non-retryable error or max retries reached
                            errors.append(f"API error from {connection.endpoint}: {error_msg}")
                            connection.circuit_breaker.record_failure()
                            break
                    
                    # Successful call
                    connection.circuit_breaker.record_success()
                    
                    # Update statistics
                    self.stats.record_request(True, response_time)
                    
                    # Store in cache if enabled
                    if self.cache_enabled and use_cache and cache_key is not None:
                        self._store_in_cache(cache_key, result)
                    
                    # Return successful result
                    return result
                    
                except asyncio.TimeoutError:
                    # Timeout should be retried with increasing delays
                    if retry_count < max_retries:
                        retry_count += 1
                        wait_time = retry_delay * (2 ** retry_count) * (0.5 + random.random())
                        logger.debug(f"Timeout from {connection.endpoint}, retrying in {wait_time:.2f}s")
                        await asyncio.sleep(wait_time)
                    else:
                        errors.append(f"Timeout from {connection.endpoint} after {retry_count} retries")
                        connection.circuit_breaker.record_failure()
                        break
                
                except Exception as e:
                    # For other exceptions, check if they're retryable
                    if self._is_retryable_exception(e) and retry_count < max_retries:
                        retry_count += 1
                        wait_time = retry_delay * (2 ** retry_count) * (0.5 + random.random())
                        logger.debug(f"Retryable exception from {connection.endpoint}, retrying in {wait_time:.2f}s: {str(e)}")
                        await asyncio.sleep(wait_time)
                    else:
                        # Record error and mark connection as failed
                        errors.append(f"Error from {connection.endpoint}: {str(e)}")
                        connection.circuit_breaker.record_failure()
                        break
        
        # All connections failed
        # Update statistics
        response_time = time.time() - start_time
        self.stats.record_request(False, response_time)
        
        # If we have a fallback response for this method, use it
        fallback_response = self._get_fallback_response(method, params)
        if fallback_response is not None:
            logger.warning(f"Using fallback response for {method} after all connections failed")
            return fallback_response
        
        # Raise exception with all errors
        error_msg = f"All connections failed: {'; '.join(errors)}"
        logger.error(error_msg)
        raise Exception(error_msg)
    
    def _get_weighted_connections(self) -> List[Connection]:
        """
        Get weighted connections list for load balancing
        
        Returns:
            List[Connection]: Weighted list of active connections
        """
        weighted_connections = []
        
        for endpoint in self.active_connections:
            connection = self.connections[endpoint]
            for _ in range(connection.weight):
                weighted_connections.append(connection)
        
        # Shuffle connections for load balancing
        random.shuffle(weighted_connections)
        
        # Sort by response time and error rate (best performing first)
        weighted_connections.sort(key=lambda c: (c.consecutive_errors, c.average_response_time))
        
        return weighted_connections
    
    def _is_retryable_error(self, error_code: int, error_msg: str) -> bool:
        """
        Determine if an API error should be retried
        
        Args:
            error_code (int): Error code
            error_msg (str): Error message
            
        Returns:
            bool: True if error should be retried
        """
        # List of error messages that indicate rate limiting
        rate_limit_messages = [
            "rate limit", "too many requests", "request rate exceeded",
            "429", "slow down", "too frequent", "rpc request rate exceeded"
        ]
        
        # List of error messages that indicate temporary issues
        temporary_issue_messages = [
            "timeout", "timed out", "service unavailable", "try again",
            "temporarily", "overloaded", "busy", "maintenance", "retry",
            "under load", "capacity", "congestion"
        ]
        
        # Solana-specific error messages
        solana_specific_messages = [
            "slot skipped", "block not available", "confirmed block not available", 
            "not found", "block cleaned up", "leadership skipped slot",
            "send transaction preflight failure", "failed to get recent blockhash",
            "instruction error", "cluster processing", "transaction version downgrade"
        ]
        
        # Convert error message to lowercase for case-insensitive comparison
        lower_msg = error_msg.lower()
        
        # Check for rate limiting errors
        if any(phrase in lower_msg for phrase in rate_limit_messages):
            logger.debug(f"Rate limiting error detected: {error_msg}")
            return True
            
        # Check for temporary issues
        if any(phrase in lower_msg for phrase in temporary_issue_messages):
            logger.debug(f"Temporary issue detected: {error_msg}")
            return True
            
        # Check for Solana-specific issues
        if any(phrase in lower_msg for phrase in solana_specific_messages):
            logger.debug(f"Solana-specific error detected: {error_msg}")
            return True
            
        # Known retryable error codes (expanded for Solana)
        # Standard HTTP error codes
        http_retryable_codes = [429, 500, 502, 503, 504, 507, 509, 522, 524]
        
        # Solana-specific error codes
        solana_retryable_codes = [
            -32005,  # Node is behind, still syncing
            -32603,  # Internal error
            -32002,  # Request being processed
            -32004,  # Method not found
            -32009,  # Slot skipped
            -32010,  # Block not available
            -32007,  # Slot not rooted
            -32027,  # Block cleaned up
            -32015,  # Runtime transaction error
        ]
        
        if error_code in http_retryable_codes or error_code in solana_retryable_codes:
            logger.debug(f"Retryable error code detected: {error_code}")
            return True
            
        # Default: don't retry
        return False
    
    def _is_retryable_exception(self, exception: Exception) -> bool:
        """
        Determine if an exception should be retried
        
        Args:
            exception (Exception): The exception
            
        Returns:
            bool: True if exception should be retried
        """
        # List of exception types or messages that should be retried
        retryable_types = [
            asyncio.TimeoutError,  # Already handled separately, but included for completeness
            ConnectionError,
            ConnectionResetError,
            ConnectionRefusedError,
            TimeoutError,
            aiohttp.ClientError,
            aiohttp.ClientConnectionError,
            aiohttp.ClientOSError,
            aiohttp.ServerConnectionError,
            aiohttp.ServerDisconnectedError,
            aiohttp.ServerTimeoutError,
            aiohttp.ClientPayloadError,
            aiohttp.ClientResponseError
        ]
        
        # Check exception type
        for exception_type in retryable_types:
            try:
                if isinstance(exception, exception_type):
                    logger.debug(f"Retryable exception type: {type(exception).__name__}")
                    return True
            except (NameError, TypeError):
                # Skip if exception type is not available
                pass
                
        # Check exception message for specific patterns
        error_msg = str(exception).lower()
        
        # Basic networking issues
        networking_patterns = [
            "timeout", "timed out", "reset by peer", "connection", 
            "network", "refused", "temporarily", "again", "retry",
            "broken pipe", "eof", "closed", "reset"
        ]
        
        # HTTP/Response specific issues
        http_patterns = [
            "status code", "http", "response", "payload", "content length",
            "content type", "header", "502", "503", "504", "429"
        ]
        
        # JSON parsing issues
        json_patterns = [
            "json", "decode", "parse", "unexpected", "invalid",
            "malformed", "syntax", "value error"
        ]
        
        # Check all pattern categories
        if any(pattern in error_msg for pattern in networking_patterns):
            logger.debug(f"Retryable networking exception: {error_msg}")
            return True
            
        if any(pattern in error_msg for pattern in http_patterns):
            logger.debug(f"Retryable HTTP exception: {error_msg}")
            return True
            
        if any(pattern in error_msg for pattern in json_patterns):
            logger.debug(f"Retryable JSON exception: {error_msg}")
            return True
            
        # Default: don't retry
        return False
        
    def _get_fallback_response(self, method: str, params: Any) -> Optional[Any]:
        """
        Get fallback response for a method when all connections fail
        
        Args:
            method (str): RPC method
            params (Any): Method parameters
            
        Returns:
            Optional[Any]: Fallback response or None
        """
        # For health checks, return a minimal healthy response
        if method == "getHealth":
            return {"jsonrpc": "2.0", "result": "ok", "id": 1, "_fallback": True}
            
        # For getLatestBlockhash, return cached value if available or None
        if method == "getLatestBlockhash":
            cached_key = (method, str(params))
            cached_blockhash = self._get_from_cache(cached_key)
            if cached_blockhash is not None:
                logger.warning("Using cached blockhash as fallback")
                return cached_blockhash
            return None
        
        # For getBalance, return last known balance if available
        if method == "getBalance":
            cached_key = (method, str(params))
            cached_balance = self._get_from_cache(cached_key)
            if cached_balance is not None:
                logger.warning("Using cached balance as fallback")
                return cached_balance
            return None
            
        # For getTokenAccountBalance, return last known balance if available
        if method == "getTokenAccountBalance":
            cached_key = (method, str(params))
            cached_token_balance = self._get_from_cache(cached_key)
            if cached_token_balance is not None:
                logger.warning("Using cached token balance as fallback")
                return cached_token_balance
            return None
            
        # For getTokenSupply, return last known supply if available
        if method == "getTokenSupply":
            cached_key = (method, str(params))
            cached_supply = self._get_from_cache(cached_key)
            if cached_supply is not None:
                logger.warning("Using cached token supply as fallback")
                return cached_supply
            return None
            
        # For getProgramAccounts with token program, can sometimes use empty array
        if method == "getProgramAccounts" and params and "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA" in str(params):
            logger.warning("Using empty accounts array as fallback for getProgramAccounts")
            return {"jsonrpc": "2.0", "result": [], "id": 1, "_fallback": True}
                
        # For most methods, there's no reasonable fallback
        logger.debug(f"No fallback available for method: {method}")
        return None
    
    def _get_from_cache(self, key: tuple) -> Optional[Any]:
        """
        Get response from cache
        
        Args:
            key (tuple): Cache key
            
        Returns:
            Optional[Any]: Cached response or None if not found
        """
        if not self.cache_enabled:
            return None
        
        cache_entry = self.cache.get(key)
        if cache_entry is None:
            return None
        
        timestamp, response = cache_entry
        
        # Check if expired
        if time.time() - timestamp > self.cache_ttl:
            del self.cache[key]
            return None
        
        # Update statistics
        self.stats.cache_hits += 1
        
        return response
    
    def _store_in_cache(self, key: tuple, response: Any):
        """
        Store response in cache
        
        Args:
            key (tuple): Cache key
            response (Any): Response to cache
        """
        if not self.cache_enabled:
            return
        
        self.cache[key] = (time.time(), response)
        
        # Update statistics
        self.stats.cache_misses += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get connection pool statistics
        
        Returns:
            Dict[str, Any]: Statistics
        """
        # Get overall statistics
        stats = self.stats.to_dict()
        
        # Add connection-specific statistics
        connection_stats = {}
        for endpoint, connection in self.connections.items():
            connection_stats[endpoint] = connection.to_dict()
        
        stats['connections'] = connection_stats
        stats['active_connections'] = len(self.active_connections)
        stats['total_connections'] = len(self.connections)
        stats['cache_size'] = len(self.cache) if self.cache_enabled else 0
        
        return stats
    
    def get_connection_health(self) -> Dict[str, str]:
        """
        Get health status of all connections
        
        Returns:
            Dict[str, str]: Health status by endpoint
        """
        health = {}
        
        for endpoint, connection in self.connections.items():
            if endpoint in self.active_connections:
                if connection.consecutive_errors > 0:
                    health[endpoint] = "degraded"
                else:
                    health[endpoint] = "healthy"
            else:
                health[endpoint] = "unhealthy"
        
        return health