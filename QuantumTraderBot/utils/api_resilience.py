"""
API Resilience Utility Module
Provides utilities for making API calls more resilient with features like:
- Retry with exponential backoff
- Circuit breaker pattern
- Fallback mechanisms
- Rate limiting protection
"""

import asyncio
import logging
import time
import random
from typing import Callable, Dict, Any, Optional, TypeVar, List
from functools import wraps

logger = logging.getLogger(__name__)

# Type for function return value
T = TypeVar('T')

class APICircuitBreaker:
    """
    Implementation of the Circuit Breaker pattern for API calls
    Prevents cascading failures by failing fast when a service is unavailable
    """
    
    def __init__(self, name: str):
        """Initialize the circuit breaker"""
        self.name = name
        self.state = "closed"  # closed (normal), open (failing), half-open (testing)
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0
        self.failure_threshold = 5
        self.reset_timeout = 60  # seconds
        self.success_threshold = 2
        
    def record_success(self):
        """Record a successful call"""
        if self.state == "half-open":
            self.success_count += 1
            if self.success_count >= self.success_threshold:
                logger.info(f"Circuit breaker {self.name} closing after {self.success_count} successful calls")
                self.state = "closed"
                self.failure_count = 0
                self.success_count = 0
        
    def record_failure(self):
        """Record a failed call"""
        self.last_failure_time = time.time()
        
        if self.state == "closed":
            self.failure_count += 1
            if self.failure_count >= self.failure_threshold:
                logger.warning(f"Circuit breaker {self.name} opening after {self.failure_count} failures")
                self.state = "open"
                
        elif self.state == "half-open":
            logger.warning(f"Circuit breaker {self.name} re-opening after failure in half-open state")
            self.state = "open"
            self.success_count = 0
    
    def allow_request(self) -> bool:
        """Determine if a request should be allowed"""
        if self.state == "closed":
            return True
            
        elif self.state == "open":
            # Check if we should move to half-open
            if time.time() - self.last_failure_time >= self.reset_timeout:
                logger.info(f"Circuit breaker {self.name} moving to half-open state")
                self.state = "half-open"
                self.success_count = 0
                return True
            return False
            
        elif self.state == "half-open":
            # In half-open state, only allow limited requests to test the service
            return True
            
        return False

# Global registry of circuit breakers
_circuit_breakers: Dict[str, APICircuitBreaker] = {}

def get_circuit_breaker(name: str) -> APICircuitBreaker:
    """Get or create a circuit breaker by name"""
    if name not in _circuit_breakers:
        _circuit_breakers[name] = APICircuitBreaker(name)
    return _circuit_breakers[name]

def reset_all_circuit_breakers():
    """Reset all circuit breakers"""
    for breaker in _circuit_breakers.values():
        breaker.state = "closed"
        breaker.failure_count = 0
        breaker.success_count = 0

class RateLimiter:
    """
    Rate limiter to prevent overwhelming external APIs
    """
    
    def __init__(self, name: str, max_calls: int, period: float):
        """
        Initialize the rate limiter
        
        Args:
            name (str): Name of the rate limiter
            max_calls (int): Maximum number of calls allowed
            period (float): Time period in seconds
        """
        self.name = name
        self.max_calls = max_calls
        self.period = period
        self.calls: List[float] = []
        
    def _cleanup_old_calls(self):
        """Remove calls outside the current period"""
        now = time.time()
        self.calls = [t for t in self.calls if now - t <= self.period]
    
    def allow_call(self) -> bool:
        """Check if a call is allowed under the rate limit"""
        self._cleanup_old_calls()
        return len(self.calls) < self.max_calls
    
    def record_call(self):
        """Record a call"""
        self.calls.append(time.time())

# Global registry of rate limiters
_rate_limiters: Dict[str, RateLimiter] = {}

def get_rate_limiter(name: str, max_calls: int = 10, period: float = 1.0) -> RateLimiter:
    """Get or create a rate limiter by name"""
    if name not in _rate_limiters:
        _rate_limiters[name] = RateLimiter(name, max_calls, period)
    return _rate_limiters[name]

def with_retry(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    backoff_factor: float = 2.0,
    circuit_breaker_name: Optional[str] = None,
    rate_limiter_name: Optional[str] = None,
    rate_limit_max_calls: int = 10,
    rate_limit_period: float = 1.0
):
    """
    Decorator for adding retry logic to async functions
    
    Args:
        max_retries (int): Maximum number of retries
        base_delay (float): Initial delay between retries (seconds)
        max_delay (float): Maximum delay between retries (seconds)
        backoff_factor (float): Multiplier for delay after each retry
        circuit_breaker_name (str, optional): Name of circuit breaker to use
        rate_limiter_name (str, optional): Name of rate limiter to use
        rate_limit_max_calls (int): Maximum calls for rate limiter
        rate_limit_period (float): Period for rate limiter (seconds)
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Get or create circuit breaker if name provided
            circuit_breaker = None
            if circuit_breaker_name:
                circuit_breaker = get_circuit_breaker(circuit_breaker_name)
                
            # Get or create rate limiter if name provided
            rate_limiter = None
            if rate_limiter_name:
                rate_limiter = get_rate_limiter(
                    rate_limiter_name, 
                    rate_limit_max_calls, 
                    rate_limit_period
                )
                
            # Check circuit breaker
            if circuit_breaker and not circuit_breaker.allow_request():
                logger.warning(f"Circuit breaker {circuit_breaker_name} is open, failing fast")
                raise Exception(f"Circuit breaker {circuit_breaker_name} is open")
                
            # Enforce rate limit if needed
            if rate_limiter:
                while not rate_limiter.allow_call():
                    delay = 0.1 * random.random() + 0.05  # Small random delay
                    logger.debug(f"Rate limit reached for {rate_limiter_name}, waiting {delay:.2f}s")
                    await asyncio.sleep(delay)
                rate_limiter.record_call()
                    
            # Attempt the call with retries
            retries = 0
            last_exception = None
            
            while retries <= max_retries:
                try:
                    result = await func(*args, **kwargs)
                    
                    # Record success if using circuit breaker
                    if circuit_breaker:
                        circuit_breaker.record_success()
                        
                    return result
                    
                except Exception as e:
                    last_exception = e
                    retries += 1
                    
                    # Record failure if using circuit breaker
                    if circuit_breaker:
                        circuit_breaker.record_failure()
                    
                    if retries > max_retries:
                        logger.error(f"Max retries ({max_retries}) exceeded: {str(e)}")
                        break
                        
                    # Calculate delay with exponential backoff and jitter
                    delay = min(max_delay, base_delay * (backoff_factor ** (retries - 1)))
                    delay = delay * (0.5 + random.random())  # Add jitter
                    
                    logger.warning(f"Retry {retries}/{max_retries} after {delay:.2f}s for: {str(e)}")
                    await asyncio.sleep(delay)
            
            if last_exception:
                raise last_exception
            else:
                logger.error("All retries failed but no exception was captured")
                raise Exception("All retries failed")
        
        return wrapper
    
    return decorator

async def with_timeout(coro, timeout: float, fallback_value: Any = None):
    """
    Execute a coroutine with a timeout
    
    Args:
        coro: Coroutine to execute
        timeout (float): Timeout in seconds
        fallback_value (Any, optional): Value to return if timeout occurs
        
    Returns:
        Result of coroutine or fallback value if timeout
    """
    try:
        return await asyncio.wait_for(coro, timeout=timeout)
    except asyncio.TimeoutError:
        logger.warning(f"Operation timed out after {timeout}s, using fallback")
        return fallback_value