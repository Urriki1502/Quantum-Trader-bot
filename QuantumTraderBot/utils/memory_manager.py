"""
MemoryManager Component
Responsible for optimizing memory usage, preventing memory leaks,
and ensuring efficient memory utilization.
"""

import asyncio
import logging
import time
import gc
import sys
import os
import psutil
import resource
import random
from typing import Dict, List, Any, Optional, Tuple, Set

logger = logging.getLogger(__name__)

class MemoryManager:
    """
    MemoryManager handles:
    - Periodic garbage collection
    - Memory usage monitoring
    - Cache cleanup and management
    - Memory leak detection
    """
    
    def __init__(self):
        """Initialize the MemoryManager"""
        # Settings
        self.gc_interval = 1800  # Garbage collection interval (seconds) - reduced from 3600
        self.max_memory_percentage = 80  # Maximum memory usage percentage
        self.cleanup_threshold = 65  # Memory usage percentage that triggers cleanup - reduced from 70
        self.min_cleanup_interval = 180  # Minimum time between cleanups (seconds) - reduced from 300
        
        # Memory pools
        self.object_pools = {}
        self.cache = {}
        self.cache_ttl = {}  # Time-to-live for cached items
        self.default_cache_ttl = 1800  # Default TTL (seconds)
        
        # State
        self.last_gc_time = 0
        self.last_cleanup_time = 0
        self.process = psutil.Process(os.getpid())
        
        # Object reference tracking for leak detection
        self.object_counts = {}
        self.last_tracking_time = 0
        self.tracking_interval = 3600  # Object tracking interval (seconds)
        
        # Enable garbage collection
        gc.enable()
        
        logger.info("MemoryManager initialized")
    
    async def monitor_memory(self) -> Dict[str, Any]:
        """
        Get current memory usage statistics
        
        Returns:
            Dict[str, Any]: Memory usage statistics
        """
        try:
            # Get memory usage
            memory_info = self.process.memory_info()
            virtual_memory = psutil.virtual_memory()
            
            # Calculate usage percentages
            total_memory = virtual_memory.total
            used_memory = memory_info.rss
            memory_percent = (used_memory / total_memory) * 100
            
            stats = {
                'rss': used_memory,
                'rss_mb': used_memory / (1024 * 1024),
                'virtual': memory_info.vms,
                'virtual_mb': memory_info.vms / (1024 * 1024),
                'percent': memory_percent,
                'total_system_mb': total_memory / (1024 * 1024),
                'available_system_mb': virtual_memory.available / (1024 * 1024),
                'allocated_objects': len(gc.get_objects()),
                'cached_items': len(self.cache)
            }
            
            logger.debug(f"Memory usage: {stats['rss_mb']:.1f} MB ({stats['percent']:.1f}%)")
            return stats
            
        except Exception as e:
            logger.error(f"Error monitoring memory: {str(e)}")
            return {
                'error': str(e),
                'rss_mb': 0,
                'percent': 0
            }
    
    async def cleanup(self) -> Dict[str, Any]:
        """
        Perform memory cleanup
        
        Returns:
            Dict[str, Any]: Cleanup results
        """
        current_time = time.time()
        
        # Get current memory usage to determine if we need immediate cleanup
        memory_stats = await self.monitor_memory()
        memory_percent = memory_stats['percent']
        
        # Check if cleanup is needed based on time or memory threshold
        time_based_cleanup = current_time - self.last_cleanup_time >= self.min_cleanup_interval
        memory_based_cleanup = memory_percent >= self.cleanup_threshold
        
        if not (time_based_cleanup or memory_based_cleanup):
            return {
                'cleaned': False,
                'reason': 'Memory usage below threshold and too soon since last cleanup',
                'next_cleanup': self.last_cleanup_time + self.min_cleanup_interval,
                'current_memory_percent': memory_percent
            }
        
        logger.info(f"Performing memory cleanup (memory usage: {memory_percent:.1f}%)")
        self.last_cleanup_time = current_time
        
        # Get memory usage before cleanup
        before_stats = await self.monitor_memory()
        before_memory = before_stats['rss']
        
        # Cleanup operations
        results = {
            'cache_items_removed': 0,
            'pool_items_removed': 0,
            'gc_collected': 0
        }
        
        # 1. Clean expired cache items
        removed_keys = []
        for key, ttl in self.cache_ttl.items():
            if ttl < current_time:
                removed_keys.append(key)
        
        for key in removed_keys:
            if key in self.cache:
                del self.cache[key]
            del self.cache_ttl[key]
        
        results['cache_items_removed'] = len(removed_keys)
        
        # 2. Run garbage collection
        gc_count = gc.collect(2)  # Full collection
        results['gc_collected'] = gc_count
        
        # Get memory usage after cleanup
        after_stats = await self.monitor_memory()
        after_memory = after_stats['rss']
        
        # Calculate memory saved
        memory_saved = before_memory - after_memory
        memory_saved_mb = memory_saved / (1024 * 1024)
        
        logger.info(f"Memory cleanup complete: {results['cache_items_removed']} cache items removed, "
                   f"{results['gc_collected']} objects collected, {memory_saved_mb:.1f} MB saved")
        
        return {
            'cleaned': True,
            'memory_before_mb': before_memory / (1024 * 1024),
            'memory_after_mb': after_memory / (1024 * 1024),
            'memory_saved_mb': memory_saved_mb,
            'memory_saved_percent': (memory_saved / before_memory * 100) if before_memory > 0 else 0,
            'results': results
        }
    
    async def perform_gc(self, force: bool = False) -> Dict[str, Any]:
        """
        Perform garbage collection
        
        Args:
            force (bool): Force garbage collection even if interval hasn't elapsed
            
        Returns:
            Dict[str, Any]: Garbage collection results
        """
        current_time = time.time()
        
        # Check if GC is needed
        if not force and current_time - self.last_gc_time < self.gc_interval:
            return {
                'performed': False,
                'reason': 'Too soon since last GC',
                'next_gc': self.last_gc_time + self.gc_interval
            }
        
        logger.info("Performing garbage collection")
        self.last_gc_time = current_time
        
        # Get memory usage before GC
        before_stats = await self.monitor_memory()
        before_memory = before_stats['rss']
        
        # Run garbage collection
        gc_counts = [gc.collect(i) for i in range(3)]
        
        # Get memory usage after GC
        after_stats = await self.monitor_memory()
        after_memory = after_stats['rss']
        
        # Calculate memory saved
        memory_saved = before_memory - after_memory
        memory_saved_mb = memory_saved / (1024 * 1024)
        
        logger.info(f"Garbage collection complete: {sum(gc_counts)} objects collected, {memory_saved_mb:.1f} MB saved")
        
        return {
            'performed': True,
            'collected': sum(gc_counts),
            'memory_before_mb': before_memory / (1024 * 1024),
            'memory_after_mb': after_memory / (1024 * 1024),
            'memory_saved_mb': memory_saved_mb,
            'memory_saved_percent': (memory_saved / before_memory * 100) if before_memory > 0 else 0,
            'gc_counts': gc_counts
        }
    
    async def track_object_counts(self) -> Dict[str, Any]:
        """
        Track object counts for memory leak detection
        
        Returns:
            Dict[str, Any]: Object count changes
        """
        current_time = time.time()
        
        # Check if tracking is needed
        if current_time - self.last_tracking_time < self.tracking_interval:
            return {
                'tracked': False,
                'reason': 'Too soon since last tracking',
                'next_tracking': self.last_tracking_time + self.tracking_interval
            }
        
        memory_stats = await self.monitor_memory()
        memory_percent = memory_stats['percent']
        
        # Force more frequent tracking if memory usage is high
        if memory_percent >= self.cleanup_threshold and current_time - self.last_tracking_time < self.tracking_interval / 2:
            logger.warning(f"Memory usage high ({memory_percent:.1f}%), performing expedited object tracking")
        else:
            logger.debug("Tracking object counts for memory leak detection")
            
        self.last_tracking_time = current_time
        
        # Get current object counts by type, but sample to reduce overhead
        sample_size = 1000  # Sample a smaller set of objects to reduce overhead
        current_counts = {}
        sampled_objects = random.sample(gc.get_objects(), min(sample_size, len(gc.get_objects())))
        
        for obj in sampled_objects:
            obj_type = type(obj).__name__
            if obj_type not in current_counts:
                current_counts[obj_type] = 0
            current_counts[obj_type] += 1
        
        # Identify the most common object types for closer monitoring
        most_common = sorted(current_counts.items(), key=lambda x: x[1], reverse=True)[:20]
        
        # Calculate changes for the most common types
        changes = {}
        for obj_type, count in most_common:
            previous = self.object_counts.get(obj_type, 0)
            # Normalize count based on sampling
            estimated_total = count * (len(gc.get_objects()) / sample_size)
            change = estimated_total - previous
            if abs(change) > 100:  # Only track significant changes
                changes[obj_type] = change
        
        # Update stored counts for most common types
        for obj_type, count in most_common:
            estimated_total = count * (len(gc.get_objects()) / sample_size)
            self.object_counts[obj_type] = estimated_total
        
        # Check for potential memory leaks (large increase in objects)
        potential_leaks = {
            obj_type: change for obj_type, change in changes.items()
            if change > 1000  # Threshold for potential leak
        }
        
        if potential_leaks:
            leak_info = ", ".join([f"{obj_type}:{change}" for obj_type, change in potential_leaks.items()])
            logger.warning(f"Potential memory leak detected: {leak_info}")
            
            # Force garbage collection if potential leaks detected
            if memory_percent > 80:  # High memory situation
                logger.warning("High memory with potential leaks detected, forcing garbage collection")
                await self.perform_gc(force=True)
        
        return {
            'tracked': True,
            'total_objects': len(gc.get_objects()),
            'sample_size': sample_size,
            'most_common_types': dict(most_common),
            'changes': changes,
            'potential_leaks': potential_leaks,
            'memory_usage_percent': memory_percent
        }
    
    def get_from_cache(self, key: str) -> Optional[Any]:
        """
        Get item from cache
        
        Args:
            key (str): Cache key
            
        Returns:
            Optional[Any]: Cached item or None if not found or expired
        """
        current_time = time.time()
        
        # Check if key exists and is not expired
        if key in self.cache and self.cache_ttl.get(key, 0) > current_time:
            return self.cache[key]
        
        # Remove expired item if it exists
        if key in self.cache and self.cache_ttl.get(key, 0) <= current_time:
            del self.cache[key]
            if key in self.cache_ttl:
                del self.cache_ttl[key]
        
        return None
    
    def set_in_cache(self, key: str, value: Any, ttl: Optional[int] = None):
        """
        Store item in cache
        
        Args:
            key (str): Cache key
            value (Any): Value to cache
            ttl (int, optional): Time-to-live in seconds
        """
        current_time = time.time()
        ttl_seconds = ttl if ttl is not None else self.default_cache_ttl
        
        self.cache[key] = value
        self.cache_ttl[key] = current_time + ttl_seconds
    
    def clear_cache(self, key_prefix: Optional[str] = None):
        """
        Clear cache items
        
        Args:
            key_prefix (str, optional): Only clear keys with this prefix
        """
        if key_prefix:
            # Remove keys matching prefix
            keys_to_remove = [k for k in self.cache.keys() if k.startswith(key_prefix)]
            for key in keys_to_remove:
                if key in self.cache:
                    del self.cache[key]
                if key in self.cache_ttl:
                    del self.cache_ttl[key]
            
            logger.debug(f"Cleared {len(keys_to_remove)} cache items with prefix '{key_prefix}'")
        else:
            # Clear all cache
            count = len(self.cache)
            self.cache = {}
            self.cache_ttl = {}
            logger.debug(f"Cleared all {count} cache items")
    
    def get_from_pool(self, pool_name: str) -> Optional[Any]:
        """
        Get object from a memory pool
        
        Args:
            pool_name (str): Pool name
            
        Returns:
            Optional[Any]: Pooled object or None if pool is empty
        """
        if pool_name not in self.object_pools:
            self.object_pools[pool_name] = []
            return None
        
        if not self.object_pools[pool_name]:
            return None
        
        return self.object_pools[pool_name].pop()
    
    def return_to_pool(self, pool_name: str, obj: Any):
        """
        Return object to a memory pool
        
        Args:
            pool_name (str): Pool name
            obj (Any): Object to return to pool
        """
        if pool_name not in self.object_pools:
            self.object_pools[pool_name] = []
        
        self.object_pools[pool_name].append(obj)
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the cache
        
        Returns:
            Dict[str, Any]: Cache statistics
        """
        current_time = time.time()
        
        # Count expired items
        expired_count = sum(1 for ttl in self.cache_ttl.values() if ttl <= current_time)
        
        # Group by prefix (first part of key before ':')
        prefix_counts = {}
        for key in self.cache:
            prefix = key.split(':')[0] if ':' in key else 'no_prefix'
            if prefix not in prefix_counts:
                prefix_counts[prefix] = 0
            prefix_counts[prefix] += 1
        
        return {
            'total_items': len(self.cache),
            'expired_items': expired_count,
            'active_items': len(self.cache) - expired_count,
            'prefix_counts': prefix_counts
        }
    
    def get_pool_stats(self) -> Dict[str, Any]:
        """
        Get statistics about memory pools
        
        Returns:
            Dict[str, Any]: Pool statistics
        """
        stats = {}
        
        for pool_name, pool in self.object_pools.items():
            stats[pool_name] = len(pool)
        
        return {
            'total_pools': len(self.object_pools),
            'total_pooled_objects': sum(len(pool) for pool in self.object_pools.values()),
            'pools': stats
        }
    
    async def set_memory_limit(self, max_mb: int) -> bool:
        """
        Set maximum memory limit for the process
        
        Args:
            max_mb (int): Maximum memory in MB
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Convert MB to bytes
            max_bytes = max_mb * 1024 * 1024
            
            # Set soft limit (will raise MemoryError when exceeded)
            resource.setrlimit(resource.RLIMIT_AS, (max_bytes, max_bytes))
            
            logger.info(f"Set memory limit to {max_mb} MB")
            return True
        except Exception as e:
            logger.error(f"Error setting memory limit: {str(e)}")
            return False
    
    async def cleanup_on_threshold(self) -> bool:
        """
        Perform cleanup if memory usage exceeds threshold
        
        Returns:
            bool: True if cleanup was performed, False otherwise
        """
        # Get current memory usage
        stats = await self.monitor_memory()
        memory_percent = stats['percent']
        
        # Check if cleanup is needed
        if memory_percent >= self.cleanup_threshold:
            logger.warning(f"Memory usage ({memory_percent:.1f}%) exceeds threshold ({self.cleanup_threshold}%), performing cleanup")
            await self.cleanup()
            return True
        
        return False
