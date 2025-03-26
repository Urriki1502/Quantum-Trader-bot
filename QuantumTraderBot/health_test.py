"""
Test script to perform a comprehensive health check on the trading bot
"""

import asyncio
import logging
import json
import sys
import os
from typing import Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

async def run_health_check():
    """Run a comprehensive health check"""
    # Import necessary components
    from core.config_manager import ConfigManager
    from core.state_manager import StateManager
    from core.health_check import HealthCheck
    
    # Initialize components
    config_manager = ConfigManager()
    state_manager = StateManager()
    
    # Create health check
    health_check = HealthCheck(config_manager, state_manager)
    
    # Run health check
    logger.info("Starting comprehensive health check...")
    results = await health_check.run_comprehensive_check()
    
    # Print summary
    logger.info(f"Health check completed. Overall status: {results['overall_status']}")
    logger.info(f"Found {len(results['issues'])} issues")
    
    # Print issues
    if results['issues']:
        logger.info("Issues found:")
        for i, issue in enumerate(results['issues'], 1):
            logger.warning(f"{i}. {issue['description']} (Severity: {issue['severity']})")
    
    # Print recommendations
    if results['recommendations']:
        logger.info("Recommendations:")
        for i, recommendation in enumerate(results['recommendations'], 1):
            logger.info(f"{i}. {recommendation}")
    
    # Perform additional checks
    await check_component_dependencies(state_manager)
    await check_system_resources()
    await check_external_connections()
    
    return results

async def check_component_dependencies(state_manager: 'StateManager'):
    """Check component dependencies"""
    logger.info("Checking component dependencies...")
    
    # Get all components
    components = state_manager.get_all_components()
    
    # Basic dependency relationships
    dependencies = {
        'trading_integration': ['pump_portal_client', 'raydium_client', 'risk_manager', 'strategy_manager'],
        'raydium_client': ['security_manager'],
        'pump_portal_client': ['security_manager'],
        'strategy_manager': ['state_manager'],
        'risk_manager': ['state_manager']
    }
    
    # Check dependencies
    for component, deps in dependencies.items():
        if component in components:
            logger.info(f"Checking dependencies for {component}...")
            for dep in deps:
                if dep not in components:
                    logger.warning(f"Missing dependency: {component} requires {dep}")
                elif components[dep].status != 'running':
                    logger.warning(f"Dependency not running: {component} requires {dep}")

async def check_system_resources():
    """Check system resources"""
    logger.info("Checking system resources...")
    
    try:
        import psutil
        
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        logger.info(f"CPU Usage: {cpu_percent}%")
        
        if cpu_percent > 80:
            logger.warning("High CPU usage detected")
        
        # Memory usage
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        memory_used_mb = memory.used / (1024 * 1024)
        memory_total_mb = memory.total / (1024 * 1024)
        
        logger.info(f"Memory Usage: {memory_percent}% ({memory_used_mb:.1f}MB / {memory_total_mb:.1f}MB)")
        
        if memory_percent > 80:
            logger.warning("High memory usage detected")
        
        # Disk usage
        disk = psutil.disk_usage('/')
        disk_percent = disk.percent
        disk_used_gb = disk.used / (1024 * 1024 * 1024)
        disk_total_gb = disk.total / (1024 * 1024 * 1024)
        
        logger.info(f"Disk Usage: {disk_percent}% ({disk_used_gb:.1f}GB / {disk_total_gb:.1f}GB)")
        
        if disk_percent > 90:
            logger.warning("High disk usage detected")
            
    except ImportError:
        logger.warning("psutil not installed, skipping system resource check")

async def check_external_connections():
    """Check external connections"""
    logger.info("Checking external connections...")
    
    # Check Solana RPC connection
    await check_solana_rpc()
    
    # Check wallet
    check_wallet()

async def check_solana_rpc():
    """Check Solana RPC connection"""
    logger.info("Checking Solana RPC connection...")
    
    try:
        import solana
        from solana.rpc.api import Client
        
        # Get RPC URL
        rpc_url = os.environ.get('SOLANA_RPC_URL')
        
        if not rpc_url:
            logger.warning("SOLANA_RPC_URL not set")
            return
        
        # Create client
        client = Client(rpc_url)
        
        # Check health
        response = await asyncio.to_thread(client.get_health)
        
        if response['result'] == 'ok':
            logger.info("Solana RPC connection healthy")
        else:
            logger.warning(f"Solana RPC connection unhealthy: {response}")
            
        # Get version
        version_response = await asyncio.to_thread(client.get_version)
        
        if 'result' in version_response:
            logger.info(f"Solana version: {version_response['result']['solana-core']}")
        
    except Exception as e:
        logger.error(f"Error checking Solana RPC connection: {str(e)}")

def check_wallet():
    """Check wallet configuration"""
    logger.info("Checking wallet configuration...")
    
    # Check wallet private key
    wallet_private_key = os.environ.get('WALLET_PRIVATE_KEY')
    
    if not wallet_private_key:
        logger.warning("WALLET_PRIVATE_KEY not set")
        return
    
    # Don't print or analyze private key for security reasons
    logger.info("Wallet private key available")
    
    try:
        from solana.keypair import Keypair
        import base58
        
        # Decode private key
        private_key_bytes = base58.b58decode(wallet_private_key)
        
        # Create keypair
        keypair = Keypair.from_secret_key(private_key_bytes)
        
        # Get public key
        public_key = str(keypair.public_key)
        
        logger.info(f"Wallet public key: {public_key}")
    except Exception as e:
        logger.error(f"Error validating wallet keypair: {str(e)}")

async def main():
    """Main entry point"""
    try:
        results = await run_health_check()
        
        # Export results
        with open('logs/health_results.json', 'w') as f:
            json.dump(results, f, indent=2)
            
        logger.info("Health check results saved to logs/health_results.json")
        
        # Exit with status code based on health
        sys.exit(0 if results['overall_status'] == 'healthy' else 1)
        
    except Exception as e:
        logger.error(f"Error running health check: {str(e)}")
        sys.exit(2)

if __name__ == "__main__":
    asyncio.run(main())