"""
Adapter Component
Responsible for data format conversion and API call forwarding
between different components and services.
"""

import logging
import json
from typing import Dict, Any, List, Optional, Union

logger = logging.getLogger(__name__)

class Adapter:
    """
    Adapter handles:
    - Conversion between different data formats
    - Forwarding API calls between systems
    - Ensuring backward compatibility
    """
    
    def __init__(self, pump_portal_client, raydium_client):
        """
        Initialize the Adapter
        
        Args:
            pump_portal_client: PumpPortalClient instance
            raydium_client: RaydiumClient instance
        """
        self.pump_portal_client = pump_portal_client
        self.raydium_client = raydium_client
        logger.info("Adapter initialized")
    
    async def convert_token_data(self, 
                                token_data: Dict[str, Any], 
                                source_format: str, 
                                target_format: str) -> Dict[str, Any]:
        """
        Convert token data between different formats
        
        Args:
            token_data (Dict[str, Any]): Token data in source format
            source_format (str): Source format identifier
            target_format (str): Target format identifier
            
        Returns:
            Dict[str, Any]: Converted token data in target format
        """
        logger.debug(f"Converting token data from {source_format} to {target_format}")
        
        if source_format == "pump_portal" and target_format == "raydium":
            return self._convert_pump_portal_to_raydium(token_data)
        
        elif source_format == "raydium" and target_format == "pump_portal":
            return self._convert_raydium_to_pump_portal(token_data)
        
        elif source_format == "pump_portal" and target_format == "internal":
            return self._convert_pump_portal_to_internal(token_data)
        
        elif source_format == "raydium" and target_format == "internal":
            return self._convert_raydium_to_internal(token_data)
        
        else:
            logger.warning(f"Unsupported conversion: {source_format} to {target_format}")
            return token_data
    
    def _convert_pump_portal_to_raydium(self, token_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert token data from PumpPortal format to Raydium format
        
        Args:
            token_data (Dict[str, Any]): Token data in PumpPortal format
            
        Returns:
            Dict[str, Any]: Token data in Raydium format
        """
        # Example conversion (adjust based on actual API formats)
        converted = {
            "mint": token_data.get("token_address"),
            "symbol": token_data.get("symbol", "").upper(),
            "name": token_data.get("name", ""),
            "decimals": token_data.get("decimals", 9),
            "liquidity": {
                "usd": token_data.get("liquidity_usd", 0)
            },
            "price": {
                "usd": token_data.get("price_usd", 0)
            }
        }
        
        # Add any additional fields needed by Raydium
        if "market_address" in token_data:
            converted["market"] = token_data["market_address"]
        
        return converted
    
    def _convert_raydium_to_pump_portal(self, token_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert token data from Raydium format to PumpPortal format
        
        Args:
            token_data (Dict[str, Any]): Token data in Raydium format
            
        Returns:
            Dict[str, Any]: Token data in PumpPortal format
        """
        # Example conversion (adjust based on actual API formats)
        converted = {
            "token_address": token_data.get("mint"),
            "symbol": token_data.get("symbol", "").upper(),
            "name": token_data.get("name", ""),
            "decimals": token_data.get("decimals", 9),
            "price_usd": token_data.get("price", {}).get("usd", 0),
            "liquidity_usd": token_data.get("liquidity", {}).get("usd", 0)
        }
        
        # Add any additional fields needed by PumpPortal
        if "market" in token_data:
            converted["market_address"] = token_data["market"]
        
        return converted
    
    def _convert_pump_portal_to_internal(self, token_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert token data from PumpPortal format to internal format
        
        Args:
            token_data (Dict[str, Any]): Token data in PumpPortal format
            
        Returns:
            Dict[str, Any]: Token data in internal format
        """
        # Example conversion to standardized internal format
        internal_data = {
            "address": token_data.get("token_address"),
            "symbol": token_data.get("symbol", "").upper(),
            "name": token_data.get("name", ""),
            "decimals": token_data.get("decimals", 9),
            "price_usd": token_data.get("price_usd", 0),
            "liquidity_usd": token_data.get("liquidity_usd", 0),
            "volume_24h_usd": token_data.get("volume_24h", 0),
            "market_cap_usd": token_data.get("market_cap", 0),
            "is_new_token": token_data.get("is_new", False),
            "discovery_time": token_data.get("discovery_time"),
            "source": "pump_portal"
        }
        
        # Calculate additional metrics if possible
        if "price_change_24h_percentage" in token_data:
            internal_data["price_change_24h"] = token_data["price_change_24h_percentage"]
        
        return internal_data
    
    def _convert_raydium_to_internal(self, token_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert token data from Raydium format to internal format
        
        Args:
            token_data (Dict[str, Any]): Token data in Raydium format
            
        Returns:
            Dict[str, Any]: Token data in internal format
        """
        # Example conversion to standardized internal format
        price_data = token_data.get("price", {})
        liquidity_data = token_data.get("liquidity", {})
        volume_data = token_data.get("volume", {})
        
        internal_data = {
            "address": token_data.get("mint"),
            "symbol": token_data.get("symbol", "").upper(),
            "name": token_data.get("name", ""),
            "decimals": token_data.get("decimals", 9),
            "price_usd": price_data.get("usd", 0),
            "liquidity_usd": liquidity_data.get("usd", 0),
            "volume_24h_usd": volume_data.get("usd", 0),
            "market_cap_usd": token_data.get("marketCap", 0),
            "source": "raydium"
        }
        
        # Calculate additional metrics if possible
        if "priceChange" in token_data:
            internal_data["price_change_24h"] = token_data["priceChange"].get("h24", 0)
        
        return internal_data
    
    async def forward_pump_portal_to_raydium(self, 
                                            method: str, 
                                            params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Forward API call from PumpPortal to Raydium with format conversion
        
        Args:
            method (str): API method name
            params (Dict[str, Any]): API parameters
            
        Returns:
            Dict[str, Any]: API response
        """
        logger.debug(f"Forwarding API call from PumpPortal to Raydium: {method}")
        
        # Convert parameters from PumpPortal format to Raydium format
        converted_params = await self._convert_api_params(
            params, "pump_portal", "raydium", method
        )
        
        # Map PumpPortal method to equivalent Raydium method
        raydium_method = self._map_pump_portal_to_raydium_method(method)
        
        # Call Raydium client with converted parameters
        try:
            result = await self.raydium_client.call_api(raydium_method, converted_params)
            return result
        except Exception as e:
            logger.error(f"Error forwarding call to Raydium: {str(e)}")
            raise
    
    async def forward_raydium_to_pump_portal(self, 
                                            method: str, 
                                            params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Forward API call from Raydium to PumpPortal with format conversion
        
        Args:
            method (str): API method name
            params (Dict[str, Any]): API parameters
            
        Returns:
            Dict[str, Any]: API response
        """
        logger.debug(f"Forwarding API call from Raydium to PumpPortal: {method}")
        
        # Convert parameters from Raydium format to PumpPortal format
        converted_params = await self._convert_api_params(
            params, "raydium", "pump_portal", method
        )
        
        # Map Raydium method to equivalent PumpPortal method
        pump_portal_method = self._map_raydium_to_pump_portal_method(method)
        
        # Call PumpPortal client with converted parameters
        try:
            result = await self.pump_portal_client.call_api(pump_portal_method, converted_params)
            return result
        except Exception as e:
            logger.error(f"Error forwarding call to PumpPortal: {str(e)}")
            raise
    
    async def _convert_api_params(self, 
                                 params: Dict[str, Any], 
                                 source_format: str, 
                                 target_format: str,
                                 method: str) -> Dict[str, Any]:
        """
        Convert API parameters between different formats
        
        Args:
            params (Dict[str, Any]): API parameters in source format
            source_format (str): Source format identifier
            target_format (str): Target format identifier
            method (str): API method being called
            
        Returns:
            Dict[str, Any]: Converted parameters in target format
        """
        # This is a generic parameter conversion function
        # In real implementation, method-specific conversions would be defined
        logger.debug(f"Converting API params for method {method}")
        
        # Simple passthrough for now - implement specific conversions as needed
        return params
    
    def _map_pump_portal_to_raydium_method(self, method: str) -> str:
        """
        Map PumpPortal API method to equivalent Raydium method
        
        Args:
            method (str): PumpPortal API method
            
        Returns:
            str: Equivalent Raydium API method
        """
        # Define method mapping
        method_map = {
            "get_token_info": "getTokenInfo",
            "get_price": "getPrice",
            "get_liquidity": "getLiquidity",
            "create_order": "createOrder"
        }
        
        return method_map.get(method, method)
    
    def _map_raydium_to_pump_portal_method(self, method: str) -> str:
        """
        Map Raydium API method to equivalent PumpPortal method
        
        Args:
            method (str): Raydium API method
            
        Returns:
            str: Equivalent PumpPortal API method
        """
        # Define method mapping (reverse of previous map)
        method_map = {
            "getTokenInfo": "get_token_info",
            "getPrice": "get_price",
            "getLiquidity": "get_liquidity",
            "createOrder": "create_order"
        }
        
        return method_map.get(method, method)
    
    async def handle_version_compatibility(self, 
                                         data: Dict[str, Any], 
                                         current_version: str,
                                         target_version: str) -> Dict[str, Any]:
        """
        Handle backward compatibility between different API versions
        
        Args:
            data (Dict[str, Any]): Data in current version format
            current_version (str): Current API version
            target_version (str): Target API version
            
        Returns:
            Dict[str, Any]: Data in target version format
        """
        logger.debug(f"Handling version compatibility from {current_version} to {target_version}")
        
        # Implement version compatibility logic here
        # This is a simple placeholder that doesn't modify the data
        # In a real implementation, you would apply transformations based on version
        return data
