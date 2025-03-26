"""
Token Contract Analyzer Component
Responsible for analyzing Solana token contracts for security issues,
potential risks, and functionality assessment.
"""

import time
import logging
import re
import base64
import json
from typing import Dict, List, Any, Optional, Tuple, Set, Union

from core.config_manager import ConfigManager
from core.state_manager import StateManager

logger = logging.getLogger(__name__)

class TokenContractAnalyzer:
    """
    Analyzes token contracts for security assessment with features:
    - Contract bytecode analysis
    - Permission model assessment
    - Honeypot detection
    - Blacklist and transfer restriction detection
    - Mint and burn capability identification
    """
    
    def __init__(self, 
                config_manager: ConfigManager,
                state_manager: StateManager):
        """
        Initialize the TokenContractAnalyzer
        
        Args:
            config_manager (ConfigManager): Configuration manager instance
            state_manager (StateManager): State manager instance
        """
        self.config_manager = config_manager
        self.state_manager = state_manager
        
        # Risk patterns
        self.risk_patterns = self._load_risk_patterns()
        
        # Analyzed contract cache
        self.analysis_cache = {}
        
        # Load honeypot heuristics
        self.honeypot_heuristics = self._load_honeypot_heuristics()
        
        # Token blacklist
        self.blacklisted_tokens = set()
        self._load_blacklist()
    
    def _load_risk_patterns(self) -> Dict[str, Any]:
        """
        Load risk patterns for contract analysis
        
        Returns:
            Dict[str, Any]: Risk patterns
        """
        # In production, these would be loaded from a regularly updated source
        return {
            'high_risk': [
                {
                    'name': 'blacklist_function',
                    'description': 'Contract includes blacklist functionality',
                    'pattern': 'blacklist|denylist|blocklist',
                    'severity': 'high'
                },
                {
                    'name': 'freeze_transfer',
                    'description': 'Contract can freeze transfers',
                    'pattern': 'freeze|pause|suspend.*transfer',
                    'severity': 'high'
                },
                {
                    'name': 'owner_drain',
                    'description': 'Owner can drain tokens',
                    'pattern': 'withdraw.*all|drain.*owner|owner.*withdraw',
                    'severity': 'high'
                }
            ],
            'medium_risk': [
                {
                    'name': 'excessive_mint',
                    'description': 'Unlimited minting capability',
                    'pattern': 'mint(?!.*cap)|unlimited.*mint',
                    'severity': 'medium'
                },
                {
                    'name': 'high_fee',
                    'description': 'Excessive transfer fee',
                    'pattern': 'fee.*1[0-9]|fee.*2[0-9]|fee.*30',
                    'severity': 'medium'
                },
                {
                    'name': 'ownership_transfer',
                    'description': 'Ownership can be transferred',
                    'pattern': 'transfer.*ownership|transfer.*owner',
                    'severity': 'medium'
                }
            ],
            'low_risk': [
                {
                    'name': 'burn_function',
                    'description': 'Burn functionality',
                    'pattern': 'burn(?!.*blackhole)',
                    'severity': 'low'
                },
                {
                    'name': 'moderate_fee',
                    'description': 'Moderate transfer fee',
                    'pattern': 'fee.*[2-9]',
                    'severity': 'low'
                }
            ]
        }
    
    def _load_honeypot_heuristics(self) -> List[Dict[str, Any]]:
        """
        Load honeypot detection heuristics
        
        Returns:
            List[Dict[str, Any]]: Honeypot heuristics
        """
        return [
            {
                'name': 'buy_sell_ratio',
                'description': 'Different tax/fee for buy and sell',
                'patterns': ['buy.*tax', 'sell.*tax', 'buy.*fee', 'sell.*fee'],
                'severity': 'high'
            },
            {
                'name': 'seller_blacklist',
                'description': 'Function to blacklist sellers',
                'patterns': ['exclude.*sell', 'block.*sell', 'deny.*sell'],
                'severity': 'high'
            },
            {
                'name': 'time_based_restriction',
                'description': 'Time-based selling restrictions',
                'patterns': ['cooldown', 'time.*sell', 'wait.*sell', 'sell.*time'],
                'severity': 'high'
            },
            {
                'name': 'balance_limit',
                'description': 'Balance-based selling restrictions',
                'patterns': ['max.*sell', 'limit.*sell', 'sell.*limit', 'percent.*balance'],
                'severity': 'high'
            },
            {
                'name': 'hidden_owner',
                'description': 'Hidden or proxy owner functions',
                'patterns': ['_owner', 'proxy.*owner', 'admin.*key', 'hidden.*admin'],
                'severity': 'high'
            }
        ]
    
    def _load_blacklist(self):
        """Load token blacklist"""
        try:
            blacklist_path = self.config_manager.get('security.blacklisted_tokens_file', './data/blacklisted_tokens.json')
            try:
                with open(blacklist_path, 'r') as f:
                    blacklist_data = json.load(f)
                    self.blacklisted_tokens = set(blacklist_data.get('tokens', []))
                    logger.info(f"Loaded {len(self.blacklisted_tokens)} blacklisted tokens")
            except FileNotFoundError:
                logger.warning(f"Blacklist file not found: {blacklist_path}")
                # Initialize empty blacklist file
                import os
                os.makedirs(os.path.dirname(blacklist_path), exist_ok=True)
                with open(blacklist_path, 'w') as f:
                    json.dump({'tokens': []}, f)
        except Exception as e:
            logger.error(f"Error loading token blacklist: {e}")
    
    def add_to_blacklist(self, token_address: str, reason: str):
        """
        Add a token to the blacklist
        
        Args:
            token_address (str): Token address
            reason (str): Reason for blacklisting
        """
        if token_address in self.blacklisted_tokens:
            return
        
        self.blacklisted_tokens.add(token_address)
        
        try:
            blacklist_path = self.config_manager.get('security.blacklisted_tokens_file', './data/blacklisted_tokens.json')
            
            # Load existing blacklist
            try:
                with open(blacklist_path, 'r') as f:
                    blacklist_data = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                blacklist_data = {'tokens': []}
            
            # Update blacklist
            if 'tokens' not in blacklist_data:
                blacklist_data['tokens'] = []
            
            if token_address not in blacklist_data['tokens']:
                blacklist_data['tokens'].append(token_address)
            
            # Add reason if tracking reasons
            if 'reasons' not in blacklist_data:
                blacklist_data['reasons'] = {}
            
            blacklist_data['reasons'][token_address] = {
                'reason': reason,
                'timestamp': time.time()
            }
            
            # Save blacklist
            with open(blacklist_path, 'w') as f:
                json.dump(blacklist_data, f, indent=2)
            
            logger.info(f"Added token to blacklist: {token_address} (Reason: {reason})")
        except Exception as e:
            logger.error(f"Error adding token to blacklist: {e}")
    
    def is_blacklisted(self, token_address: str) -> bool:
        """
        Check if a token is blacklisted
        
        Args:
            token_address (str): Token address
            
        Returns:
            bool: True if token is blacklisted
        """
        return token_address in self.blacklisted_tokens
    
    async def analyze_token_contract(self, 
                                   token_address: str, 
                                   contract_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Analyze a token contract for risk factors
        
        Args:
            token_address (str): Token address
            contract_data (Dict[str, Any], optional): Contract data if already fetched
            
        Returns:
            Dict[str, Any]: Analysis results
        """
        # Check cache first
        if token_address in self.analysis_cache:
            cache_entry = self.analysis_cache[token_address]
            
            # Use cache if still valid
            cache_validity = self.config_manager.get('security.contract_analysis_cache_sec', 3600)
            if time.time() - cache_entry['timestamp'] < cache_validity:
                logger.debug(f"Using cached contract analysis for {token_address}")
                return cache_entry['analysis']
        
        # Check if token is blacklisted
        if self.is_blacklisted(token_address):
            logger.warning(f"Token {token_address} is blacklisted")
            return {
                'token_address': token_address,
                'analysis_success': True,
                'is_blacklisted': True,
                'risk_level': 'critical',
                'risk_score': 100,
                'findings': [{
                    'name': 'blacklisted_token',
                    'description': 'Token is on blacklist',
                    'severity': 'critical',
                    'category': 'blacklist'
                }],
                'contract_verified': False,
                'analyzed_at': time.time()
            }
        
        # Fetch contract data if not provided
        # In production, this would fetch from chain or explorer API
        if not contract_data:
            contract_data = await self._fetch_contract_data(token_address)
        
        if not contract_data:
            logger.warning(f"No contract data found for {token_address}")
            return {
                'token_address': token_address,
                'analysis_success': False,
                'error': 'Contract data not available',
                'analyzed_at': time.time()
            }
        
        # Extract contract content
        contract_verified = contract_data.get('verified', False)
        contract_code = contract_data.get('source_code', '')
        bytecode = contract_data.get('bytecode', '')
        
        findings = []
        risk_score = 0
        
        # Only detailed analysis for verified contracts
        if contract_verified and contract_code:
            # Analyze code patterns
            pattern_findings = self._analyze_code_patterns(contract_code)
            findings.extend(pattern_findings)
            
            # Detect honeypot characteristics
            honeypot_findings = self._detect_honeypot_patterns(contract_code)
            findings.extend(honeypot_findings)
            
            # Check for suspicious functions
            function_findings = self._analyze_functions(contract_code)
            findings.extend(function_findings)
            
            # Analyze permissions
            permission_findings = self._analyze_permissions(contract_code)
            findings.extend(permission_findings)
        else:
            # For unverified contracts, do bytecode analysis
            bytecode_findings = self._analyze_bytecode(bytecode)
            findings.extend(bytecode_findings)
            
            # Add unverified contract as a finding
            findings.append({
                'name': 'unverified_contract',
                'description': 'Contract source code is not verified',
                'severity': 'medium',
                'category': 'transparency'
            })
        
        # Calculate risk score
        risk_score = self._calculate_risk_score(findings)
        
        # Determine overall risk level
        risk_level = 'low'
        if risk_score >= 75:
            risk_level = 'critical'
        elif risk_score >= 50:
            risk_level = 'high'
        elif risk_score >= 25:
            risk_level = 'medium'
        
        # Prepare results
        analysis_results = {
            'token_address': token_address,
            'analysis_success': True,
            'is_blacklisted': False,
            'risk_level': risk_level,
            'risk_score': risk_score,
            'findings': findings,
            'contract_verified': contract_verified,
            'analyzed_at': time.time()
        }
        
        # Cache results
        self.analysis_cache[token_address] = {
            'timestamp': time.time(),
            'analysis': analysis_results
        }
        
        # Log high-risk findings
        if risk_level in ('high', 'critical'):
            logger.warning(f"High-risk token detected {token_address}: {risk_score} risk score, {len(findings)} findings")
            
            # Add to blacklist if critical and automatic blacklisting is enabled
            auto_blacklist = self.config_manager.get('security.auto_blacklist_critical', True)
            if risk_level == 'critical' and auto_blacklist:
                self.add_to_blacklist(token_address, f"Critical risk score: {risk_score}")
        
        return analysis_results
    
    async def _fetch_contract_data(self, token_address: str) -> Optional[Dict[str, Any]]:
        """
        Fetch contract data
        
        Args:
            token_address (str): Token address
            
        Returns:
            Optional[Dict[str, Any]]: Contract data or None if not found
        """
        logger.debug(f"Fetching contract data for {token_address}")
        
        # In production, this would make API call to explorer or query chain
        # Placeholder for now
        
        # Try to load from mock/test data if available
        try:
            contracts_file = self.config_manager.get('security.test_contracts_file', './data/test_contracts.json')
            
            try:
                with open(contracts_file, 'r') as f:
                    contracts_data = json.load(f)
                    if token_address in contracts_data:
                        logger.debug(f"Loaded contract data for {token_address} from test file")
                        return contracts_data[token_address]
            except FileNotFoundError:
                logger.debug(f"Test contracts file not found: {contracts_file}")
        except Exception as e:
            logger.error(f"Error loading test contract data: {e}")
        
        return None
    
    def _analyze_code_patterns(self, contract_code: str) -> List[Dict[str, Any]]:
        """
        Analyze code for risk patterns
        
        Args:
            contract_code (str): Contract source code
            
        Returns:
            List[Dict[str, Any]]: Pattern findings
        """
        findings = []
        
        # Apply all patterns
        for risk_level, patterns in self.risk_patterns.items():
            for pattern_def in patterns:
                pattern = pattern_def['pattern']
                name = pattern_def['name']
                description = pattern_def['description']
                severity = pattern_def['severity']
                
                if re.search(pattern, contract_code, re.IGNORECASE):
                    findings.append({
                        'name': name,
                        'description': description,
                        'severity': severity,
                        'category': 'code_pattern',
                        'matches': True
                    })
        
        return findings
    
    def _detect_honeypot_patterns(self, contract_code: str) -> List[Dict[str, Any]]:
        """
        Detect honeypot characteristics in contract code
        
        Args:
            contract_code (str): Contract source code
            
        Returns:
            List[Dict[str, Any]]: Honeypot findings
        """
        findings = []
        
        # Check for honeypot patterns
        for heuristic in self.honeypot_heuristics:
            pattern_matches = []
            
            for pattern in heuristic['patterns']:
                if re.search(pattern, contract_code, re.IGNORECASE):
                    pattern_matches.append(pattern)
            
            if pattern_matches:
                findings.append({
                    'name': f"honeypot_{heuristic['name']}",
                    'description': f"Possible honeypot: {heuristic['description']}",
                    'severity': heuristic['severity'],
                    'category': 'honeypot',
                    'matching_patterns': pattern_matches
                })
        
        return findings
    
    def _analyze_functions(self, contract_code: str) -> List[Dict[str, Any]]:
        """
        Analyze contract functions for suspicious features
        
        Args:
            contract_code (str): Contract source code
            
        Returns:
            List[Dict[str, Any]]: Function findings
        """
        findings = []
        
        # Extract functions
        function_pattern = r'(?:function|pub fn)\s+(\w+)'
        functions = re.findall(function_pattern, contract_code)
        
        # Check for suspicious function names
        suspicious_functions = [
            ('owner', 'admin', 'administrator'),
            ('blacklist', 'blocklist', 'deny'),
            ('pause', 'freeze', 'lock'),
            ('mint', 'issue', 'create'),
            ('withdraw', 'drain', 'collect'),
            ('exclude', 'exempt', 'whitelist')
        ]
        
        for function in functions:
            for suspect_group in suspicious_functions:
                for suspect in suspect_group:
                    if suspect.lower() in function.lower():
                        findings.append({
                            'name': 'suspicious_function',
                            'description': f"Suspicious function '{function}' detected",
                            'severity': 'medium',
                            'category': 'suspicious_function',
                            'function_name': function
                        })
                        break
                else:
                    continue
                break
        
        return findings
    
    def _analyze_permissions(self, contract_code: str) -> List[Dict[str, Any]]:
        """
        Analyze contract permission model
        
        Args:
            contract_code (str): Contract source code
            
        Returns:
            List[Dict[str, Any]]: Permission findings
        """
        findings = []
        
        # Check for owner-only functions
        owner_patterns = [
            r'only\s*\(\s*owner\s*\)',
            r'require\s*\(\s*msg\.sender\s*==\s*owner\s*\)',
            r'require\s*\(\s*isOwner\(\)\s*\)',
            r'if\s*\(\s*msg\.sender\s*!=\s*owner\s*\)\s*revert'
        ]
        
        owner_functions_count = 0
        for pattern in owner_patterns:
            matches = re.findall(pattern, contract_code)
            owner_functions_count += len(matches)
        
        if owner_functions_count > 5:
            findings.append({
                'name': 'excessive_owner_functions',
                'description': f"Excessive number of owner-only functions: {owner_functions_count}",
                'severity': 'medium',
                'category': 'permissions',
                'count': owner_functions_count
            })
        
        # Check for renounced ownership
        renounce_pattern = r'(renounce|revoke|forfeit|surrender)\s*Ownership'
        if not re.search(renounce_pattern, contract_code, re.IGNORECASE):
            findings.append({
                'name': 'no_renounce_ownership',
                'description': "No function to renounce ownership found",
                'severity': 'low',
                'category': 'permissions'
            })
        
        return findings
    
    def _analyze_bytecode(self, bytecode: str) -> List[Dict[str, Any]]:
        """
        Analyze contract bytecode for indicators of risk
        
        Args:
            bytecode (str): Contract bytecode
            
        Returns:
            List[Dict[str, Any]]: Bytecode findings
        """
        findings = []
        
        # This is a placeholder as bytecode analysis requires more complex heuristics
        # In production, would do pattern matching against known patterns
        
        # Check for unusually small bytecode (might be proxy)
        if len(bytecode) < 500:
            findings.append({
                'name': 'small_bytecode',
                'description': "Unusually small bytecode, may be a proxy contract",
                'severity': 'medium',
                'category': 'bytecode_analysis',
                'bytecode_length': len(bytecode)
            })
        
        return findings
    
    def _calculate_risk_score(self, findings: List[Dict[str, Any]]) -> int:
        """
        Calculate overall risk score from findings
        
        Args:
            findings (List[Dict[str, Any]]): List of findings
            
        Returns:
            int: Risk score (0-100)
        """
        if not findings:
            return 0
        
        # Severity weights
        weights = {
            'critical': 25,
            'high': 15,
            'medium': 7,
            'low': 3
        }
        
        # Calculate weighted score
        score = 0
        for finding in findings:
            severity = finding.get('severity', 'low')
            score += weights.get(severity, 1)
        
        # Cap at 100
        return min(score, 100)
    
    def get_analysis_stats(self) -> Dict[str, Any]:
        """
        Get statistics about contract analysis
        
        Returns:
            Dict[str, Any]: Analysis statistics
        """
        return {
            'analyzed_contracts': len(self.analysis_cache),
            'blacklisted_tokens': len(self.blacklisted_tokens),
            'cache_size': len(self.analysis_cache)
        }
    
    def clear_analysis_cache(self):
        """Clear analysis cache"""
        self.analysis_cache = {}
        logger.info("Contract analysis cache cleared")