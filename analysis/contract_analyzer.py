"""
ContractAnalyzer Component
Responsible for analyzing token smart contracts to detect
potential issues, risks, and suspicious features.
"""

import asyncio
import logging
import time
import re
from typing import Dict, Any, List, Optional
import aiohttp

logger = logging.getLogger(__name__)

class ContractAnalyzer:
    """
    ContractAnalyzer handles:
    - Smart contract code analysis
    - Detection of suspicious functions
    - Risk assessment for new tokens
    - Identification of potential scams
    """
    
    def __init__(self):
        """Initialize the ContractAnalyzer"""
        # Analysis results cache
        self.analyzed_contracts = {}
        
        # Known malicious patterns - Super Advanced version with more detailed patterns
        self.suspicious_patterns = {
            # Ownership & control patterns
            'ownership_transfer': r'transferOwnership|newOwner|OwnershipTransferred|changeOwner|setOwner',
            'privileged_functions': r'onlyOwner|onlyAdmin|onlyController|require\(\s*msg\.sender\s*==\s*owner\s*\)',
            'pause_functions': r'pause\s*\(|unpause\s*\(|whenNotPaused|setPause|freezeTrading|setTradingStatus',
            'proxy_patterns': r'delegatecall|upgradeToAndCall|upgradeableTo|setImplementation',
            
            # User restriction patterns
            'blacklist': r'blacklist|blocklist|banUser|ban\s*\(|blockAddress|addToBlacklist|blacklisted',
            'whitelist': r'whitelist|addToWhitelist|whitelistedAddresses|isWhitelisted',
            
            # Fee/tax patterns
            'high_tax': r'setFee|updateFee|setTax|changeTax|setBuyFee|setSellFee|setTransferFee',
            'hidden_fee': r'_fee|swapAndLiquify|takeFee|calculateFee|distributeFee',
            
            # Supply manipulation
            'mint': r'mint\s*\(|createToken|issueToken|_mint|addSupply|increaseSupply',
            'burn_manipulation': r'burn\s*\(|burnFrom|_burn|decreaseSupply|reduceSupply',
            
            # Trading restrictions
            'honeypot': r'onlyTradeWhitelisted|disableTrading|canTrade|tradingEnabled|setTradingEnabled',
            'anti_bot': r'antiBot|botProtection|sniper|sniperProtection|maxTxAmount',
            
            # Dangerous operations
            'self_destruct': r'selfdestruct|suicide|kill\s*\(|destroyContract',
            'hidden_owner': r'_owner|OWNER|owner\(\)|_admin|_controller|hiddenOwner',
            
            # Backdoors
            'backdoor': r'emergencyWithdraw|withdrawAll|drainToken|rescueToken|claimTokens',
            'time_manipulation': r'block\.timestamp|now\s+[<>=]|block\.number',
            
            # Advanced exploit patterns
            'reentrancy': r'\.call{value:',
            'flash_loans': r'flashLoan|flash_loan|IERC3156FlashLender',
            'arbitrary_calls': r'callFunction|execute\s*\(|call\s*\(address',
            'risky_assembly': r'assembly\s*{|inline assembly'
        }
        
        # Risk factors and weights - Adjusted for more precision
        self.risk_factors = {
            # Primary factors (most concerning)
            'honeypot': 80,
            'self_destruct': 75,
            'backdoor': 70,
            'flash_loans': 65,
            
            # High risk factors
            'mint': 55,
            'high_tax': 50,
            'hidden_fee': 50,
            'arbitrary_calls': 45,
            'risky_assembly': 40,
            
            # Medium risk factors
            'pause_functions': 35,
            'blacklist': 30,
            'proxy_patterns': 30,
            'time_manipulation': 25,
            'reentrancy': 25,
            
            # Lower risk factors
            'ownership_transfer': 20,
            'privileged_functions': 15,
            'burn_manipulation': 15,
            'hidden_owner': 15,
            'whitelist': 10,
            'anti_bot': 10
        }
        
        # Recent known scam contracts (would be regularly updated in production)
        self.known_scam_addresses = set([
            "So1aScamxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
            "So1aRugPu11xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
        ])
        
        # Token contract audit history (would be populated from external API in production)
        self.audit_history = {}
        
        # External solidity scanner API endpoint
        # This would be replaced with a real service in production
        self.scanner_api_url = "https://api.solidity-scanner.example.com"
        
        # Session for API calls
        self.session = None
        
        logger.info("Advanced ContractAnalyzer initialized with expanded detection capabilities")
    
    async def analyze_contract(self, token_address: str) -> Dict[str, Any]:
        """
        Analyze a token contract for potential risks using advanced detection
        
        Args:
            token_address (str): Token contract address
            
        Returns:
            Dict[str, Any]: Analysis results
        """
        logger.info(f"Analyzing contract for token {token_address}")
        
        # First check for known scams
        if token_address in self.known_scam_addresses:
            logger.warning(f"Token {token_address} matches known scam contract address")
            return {
                'token_address': token_address,
                'suspicious_features': ['known_scam'],
                'risk_score': 100,
                'risk_level': 'extreme',
                'is_potential_scam': True,
                'source_verified': False,
                'analysis_time': time.time(),
                'explanations': ['This contract address is in our database of known scam tokens.'],
                'recommendation': 'AVOID - Known scam token'
            }
        
        # Check if already analyzed
        if token_address in self.analyzed_contracts:
            cached_analysis = self.analyzed_contracts[token_address]
            # Check if cache is less than 1 hour old
            if time.time() - cached_analysis.get('analysis_time', 0) < 3600:
                logger.debug(f"Using cached analysis for {token_address}")
                return cached_analysis
            else:
                logger.debug(f"Cached analysis for {token_address} is outdated, refreshing")
        
        # Fetch contract code
        contract_code = await self._fetch_contract_code(token_address)
        
        # Analyze code for suspicious patterns
        suspicious_features = []
        risk_score = 0
        severity_counts = {'critical': 0, 'high': 0, 'medium': 0, 'low': 0}
        
        if contract_code:
            # Advanced pattern detection
            for pattern_name, regex in self.suspicious_patterns.items():
                if re.search(regex, contract_code, re.IGNORECASE):
                    suspicious_features.append(pattern_name)
                    pattern_score = self.risk_factors.get(pattern_name, 0)
                    risk_score += pattern_score
                    
                    # Categorize by severity
                    if pattern_score >= 65:
                        severity_counts['critical'] += 1
                    elif pattern_score >= 40:
                        severity_counts['high'] += 1
                    elif pattern_score >= 20:
                        severity_counts['medium'] += 1
                    else:
                        severity_counts['low'] += 1
            
            # Additional contextual analysis
            if 'honeypot' in suspicious_features and 'high_tax' in suspicious_features:
                # Combination of honeypot + high tax is extra suspicious
                risk_score += 20
                suspicious_features.append('combined_sell_prevention')
            
            if 'mint' in suspicious_features and 'hidden_owner' in suspicious_features:
                # Hidden minting capability is very suspicious
                risk_score += 15
                suspicious_features.append('hidden_inflation_risk')
            
            # Cap risk score at 100
            risk_score = min(risk_score, 100)
        else:
            # Could not fetch contract code
            suspicious_features.append('unverified_code')
            risk_score = 50  # Unverified code is moderately risky
            severity_counts['high'] += 1
        
        # Generate comprehensive analysis
        analysis = {
            'token_address': token_address,
            'suspicious_features': suspicious_features,
            'risk_score': risk_score,
            'risk_level': self._get_risk_level(risk_score),
            'risk_breakdown': {
                'critical_issues': severity_counts['critical'],
                'high_issues': severity_counts['high'],
                'medium_issues': severity_counts['medium'],
                'low_issues': severity_counts['low']
            },
            'is_potential_scam': risk_score > 70,
            'source_verified': bool(contract_code),
            'analysis_time': time.time(),
            'analysis_version': '2.0'
        }
        
        # Intelligent recommendation based on risk profile
        if risk_score >= 80:
            analysis['recommendation'] = 'AVOID - Extremely high risk'
        elif risk_score >= 65:
            analysis['recommendation'] = 'HIGH CAUTION - Trade only with minimal capital'
        elif risk_score >= 40:
            analysis['recommendation'] = 'CAUTION - Potential risks identified'
        elif risk_score >= 20:
            analysis['recommendation'] = 'MODERATE RISK - Common contract patterns detected'
        else:
            analysis['recommendation'] = 'LOW RISK - Few suspicious patterns detected'
        
        # Add detailed explanations for suspicious features
        if suspicious_features:
            explanations = []
            for feature in suspicious_features:
                explanation = self._get_feature_explanation(feature)
                if explanation:
                    explanations.append(explanation)
            
            analysis['explanations'] = explanations
        
        # Cache the analysis
        self.analyzed_contracts[token_address] = analysis
        
        logger.info(f"Contract analysis for {token_address}: Risk score {risk_score}, Risk level: {analysis['risk_level']}")
        return analysis
    
    async def _fetch_contract_code(self, token_address: str) -> Optional[str]:
        """
        Fetch contract source code from blockchain explorer
        
        Args:
            token_address (str): Token contract address
            
        Returns:
            Optional[str]: Contract source code or None if not available
        """
        logger.debug(f"Fetching contract code for {token_address}")
        
        try:
            # In a real implementation, this would use Solana explorer API
            # or similar service to fetch the verified contract code
            
            # Initialize session if not already done
            if not self.session:
                self.session = aiohttp.ClientSession()
            
            # This is a placeholder for the actual implementation that would make a real API call
            # In a production environment, we would call the Solana Explorer API or another blockchain explorer
            
            # For demo purposes, we'll simulate the response based on the token address
            # Use the last character of the address to determine the response type (for simulation only)
            last_char = token_address[-1].lower()
            
            if last_char in '01234': # 50% chance of verified code with few issues
                # Simulate a token with minimal issues
                return f"""
                // SPDX-License-Identifier: MIT
                pragma solidity ^0.8.10;
                
                import "@openzeppelin/contracts/token/ERC20/ERC20.sol";
                import "@openzeppelin/contracts/access/Ownable.sol";
                
                contract TokenExample is ERC20, Ownable {{
                    bool public tradingEnabled = true;
                    mapping(address => bool) public blacklist;
                    
                    constructor() ERC20("ExampleToken", "EX") {{
                        _mint(msg.sender, 1000000000 * 10**18);
                    }}
                    
                    function transferOwnership(address newOwner) public override onlyOwner {{
                        require(newOwner != address(0), "New owner cannot be zero address");
                        super.transferOwnership(newOwner);
                    }}
                    
                    function enableTrading(bool _enabled) external onlyOwner {{
                        tradingEnabled = _enabled;
                    }}
                    
                    function addToBlacklist(address user) external onlyOwner {{
                        blacklist[user] = true;
                    }}
                    
                    function removeFromBlacklist(address user) external onlyOwner {{
                        blacklist[user] = false;
                    }}
                    
                    function _beforeTokenTransfer(address from, address to, uint256 amount) internal override {{
                        require(tradingEnabled || from == owner() || to == owner(), "Trading not enabled");
                        require(!blacklist[from] && !blacklist[to], "Address is blacklisted");
                        super._beforeTokenTransfer(from, to, amount);
                    }}
                }}
                """
            elif last_char in '56789': # 50% chance of verified code with many issues
                # Simulate a token with many suspicious features (honeypot-like)
                return f"""
                // SPDX-License-Identifier: UNLICENSED
                pragma solidity ^0.8.10;
                
                contract HighRiskToken {{
                    address private _owner;
                    bool public tradingEnabled = false;
                    bool private _inSwap;
                    uint256 private _sellTax = 5;
                    uint256 private _buyTax = 5;
                    mapping(address => bool) private _isExcludedFromFees;
                    mapping(address => bool) public blacklist;
                    mapping(address => bool) private _isWhitelisted;
                    
                    constructor() {{
                        _owner = msg.sender;
                        _isExcludedFromFees[msg.sender] = true;
                        _isWhitelisted[msg.sender] = true;
                    }}
                    
                    modifier onlyOwner() {{
                        require(msg.sender == _owner, "Not authorized");
                        _;
                    }}
                    
                    function transferOwnership(address newOwner) public onlyOwner {{
                        _owner = newOwner;
                    }}
                    
                    function enableTrading() external onlyOwner {{
                        tradingEnabled = true;
                    }}
                    
                    function disableTrading() external onlyOwner {{
                        tradingEnabled = false;
                    }}
                    
                    function setSellTax(uint256 newTax) external onlyOwner {{
                        require(newTax <= 99, "Tax too high");
                        _sellTax = newTax;
                    }}
                    
                    function setBuyTax(uint256 newTax) external onlyOwner {{
                        require(newTax <= 99, "Tax too high");
                        _buyTax = newTax;
                    }}
                    
                    function addToBlacklist(address user) external onlyOwner {{
                        blacklist[user] = true;
                    }}
                    
                    function removeFromBlacklist(address user) external onlyOwner {{
                        blacklist[user] = false;
                    }}
                    
                    function whitelistAddress(address user, bool status) external onlyOwner {{
                        _isWhitelisted[user] = status;
                    }}
                    
                    function mint(address to, uint256 amount) external onlyOwner {{
                        // Mint tokens
                    }}
                    
                    function emergencyWithdraw() external onlyOwner {{
                        // Transfer all contract funds to owner
                    }}
                    
                    function transfer(address to, uint256 amount) public returns (bool) {{
                        require(tradingEnabled || msg.sender == _owner || _isWhitelisted[msg.sender], "Trading not enabled");
                        require(!blacklist[msg.sender], "Address is blacklisted");
                        
                        // Apply taxes based on various conditions
                        uint256 taxAmount = 0;
                        if (!_isExcludedFromFees[msg.sender]) {{
                            if (block.timestamp < 1 days) {{
                                taxAmount = amount * 90 / 100; // 90% tax in first day
                            }} else {{
                                taxAmount = amount * _sellTax / 100;
                            }}
                        }}
                        
                        // Complex logic to handle transfers with taxes
                        return true;
                    }}
                    
                    // Low-level function that can be dangerous
                    function executeCustomTransaction(address target, bytes memory data) external onlyOwner {{
                        (bool success, ) = target.call(data);
                        require(success, "Transaction failed");
                    }}
                    
                    // Potentially dangerous self-destruct function
                    function destroyContract() external onlyOwner {{
                        selfdestruct(payable(_owner));
                    }}
                }}
                """
            else:
                # Unverified contract code
                return None
            
        except Exception as e:
            logger.error(f"Error fetching contract code: {str(e)}")
            return None
    
    def _get_risk_level(self, risk_score: float) -> str:
        """
        Convert risk score to risk level
        
        Args:
            risk_score (float): Risk score (0-100)
            
        Returns:
            str: Risk level (low, medium, high, extreme)
        """
        if risk_score < 20:
            return 'low'
        elif risk_score < 50:
            return 'medium'
        elif risk_score < 75:
            return 'high'
        else:
            return 'extreme'
    
    def _get_feature_explanation(self, feature: str) -> Optional[str]:
        """
        Get explanation for a suspicious feature
        
        Args:
            feature (str): Feature identifier
            
        Returns:
            Optional[str]: Explanation or None if not available
        """
        explanations = {
            # Ownership & control explanations
            'ownership_transfer': 'Contract allows ownership transfer, which can be used for rug pulls if transferred to a malicious address.',
            'privileged_functions': 'Contract has restricted functions accessible only to privileged roles, potentially allowing centralized control.',
            'pause_functions': 'Contract can be paused, potentially preventing users from selling tokens during price drops.',
            'proxy_patterns': 'Contract uses proxy patterns that may allow its code to be replaced entirely, changing how the token functions.',
            
            # User restriction explanations
            'blacklist': 'Contract includes blacklist functionality that can block specific addresses from trading.',
            'whitelist': 'Contract restricts trading to whitelisted addresses, allowing selective control of who can trade.',
            
            # Fee/tax explanations
            'high_tax': 'Contract has functions to change transaction taxes/fees, which could be set very high to prevent selling.',
            'hidden_fee': 'Contract contains mechanisms for taking hidden fees during transactions, potentially extracting value from users.',
            
            # Supply manipulation explanations
            'mint': 'Contract allows minting of new tokens, which can lead to inflation and value dilution without notice.',
            'burn_manipulation': 'Contract has special burn functionality that may be used to manipulate token supply and metrics.',
            
            # Trading restrictions explanations
            'honeypot': 'Contract contains functions that can restrict trading, making it impossible to sell tokens (honeypot).',
            'anti_bot': 'Contract has anti-bot measures that can be used legitimately but may also restrict normal trading.',
            
            # Dangerous operations explanations
            'self_destruct': 'Contract can self-destruct, potentially locking funds forever or destroying the token contract.',
            'hidden_owner': 'Contract uses obscured ownership patterns that hide who can control critical functions.',
            
            # Backdoors explanations
            'backdoor': 'Contract contains emergency functions that can drain tokens or funds from the contract or users.',
            'time_manipulation': 'Contract uses block time or number in security-critical ways that could be manipulated by miners.',
            
            # Advanced exploit explanations
            'reentrancy': 'Contract contains patterns vulnerable to reentrancy attacks, potentially allowing funds to be drained.',
            'flash_loans': 'Contract interacts with flash loans, which can be used for price manipulation and exploits.',
            'arbitrary_calls': 'Contract allows execution of arbitrary external calls, which may lead to unexpected behavior.',
            'risky_assembly': 'Contract uses low-level assembly code that bypasses Solidity safety checks.',
            
            # Combination patterns
            'combined_sell_prevention': 'Contract combines multiple mechanisms (trading disablement AND high taxes) that together can completely block selling.',
            'hidden_inflation_risk': 'Contract has hidden minting capabilities controlled by obscured owner, extreme risk of stealth inflation.',
            
            # Known issues
            'known_scam': 'This contract address matches a known scam token in our database. Avoid at all costs.',
            
            # Other explanations
            'unverified_code': 'Contract source code is not verified on-chain, making it impossible to audit for malicious code.'
        }
        
        explanation = explanations.get(feature)
        if explanation:
            return explanation
        
        # Generic explanation for unknown features
        return f"Contract contains a potentially risky feature: {feature.replace('_', ' ')}"
    
    async def scan_with_external_service(self, token_address: str) -> Dict[str, Any]:
        """
        Scan contract with external security service
        
        Args:
            token_address (str): Token contract address
            
        Returns:
            Dict[str, Any]: Scan results
        """
        logger.debug(f"Scanning contract with external service: {token_address}")
        
        try:
            # In a real implementation, this would call an actual API
            # For simulation, we'll return a simulated response
            
            # Use last two characters of address to determine result (for simulation)
            last_chars = token_address[-2:].lower()
            
            # Simulate some randomness in the response
            vulnerability_count = int(last_chars[0], 16) % 5 if last_chars[0].isalnum() else 0
            
            vulnerabilities = []
            for i in range(vulnerability_count):
                vuln_type = ['reentrancy', 'overflow', 'access-control', 'logic-error'][i % 4]
                vulnerabilities.append({
                    'type': vuln_type,
                    'severity': ['low', 'medium', 'high', 'critical'][i % 4],
                    'line': 100 + (i * 20),
                    'description': f"Potential {vuln_type} vulnerability detected"
                })
            
            scan_result = {
                'token_address': token_address,
                'scan_id': f"scan_{int(time.time())}",
                'vulnerabilities': vulnerabilities,
                'vulnerability_count': vulnerability_count,
                'scan_time': time.time(),
                'status': 'completed'
            }
            
            return scan_result
            
        except Exception as e:
            logger.error(f"Error scanning contract with external service: {str(e)}")
            return {
                'token_address': token_address,
                'error': str(e),
                'status': 'failed',
                'scan_time': time.time()
            }
    
    def get_analysis_history(self) -> List[Dict[str, Any]]:
        """
        Get history of analyzed contracts
        
        Returns:
            List[Dict[str, Any]]: List of analysis results
        """
        # Convert from dict to list
        history = list(self.analyzed_contracts.values())
        
        # Sort by analysis time (most recent first)
        history.sort(key=lambda x: x.get('analysis_time', 0), reverse=True)
        
        return history
    
    def clear_cache(self, token_address: Optional[str] = None):
        """
        Clear analysis cache
        
        Args:
            token_address (str, optional): Specific token to clear, or all if None
        """
        if token_address:
            if token_address in self.analyzed_contracts:
                del self.analyzed_contracts[token_address]
                logger.debug(f"Cleared cache for {token_address}")
        else:
            self.analyzed_contracts = {}
            logger.debug("Cleared entire analysis cache")
