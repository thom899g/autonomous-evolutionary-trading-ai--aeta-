"""
AETA Configuration Management
Centralized configuration with environment-aware settings and validation
"""
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from pathlib import Path
import structlog
from dotenv import load_dotenv

load_dotenv()

logger = structlog.get_logger(__name__)

@dataclass
class TradingConfig:
    """Trading-specific configuration"""
    initial_capital: float = 100000.0
    max_position_size: float = 0.1  # 10% of portfolio
    stop_loss_pct: float = 0.02  # 2% stop loss
    take_profit_pct: float = 0.05  # 5% take profit
    max_daily_loss: float = 0.03  # 3% max daily loss
    commission_rate: float = 0.001  # 0.1% commission

@dataclass
class RLConfig:
    """Reinforcement Learning configuration"""
    learning_rate: float = 0.0003
    gamma: float = 0.99
    batch_size: int = 64
    buffer_size: int = 100000
    tau: float = 0.005
    exploration_noise: float = 0.1
    policy_frequency: int = 2
    target_update_frequency: int = 1000

@dataclass
class DataConfig:
    """Data pipeline configuration"""
    data_sources: List[str] = None
    update_frequency: int = 300  # seconds
    history_days: int = 365
    feature_window: int = 50
    validation_split: float = 0.2
    
    def __post_init__(self):
        if self.data_sources is None:
            self.data_sources = ["yfinance", "ccxt"]

@dataclass
class FirebaseConfig:
    """Firebase configuration"""
    project_id: str = ""
    credentials_path: str = ""
    collection_prefix: str = "aeta_"
    
    def __post_init__(self):
        if not self.project_id:
            self.project_id = os.getenv("FIREBASE_PROJECT_ID", "")
        if not self.credentials_path:
            self.credentials_path = os.getenv("FIREBASE_CREDENTIALS_PATH", "")

@dataclass
class AETAConfig:
    """Master configuration for AETA system"""
    trading: TradingConfig
    rl: RLConfig
    data: DataConfig
    firebase: FirebaseConfig
    log_level: str = "INFO"
    environment: str = "development"
    
    @classmethod
    def load(cls) -> 'AETAConfig':
        """Load configuration with environment overrides"""
        env = os.getenv("ENVIRONMENT", "development")
        
        # Initialize configurations
        trading = TradingConfig()
        rl = RLConfig()
        data = DataConfig()
        firebase = FirebaseConfig()
        
        # Environment-specific overrides
        if env == "production":
            trading.max_position_size = 0.05
            rl.exploration_noise = 0.05
            log_level = "WARNING"
        elif env == "staging":
            log_level = "INFO"
        else:
            log_level = "DEBUG"
            
        return cls(
            trading=trading,
            rl=rl,
            data=data,
            firebase=firebase,
            log_level=log_level,
            environment=env
        )
    
    def validate(self) -> bool:
        """Validate configuration parameters"""
        errors = []
        
        if self.trading.initial_capital <= 0:
            errors.append("Initial capital must be positive")
        if not 0 < self.trading.max_position_size <= 1:
            errors.append("Max position size must be between 0 and 1")
        if self.firebase.project_id == "":
            errors.append("Firebase project ID is required")
            
        if errors:
            logger.error("Configuration validation failed", errors=errors)
            return False
        return True

# Global configuration instance
config = AETAConfig.load()

if not config.validate():
    logger.critical("Invalid configuration")
    sys.exit(1)