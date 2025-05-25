#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import sys
import json
import time
import logging
import threading
import traceback
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from contextlib import contextmanager
import warnings
warnings.filterwarnings('ignore')

# Core dependencies
import pandas as pd
import numpy as np
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import sqlite3
import pickle
from pathlib import Path

# ML and TA libraries
import ta
import talib
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LeakyReLU, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from imblearn.under_sampling import RandomUnderSampler
import shap

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Email alerts
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart


# In[2]:


# Set random seeds for reproducibility
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)


# In[3]:


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_system.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


# In[4]:


@dataclass
class TradingConfig:
    """Configuration class for the trading system"""
    # Data settings
    symbol: str = "BTCUSDT"
    interval: str = "4h"
    lookback_days: int = 365

    # Model settings
    model_retrain_days: int = 30
    min_samples_for_training: int = 1000
    validation_split: float = 0.2

    # Risk management
    max_position_size: float = 0.1  # 10% of portfolio max
    max_daily_drawdown: float = 0.02  # 2% max daily loss
    max_total_drawdown: float = 0.15  # 15% max total drawdown
    stop_loss_pct: float = 0.05  # 5% stop loss
    take_profit_pct: float = 0.10  # 10% take profit

    # Trading costs
    trading_fee: float = 0.001  # 0.1% fee
    slippage: float = 0.0005  # 0.05% slippage

    # Portfolio settings
    initial_capital: float = 100000  # $100k
    min_trade_size: float = 100  # $100 minimum trade

    # API settings
    api_timeout: int = 30
    max_retries: int = 3
    rate_limit_calls: int = 1000
    rate_limit_window: int = 60

    # Monitoring
    alert_email: str = ""
    smtp_server: str = "smtp.gmail.com"
    smtp_port: int = 587
    email_user: str = ""
    email_password: str = ""

    # Paths
    data_path: str = "data"
    models_path: str = "models"
    logs_path: str = "logs"


# In[5]:


class DataValidator:
    """Validates market data quality and consistency"""

    @staticmethod
    def validate_ohlcv(df: pd.DataFrame) -> bool:
        """Validate OHLCV data integrity"""
        try:
            # Check required columns
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_cols):
                logger.error("Missing required OHLCV columns")
                return False

            # Check for NaN values
            if df[required_cols].isnull().any().any():
                logger.warning("Found NaN values in OHLCV data")
                return False

            # Validate OHLC relationships
            invalid_ohlc = (
                (df['high'] < df['low']) |
                (df['high'] < df['open']) |
                (df['high'] < df['close']) |
                (df['low'] > df['open']) |
                (df['low'] > df['close'])
            )

            if invalid_ohlc.any():
                logger.error(f"Found {invalid_ohlc.sum()} invalid OHLC relationships")
                return False

            # Check for extreme price movements (>50% in one period)
            price_changes = df['close'].pct_change().abs()
            extreme_moves = price_changes > 0.5

            if extreme_moves.any():
                logger.warning(f"Found {extreme_moves.sum()} extreme price movements")

            # Check data freshness (last data point should be recent)
            if isinstance(df.index, pd.DatetimeIndex):
                last_update = df.index[-1]
                staleness = pd.Timestamp.now() - last_update
                if staleness > timedelta(hours=8):
                    logger.warning(f"Data is stale by {staleness}")

            return True

        except Exception as e:
            logger.error(f"Data validation error: {e}")
            return False


# In[6]:


class RiskManager:
    """Comprehensive risk management system"""

    def __init__(self, config: TradingConfig):
        self.config = config
        self.daily_pnl = 0
        self.total_pnl = 0
        self.max_equity = config.initial_capital
        self.positions = {}
        self.trade_history = []

    def calculate_position_size(self, signal: float, current_price: float, 
                              portfolio_value: float) -> float:
        """Calculate safe position size based on risk parameters"""
        try:
            # Base position size from signal strength
            base_size = abs(signal) * self.config.max_position_size * portfolio_value

            # Apply minimum trade size
            if base_size < self.config.min_trade_size:
                return 0

            # Check daily drawdown limit
            if self.daily_pnl < -self.config.max_daily_drawdown * portfolio_value:
                logger.warning("Daily drawdown limit reached, blocking trades")
                return 0

            # Check total drawdown limit
            current_drawdown = (self.max_equity - portfolio_value) / self.max_equity
            if current_drawdown > self.config.max_total_drawdown:
                logger.error("Maximum drawdown exceeded, blocking trades")
                return 0

            # Position sizing based on volatility
            volatility_multiplier = self._calculate_volatility_multiplier(current_price)
            adjusted_size = base_size * volatility_multiplier

            return min(adjusted_size, self.config.max_position_size * portfolio_value)

        except Exception as e:
            logger.error(f"Position sizing error: {e}")
            return 0

    def _calculate_volatility_multiplier(self, current_price: float) -> float:
        """Adjust position size based on recent volatility"""
        # Simplified volatility adjustment (in production, use proper volatility metrics)
        return 1.0

    def check_stop_loss(self, symbol: str, current_price: float, 
                       entry_price: float, position_side: str) -> bool:
        """Check if stop loss should be triggered"""
        if position_side == "long":
            loss_pct = (entry_price - current_price) / entry_price
        else:
            loss_pct = (current_price - entry_price) / entry_price

        return loss_pct > self.config.stop_loss_pct

    def check_take_profit(self, symbol: str, current_price: float, 
                         entry_price: float, position_side: str) -> bool:
        """Check if take profit should be triggered"""
        if position_side == "long":
            profit_pct = (current_price - entry_price) / entry_price
        else:
            profit_pct = (entry_price - current_price) / entry_price

        return profit_pct > self.config.take_profit_pct

    def update_pnl(self, pnl: float):
        """Update PnL tracking"""
        self.daily_pnl += pnl
        self.total_pnl += pnl

        # Reset daily PnL at midnight
        if datetime.now().hour == 0 and datetime.now().minute < 5:
            self.daily_pnl = 0


# In[7]:


class AlertManager:
    """Handles system alerts and notifications"""

    def __init__(self, config: TradingConfig):
        self.config = config

    def send_alert(self, alert_type: str, message: str, severity: str = "INFO"):
        """Send alert via email and logging"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        formatted_message = f"[{severity}] {alert_type}: {message} at {timestamp}"

        # Log the alert
        if severity == "ERROR":
            logger.error(formatted_message)
        elif severity == "WARNING":
            logger.warning(formatted_message)
        else:
            logger.info(formatted_message)

        # Send email if configured
        if self.config.alert_email and severity in ["ERROR", "WARNING"]:
            self._send_email_alert(alert_type, formatted_message)

    def _send_email_alert(self, alert_type: str, message: str):
        """Send email alert"""
        try:
            if not all([self.config.email_user, self.config.email_password]):
                return

            msg = MIMEMultipart()
            msg['From'] = self.config.email_user
            msg['To'] = self.config.alert_email
            msg['Subject'] = f"Trading System Alert: {alert_type}"

            body = message
            msg.attach(MIMEText(body, 'plain'))

            server = smtplib.SMTP(self.config.smtp_server, self.config.smtp_port)
            server.starttls()
            server.login(self.config.email_user, self.config.email_password)
            text = msg.as_string()
            server.sendmail(self.config.email_user, self.config.alert_email, text)
            server.quit()

        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")


# In[8]:


class DataManager:
    """Handles data collection, storage, and retrieval"""

    def __init__(self, config: TradingConfig):
        self.config = config
        self.session = self._create_session()
        self.db_path = Path(config.data_path) / "trading_data.db"
        self.validator = DataValidator()
        self._init_database()

    def _create_session(self) -> requests.Session:
        """Create HTTP session with retry strategy"""
        session = requests.Session()
        retry_strategy = Retry(
            total=self.config.max_retries,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS"],
            backoff_factor=1
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session

    def _init_database(self):
        """Initialize SQLite database for data storage"""
        Path(self.config.data_path).mkdir(exist_ok=True)

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS market_data (
                    timestamp TEXT PRIMARY KEY,
                    symbol TEXT,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume REAL,
                    created_at TEXT
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    symbol TEXT,
                    side TEXT,
                    size REAL,
                    price REAL,
                    pnl REAL,
                    created_at TEXT
                )
            """)

    def fetch_ohlcv(self, symbol: str, interval: str = '4h', 
                   start_time: Optional[str] = None, limit: int = 1000) -> pd.DataFrame:
        """Fetch OHLCV data from Binance API with error handling"""
        try:
            url = 'https://api.binance.com/api/v3/klines'
            params = {
                'symbol': symbol,
                'interval': interval,
                'limit': min(limit, 1000)  # Binance limit
            }

            if start_time:
                params['startTime'] = int(pd.Timestamp(start_time).timestamp() * 1000)

            all_data = []

            while len(all_data) < limit:
                response = self.session.get(
                    url, 
                    params=params, 
                    timeout=self.config.api_timeout
                )
                response.raise_for_status()

                batch = response.json()
                if not batch:
                    break

                all_data.extend(batch)

                if len(batch) < params['limit']:
                    break

                # Update start time for next batch
                params['startTime'] = batch[-1][0] + 1

                # Rate limiting
                time.sleep(0.1)

            # Convert to DataFrame
            df = pd.DataFrame(all_data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ])

            # Clean and format data
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)

            numeric_cols = ['open', 'high', 'low', 'close', 'volume']
            df[numeric_cols] = df[numeric_cols].astype(float)

            # Validate data
            if not self.validator.validate_ohlcv(df[numeric_cols]):
                raise ValueError("Data validation failed")

            # Store in database
            self._store_market_data(df, symbol)

            return df[numeric_cols]

        except Exception as e:
            logger.error(f"Error fetching OHLCV data: {e}")
            # Try to return cached data
            return self._get_cached_data(symbol, interval)

    def _store_market_data(self, df: pd.DataFrame, symbol: str):
        """Store market data in database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                for idx, row in df.iterrows():
                    conn.execute("""
                        INSERT OR REPLACE INTO market_data 
                        (timestamp, symbol, open, high, low, close, volume, created_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        idx.isoformat(), symbol, row['open'], row['high'],
                        row['low'], row['close'], row['volume'], 
                        datetime.now().isoformat()
                    ))
        except Exception as e:
            logger.error(f"Failed to store market data: {e}")

    def _get_cached_data(self, symbol: str, interval: str) -> pd.DataFrame:
        """Retrieve cached data from database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                df = pd.read_sql("""
                    SELECT timestamp, open, high, low, close, volume
                    FROM market_data 
                    WHERE symbol = ?
                    ORDER BY timestamp DESC
                    LIMIT 1000
                """, conn, params=(symbol,))

                if not df.empty:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    df.set_index('timestamp', inplace=True)
                    return df

        except Exception as e:
            logger.error(f"Failed to retrieve cached data: {e}")

        return pd.DataFrame()


# In[9]:


class FeatureEngineer:
    """Advanced feature engineering for trading signals"""

    def __init__(self):
        self.required_history = 100  # Minimum bars needed for indicators

    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute comprehensive technical indicators"""
        try:
            df = df.copy()

            # Basic price features
            df['returns'] = df['close'].pct_change()
            df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
            df['price_range'] = (df['high'] - df['low']) / df['close']

            # Moving averages and trends
            for period in [5, 10, 20, 50]:
                df[f'sma_{period}'] = df['close'].rolling(period).mean()
                df[f'ema_{period}'] = df['close'].ewm(span=period).mean()
                df[f'price_to_sma_{period}'] = df['close'] / df[f'sma_{period}'] - 1

            # Volatility indicators
            df['volatility_20'] = df['returns'].rolling(20).std()
            df['atr_14'] = ta.volatility.AverageTrueRange(
                df['high'], df['low'], df['close'], window=14
            ).average_true_range()

            # Momentum indicators
            df['rsi_14'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
            df['rsi_7'] = ta.momentum.RSIIndicator(df['close'], window=7).rsi()
            df['macd'] = ta.trend.MACD(df['close']).macd()
            df['macd_signal'] = ta.trend.MACD(df['close']).macd_signal()
            df['macd_diff'] = df['macd'] - df['macd_signal']

            # Bollinger Bands
            bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
            df['bb_upper'] = bb.bollinger_hband()
            df['bb_middle'] = bb.bollinger_mavg()
            df['bb_lower'] = bb.bollinger_lband()
            df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']

            # Volume indicators
            df['volume_sma_20'] = df['volume'].rolling(20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma_20']
            df['obv'] = ta.volume.OnBalanceVolumeIndicator(df['close'], df['volume']).on_balance_volume()

            # Support and resistance levels
            df['support'] = df['low'].rolling(20).min()
            df['resistance'] = df['high'].rolling(20).max()
            df['support_distance'] = (df['close'] - df['support']) / df['close']
            df['resistance_distance'] = (df['resistance'] - df['close']) / df['close']

            # Candlestick patterns (simplified)
            df['body_size'] = abs(df['close'] - df['open']) / df['close']
            df['upper_shadow'] = (df['high'] - np.maximum(df['open'], df['close'])) / df['close']
            df['lower_shadow'] = (np.minimum(df['open'], df['close']) - df['low']) / df['close']

            # Time-based features
            if isinstance(df.index, pd.DatetimeIndex):
                df['hour'] = df.index.hour
                df['day_of_week'] = df.index.dayofweek
                df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

            # Market regime indicators
            df['trend_strength'] = abs(df['close'].rolling(20).apply(
                lambda x: np.corrcoef(np.arange(len(x)), x)[0, 1], raw=False
            ))

            # Remove rows with insufficient data
            df = df.dropna()

            return df

        except Exception as e:
            logger.error(f"Feature engineering error: {e}")
            return df.dropna()

    def create_labels(self, df: pd.DataFrame, horizon: int = 8, 
                     thresholds: List[float] = [0.02, 0.05]) -> pd.DataFrame:
        """Create multi-class labels for future returns"""
        try:
            df = df.copy()

            # Calculate forward returns
            future_returns = df['close'].shift(-horizon) / df['close'] - 1

            # Create labels based on thresholds
            conditions = [
                future_returns > thresholds[1],      # Strong bullish
                future_returns > thresholds[0],      # Mild bullish  
                future_returns.abs() <= thresholds[0],  # Neutral
                future_returns < -thresholds[0],     # Mild bearish
                future_returns < -thresholds[1]      # Strong bearish
            ]

            labels = [4, 3, 2, 1, 0]  # 4=StrongBull, 3=MildBull, 2=Neutral, 1=MildBear, 0=StrongBear
            df['label'] = np.select(conditions, labels, default=2)
            df['future_return'] = future_returns

            return df.dropna()

        except Exception as e:
            logger.error(f"Label creation error: {e}")
            return df


# In[10]:


class MLModel:
    """Enhanced ML model with proper validation and monitoring"""

    def __init__(self, config: TradingConfig):
        self.config = config
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.model_path = Path(config.models_path)
        self.model_path.mkdir(exist_ok=True)

        # Performance tracking
        self.training_history = []
        self.prediction_log = []

    def prepare_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Prepare data for training with proper preprocessing"""
        try:
            # Separate features and labels
            label_col = 'label'
            feature_cols = [col for col in df.columns if col not in [label_col, 'future_return']]

            X = df[feature_cols].values
            y = df[label_col].values

            # Handle infinite and extreme values
            X = np.nan_to_num(X, nan=0, posinf=1e6, neginf=-1e6)

            # Store feature columns for later use
            self.feature_columns = feature_cols

            return X, y, feature_cols

        except Exception as e:
            logger.error(f"Data preparation error: {e}")
            raise

    def build_model(self, input_dim: int, num_classes: int = 5) -> Sequential:
        """Build enhanced neural network model"""
        model = Sequential([
            Dense(256, input_dim=input_dim),
            BatchNormalization(),
            LeakyReLU(0.1),
            Dropout(0.3),

            Dense(128),
            BatchNormalization(),
            LeakyReLU(0.1),
            Dropout(0.2),

            Dense(64),
            BatchNormalization(),
            LeakyReLU(0.1),
            Dropout(0.1),

            Dense(32),
            LeakyReLU(0.1),

            Dense(num_classes, activation='softmax')
        ])

        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        return model

    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: np.ndarray, y_val: np.ndarray) -> Dict:
        """Train model with comprehensive validation"""
        try:
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_val_scaled = self.scaler.transform(X_val)

            # Handle class imbalance
            rus = RandomUnderSampler(random_state=SEED)
            X_train_balanced, y_train_balanced = rus.fit_resample(X_train_scaled, y_train)

            # Build and train model
            self.model = self.build_model(X_train_balanced.shape[1])

            callbacks = [
                EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    restore_best_weights=True,
                    verbose=1
                ),
                ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=5,
                    min_lr=1e-6,
                    verbose=1
                )
            ]

            history = self.model.fit(
                X_train_balanced, y_train_balanced,
                validation_data=(X_val_scaled, y_val),
                epochs=100,
                batch_size=64,
                callbacks=callbacks,
                verbose=1
            )

            # Evaluate model
            val_predictions = np.argmax(self.model.predict(X_val_scaled), axis=1)
            val_accuracy = accuracy_score(y_val, val_predictions)

            # Log training results
            training_result = {
                'timestamp': datetime.now().isoformat(),
                'val_accuracy': val_accuracy,
                'training_samples': len(X_train_balanced),
                'validation_samples': len(X_val),
                'epochs_trained': len(history.history['loss'])
            }

            self.training_history.append(training_result)

            # Save model
            self.save_model()

            logger.info(f"Model trained successfully. Validation accuracy: {val_accuracy:.4f}")

            return training_result

        except Exception as e:
            logger.error(f"Model training error: {e}")
            raise

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions with confidence scores"""
        try:
            if self.model is None:
                raise ValueError("Model not trained or loaded")

            X_scaled = self.scaler.transform(X)
            probabilities = self.model.predict(X_scaled)
            predictions = np.argmax(probabilities, axis=1)
            confidence = np.max(probabilities, axis=1)

            # Log predictions for monitoring
            self.prediction_log.append({
                'timestamp': datetime.now().isoformat(),
                'predictions': predictions.tolist(),
                'confidence': confidence.tolist()
            })

            # Keep only recent predictions
            if len(self.prediction_log) > 1000:
                self.prediction_log = self.prediction_log[-1000:]

            return predictions, confidence

        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return np.array([2]), np.array([0.2])  # Default neutral prediction

    def save_model(self):
        """Save model and scaler"""
        try:
            model_file = self.model_path / f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.h5"
            scaler_file = self.model_path / f"scaler_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"

            self.model.save(model_file)

            with open(scaler_file, 'wb') as f:
                pickle.dump({
                    'scaler': self.scaler,
                    'feature_columns': self.feature_columns,
                    'training_history': self.training_history[-5:]  # Keep last 5 training sessions
                }, f)

            logger.info(f"Model saved to {model_file}")

        except Exception as e:
            logger.error(f"Failed to save model: {e}")

    def load_model(self, model_file: str = None, scaler_file: str = None):
        """Load saved model and scaler"""
        try:
            if model_file is None:
                # Find most recent model
                model_files = list(self.model_path.glob("model_*.h5"))
                if not model_files:
                    raise FileNotFoundError("No saved models found")
                model_file = max(model_files, key=os.path.getctime)

            if scaler_file is None:
                # Find corresponding scaler
                timestamp = model_file.stem.split('_', 1)[1]
                scaler_file = self.model_path / f"scaler_{timestamp}.pkl"

            self.model = load_model(model_file)

            with open(scaler_file, 'rb') as f:
                saved_data = pickle.load(f)
                self.scaler = saved_data['scaler']
                self.feature_columns = saved_data['feature_columns']
                if 'training_history' in saved_data:
                    self.training_history.extend(saved_data['training_history'])

            logger.info(f"Model loaded from {model_file}")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise


# In[11]:


class TradingStrategy:
    """Main trading strategy implementation"""

    def __init__(self, config: TradingConfig):
        self.config = config
        self.data_manager = DataManager(config)
        self.feature_engineer = FeatureEngineer()
        self.model = MLModel(config)
        self.risk_manager = RiskManager(config)
        self.alert_manager = AlertManager(config)

        # Portfolio tracking
        self.portfolio_value = config.initial_capital
        self.positions = {}
        self.trade_log = []

        # Model performance monitoring
        self.model_performance = []
        self.last_retrain_date = None

        # Strategy state
        self.is_running = False
        self.last_update = None

    def initialize(self):
        """Initialize the trading strategy"""
        try:
            logger.info("Initializing trading strategy...")

            # Try to load existing model
            try:
                self.model.load_model()
                logger.info("Loaded existing model")
            except:
                logger.info("No existing model found, will train new model")
                self.retrain_model()

            # Initialize risk manager
            self.risk_manager.max_equity = self.portfolio_value

            logger.info("Trading strategy initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize strategy: {e}")
            raise

    def retrain_model(self, force_retrain: bool = False):
        """Retrain the ML model with latest data"""
        try:
            # Check if retraining is needed
            if not force_retrain and self.last_retrain_date:
                days_since_retrain = (datetime.now() - self.last_retrain_date).days
                if days_since_retrain < self.config.model_retrain_days:
                    logger.info(f"Model retrain not needed (last retrain: {days_since_retrain} days ago)")
                    return

            logger.info("Starting model retraining...")

            # Fetch training data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=self.config.lookback_days)

            df = self.data_manager.fetch_ohlcv(
                self.config.symbol,
                self.config.interval,
                start_date.strftime('%Y-%m-%d'),
                limit=2000
            )

            if len(df) < self.config.min_samples_for_training:
                raise ValueError(f"Insufficient data for training: {len(df)} samples")

            # Feature engineering
            df_features = self.feature_engineer.compute_features(df)
            df_labeled = self.feature_engineer.create_labels(df_features)

            # Prepare data
            X, y, feature_cols = self.model.prepare_data(df_labeled)

            # Time-based train/validation split
            split_idx = int(len(X) * (1 - self.config.validation_split))
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]

            # Train model
            training_result = self.model.train(X_train, y_train, X_val, y_val)

            self.last_retrain_date = datetime.now()

            # Send success alert
            self.alert_manager.send_alert(
                "MODEL_RETRAIN",
                f"Model retrained successfully. Validation accuracy: {training_result['val_accuracy']:.4f}",
                "INFO"
            )

            logger.info("Model retraining completed successfully")

        except Exception as e:
            error_msg = f"Model retraining failed: {e}"
            logger.error(error_msg)
            self.alert_manager.send_alert("MODEL_RETRAIN_FAILED", error_msg, "ERROR")
            raise

    def generate_signal(self) -> Tuple[float, float, Dict]:
        """Generate trading signal with confidence and metadata"""
        try:
            # Fetch latest data
            df = self.data_manager.fetch_ohlcv(
                self.config.symbol,
                self.config.interval,
                limit=200
            )

            # Feature engineering
            df_features = self.feature_engineer.compute_features(df)

            if len(df_features) == 0:
                return 0.0, 0.0, {"error": "No features computed"}

            # Get latest features
            latest_features = df_features.iloc[-1:][self.model.feature_columns].values

            # Make prediction
            predictions, confidence = self.model.predict(latest_features)

            # Convert prediction to signal
            # 0=StrongBear, 1=MildBear, 2=Neutral, 3=MildBull, 4=StrongBull
            signal_mapping = {0: -1.0, 1: -0.5, 2: 0.0, 3: 0.5, 4: 1.0}
            signal = signal_mapping[predictions[0]]

            # Adjust signal based on confidence
            adjusted_signal = signal * confidence[0]

            # Metadata
            metadata = {
                "prediction": int(predictions[0]),
                "confidence": float(confidence[0]),
                "raw_signal": signal,
                "adjusted_signal": adjusted_signal,
                "current_price": float(df_features['close'].iloc[-1]),
                "timestamp": df_features.index[-1].isoformat()
            }

            return adjusted_signal, confidence[0], metadata

        except Exception as e:
            logger.error(f"Signal generation error: {e}")
            return 0.0, 0.0, {"error": str(e)}

    def execute_trade(self, signal: float, confidence: float, metadata: Dict) -> Optional[Dict]:
        """Execute trade based on signal"""
        try:
            current_price = metadata.get("current_price", 0)
            if current_price == 0:
                return None

            # Check if signal is strong enough
            min_confidence = 0.6
            if confidence < min_confidence:
                logger.info(f"Signal confidence too low: {confidence:.3f} < {min_confidence}")
                return None

            # Calculate position size
            position_size = self.risk_manager.calculate_position_size(
                signal, current_price, self.portfolio_value
            )

            if position_size == 0:
                logger.info("Risk manager blocked trade")
                return None

            # Determine trade side
            side = "long" if signal > 0 else "short"

            # Simulate trade execution (in production, integrate with exchange API)
            trade = {
                "timestamp": datetime.now().isoformat(),
                "symbol": self.config.symbol,
                "side": side,
                "size": position_size,
                "price": current_price,
                "signal": signal,
                "confidence": confidence,
                "status": "executed"
            }

            # Update positions
            self.positions[self.config.symbol] = {
                "side": side,
                "size": position_size,
                "entry_price": current_price,
                "entry_time": datetime.now(),
                "stop_loss": current_price * (1 - self.config.stop_loss_pct) if side == "long" 
                           else current_price * (1 + self.config.stop_loss_pct),
                "take_profit": current_price * (1 + self.config.take_profit_pct) if side == "long"
                             else current_price * (1 - self.config.take_profit_pct)
            }

            # Log trade
            self.trade_log.append(trade)

            # Store in database
            self._store_trade(trade)

            logger.info(f"Executed {side} trade: {position_size:.2f} @ {current_price:.2f}")

            # Send alert
            self.alert_manager.send_alert(
                "TRADE_EXECUTED",
                f"Executed {side} trade for {self.config.symbol}: {position_size:.2f} @ {current_price:.2f}",
                "INFO"
            )

            return trade

        except Exception as e:
            logger.error(f"Trade execution error: {e}")
            self.alert_manager.send_alert("TRADE_EXECUTION_FAILED", str(e), "ERROR")
            return None

    def check_exit_conditions(self) -> List[Dict]:
        """Check if any positions should be closed"""
        exits = []

        try:
            # Get current price
            df = self.data_manager.fetch_ohlcv(self.config.symbol, limit=1)
            if df.empty:
                return exits

            current_price = df['close'].iloc[-1]

            for symbol, position in self.positions.items():
                should_exit = False
                exit_reason = ""

                # Check stop loss
                if self.risk_manager.check_stop_loss(
                    symbol, current_price, position['entry_price'], position['side']
                ):
                    should_exit = True
                    exit_reason = "stop_loss"

                # Check take profit
                elif self.risk_manager.check_take_profit(
                    symbol, current_price, position['entry_price'], position['side']
                ):
                    should_exit = True
                    exit_reason = "take_profit"

                # Check time-based exit (optional)
                position_age = datetime.now() - position['entry_time']
                if position_age > timedelta(days=7):  # Close positions older than 1 week
                    should_exit = True
                    exit_reason = "time_based"

                if should_exit:
                    exit_trade = self._close_position(symbol, current_price, exit_reason)
                    if exit_trade:
                        exits.append(exit_trade)

        except Exception as e:
            logger.error(f"Error checking exit conditions: {e}")

        return exits

    def _close_position(self, symbol: str, current_price: float, reason: str) -> Optional[Dict]:
        """Close a position"""
        try:
            if symbol not in self.positions:
                return None

            position = self.positions[symbol]

            # Calculate PnL
            if position['side'] == "long":
                pnl = (current_price - position['entry_price']) * position['size'] / position['entry_price']
            else:
                pnl = (position['entry_price'] - current_price) * position['size'] / position['entry_price']

            # Account for trading costs
            trading_cost = position['size'] * (self.config.trading_fee + self.config.slippage)
            net_pnl = pnl - trading_cost

            # Create exit trade record
            exit_trade = {
                "timestamp": datetime.now().isoformat(),
                "symbol": symbol,
                "side": "sell" if position['side'] == "long" else "buy",
                "size": position['size'],
                "price": current_price,
                "pnl": net_pnl,
                "reason": reason,
                "status": "executed"
            }

            # Update portfolio
            self.portfolio_value += net_pnl
            self.risk_manager.update_pnl(net_pnl)

            # Update max equity for drawdown calculation
            if self.portfolio_value > self.risk_manager.max_equity:
                self.risk_manager.max_equity = self.portfolio_value

            # Remove position
            del self.positions[symbol]

            # Log trade
            self.trade_log.append(exit_trade)
            self._store_trade(exit_trade)

            logger.info(f"Closed {position['side']} position: PnL = {net_pnl:.2f} ({reason})")

            # Send alert for significant PnL
            if abs(net_pnl) > self.config.initial_capital * 0.01:  # > 1% of capital
                self.alert_manager.send_alert(
                    "POSITION_CLOSED",
                    f"Closed {position['side']} position in {symbol}: PnL = {net_pnl:.2f} ({reason})",
                    "INFO"
                )

            return exit_trade

        except Exception as e:
            logger.error(f"Error closing position: {e}")
            return None

    def _store_trade(self, trade: Dict):
        """Store trade in database"""
        try:
            with sqlite3.connect(self.data_manager.db_path) as conn:
                conn.execute("""
                    INSERT INTO trades (timestamp, symbol, side, size, price, pnl, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    trade['timestamp'], trade['symbol'], trade['side'],
                    trade['size'], trade['price'], trade.get('pnl', 0),
                    datetime.now().isoformat()
                ))
        except Exception as e:
            logger.error(f"Failed to store trade: {e}")

    def run_strategy_cycle(self):
        """Run one complete strategy cycle"""
        try:
            logger.info("Running strategy cycle...")

            # Check if model needs retraining
            self.retrain_model()

            # Generate trading signal
            signal, confidence, metadata = self.generate_signal()

            logger.info(f"Generated signal: {signal:.3f} (confidence: {confidence:.3f})")

            # Check exit conditions first
            exits = self.check_exit_conditions()

            # Execute new trades if signal is strong enough
            if abs(signal) > 0.1 and len(self.positions) == 0:  # Only trade if no existing positions
                trade = self.execute_trade(signal, confidence, metadata)

            # Update last run time
            self.last_update = datetime.now()

            # Monitor model performance
            self._monitor_model_performance(metadata)

        except Exception as e:
            logger.error(f"Strategy cycle error: {e}")
            self.alert_manager.send_alert("STRATEGY_CYCLE_ERROR", str(e), "ERROR")

    def _monitor_model_performance(self, metadata: Dict):
        """Monitor model performance and detect degradation"""
        try:
            # Store prediction for analysis
            self.model_performance.append({
                "timestamp": datetime.now().isoformat(),
                "confidence": metadata.get("confidence", 0),
                "prediction": metadata.get("prediction", 2)
            })

            # Keep only recent performance data
            if len(self.model_performance) > 500:
                self.model_performance = self.model_performance[-500:]

            # Check for performance degradation
            if len(self.model_performance) >= 50:
                recent_confidence = [p["confidence"] for p in self.model_performance[-50:]]
                avg_confidence = np.mean(recent_confidence)

                if avg_confidence < 0.4:  # Low confidence threshold
                    self.alert_manager.send_alert(
                        "MODEL_PERFORMANCE_WARNING",
                        f"Model confidence declining: {avg_confidence:.3f}",
                        "WARNING"
                    )

        except Exception as e:
            logger.error(f"Model monitoring error: {e}")

    def get_portfolio_summary(self) -> Dict:
        """Get current portfolio summary"""
        try:
            total_pnl = self.portfolio_value - self.config.initial_capital
            total_return = total_pnl / self.config.initial_capital

            current_equity = self.portfolio_value
            if hasattr(self.risk_manager, 'max_equity'):
                current_drawdown = (self.risk_manager.max_equity - current_equity) / self.risk_manager.max_equity
            else:
                current_drawdown = 0

            summary = {
                "timestamp": datetime.now().isoformat(),
                "portfolio_value": self.portfolio_value,
                "total_pnl": total_pnl,
                "total_return": total_return,
                "current_drawdown": current_drawdown,
                "positions": len(self.positions),
                "total_trades": len(self.trade_log),
                "is_running": self.is_running,
                "last_update": self.last_update.isoformat() if self.last_update else None
            }

            return summary

        except Exception as e:
            logger.error(f"Error generating portfolio summary: {e}")
            return {}

    def start_automated_trading(self, run_interval_minutes: int = 240):  # 4 hours default
        """Start automated trading loop"""
        try:
            self.is_running = True
            logger.info(f"Starting automated trading (interval: {run_interval_minutes} minutes)")

            def trading_loop():
                while self.is_running:
                    try:
                        self.run_strategy_cycle()
                        time.sleep(run_interval_minutes * 60)  # Convert to seconds
                    except Exception as e:
                        logger.error(f"Trading loop error: {e}")
                        self.alert_manager.send_alert("TRADING_LOOP_ERROR", str(e), "ERROR")
                        time.sleep(300)  # Wait 5 minutes before retrying

            # Start trading loop in separate thread
            trading_thread = threading.Thread(target=trading_loop, daemon=True)
            trading_thread.start()

            return trading_thread

        except Exception as e:
            logger.error(f"Failed to start automated trading: {e}")
            self.is_running = False
            raise

    def stop_automated_trading(self):
        """Stop automated trading"""
        self.is_running = False
        logger.info("Automated trading stopped")


# In[12]:


class TradingSystemManager:
    """Main system manager for the trading system"""

    def __init__(self, config_file: str = "trading_config.json"):
        self.config = self._load_config(config_file)
        self.strategy = TradingStrategy(self.config)

    def _load_config(self, config_file: str) -> TradingConfig:
        """Load configuration from file or create default"""
        try:
            if Path(config_file).exists():
                with open(config_file, 'r') as f:
                    config_dict = json.load(f)
                return TradingConfig(**config_dict)
            else:
                # Create default config
                config = TradingConfig()
                self._save_config(config, config_file)
                return config
        except Exception as e:
            logger.warning(f"Failed to load config: {e}, using defaults")
            return TradingConfig()

    def _save_config(self, config: TradingConfig, config_file: str):
        """Save configuration to file"""
        try:
            with open(config_file, 'w') as f:
                json.dump(asdict(config), f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save config: {e}")

    def initialize_system(self):
        """Initialize the complete trading system"""
        try:
            logger.info("Initializing trading system...")

            # Create necessary directories
            for path in [self.config.data_path, self.config.models_path, self.config.logs_path]:
                Path(path).mkdir(exist_ok=True)

            # Initialize strategy
            self.strategy.initialize()

            logger.info("Trading system initialized successfully")

        except Exception as e:
            logger.error(f"System initialization failed: {e}")
            raise

    def run_backtest(self, start_date: str, end_date: str, debug: bool = True) -> Dict:
        """Run comprehensive backtest with debug output"""
        try:
            logger.info(f"Running backtest from {start_date} to {end_date}")

            # Fetch historical data
            df = self.strategy.data_manager.fetch_ohlcv(
                self.config.symbol,
                self.config.interval,
                start_date,
                limit=5000
            )

            if df.empty:
                logger.error("No data fetched for backtest")
                return {}

            if debug:
                print(f"📊 Data fetched: {len(df)} rows from {df.index[0]} to {df.index[-1]}")

            # Feature engineering
            df_features = self.strategy.feature_engineer.compute_features(df)
            df_labeled = self.strategy.feature_engineer.create_labels(df_features)

            if debug:
                print(f"📈 Features computed: {len(df_features.columns)} columns")
                print(f"🏷️  Labels created: {df_labeled['label'].value_counts().to_dict()}")

            # Split data for training and testing
            split_date = pd.Timestamp(start_date) + (pd.Timestamp(end_date) - pd.Timestamp(start_date)) * 0.7

            train_data = df_labeled[df_labeled.index <= split_date]
            test_data = df_labeled[df_labeled.index > split_date]

            if debug:
                print(f"📅 Data split: Train={len(train_data)}, Test={len(test_data)} at {split_date}")

            # Ensure we have enough data
            if len(train_data) < 100 or len(test_data) < 50:
                logger.error("Insufficient data for backtesting")
                return {}

            # Train model on historical data
            try:
                X_train, y_train, feature_cols = self.strategy.model.prepare_data(train_data)
                X_train_scaled = self.strategy.model.scaler.fit_transform(X_train)

                # Simple train/val split for backtest
                val_split = int(len(X_train_scaled) * 0.8)
                self.strategy.model.train(
                    X_train_scaled[:val_split], y_train[:val_split],
                    X_train_scaled[val_split:], y_train[val_split:]
                )

                if debug:
                    print(f"🤖 Model trained on {len(X_train)} samples with {len(feature_cols)} features")

            except Exception as e:
                logger.error(f"Model training failed: {e}")
                return {}

            # Run backtest simulation
            backtest_results = self._simulate_backtest(test_data, debug=debug)

            logger.info("Backtest completed successfully")
            return backtest_results

        except Exception as e:
            logger.error(f"Backtest failed: {e}")
            import traceback
            traceback.print_exc()
            return {}

    def _simulate_backtest(self, test_data: pd.DataFrame, debug: bool = True) -> Dict:
        """Simulate trading on historical data with detailed diagnostics"""
        try:
            # Initialize portfolio
            portfolio_value = self.config.initial_capital
            cash = self.config.initial_capital
            positions = {}
            trades = []
            equity_curve = []

            # Enhanced debug counters and tracking
            signal_count = 0
            entry_attempts = 0
            successful_entries = 0
            exit_count = 0

            # Diagnostic tracking
            signal_stats = {
                'total_signals': 0,
                'strong_signals': 0,  # abs(signal) > 0.3
                'high_confidence': 0,  # confidence > 0.6
                'both_conditions': 0,  # both strong signal and high confidence
                'no_position': 0,     # times when no position was held
                'sufficient_size': 0, # times when position size was sufficient
                'blocked_reasons': {
                    'weak_signal': 0,
                    'low_confidence': 0,
                    'position_exists': 0,
                    'insufficient_size': 0,
                    'insufficient_cash': 0,
                    'feature_issues': 0
                }
            }

            feature_issues = 0

            if debug:
                print(f"\n🔄 Starting backtest simulation with ${portfolio_value:,.2f}")
                print(f"📋 Configuration check:")
                print(f"   Max position size: {self.config.max_position_size:.1%}")
                print(f"   Min trade size: ${self.config.min_trade_size:,.2f}")
                print(f"   Stop loss: {self.config.stop_loss_pct:.1%}")
                print(f"   Take profit: {self.config.take_profit_pct:.1%}")
                print(f"   Trading fee: {self.config.trading_fee:.4f}")
                print(f"   Slippage: {self.config.slippage:.4f}")
                print("=" * 60)

            for i, (timestamp, row) in enumerate(test_data.iterrows()):
                try:
                    # Feature validation
                    if not hasattr(self.strategy.model, 'feature_columns'):
                        if debug and feature_issues == 0:
                            print("❌ Model feature columns not defined")
                        feature_issues += 1
                        continue

                    # Get features for this timestamp - handle missing columns gracefully
                    available_features = [col for col in self.strategy.model.feature_columns if col in row.index]
                    if len(available_features) < len(self.strategy.model.feature_columns) * 0.8:
                        if debug and feature_issues == 0:
                            print(f"❌ Too many missing features: {len(available_features)}/{len(self.strategy.model.feature_columns)}")
                        signal_stats['blocked_reasons']['feature_issues'] += 1
                        feature_issues += 1
                        continue

                    features = row[available_features].values.reshape(1, -1)

                    # Handle NaN values
                    if np.any(np.isnan(features)):
                        if debug and feature_issues == 0:
                            print("❌ NaN values in features")
                        signal_stats['blocked_reasons']['feature_issues'] += 1
                        feature_issues += 1
                        continue

                    # Scale features
                    features_scaled = self.strategy.model.scaler.transform(features)

                    # Generate signal
                    predictions, confidence = self.strategy.model.predict(features_scaled)
                    signal_mapping = {0: -1.0, 1: -0.5, 2: 0.0, 3: 0.5, 4: 1.0}
                    raw_signal = signal_mapping.get(predictions[0], 0.0)
                    signal = raw_signal * confidence[0]

                    current_price = row['close']
                    signal_count += 1
                    signal_stats['total_signals'] += 1

                    # Detailed signal analysis
                    strong_signal = abs(signal) > 0.3
                    high_confidence = confidence[0] > 0.6
                    no_position = len(positions) == 0

                    if strong_signal:
                        signal_stats['strong_signals'] += 1
                    if high_confidence:
                        signal_stats['high_confidence'] += 1
                    if strong_signal and high_confidence:
                        signal_stats['both_conditions'] += 1
                    if no_position:
                        signal_stats['no_position'] += 1

                    # Debug first 10 signals regardless of entry
                    if debug and signal_count <= 10:
                        print(f"🔍 Signal {signal_count:3d} | {timestamp.strftime('%m-%d %H:%M')} | "
                              f"Raw: {raw_signal:6.3f} | Conf: {confidence[0]:6.3f} | "
                              f"Final: {signal:6.3f} | Price: ${current_price:8.2f}")
                        print(f"           Conditions: Strong={strong_signal} | HighConf={high_confidence} | NoPos={no_position}")

                    # Entry logic with detailed tracking
                    if strong_signal and high_confidence and no_position:
                        entry_attempts += 1

                        # Calculate position size based on available cash
                        max_position_value = min(
                            abs(signal) * self.config.max_position_size * cash,
                            self.config.max_position_size * cash,
                            cash * 0.95  # Leave some cash buffer
                        )

                        # Check minimum trade size
                        if max_position_value >= self.config.min_trade_size:
                            signal_stats['sufficient_size'] += 1

                            side = "long" if signal > 0 else "short"

                            # Calculate actual position size (number of units)
                            position_units = max_position_value / current_price
                            position_value = position_units * current_price

                            # Account for trading costs
                            trading_cost = position_value * (self.config.trading_fee + self.config.slippage)
                            total_cost = position_value + trading_cost

                            if total_cost <= cash:
                                positions[self.config.symbol] = {
                                    "side": side,
                                    "units": position_units,
                                    "value": position_value,
                                    "entry_price": current_price,
                                    "entry_time": timestamp,
                                    "trading_cost": trading_cost
                                }

                                cash -= total_cost
                                successful_entries += 1

                                trades.append({
                                    "timestamp": timestamp,
                                    "side": side,
                                    "units": position_units,
                                    "value": position_value,
                                    "price": current_price,
                                    "cost": trading_cost,
                                    "type": "entry",
                                    "signal": signal,
                                    "confidence": confidence[0]
                                })

                                if debug:
                                    print(f"✅ ENTRY  {timestamp.strftime('%Y-%m-%d %H:%M')} | {side.upper():5} | "
                                          f"${position_value:8.2f} @ ${current_price:8.2f} | "
                                          f"Signal: {signal:6.3f} | Conf: {confidence[0]:6.3f}")
                            else:
                                signal_stats['blocked_reasons']['insufficient_cash'] += 1
                                if debug and signal_stats['blocked_reasons']['insufficient_cash'] <= 5:
                                    print(f"💰 Insufficient cash: need ${total_cost:,.2f}, have ${cash:,.2f}")
                        else:
                            signal_stats['blocked_reasons']['insufficient_size'] += 1
                            if debug and signal_stats['blocked_reasons']['insufficient_size'] <= 5:
                                print(f"📏 Position too small: ${max_position_value:,.2f} < ${self.config.min_trade_size:,.2f}")
                    else:
                        # Track why entry was blocked
                        if not strong_signal:
                            signal_stats['blocked_reasons']['weak_signal'] += 1
                        if not high_confidence:
                            signal_stats['blocked_reasons']['low_confidence'] += 1
                        if not no_position:
                            signal_stats['blocked_reasons']['position_exists'] += 1

                    # Exit logic (unchanged from previous version)
                    if self.config.symbol in positions:
                        position = positions[self.config.symbol]

                        # Calculate current position value
                        current_position_value = position['units'] * current_price

                        # Calculate unrealized PnL
                        if position['side'] == "long":
                            unrealized_pnl = current_position_value - position['value']
                        else:  # short position
                            unrealized_pnl = position['value'] - current_position_value

                        # Calculate PnL percentage
                        pnl_pct = unrealized_pnl / position['value']

                        # Check exit conditions
                        should_exit = False
                        exit_reason = ""

                        if pnl_pct < -self.config.stop_loss_pct:
                            should_exit = True
                            exit_reason = "stop_loss"
                        elif pnl_pct > self.config.take_profit_pct:
                            should_exit = True
                            exit_reason = "take_profit"
                        elif abs(signal) > 0.5 and np.sign(signal) != np.sign(raw_signal):
                            should_exit = True
                            exit_reason = "signal_reversal"

                        if should_exit:
                            # Calculate exit costs
                            exit_cost = current_position_value * (self.config.trading_fee + self.config.slippage)
                            net_proceeds = current_position_value - exit_cost

                            # Calculate net PnL
                            net_pnl = unrealized_pnl - position['trading_cost'] - exit_cost

                            # Update cash
                            if position['side'] == "long":
                                cash += net_proceeds
                            else:  # short position
                                cash += position['value'] + (position['value'] - net_proceeds)

                            exit_count += 1

                            trades.append({
                                "timestamp": timestamp,
                                "side": "sell" if position['side'] == "long" else "cover",
                                "units": position['units'],
                                "value": current_position_value,
                                "price": current_price,
                                "cost": exit_cost,
                                "pnl": net_pnl,
                                "pnl_pct": pnl_pct,
                                "reason": exit_reason,
                                "type": "exit",
                                "hold_time": (timestamp - position['entry_time']).total_seconds() / 3600
                            })

                            if debug:
                                print(f"📉 EXIT   {timestamp.strftime('%Y-%m-%d %H:%M')} | {trades[-1]['side'].upper():5} | "
                                      f"${current_position_value:8.2f} @ ${current_price:8.2f} | "
                                      f"PnL: ${net_pnl:8.2f} ({pnl_pct:6.1%}) | {exit_reason}")

                            del positions[self.config.symbol]

                    # Calculate current portfolio value
                    current_portfolio_value = cash
                    if self.config.symbol in positions:
                        position = positions[self.config.symbol]
                        current_position_value = position['units'] * current_price

                        if position['side'] == "long":
                            current_portfolio_value += current_position_value
                        else:  # short position
                            unrealized_pnl = position['value'] - current_position_value
                            current_portfolio_value += position['value'] + unrealized_pnl

                    # Record equity curve
                    equity_curve.append({
                        "timestamp": timestamp,
                        "equity": current_portfolio_value,
                        "price": current_price,
                        "cash": cash,
                        "position_value": current_portfolio_value - cash
                    })

                except Exception as e:
                    if debug:
                        print(f"⚠️  Error processing row {i}: {e}")
                    continue

            # Close any remaining positions at the end
            if positions:
                for symbol, position in positions.items():
                    final_price = test_data.iloc[-1]['close']
                    current_position_value = position['units'] * final_price
                    exit_cost = current_position_value * (self.config.trading_fee + self.config.slippage)

                    if position['side'] == "long":
                        net_pnl = current_position_value - position['value'] - position['trading_cost'] - exit_cost
                        cash += current_position_value - exit_cost
                    else:
                        net_pnl = position['value'] - current_position_value - position['trading_cost'] - exit_cost
                        cash += position['value'] + (position['value'] - current_position_value) - exit_cost

                    trades.append({
                        "timestamp": test_data.index[-1],
                        "side": "sell" if position['side'] == "long" else "cover",
                        "units": position['units'],
                        "value": current_position_value,
                        "price": final_price,
                        "cost": exit_cost,
                        "pnl": net_pnl,
                        "reason": "end_of_backtest",
                        "type": "exit"
                    })

            if debug:
                print("=" * 60)
                print(f"📊 Detailed Signal Analysis:")
                print(f"   Total signals: {signal_stats['total_signals']}")
                print(f"   Strong signals (>0.3): {signal_stats['strong_signals']} ({signal_stats['strong_signals']/signal_stats['total_signals']*100:.1f}%)")
                print(f"   High confidence (>0.6): {signal_stats['high_confidence']} ({signal_stats['high_confidence']/signal_stats['total_signals']*100:.1f}%)")
                print(f"   Both conditions met: {signal_stats['both_conditions']} ({signal_stats['both_conditions']/signal_stats['total_signals']*100:.1f}%)")
                print(f"   Times with no position: {signal_stats['no_position']} ({signal_stats['no_position']/signal_stats['total_signals']*100:.1f}%)")
                print(f"   Sufficient position size: {signal_stats['sufficient_size']}")

                print(f"\n🚫 Blocking reasons:")
                for reason, count in signal_stats['blocked_reasons'].items():
                    if count > 0:
                        print(f"   {reason.replace('_', ' ').title()}: {count}")

                print(f"\n📈 Execution Summary:")
                print(f"   Entry attempts: {entry_attempts}")
                print(f"   Successful entries: {successful_entries}")
                print(f"   Exits: {exit_count}")
                print(f"   Feature issues: {feature_issues}")

            # Calculate performance metrics (unchanged)
            if not equity_curve:
                logger.error("No equity curve data generated")
                return {}

            equity_series = pd.Series([point["equity"] for point in equity_curve])
            price_series = pd.Series([point["price"] for point in equity_curve])

            final_value = equity_series.iloc[-1]
            total_return = (final_value - self.config.initial_capital) / self.config.initial_capital

            returns = equity_series.pct_change().dropna()

            if len(returns) > 0 and returns.std() > 0:
                periods_per_year = 365 * 24 if self.config.interval == '1h' else 365
                sharpe_ratio = np.sqrt(periods_per_year) * returns.mean() / returns.std()
            else:
                sharpe_ratio = 0

            rolling_max = equity_series.expanding().max()
            drawdown = (equity_series - rolling_max) / rolling_max
            max_drawdown = drawdown.min()

            buy_hold_return = (price_series.iloc[-1] - price_series.iloc[0]) / price_series.iloc[0]

            entry_trades = [t for t in trades if t["type"] == "entry"]
            exit_trades = [t for t in trades if t["type"] == "exit" and "pnl" in t]

            winning_trades = len([t for t in exit_trades if t["pnl"] > 0])
            losing_trades = len([t for t in exit_trades if t["pnl"] <= 0])

            if exit_trades:
                avg_pnl = np.mean([t["pnl"] for t in exit_trades])
                avg_winning_pnl = np.mean([t["pnl"] for t in exit_trades if t["pnl"] > 0]) if winning_trades > 0 else 0
                avg_losing_pnl = np.mean([t["pnl"] for t in exit_trades if t["pnl"] <= 0]) if losing_trades > 0 else 0
            else:
                avg_pnl = avg_winning_pnl = avg_losing_pnl = 0

            results = {
                "start_date": test_data.index[0].isoformat(),
                "end_date": test_data.index[-1].isoformat(),
                "initial_capital": self.config.initial_capital,
                "final_value": final_value,
                "final_cash": cash,
                "total_return": total_return,
                "buy_hold_return": buy_hold_return,
                "alpha": total_return - buy_hold_return,
                "sharpe_ratio": sharpe_ratio,
                "max_drawdown": max_drawdown,
                "total_trades": len(entry_trades),
                "completed_trades": len(exit_trades),
                "winning_trades": winning_trades,
                "losing_trades": losing_trades,
                "win_rate": winning_trades / len(exit_trades) if exit_trades else 0,
                "avg_trade_pnl": avg_pnl,
                "avg_winning_trade": avg_winning_pnl,
                "avg_losing_trade": avg_losing_pnl,
                "equity_curve": equity_curve,
                "trades": trades,
                "debug_stats": {
                    "signals_generated": signal_count,
                    "entry_attempts": entry_attempts,
                    "successful_entries": successful_entries,
                    "exits": exit_count,
                    "feature_issues": feature_issues
                },
                "signal_analysis": signal_stats
            }

            return results

        except Exception as e:
            logger.error(f"Backtest simulation error: {e}")
            import traceback
            traceback.print_exc()
            return {}

    def generate_report(self) -> str:
        """Generate comprehensive system report"""
        try:
            report = []
            report.append("=" * 60)
            report.append("CRYPTO TRADING SYSTEM REPORT")
            report.append("=" * 60)
            report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            report.append("")

            # Portfolio summary
            portfolio = self.strategy.get_portfolio_summary()
            report.append("PORTFOLIO SUMMARY:")
            report.append(f"  Current Value: ${portfolio.get('portfolio_value', 0):,.2f}")
            report.append(f"  Total P&L: ${portfolio.get('total_pnl', 0):,.2f}")
            report.append(f"  Total Return: {portfolio.get('total_return', 0):.2%}")
            report.append(f"  Current Drawdown: {portfolio.get('current_drawdown', 0):.2%}")
            report.append(f"  Open Positions: {portfolio.get('positions', 0)}")
            report.append(f"  Total Trades: {portfolio.get('total_trades', 0)}")
            report.append("")

            # Model performance
            if hasattr(self.strategy.model, 'training_history') and self.strategy.model.training_history:
                latest_training = self.strategy.model.training_history[-1]
                report.append("MODEL PERFORMANCE:")
                report.append(f"  Last Retrain: {latest_training.get('timestamp', 'N/A')}")
                report.append(f"  Validation Accuracy: {latest_training.get('val_accuracy', 0):.4f}")
                report.append(f"  Training Samples: {latest_training.get('training_samples', 0):,}")
                report.append("")

            # Recent trades
            if hasattr(self.strategy, 'trade_log') and self.strategy.trade_log:
                report.append("RECENT TRADES:")
                for trade in self.strategy.trade_log[-5:]:
                    timestamp = trade.get('timestamp', '')[:19]  # Remove microseconds
                    side = trade.get('side', '')
                    size = trade.get('size', 0)
                    price = trade.get('price', 0)
                    pnl = trade.get('pnl', 0)
                    report.append(f"  {timestamp} | {side.upper():5} | ${size:8.2f} @ ${price:8.2f} | P&L: ${pnl:8.2f}")
                report.append("")

            # System status
            report.append("SYSTEM STATUS:")
            report.append(f"  Running: {getattr(self.strategy, 'is_running', False)}")
            report.append(f"  Last Update: {portfolio.get('last_update', 'N/A')}")
            report.append(f"  Symbol: {self.config.symbol}")
            report.append(f"  Interval: {self.config.interval}")
            report.append("")

            return "\n".join(report)

        except Exception as e:
            logger.error(f"Report generation error: {e}")
            return f"Error generating report: {e}"

    def save_backtest_results(self, results: Dict, filename: str = None):
        """Save backtest results with trade details"""
        if not filename:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"backtest_results_{timestamp}.json"

        try:
            # Convert timestamps to strings for JSON serialization
            results_copy = results.copy()
            if 'trades' in results_copy:
                for trade in results_copy['trades']:
                    if 'timestamp' in trade:
                        trade['timestamp'] = trade['timestamp'].isoformat()

            if 'equity_curve' in results_copy:
                for point in results_copy['equity_curve']:
                    if 'timestamp' in point:
                        point['timestamp'] = point['timestamp'].isoformat()

            with open(filename, 'w') as f:
                json.dump(results_copy, f, indent=2, default=str)

            print(f"✅ Backtest results saved to {filename}")

        except Exception as e:
            print(f"❌ Failed to save backtest results: {e}")

# Additional diagnostic function to check model and config
def diagnose_trading_system(system):
    """Diagnose potential issues with the trading system"""
    print("\n🔍 SYSTEM DIAGNOSTICS")
    print("=" * 50)

    # Check config
    print("📋 Configuration:")
    print(f"   Symbol: {system.config.symbol}")
    print(f"   Initial capital: ${system.config.initial_capital:,.2f}")
    print(f"   Max position size: {system.config.max_position_size:.1%}")
    print(f"   Min trade size: ${system.config.min_trade_size:,.2f}")
    print(f"   Stop loss: {system.config.stop_loss_pct:.1%}")
    print(f"   Take profit: {system.config.take_profit_pct:.1%}")

    # Check model
    print(f"\n🤖 Model Status:")
    if hasattr(system.strategy.model, 'feature_columns'):
        print(f"   Feature columns: {len(system.strategy.model.feature_columns)}")
        print(f"   Features: {system.strategy.model.feature_columns[:5]}...")
    else:
        print("   ❌ No feature columns defined")

    if hasattr(system.strategy.model, 'scaler') and system.strategy.model.scaler:
        print(f"   Scaler: {type(system.strategy.model.scaler).__name__}")
    else:
        print("   ❌ No scaler defined")

    if hasattr(system.strategy.model, 'model') and system.strategy.model.model:
        print(f"   Model: {type(system.strategy.model.model).__name__}")
    else:
        print("   ❌ No model defined")

    # Test with sample data
    print(f"\n🧪 Quick Test:")
    try:
        # Try to fetch a small amount of data
        df = system.strategy.data_manager.fetch_ohlcv(
            system.config.symbol, 
            system.config.interval, 
            "2024-01-01", 
            limit=100
        )
        print(f"   Data fetch: ✅ {len(df)} rows")

        # Try feature engineering
        df_features = system.strategy.feature_engineer.compute_features(df)
        print(f"   Feature engineering: ✅ {len(df_features.columns)} features")

        # Try prediction on last row
        if hasattr(system.strategy.model, 'feature_columns'):
            last_row = df_features.iloc[-1]
            available_features = [col for col in system.strategy.model.feature_columns if col in last_row.index]

            if len(available_features) > 0:
                features = last_row[available_features].values.reshape(1, -1)
                if not np.any(np.isnan(features)):
                    features_scaled = system.strategy.model.scaler.transform(features)
                    predictions, confidence = system.strategy.model.predict(features_scaled)
                    print(f"   Model prediction: ✅ Pred={predictions[0]}, Conf={confidence[0]:.3f}")
                else:
                    print(f"   Model prediction: ❌ NaN values in features")
            else:
                print(f"   Model prediction: ❌ No matching features")

    except Exception as e:
        print(f"   Test failed: ❌ {e}")

    print("=" * 50)


# In[13]:


def main():
    """Main function to run the trading system"""
    try:
        # Initialize system
        print("🚀 Initializing Crypto Trading System...")
        system = TradingSystemManager()
        system.initialize_system()
        diagnose_trading_system(system)
        # Display system configuration
        print(f"📊 Trading Configuration:")
        print(f"   Symbol: {system.config.symbol}")
        print(f"   Interval: {system.config.interval}")
        print(f"   Initial Capital: ${system.config.initial_capital:,.2f}")
        print(f"   Max Position Size: {system.config.max_position_size:.1%}")
        print(f"   Stop Loss: {system.config.stop_loss_pct:.1%}")
        print(f"   Take Profit: {system.config.take_profit_pct:.1%}")
        print()

        # Menu-driven interface
        while True:
            print("=" * 60)
            print("CRYPTO TRADING SYSTEM - MAIN MENU")
            print("=" * 60)
            print("1. Run Backtest")
            print("2. Generate Single Trading Signal")
            print("3. Start Live Trading (Paper Trading)")
            print("4. View Portfolio Status")
            print("5. View System Report")
            print("6. Retrain Model")
            print("7. Run Strategy Analysis")
            print("8. Export Trade History")
            print("9. System Configuration")
            print("0. Exit")
            print("=" * 60)

            try:
                choice = input("Enter your choice (0-9): ").strip()

                if choice == '0':
                    # Graceful shutdown
                    print("🛑 Shutting down trading system...")
                    if system.strategy.is_running:
                        system.strategy.stop_automated_trading()
                        print("✅ Automated trading stopped")
                    print("👋 Goodbye!")
                    break

                elif choice == '1':
                    # Run backtest
                    print("\n📈 BACKTESTING MODULE")
                    print("-" * 40)

                    # Get date range from user
                    default_start = "2023-01-01"
                    default_end = "2023-12-31"

                    start_date = input(f"Start date (YYYY-MM-DD) [{default_start}]: ").strip() or default_start
                    end_date = input(f"End date (YYYY-MM-DD) [{default_end}]: ").strip() or default_end
                    debug = input("Enable debug output? (Y/n): ").strip().lower() != 'n'

                    print(f"🔄 Running backtest from {start_date} to {end_date}...")
                    backtest_results = system.run_backtest(start_date, end_date, debug=debug)

                    if backtest_results:
                        print("\n📊 BACKTEST RESULTS:")
                        print(f"   Period: {backtest_results.get('start_date', 'N/A')[:10]} to {backtest_results.get('end_date', 'N/A')[:10]}")
                        print(f"   Initial Capital: ${backtest_results.get('initial_capital', 0):,.2f}")
                        print(f"   Final Value: ${backtest_results.get('final_value', 0):,.2f}")
                        print(f"   Final Cash: ${backtest_results.get('final_cash', 0):,.2f}")
                        print(f"   Total Return: {backtest_results.get('total_return', 0):.2%}")
                        print(f"   Buy & Hold Return: {backtest_results.get('buy_hold_return', 0):.2%}")
                        print(f"   Alpha vs Buy & Hold: {backtest_results.get('alpha', 0):.2%}")
                        print(f"   Sharpe Ratio: {backtest_results.get('sharpe_ratio', 0):.3f}")
                        print(f"   Maximum Drawdown: {backtest_results.get('max_drawdown', 0):.2%}")
                        print(f"   Total Trades: {backtest_results.get('total_trades', 0)}")
                        print(f"   Completed Trades: {backtest_results.get('completed_trades', 0)}")
                        print(f"   Winning Trades: {backtest_results.get('winning_trades', 0)}")
                        print(f"   Losing Trades: {backtest_results.get('losing_trades', 0)}")
                        print(f"   Win Rate: {backtest_results.get('win_rate', 0):.1%}")
                        print(f"   Avg Trade P&L: ${backtest_results.get('avg_trade_pnl', 0):.2f}")
                        print(f"   Avg Winning Trade: ${backtest_results.get('avg_winning_trade', 0):.2f}")
                        print(f"   Avg Losing Trade: ${backtest_results.get('avg_losing_trade', 0):.2f}")

                        # Show recent trades
                        if 'trades' in backtest_results and backtest_results['trades']:
                            print(f"\n📋 RECENT TRADES (Last 10):")
                            recent_trades = backtest_results['trades'][-10:]
                            for trade in recent_trades:
                                timestamp = trade['timestamp'][:19] if isinstance(trade['timestamp'], str) else trade['timestamp'].strftime('%Y-%m-%d %H:%M')
                                trade_type = trade.get('type', 'unknown').upper()
                                side = trade.get('side', 'unknown').upper()
                                price = trade.get('price', 0)
                                value = trade.get('value', 0)
                                pnl = trade.get('pnl', 0) if 'pnl' in trade else 0

                                if trade.get('type') == 'entry':
                                    print(f"   {timestamp} | {trade_type:5} {side:5} | ${value:8.2f} @ ${price:8.2f}")
                                else:
                                    reason = trade.get('reason', 'unknown')
                                    print(f"   {timestamp} | {trade_type:5} {side:5} | ${value:8.2f} @ ${price:8.2f} | PnL: ${pnl:8.2f} | {reason}")

                        # Debug statistics
                        if 'debug_stats' in backtest_results:
                            debug_stats = backtest_results['debug_stats']
                            print(f"\n🔍 DEBUG STATISTICS:")
                            print(f"   Signals Generated: {debug_stats.get('signals_generated', 0)}")
                            print(f"   Entry Attempts: {debug_stats.get('entry_attempts', 0)}")
                            print(f"   Successful Entries: {debug_stats.get('successful_entries', 0)}")
                            print(f"   Exits: {debug_stats.get('exits', 0)}")

                        # Offer to save results
                        save = input("\nSave detailed results to file? (y/N): ").lower().strip()
                        if save == 'y':
                            system.save_backtest_results(backtest_results)
                    else:
                        print("❌ Backtest failed or returned no results")
                        print("   Check logs for detailed error information")

                elif choice == '2':
                    # Generate single signal
                    print("\n🎯 SIGNAL GENERATION")
                    print("-" * 40)

                    signal, confidence, metadata = system.strategy.generate_signal()

                    print(f"📡 Current Trading Signal:")
                    print(f"   Signal Strength: {signal:.3f}")
                    print(f"   Confidence: {confidence:.1%}")
                    print(f"   Prediction: {metadata.get('prediction', 'N/A')}")
                    print(f"   Current Price: ${metadata.get('current_price', 0):,.2f}")
                    print(f"   Timestamp: {metadata.get('timestamp', 'N/A')}")

                    # Signal interpretation
                    if signal > 0.3:
                        print("   📈 STRONG BUY signal")
                    elif signal > 0.1:
                        print("   📊 MILD BUY signal")
                    elif signal < -0.3:
                        print("   📉 STRONG SELL signal")
                    elif signal < -0.1:
                        print("   📊 MILD SELL signal")
                    else:
                        print("   ➡️  NEUTRAL signal")

                elif choice == '3':
                    # Start live trading
                    print("\n🤖 LIVE TRADING MODULE")
                    print("-" * 40)

                    if system.strategy.is_running:
                        print("⚠️  Trading is already running!")
                        stop = input("Stop current trading session? (y/N): ").lower().strip()
                        if stop == 'y':
                            system.strategy.stop_automated_trading()
                            print("✅ Trading stopped")
                    else:
                        print("🚨 WARNING: This will start paper trading with real market data")
                        print("💡 No real money will be used, but signals will be generated continuously")

                        confirm = input("Start automated trading? (y/N): ").lower().strip()
                        if confirm == 'y':
                            # Get trading interval
                            intervals = {
                                '1': 60,    # 1 hour
                                '2': 240,   # 4 hours (default)
                                '3': 1440,  # 24 hours
                                '4': 360    # 6 hours
                            }

                            print("Select trading interval:")
                            print("1. Every 1 hour")
                            print("2. Every 4 hours (recommended)")
                            print("3. Every 24 hours")
                            print("4. Every 6 hours")

                            interval_choice = input("Choice (1-4) [2]: ").strip() or '2'
                            interval_minutes = intervals.get(interval_choice, 240)

                            print(f"🚀 Starting automated trading (interval: {interval_minutes//60} hours)...")
                            trading_thread = system.strategy.start_automated_trading(interval_minutes)
                            print("✅ Automated trading started!")
                            print("💡 Use option 4 to monitor portfolio status")
                            print("💡 Use option 0 to stop and exit")

                elif choice == '4':
                    # View portfolio status
                    print("\n💼 PORTFOLIO STATUS")
                    print("-" * 40)

                    portfolio = system.strategy.get_portfolio_summary()

                    print(f"💰 Portfolio Value: ${portfolio.get('portfolio_value', 0):,.2f}")
                    print(f"📈 Total P&L: ${portfolio.get('total_pnl', 0):,.2f}")
                    print(f"📊 Total Return: {portfolio.get('total_return', 0):.2%}")
                    print(f"📉 Current Drawdown: {portfolio.get('current_drawdown', 0):.2%}")
                    print(f"🎯 Open Positions: {portfolio.get('positions', 0)}")
                    print(f"📋 Total Trades: {portfolio.get('total_trades', 0)}")
                    print(f"🤖 System Running: {'✅ Yes' if portfolio.get('is_running', False) else '❌ No'}")
                    print(f"🕐 Last Update: {portfolio.get('last_update', 'Never')}")

                    # Show current positions
                    if system.strategy.positions:
                        print("\n🎯 CURRENT POSITIONS:")
                        for symbol, position in system.strategy.positions.items():
                            print(f"   {symbol}: {position['side'].upper()} ${position['size']:,.2f} @ ${position['entry_price']:,.2f}")

                    # Show recent trades
                    if system.strategy.trade_log:
                        print("\n📝 RECENT TRADES (Last 5):")
                        for trade in system.strategy.trade_log[-5:]:
                            timestamp = trade.get('timestamp', '')[:19]
                            side = trade.get('side', '').upper()
                            size = trade.get('size', 0)
                            price = trade.get('price', 0)
                            pnl = trade.get('pnl', 0)
                            pnl_str = f"P&L: ${pnl:+.2f}" if pnl != 0 else ""
                            print(f"   {timestamp} | {side:5} | ${size:8.2f} @ ${price:8.2f} | {pnl_str}")

                elif choice == '5':
                    # System report
                    print("\n📋 SYSTEM REPORT")
                    print("-" * 40)

                    report = system.generate_report()
                    print(report)

                    # Offer to save report
                    save = input("\nSave report to file? (y/N): ").lower().strip()
                    if save == 'y':
                        filename = f"trading_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                        try:
                            with open(filename, 'w') as f:
                                f.write(report)
                            print(f"✅ Report saved to {filename}")
                        except Exception as e:
                            print(f"❌ Failed to save report: {e}")

                elif choice == '6':
                    # Retrain model
                    print("\n🧠 MODEL RETRAINING")
                    print("-" * 40)

                    print("⚠️  This will retrain the ML model with latest data")
                    print("⏱️  This process may take several minutes...")

                    confirm = input("Proceed with retraining? (y/N): ").lower().strip()
                    if confirm == 'y':
                        try:
                            print("🔄 Retraining model...")
                            system.strategy.retrain_model(force_retrain=True)
                            print("✅ Model retrained successfully!")
                        except Exception as e:
                            print(f"❌ Model retraining failed: {e}")

                elif choice == '7':
                    # Strategy analysis
                    print("\n📊 STRATEGY ANALYSIS")
                    print("-" * 40)

                    # Analyze recent model performance
                    if hasattr(system.strategy, 'model_performance') and system.strategy.model_performance:
                        recent_performance = system.strategy.model_performance[-50:]
                        avg_confidence = np.mean([p["confidence"] for p in recent_performance])
                        print(f"📈 Recent Model Performance:")
                        print(f"   Average Confidence: {avg_confidence:.1%}")
                        print(f"   Predictions Analyzed: {len(recent_performance)}")

                        # Confidence distribution
                        confidences = [p["confidence"] for p in recent_performance]
                        high_conf = sum(1 for c in confidences if c > 0.7)
                        med_conf = sum(1 for c in confidences if 0.4 <= c <= 0.7)
                        low_conf = sum(1 for c in confidences if c < 0.4)

                        print(f"   High Confidence (>70%): {high_conf} ({high_conf/len(confidences):.1%})")
                        print(f"   Medium Confidence (40-70%): {med_conf} ({med_conf/len(confidences):.1%})")
                        print(f"   Low Confidence (<40%): {low_conf} ({low_conf/len(confidences):.1%})")

                    # Risk metrics
                    risk_mgr = system.strategy.risk_manager
                    print(f"\n⚠️  Risk Metrics:")
                    print(f"   Daily P&L: ${risk_mgr.daily_pnl:,.2f}")
                    print(f"   Total P&L: ${risk_mgr.total_pnl:,.2f}")
                    print(f"   Max Equity: ${risk_mgr.max_equity:,.2f}")

                    current_drawdown = (risk_mgr.max_equity - system.strategy.portfolio_value) / risk_mgr.max_equity
                    print(f"   Current Drawdown: {current_drawdown:.2%}")
                    print(f"   Max Allowed Drawdown: {system.config.max_total_drawdown:.2%}")

                elif choice == '8':
                    # Export trade history
                    print("\n💾 EXPORT TRADE HISTORY")
                    print("-" * 40)

                    if not system.strategy.trade_log:
                        print("❌ No trades to export")
                    else:
                        filename = f"trade_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                        try:
                            df_trades = pd.DataFrame(system.strategy.trade_log)
                            df_trades.to_csv(filename, index=False)
                            print(f"✅ Trade history exported to {filename}")
                            print(f"📊 Total trades exported: {len(system.strategy.trade_log)}")
                        except Exception as e:
                            print(f"❌ Export failed: {e}")

                elif choice == '9':
                    # System configuration
                    print("\n⚙️  SYSTEM CONFIGURATION")
                    print("-" * 40)

                    print(f"Current Configuration:")
                    print(f"   Symbol: {system.config.symbol}")
                    print(f"   Interval: {system.config.interval}")
                    print(f"   Max Position Size: {system.config.max_position_size:.1%}")
                    print(f"   Stop Loss: {system.config.stop_loss_pct:.1%}")
                    print(f"   Take Profit: {system.config.take_profit_pct:.1%}")
                    print(f"   Trading Fee: {system.config.trading_fee:.3%}")
                    print(f"   Max Daily Drawdown: {system.config.max_daily_drawdown:.1%}")
                    print(f"   Max Total Drawdown: {system.config.max_total_drawdown:.1%}")

                    print("\n💡 Configuration can be modified in 'trading_config.json'")
                    print("⚠️  Restart system after configuration changes")

                else:
                    print("❌ Invalid choice. Please select 0-9.")

                # Wait for user input before returning to menu
                if choice != '0':
                    input("\nPress Enter to continue...")

            except KeyboardInterrupt:
                print("\n\n🛑 Interrupted by user")
                if system.strategy.is_running:
                    system.strategy.stop_automated_trading()
                break
            except Exception as e:
                print(f"❌ Error in menu option: {e}")
                logger.error(f"Menu error: {e}")
                input("Press Enter to continue...")

    except Exception as e:
        logger.error(f"Critical system error: {e}")
        print(f"💥 Critical Error: {e}")
        print("📋 Check trading_system.log for detailed error information")

        # Emergency shutdown
        try:
            if 'system' in locals() and system.strategy.is_running:
                system.strategy.stop_automated_trading()
                print("🛑 Emergency shutdown completed")
        except:
            pass

    finally:
        print("🔄 System cleanup completed")


# In[ ]:


if __name__ == "__main__":
    # Set up signal handlers for graceful shutdown
    import signal

    def signal_handler(signum, frame):
        print("\n🛑 Received shutdown signal...")
        # This will be caught by the KeyboardInterrupt handler
        raise KeyboardInterrupt()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Run the main function
    main()


# In[ ]:




