import pandas as pd
import numpy as np
import requests
from scipy.stats import t
import time

def fetch_binance_data(symbol="BTCUSDT", interval="1h", limit=1000):
    """
    Fetch historical klines from Binance.
    """
    url = f"https://data-api.binance.vision/api/v3/klines"
    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": limit
    }
    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json()
    
    df = pd.DataFrame(data, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
    ])
    
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    
    return df

def calculate_log_returns(df):
    """
    Compute log returns.
    """
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    return df

def calculate_volatility(df, window=30):
    """
    Compute rolling volatility.
    """
    df["volatility"] = df["log_return"].rolling(window=window).std()
    return df

def simulate_gbm_student_t(current_price, returns_window, n_simulations=10000):
    """
    Simulate 10,000 possible next hours using GBM with Student-t distribution fitted to returns.
    """
    if len(returns_window) < 10:
        # Fallback to normal if not enough data
        vol = returns_window.std()
        returns_sim = np.random.normal(0, vol, n_simulations)
    else:
        # NOTE: scipy t.fit() returns scale ≠ std dev. 
        # Corrected std = scale * sqrt(df/(df-2)). 
        # Using raw volatility as scale over-estimates range (coverage ~0.97).
        # t.fit() on returns window handles this calibration correctly.
        df_fit, loc_fit, scale_fit = t.fit(returns_window)
        
        # Sample from the fitted distribution
        returns_sim = t.rvs(df=df_fit, loc=loc_fit, scale=scale_fit, size=n_simulations)
    
    # Calculate simulated prices
    prices_sim = current_price * np.exp(returns_sim)
    
    # Extract 95% range (2.5th and 97.5th percentiles)
    low = np.percentile(prices_sim, 2.5)
    high = np.percentile(prices_sim, 97.5)
    
    return low, high

def evaluate_metrics(predictions, actuals):
    """
    Compute Coverage, Average Width, and Winkler Score.
    predictions: list of tuples (low, high)
    actuals: list of actual prices
    """
    n = len(predictions)
    hits = 0
    total_width = 0
    winkler_scores = []
    
    for (low, high), actual in zip(predictions, actuals):
        # Coverage
        if low <= actual <= high:
            hits += 1
            winkler_scores.append(high - low)
        else:
            # Winkler Score Penalty
            width = high - low
            if actual < low:
                penalty = (2 / 0.05) * (low - actual)
            else:
                penalty = (2 / 0.05) * (actual - high)
            winkler_scores.append(width + penalty)
        
        total_width += (high - low)
    
    coverage = hits / n
    avg_width = total_width / n
    mean_winkler = np.mean(winkler_scores)
    
    return {
        "coverage": coverage,
        "avg_width": avg_width,
        "mean_winkler": mean_winkler
    }
