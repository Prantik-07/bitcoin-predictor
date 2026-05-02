import pandas as pd
import numpy as np
import json
from utils import fetch_binance_data, calculate_log_returns, calculate_volatility, simulate_gbm_student_t, evaluate_metrics

def run_backtest(df, window=30):
    predictions = []
    actuals = []
    results_records = []
    
    # We need to predict for the last 720 bars
    total_bars = len(df)
    start_idx = total_bars - 720
    
    actual_count = total_bars - start_idx
    print(f"Running backtest on {actual_count} bars (Target: 720)...")
    
    for i in range(start_idx, total_bars):
        current_data = df.iloc[:i]
        latest_price = current_data.iloc[-1]["close"]
        
        # Pass the window of returns
        returns_window = current_data["log_return"].tail(window).dropna()
        
        # Predict
        low, high = simulate_gbm_student_t(latest_price, returns_window)
        
        # Actual
        actual = df.iloc[i]["close"]
        timestamp = df.iloc[i]["timestamp"]
        
        predictions.append((low, high))
        actuals.append(actual)
        
        results_records.append({
            "timestamp": str(timestamp),
            "low": float(low),
            "high": float(high),
            "actual": float(actual)
        })
        
    metrics = evaluate_metrics(predictions, actuals)
    return metrics, results_records

def main():
    print("Fetching data...")
    df = fetch_binance_data(limit=1000)
    df = calculate_log_returns(df)
    
    # Experiment with different windows
    windows = [10, 20, 30, 50, 100]
    best_window = 30
    best_coverage_diff = 1.0
    best_metrics = None
    best_records = None
    
    print("Running optimization...")
    for w in windows:
        metrics, records = run_backtest(df, window=w)
        print(f"Window {w}: Coverage={metrics['coverage']:.4f}, Winkler={metrics['mean_winkler']:.2f}")
        
        coverage_diff = abs(metrics['coverage'] - 0.95)
        if coverage_diff < best_coverage_diff:
            best_coverage_diff = coverage_diff
            best_window = w
            best_metrics = metrics
            best_records = records

    print(f"\nBest configuration: Window {best_window}")
    print(f"Best Metrics: {best_metrics}")
    
    # Save best results
    print(f"Saving results to backtest_results.jsonl...")
    with open("backtest_results.jsonl", "w") as f:
        for record in best_records:
            f.write(json.dumps(record) + "\n")
    
    # Save metrics to a file for easy access by the dashboard
    with open("backtest_metrics.json", "w") as f:
        json.dump(best_metrics, f)

if __name__ == "__main__":
    main()
