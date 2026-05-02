import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from utils import fetch_binance_data, calculate_log_returns, simulate_gbm_student_t
import sqlite3
import json
import os
from datetime import datetime

# Configuration
VOLATILITY_WINDOW = 30 # Chosen from window optimization experiment - see README

# Page config
st.set_page_config(page_title="Bitcoin Next Hour Predictor | AlphaI × Polaris", layout="wide")

# Database setup
DB_FILE = "predictions.db"

def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS predictions
                 (timestamp TEXT, low REAL, high REAL, actual REAL, hit INTEGER)''')
    conn.commit()
    conn.close()

def save_prediction(timestamp, low, high, actual=None):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    # Check if prediction for this timestamp already exists
    c.execute("SELECT * FROM predictions WHERE timestamp=?", (str(timestamp),))
    if c.fetchone() is None:
        c.execute("INSERT INTO predictions (timestamp, low, high, actual, hit) VALUES (?, ?, ?, ?, ?)",
                  (str(timestamp), low, high, actual, None))
    conn.commit()
    conn.close()

def update_actuals(df):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    # Find predictions without actuals
    c.execute("SELECT timestamp FROM predictions WHERE actual IS NULL")
    rows = c.fetchall()
    for row in rows:
        ts = row[0]
        # Match with df
        match = df[df['timestamp'].astype(str) == ts]
        if not match.empty:
            actual = float(match.iloc[0]['close'])
            low = c.execute("SELECT low FROM predictions WHERE timestamp=?", (ts,)).fetchone()[0]
            high = c.execute("SELECT high FROM predictions WHERE timestamp=?", (ts,)).fetchone()[0]
            hit = 1 if low <= actual <= high else 0
            c.execute("UPDATE predictions SET actual=?, hit=? WHERE timestamp=?", (actual, hit, ts))
    conn.commit()
    conn.close()

def get_predictions_history():
    conn = sqlite3.connect(DB_FILE)
    df_hist = pd.read_sql_query("SELECT * FROM predictions ORDER BY timestamp DESC", conn)
    conn.close()
    return df_hist

# Initialize
init_db()

st.title("BTCUSDT Next Hour Price Predictor")
st.write("AlphaI × Polaris Challenge Submission")

# Sidebar - Metrics from Backtest
st.sidebar.header("Backtest Metrics (Part A)")
if os.path.exists("backtest_metrics.json"):
    with open("backtest_metrics.json", "r") as f:
        metrics = json.load(f)
    st.sidebar.metric("Coverage", f"{metrics['coverage']:.4f}")
    st.sidebar.metric("Avg Width", f"${metrics['avg_width']:.2f}")
    st.sidebar.metric("Winkler Score", f"{metrics['mean_winkler']:.2f}")
    st.sidebar.caption("Winkler score = avg range width + penalty for misses. Lower is better. Reference: ~1757.")
else:
    st.sidebar.warning("Run backtest.py first to see metrics.")

# Main content
with st.spinner("Fetching latest data from Binance..."):
    df = fetch_binance_data(limit=500)
    df = calculate_log_returns(df)
    update_actuals(df)

latest_bar = df.iloc[-1]
current_price = latest_bar['close']
current_ts = latest_bar['timestamp']

# Prediction for next hour
returns_window = df['log_return'].tail(VOLATILITY_WINDOW).dropna()
low, high = simulate_gbm_student_t(current_price, returns_window)

# Save current prediction
next_ts = current_ts + pd.Timedelta(hours=1)
save_prediction(next_ts, low, high)

# Display metrics
col1, col2, col3 = st.columns(3)
col1.metric("Current BTC Price", f"${current_price:,.2f}")
col2.metric("Predicted Low (95%)", f"${low:,.2f}")
col3.metric("Predicted High (95%)", f"${high:,.2f}")

# Chart
st.subheader("Price History and Next Hour Prediction")
plot_df = df.tail(50).copy()

fig = go.Figure()

# Historical Price
fig.add_trace(go.Scatter(
    x=plot_df['timestamp'],
    y=plot_df['close'],
    mode='lines+markers',
    name='Actual Price',
    line=dict(color='blue')
))

# Prediction Range Ribbon for the next hour
# We create a small range for the next hour
fig.add_trace(go.Scatter(
    x=[current_ts, next_ts],
    y=[current_price, high],
    mode='lines',
    line=dict(width=0),
    showlegend=False
))
fig.add_trace(go.Scatter(
    x=[current_ts, next_ts],
    y=[current_price, low],
    mode='lines',
    line=dict(width=0),
    fill='tonexty',
    fillcolor='rgba(0, 255, 0, 0.2)',
    name='95% Prediction Range'
))

# Add a dashed line for current price
fig.add_hline(y=current_price, line_dash="dash", line_color="gray", annotation_text="Current Price")

fig.update_layout(
    xaxis_title="Time",
    yaxis_title="BTC Price (USD)",
    hovermode="x unified",
    template="plotly_white",
    height=600
)

st.plotly_chart(fig, use_container_width=True)

# Part C: Prediction History
st.subheader("Prediction History (Part C)")
history_df = get_predictions_history()
if not history_df.empty:
    st.dataframe(history_df)
    
    # Historical Performance
    valid_hist = history_df.dropna(subset=['actual'])
    if not valid_hist.empty:
        live_coverage = valid_hist['hit'].mean()
        st.metric("Live Coverage", f"{live_coverage:.2%}")
        
        # Plot history vs actuals
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Scatter(x=valid_hist['timestamp'], y=valid_hist['actual'], mode='markers', name='Actual'))
        fig_hist.add_trace(go.Scatter(x=valid_hist['timestamp'], y=valid_hist['high'], mode='lines', name='Upper Bound', line=dict(width=0)))
        fig_hist.add_trace(go.Scatter(x=valid_hist['timestamp'], y=valid_hist['low'], mode='lines', name='Lower Bound', line=dict(width=0), fill='tonexty'))
        fig_hist.update_layout(title="Live Prediction Performance", xaxis_title="Time", yaxis_title="Price")
        st.plotly_chart(fig_hist, use_container_width=True)
else:
    st.info("History will appear after more data is collected.")
