import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from utils import fetch_binance_data, calculate_log_returns, simulate_gbm_student_t
from supabase import create_client
import json
import os
from datetime import datetime

# Configuration
VOLATILITY_WINDOW = 30 # Chosen from window optimization experiment - see README

# Page config
st.set_page_config(page_title="Bitcoin Next Hour Predictor | AlphaI × Polaris", layout="wide")

@st.cache_resource
def get_db():
    url = st.secrets["supabase"]["url"]
    key = st.secrets["supabase"]["key"]
    return create_client(url, key)

def save_prediction(timestamp, low, high, actual=None):
    db = get_db()
    # Check if prediction for this timestamp already exists
    res = db.table("predictions").select("*").eq("timestamp", str(timestamp)).execute()
    if len(res.data) == 0:
        db.table("predictions").insert({
            "timestamp": str(timestamp),
            "low": low,
            "high": high,
            "actual": actual,
            "hit": None
        }).execute()

def update_actuals(df):
    db = get_db()
    # Find predictions without actuals
    res = db.table("predictions").select("*").is_("actual", "null").execute()
    rows = res.data
    for row in rows:
        ts = row["timestamp"]
        # Match with df
        match = df[df['timestamp'].astype(str) == ts]
        if not match.empty:
            actual = float(match.iloc[0]['close'])
            low = row["low"]
            high = row["high"]
            hit = 1 if low <= actual <= high else 0
            db.table("predictions").update({"actual": actual, "hit": hit}).eq("timestamp", ts).execute()

def get_predictions_history():
    db = get_db()
    res = db.table("predictions").select("*").order("timestamp", desc=True).execute()
    return pd.DataFrame(res.data)

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
