import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests as req
from utils import fetch_binance_data, calculate_log_returns, simulate_gbm_student_t
import json
import os

# Configuration
VOLATILITY_WINDOW = 30  # Chosen from window optimization experiment - see README

# Page config
st.set_page_config(page_title="Bitcoin Next Hour Predictor | AlphaI × Polaris", layout="wide")


# --- Supabase via REST API (avoids httpx IPv6 bug on Streamlit Cloud) ---
def get_sb_headers():
    key = st.secrets["supabase"]["key"]
    return {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "Prefer": "return=minimal"
    }

def get_sb_url(table="predictions"):
    base = st.secrets["supabase"]["url"]
    return f"{base}/rest/v1/{table}"


def save_prediction(timestamp: str, low: float, high: float):
    try:
        url = get_sb_url()
        headers = get_sb_headers()
        # Check if already exists
        check = req.get(
            url,
            headers={**headers, "Prefer": ""},
            params={"select": "id", "timestamp": f"eq.{timestamp}"},
            timeout=10
        )
        if check.status_code == 200 and len(check.json()) > 0:
            return  # Already saved
        # Insert new prediction
        payload = {
            "timestamp": timestamp,
            "low": float(low),
            "high": float(high)
        }
        req.post(
            url,
            headers=headers,
            json=payload,
            timeout=10
        )
    except Exception:
        pass


def update_actuals(df: pd.DataFrame):
    try:
        url = get_sb_url()
        headers = get_sb_headers()
        pending_res = req.get(
            url,
            headers={**headers, "Prefer": ""},
            params={"actual": "is.null", "select": "*"},
            timeout=5
        )
        if not pending_res.ok:
            return
        for row in pending_res.json():
            ts = row["timestamp"]
            # Flexible matching for timestamp strings
            match = df[df["timestamp"].astype(str).str.startswith(ts[:16])]
            if not match.empty:
                actual = float(match.iloc[0]["close"])
                hit = "true" if row["low"] <= actual <= row["high"] else "false"
                req.patch(
                    url,
                    headers=headers,
                    params={"id": f"eq.{row['id']}"},
                    json={"actual": actual, "hit": hit},
                    timeout=5
                )
    except Exception:
        pass


def get_history() -> pd.DataFrame:
    try:
        url = get_sb_url()
        headers = {**get_sb_headers(), "Prefer": ""}
        res = req.get(
            url,
            headers=headers,
            params={"select": "*", "order": "timestamp.desc"},
            timeout=5
        )
        if res.ok:
            return pd.DataFrame(res.json())
    except Exception:
        pass
    return pd.DataFrame()


# Sidebar
st.sidebar.header("Backtest Metrics (Part A)")
if os.path.exists("backtest_metrics.json"):
    with open("backtest_metrics.json", "r") as f:
        metrics = json.load(f)
    st.sidebar.metric("Coverage", f"{metrics['coverage']:.4f}")
    st.sidebar.metric("Avg Width", f"${metrics['avg_width']:.2f}")
    st.sidebar.metric("Winkler Score", f"{metrics['mean_winkler']:.2f}")
    st.sidebar.caption(
        "Winkler score = avg range width + penalty for misses. "
        "Lower is better. Reference: ~1757."
    )
else:
    st.sidebar.warning("Run backtest.py first to see metrics.")

# Main
st.title("BTCUSDT Next Hour Price Predictor")
st.write("AlphaI × Polaris Challenge Submission")

# DEBUG - remove after fixing
try:
    r = req.get(f"{st.secrets['supabase']['url']}/rest/v1/predictions?select=id&limit=1",
                headers={"apikey": st.secrets["supabase"]["key"]}, timeout=10)
    st.write(f"DB status: {r.status_code} | response: {r.text[:200]}")
except Exception as e:
    st.write(f"DB FAILED: {e}")

with st.spinner("Fetching latest data from Binance..."):
    df = fetch_binance_data(limit=500)
    df = calculate_log_returns(df)
    update_actuals(df)

latest_bar = df.iloc[-1]
current_price = latest_bar["close"]
current_ts = latest_bar["timestamp"]
next_ts = current_ts + pd.Timedelta(hours=1)

returns_window = df["log_return"].tail(VOLATILITY_WINDOW).dropna()
low, high = simulate_gbm_student_t(current_price, returns_window)

save_prediction(str(next_ts), low, high)

col1, col2, col3 = st.columns(3)
col1.metric("Current BTC Price", f"${current_price:,.2f}")
col2.metric("Predicted Low (95%)", f"${low:,.2f}")
col3.metric("Predicted High (95%)", f"${high:,.2f}")

st.subheader("Price History and Next Hour Prediction")
plot_df = df.tail(50).copy()

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=plot_df["timestamp"], y=plot_df["close"],
    mode="lines+markers", name="Actual Price",
    line=dict(color="royalblue", width=2), marker=dict(size=3)
))
fig.add_trace(go.Scatter(
    x=[current_ts, next_ts], y=[current_price, high],
    mode="lines", line=dict(width=0), showlegend=False
))
fig.add_trace(go.Scatter(
    x=[current_ts, next_ts], y=[current_price, low],
    mode="lines", line=dict(width=0),
    fill="tonexty", fillcolor="rgba(0, 200, 100, 0.25)",
    name="95% Prediction Range"
))
fig.add_hline(y=current_price, line_dash="dash", line_color="gray",
              annotation_text="Current Price")
fig.update_layout(
    xaxis_title="Time", yaxis_title="BTC Price (USD)",
    hovermode="x unified", template="plotly_dark", height=500
)
st.plotly_chart(fig, use_container_width=True)

st.subheader("Prediction History (Part C)")
history_df = get_history()

if not history_df.empty:
    st.dataframe(history_df[["timestamp", "low", "high", "actual", "hit"]])
    resolved = history_df.dropna(subset=["actual"])
    if not resolved.empty:
        live_hits = (resolved["hit"] == "true").sum()
        live_coverage = live_hits / len(resolved)
        st.metric("Live Coverage (resolved predictions)", f"{live_coverage:.2%}")
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Scatter(
            x=resolved["timestamp"], y=resolved["actual"],
            mode="markers", name="Actual Price",
            marker=dict(color="white", size=6)
        ))
        fig_hist.add_trace(go.Scatter(
            x=resolved["timestamp"], y=resolved["high"],
            mode="lines", line=dict(width=0), showlegend=False
        ))
        fig_hist.add_trace(go.Scatter(
            x=resolved["timestamp"], y=resolved["low"],
            mode="lines", line=dict(width=0),
            fill="tonexty", fillcolor="rgba(0, 200, 100, 0.2)",
            name="Predicted Range"
        ))
        fig_hist.update_layout(
            title="Live Prediction Performance",
            xaxis_title="Time", yaxis_title="Price (USD)",
            template="plotly_dark", height=400
        )
        st.plotly_chart(fig_hist, use_container_width=True)
    else:
        st.info("Actuals will fill in automatically as each hour closes.")
else:
    st.info("History will grow with each visit to the dashboard.")
