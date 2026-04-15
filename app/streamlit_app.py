"""
TSA Checkpoint Throughput Forecasting — Streamlit Demo App
Run:  streamlit run streamlit_app.py
Env:  S3_BUCKET  (default: tsa-throughput-model)
      AWS_DEFAULT_REGION (default: us-east-1)
"""

import os
import io
import joblib
import boto3
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import LabelEncoder

# ── Config ────────────────────────────────────────────────────────────────────
S3_BUCKET   = os.getenv("S3_BUCKET", "tsa-throughput-model")
MODEL_KEY   = "lgbmModel.pkl"
DATA_KEY    = "modelDf.csv"

AIRPORTS = [
    'ATL','LAX','DFW','DEN','ORD','JFK','MCO','LAS','CLT','MIA',
    'PHX','IAH','BOS','FLL','MSP','LGA','DTW','SEA','SFO','EWR'
]

DAY_NAMES = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']

# ── Page setup ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TSA Throughput Forecaster",
    page_icon="✈️",
    layout="wide"
)

st.title("✈️  TSA Checkpoint Throughput Forecaster")
st.caption(
    "Predicts hourly passenger volume at TSA checkpoints — "
    "designed for TSA workforce planners making staffing decisions 12+ months ahead."
)

# ── Load model from S3 (cached) ───────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model from S3…")
def load_model():
    s3 = boto3.client("s3")
    buf = io.BytesIO()
    s3.download_fileobj(S3_BUCKET, MODEL_KEY, buf)
    buf.seek(0)
    return joblib.load(buf)   # returns dict: model, leAirport, leCheckpoint, features

@st.cache_data(show_spinner="Loading historical data from S3…")
def load_history():
    s3 = boto3.client("s3")
    buf = io.BytesIO()
    s3.download_fileobj(S3_BUCKET, DATA_KEY, buf)
    buf.seek(0)
    return pd.read_csv(buf, parse_dates=["date"])

# ── Sidebar — user inputs ──────────────────────────────────────────────────────
with st.sidebar:
    st.header("Forecast Parameters")

    airport = st.selectbox("Airport", AIRPORTS, index=AIRPORTS.index("ATL"))

    col1, col2 = st.columns(2)
    with col1:
        month = st.selectbox("Month", list(range(1, 13)),
                             format_func=lambda m: pd.Timestamp(2025, m, 1).strftime("%b"),
                             index=5)
    with col2:
        day_of_week = st.selectbox("Day of week", list(range(7)),
                                   format_func=lambda d: DAY_NAMES[d],
                                   index=4)   # default Friday

    is_holiday = st.checkbox("Federal holiday?", value=False)
    days_to_holiday = st.slider("Days to nearest holiday", 0, 30, 7)

    st.divider()
    st.markdown("**Scheduled flights (optional)**")
    departures = st.number_input("Hourly departure count (peak hour)", 0, 80, 30)
    avg_seats  = st.number_input("Avg seats per flight", 50, 300, 155)
    load_factor = st.slider("Avg load factor", 0.5, 1.0, 0.78)

    run_btn = st.button("Generate Forecast", type="primary", use_container_width=True)

# ── Main panel ────────────────────────────────────────────────────────────────
if not run_btn:
    st.info("Set parameters in the sidebar and click **Generate Forecast** to begin.")
    st.stop()

try:
    artifact = load_model()
except Exception as e:
    st.error(f"Could not load model from S3: {e}")
    st.stop()

model       = artifact["model"]
le_airport  = artifact["leAirport"]
le_checkpoint = artifact["leCheckpoint"]
features    = artifact["features"]

# ── Build prediction rows for every hour × every checkpoint ───────────────────
checkpoints = list(le_checkpoint.classes_)
airport_checkpoints = [c for c in checkpoints if c.startswith(airport) or True]
# Filter to checkpoints that belong to selected airport using history
try:
    history = load_history()
    cp_list = sorted(history.loc[history["airportCode"] == airport, "checkpointName"].unique())
except Exception:
    cp_list = ["Main Checkpoint"]   # fallback

hours = list(range(4, 24))   # 4 AM – 11 PM (operational window)

rows = []
for hour in hours:
    for cp in cp_list:
        exp_vol = departures * avg_seats * load_factor if hour == departures else \
                  departures * avg_seats * load_factor * (0.5 + 0.5 * np.sin((hour - 6) * np.pi / 12))
        rows.append({
            "hour":                    hour,
            "airportCode_lbl":         le_airport.transform([airport])[0]
                                       if airport in le_airport.classes_ else 0,
            "checkpointName_lbl":      le_checkpoint.transform([cp])[0]
                                       if cp in le_checkpoint.classes_ else 0,
            "hourlyDepartureCount":    float(departures),
            "avgSeatsPerFlight":       float(avg_seats),
            "avgLoadFactor":           float(load_factor),
            "expectedPassengerVolume": float(exp_vol),
            "dayOfWeek":               day_of_week,
            "month":                   month,
            "isWeekend":               int(day_of_week >= 5),
            "isHoliday":               int(is_holiday),
            "daysToNearestHoliday":    days_to_holiday,
        })

pred_df = pd.DataFrame(rows)[features]
preds   = model.predict(pred_df)

result_df = pd.DataFrame({
    "hour":       [r["hour"]        for r in rows],
    "checkpoint": [cp_list[i % len(cp_list)] for i, r in enumerate(rows)],
    "predicted":  np.maximum(preds, 0),
})

# ── Aggregate to airport-hour totals ─────────────────────────────────────────
hourly_total = result_df.groupby("hour")["predicted"].sum().reset_index()

# ── Historical average — summed across all checkpoints per hour ───────────────
try:
    _hist_filtered = history.loc[
        (history["airportCode"] == airport) &
        (history["month"] == month) &
        (history["dayOfWeek"] == day_of_week)
    ]
    # Step 1: sum across checkpoints for each date-hour (total airport throughput)
    # Step 2: average across different dates → typical throughput for this slot
    hist_avg = (
        _hist_filtered
        .groupby(["date", "hour"])["totalPassengers"].sum()
        .reset_index()
        .groupby("hour")["totalPassengers"].mean()
        .reset_index()
        .rename(columns={"totalPassengers": "historical_avg"})
    )
    hourly_total = hourly_total.merge(hist_avg, on="hour", how="left")
    show_hist = True
except Exception:
    show_hist = False

FONT = dict(size=14, color="black")
TICK = dict(size=13, color="black")

# ── Charts ────────────────────────────────────────────────────────────────────
st.subheader(f"Hourly Throughput Forecast — {airport}  ·  {DAY_NAMES[day_of_week]}, {pd.Timestamp(2025,month,1).strftime('%B')}")

fig = go.Figure()
fig.add_trace(go.Bar(
    x=hourly_total["hour"],
    y=hourly_total["predicted"],
    name="LightGBM Forecast",
    marker_color="#4472C4",
    opacity=0.85,
))
if show_hist and "historical_avg" in hourly_total.columns:
    fig.add_trace(go.Scatter(
        x=hourly_total["hour"],
        y=hourly_total["historical_avg"],
        name="Historical Average",
        mode="lines+markers",
        line=dict(color="#D84315", width=2.5, dash="dot"),
        marker=dict(size=6),
    ))
fig.update_layout(
    xaxis_title="Hour of Day",
    yaxis_title="Predicted Passengers",
    legend=dict(
        orientation="h", yanchor="top", y=-0.18,
        xanchor="center", x=0.5,
        font=dict(size=14, color="black"),
        bgcolor="rgba(0,0,0,0)",
    ),
    height=450,
    plot_bgcolor="white",
    paper_bgcolor="white",
    font=dict(size=14, color="black"),
    xaxis=dict(title_font=FONT, tickfont=TICK),
    yaxis=dict(title_font=FONT, tickfont=TICK),
    margin=dict(b=80),
)
fig.update_xaxes(tickmode="linear", dtick=1, gridcolor="#EEEEEE")
fig.update_yaxes(gridcolor="#EEEEEE")
st.plotly_chart(fig, use_container_width=True)

# ── Per-checkpoint breakdown ──────────────────────────────────────────────────
st.subheader("Breakdown by Checkpoint (peak hour)")
peak_hour = hourly_total.loc[hourly_total["predicted"].idxmax(), "hour"]
peak_df   = result_df[result_df["hour"] == peak_hour].sort_values("predicted", ascending=True)

fig2 = px.bar(peak_df, x="predicted", y="checkpoint", orientation="h",
              labels={"predicted": "Predicted Passengers", "checkpoint": ""},
              color_discrete_sequence=["#7B9CD9"])
fig2.update_layout(height=max(300, len(cp_list) * 45), plot_bgcolor="white",
                   paper_bgcolor="white", font=dict(size=13, color="black"),
                   xaxis=dict(title_font=FONT, tickfont=TICK),
                   yaxis=dict(tickfont=TICK))
fig2.update_xaxes(gridcolor="#EEEEEE")
st.plotly_chart(fig2, use_container_width=True)

# ── Summary metrics ───────────────────────────────────────────────────────────
st.subheader("Summary")
col1, col2, col3 = st.columns(3)
col1.metric("Peak hour",              f"{peak_hour}:00")
col2.metric("Peak predicted volume",  f"{int(hourly_total['predicted'].max()):,} pax")
col3.metric("Total daily forecast",   f"{int(hourly_total['predicted'].sum()):,} pax")

if show_hist and "historical_avg" in hourly_total.columns:
    hist_total = hourly_total["historical_avg"].sum()
    delta_pct  = (hourly_total["predicted"].sum() - hist_total) / hist_total * 100
    st.caption(
        f"Model forecast is **{delta_pct:+.1f}%** vs. historical average "
        f"({int(hist_total):,} pax/day) for this airport / month / day-of-week."
    )

st.divider()
st.caption(
    "Model: LightGBM  ·  Test R² = 0.85  ·  Test MAE = 124 pax/checkpoint-hour  ·  "
    "Data: TSA throughput + BTS on-time + T-100 load factors (2022–2025)"
)
