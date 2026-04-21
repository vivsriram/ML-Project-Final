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

OCCUPANCY_OPTIONS = {
    "Low season — ~60% of seats filled":      0.60,
    "Typical — ~78% of seats filled":         0.78,
    "High season — ~88% of seats filled":     0.88,
    "Near-full — ~95% of seats filled":       0.95,
}

# ── Page setup ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Checkpoint Throughput Forecaster",
    page_icon="✈️",
    layout="wide"
)

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Sans+3:ital,wght@0,400;0,600;0,700;1,400&family=Merriweather:ital,wght@0,400;0,700;1,400&display=swap');

/* ── Global typography ── */
html, body, [class*="css"], p, li, span, div, label, input, select {
    font-family: 'Source Sans 3', 'Helvetica Neue', Arial, sans-serif !important;
    font-size: 15px;
    color: #1a1a1a;
}

/* ── App background ── */
[data-testid="stAppViewContainer"],
[data-testid="stMainBlockContainer"] {
    background-color: #f4f4f1;
}

/* ── Top rule bar ── */
[data-testid="stAppViewContainer"]::before {
    content: "";
    display: block;
    height: 5px;
    background: #001f4d;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background-color: #001f4d !important;
    border-right: 4px solid #9b7e00;
}
[data-testid="stSidebar"] * {
    color: #e8eef7 !important;
    font-family: 'Source Sans 3', Arial, sans-serif !important;
}
[data-testid="stSidebar"] input,
[data-testid="stSidebar"] select,
[data-testid="stSidebar"] [data-baseweb="select"] *,
[data-testid="stSidebar"] [data-baseweb="input"] *,
[data-testid="stSidebar"] [role="listbox"] *,
[data-testid="stSidebar"] [role="option"] {
    color: #1a1a1a !important;
    background-color: #ffffff !important;
}
[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background-color: #ffffff !important;
    color: #1a1a1a !important;
}
[data-testid="stSidebar"] [data-baseweb="select"] svg {
    color: #1a1a1a !important;
    fill: #1a1a1a !important;
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    font-family: 'Merriweather', Georgia, serif !important;
    color: #ffffff !important;
    font-size: 0.95rem !important;
    font-weight: 700 !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    border-bottom: 1px solid #9b7e00;
    padding-bottom: 6px;
    margin-bottom: 12px;
}
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stNumberInput label,
[data-testid="stSidebar"] .stCheckbox label {
    color: #b8ccdf !important;
    font-size: 0.78rem !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.07em !important;
}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
    color: #b8ccdf !important;
    font-size: 0.82rem;
}
[data-testid="stSidebar"] .stButton > button {
    background-color: #9b7e00 !important;
    color: #ffffff !important;
    font-family: 'Merriweather', Georgia, serif !important;
    font-weight: 700 !important;
    font-size: 0.85rem !important;
    border: none !important;
    border-radius: 1px !important;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    padding: 10px !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background-color: #b89200 !important;
}

/* ── Main headings ── */
h1 {
    font-family: 'Merriweather', Georgia, serif !important;
    color: #001f4d !important;
    font-size: 1.65rem !important;
    font-weight: 700 !important;
    border-bottom: 3px solid #001f4d;
    padding-bottom: 10px;
    margin-bottom: 4px !important;
    letter-spacing: 0.01em;
}
h2, h3 {
    font-family: 'Merriweather', Georgia, serif !important;
    color: #001f4d !important;
    font-size: 0.95rem !important;
    font-weight: 700 !important;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    border-left: 4px solid #9b7e00;
    padding-left: 10px;
    margin-top: 1.6rem !important;
}

/* ── Caption ── */
[data-testid="stCaptionContainer"] p {
    color: #555555 !important;
    font-size: 0.8rem !important;
    font-style: italic;
    font-family: 'Source Sans 3', Arial, sans-serif !important;
}

/* ── Metric cards ── */
[data-testid="stMetric"] {
    background: #ffffff;
    border: 1px solid #ccd5df;
    border-top: 4px solid #001f4d;
    border-radius: 1px;
    padding: 14px 18px !important;
}
[data-testid="stMetricLabel"] p {
    color: #4a5a6a !important;
    font-size: 0.72rem !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
    font-family: 'Source Sans 3', Arial, sans-serif !important;
}
[data-testid="stMetricValue"] {
    color: #001f4d !important;
    font-family: 'Merriweather', Georgia, serif !important;
    font-size: 1.7rem !important;
    font-weight: 700 !important;
}

/* ── Alert / info box ── */
[data-testid="stAlert"] {
    background-color: #e4ecf7 !important;
    border-left: 5px solid #001f4d !important;
    border-radius: 1px !important;
}
[data-testid="stAlert"] p {
    color: #001f4d !important;
    font-family: 'Source Sans 3', Arial, sans-serif !important;
}

/* ── Divider ── */
hr {
    border-color: #ccd5df !important;
    margin: 1.5rem 0 !important;
}

/* ── Sidebar dropdown / input text fix ── */
[data-testid="stSidebar"] [data-baseweb="select"] > div,
[data-testid="stSidebar"] [data-baseweb="select"] input,
[data-testid="stSidebar"] [data-baseweb="input"] input {
    background-color: #ffffff !important;
    color: #1a1a1a !important;
}
[data-testid="stSidebar"] [data-baseweb="select"] span,
[data-testid="stSidebar"] [data-baseweb="select"] [class*="valueContainer"] * {
    color: #1a1a1a !important;
}
[data-testid="stSidebar"] [data-baseweb="select"] svg {
    fill: #1a1a1a !important;
}
/* dropdown option list */
[data-baseweb="popover"] *,
[data-baseweb="menu"] *,
[role="option"] {
    color: #1a1a1a !important;
    background-color: #ffffff !important;
}
[role="option"]:hover,
[role="option"][aria-selected="true"] {
    background-color: #e4ecf7 !important;
    color: #001f4d !important;
}
</style>

<div style="
    background: #001f4d;
    color: white;
    padding: 22px 28px 20px;
    text-align: center;
    margin-bottom: 10px;
    border-bottom: 4px solid #9b7e00;
">
  <div style="font-size: 2.6rem; line-height: 1; margin-bottom: 10px;">✈️</div>
  <div style="
      font-family: 'Merriweather', Georgia, serif;
      font-size: 1.7rem;
      font-weight: 700;
      letter-spacing: 0.03em;
      line-height: 1.3;
      color: #ffffff;
  ">
    Checkpoint Throughput Forecaster
  </div>
  <div style="
      font-family: 'Source Sans 3', Arial, sans-serif;
      font-size: 0.78rem;
      letter-spacing: 0.16em;
      color: #a8bccc;
      text-transform: uppercase;
      margin-top: 8px;
  ">
    Airport Security Operations &nbsp;·&nbsp; Workforce Planning Tool
  </div>
</div>
""", unsafe_allow_html=True)

st.title("Checkpoint Throughput Forecaster")
st.caption(
    "Predicts hourly passenger volume at airport security checkpoints. "
    "Designed for workforce planners making staffing decisions 12+ months ahead."
)

# ── Load model from S3 (cached) ───────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model…")
def load_model():
    s3 = boto3.client("s3")
    buf = io.BytesIO()
    s3.download_fileobj(S3_BUCKET, MODEL_KEY, buf)
    buf.seek(0)
    return joblib.load(buf)

@st.cache_data(show_spinner="Loading historical data…")
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
                                   index=4)

    is_holiday = st.checkbox("Federal holiday?", value=False)
    days_to_holiday = st.slider("Days to nearest holiday", 0, 30, 7)

    st.divider()
    use_flight_data = st.checkbox(
        "Include scheduled flight data",
        value=False,
        help="Check this if you have flight schedule data for the selected airport and date. "
             "Without it, the model relies solely on historical throughput patterns."
    )
    if use_flight_data:
        departures = st.number_input("Hourly departure count (peak hour)", 0, 80, 30)
        avg_seats  = st.number_input("Avg seats per flight", 50, 300, 155)
        occupancy_label = st.selectbox(
            "Typical flight occupancy",
            options=list(OCCUPANCY_OPTIONS.keys()),
            index=1,
        )
        load_factor = OCCUPANCY_OPTIONS[occupancy_label]
    else:
        departures  = 0
        avg_seats   = 0
        load_factor = 0.0

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

model         = artifact["model"]
le_airport    = artifact["leAirport"]
le_checkpoint = artifact["leCheckpoint"]
features      = artifact["features"]

# ── Build prediction rows for every hour × every checkpoint ───────────────────
try:
    history = load_history()
    cp_list = sorted(history.loc[history["airportCode"] == airport, "checkpointName"].unique())
except Exception:
    cp_list = ["Main Checkpoint"]

hours = list(range(4, 24))

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
    "hour":       [r["hour"]                        for r in rows],
    "checkpoint": [cp_list[i % len(cp_list)]        for i in range(len(rows))],
    "predicted":  np.maximum(preds, 0),
})

# ── Aggregate to airport-hour totals ─────────────────────────────────────────
hourly_total = result_df.groupby("hour")["predicted"].sum().reset_index()

# ── Historical average ────────────────────────────────────────────────────────
try:
    _hist_filtered = history.loc[
        (history["airportCode"] == airport) &
        (history["month"] == month) &
        (history["dayOfWeek"] == day_of_week)
    ]
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

FONT = dict(family="Merriweather, Georgia, serif", size=13, color="#1a1a1a")
TICK = dict(family="Source Sans 3, Arial, sans-serif", size=12, color="#333333")

# ── Charts ────────────────────────────────────────────────────────────────────
st.subheader(f"Hourly Throughput Forecast — {airport}  ·  {DAY_NAMES[day_of_week]}, {pd.Timestamp(2025,month,1).strftime('%B')}")

fig = go.Figure()
fig.add_trace(go.Bar(
    x=hourly_total["hour"],
    y=hourly_total["predicted"],
    name="Model Forecast",
    marker_color="#001f4d",
    opacity=0.88,
))
if show_hist and "historical_avg" in hourly_total.columns:
    fig.add_trace(go.Scatter(
        x=hourly_total["hour"],
        y=hourly_total["historical_avg"],
        name="Historical Average",
        mode="lines+markers",
        line=dict(color="#9b7e00", width=2.5, dash="dot"),
        marker=dict(size=6),
    ))
fig.update_layout(
    xaxis_title="Hour of Day",
    yaxis_title="Predicted Passengers",
    legend=dict(
        orientation="h", yanchor="top", y=-0.18,
        xanchor="center", x=0.5,
        font=dict(family="Source Sans 3, Arial, sans-serif", size=13, color="#1a1a1a"),
        bgcolor="rgba(0,0,0,0)",
    ),
    height=450,
    plot_bgcolor="white",
    paper_bgcolor="#f4f4f1",
    font=FONT,
    xaxis=dict(title_font=FONT, tickfont=TICK),
    yaxis=dict(title_font=FONT, tickfont=TICK),
    margin=dict(b=80),
)
fig.update_xaxes(tickmode="linear", dtick=1, gridcolor="#e0e0e0", linecolor="#cccccc")
fig.update_yaxes(gridcolor="#e0e0e0", linecolor="#cccccc")
st.plotly_chart(fig, use_container_width=True)

# ── Per-checkpoint breakdown ──────────────────────────────────────────────────
st.subheader("Breakdown by Checkpoint (peak hour)")
peak_hour = hourly_total.loc[hourly_total["predicted"].idxmax(), "hour"]
peak_df   = result_df[result_df["hour"] == peak_hour].sort_values("predicted", ascending=True)

fig2 = px.bar(peak_df, x="predicted", y="checkpoint", orientation="h",
              labels={"predicted": "Predicted Passengers", "checkpoint": ""},
              color_discrete_sequence=["#1a3a6b"])
fig2.update_layout(
    height=max(300, len(cp_list) * 45),
    plot_bgcolor="white",
    paper_bgcolor="#f4f4f1",
    font=FONT,
    xaxis=dict(title_font=FONT, tickfont=TICK, gridcolor="#e0e0e0"),
    yaxis=dict(tickfont=TICK),
)
fig2.update_xaxes(gridcolor="#e0e0e0")
st.plotly_chart(fig2, use_container_width=True)

# ── Summary metrics ───────────────────────────────────────────────────────────
st.subheader("Summary")
col1, col2, col3 = st.columns(3)
col1.metric("Peak hour",             f"{peak_hour}:00")
col2.metric("Peak predicted volume", f"{int(hourly_total['predicted'].max()):,} pax")
col3.metric("Total daily forecast",  f"{int(hourly_total['predicted'].sum()):,} pax")

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
