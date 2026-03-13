# app.py


from __future__ import annotations

from pathlib import Path
from datetime import datetime

import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt



# Unit conversion

MGDL_PER_MMOLL = 18.0


def mgdl_to_mmoll(x: float) -> float:
    return float(x) / MGDL_PER_MMOLL


def mmoll_to_mgdl(x: float) -> float:
    return float(x) * MGDL_PER_MMOLL



# Paths 

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data_app"
LOG_PATH = DATA_DIR / "user_log.csv"
MODEL_PATH = BASE_DIR / "models" / "app_history_model.joblib"


# Model + log helpers

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at:\n{MODEL_PATH}\n\n"
            "Please run:\n"
            "    python model_app.py\n"
            "to create models/app_history_model.joblib"
        )
    return joblib.load(MODEL_PATH)


def load_log() -> pd.DataFrame:
    if not LOG_PATH.exists():
        return pd.DataFrame(columns=["timestamp", "bg", "carbs", "insulin", "steps"])
    return pd.read_csv(LOG_PATH, parse_dates=["timestamp"])


def save_log(df: pd.DataFrame) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(LOG_PATH, index=False)


def prep_log(df: pd.DataFrame) -> pd.DataFrame:
    """Clean types + sort for stable behavior. Stored bg is mmol/L."""
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")

    for c in ["bg", "carbs", "insulin", "steps"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # bg is required; other fields can be 0 if missing
    df = df.dropna(subset=["bg"])
    df["carbs"] = df["carbs"].fillna(0.0)
    df["insulin"] = df["insulin"].fillna(0.0)
    df["steps"] = df["steps"].fillna(0.0)

    return df



# Feature builder (MUST match model_app.py)
# Stored bg is mmol/L and model expects mmol/L.

def build_features_from_log(df: pd.DataFrame, now_ts: pd.Timestamp) -> pd.DataFrame | None:
    """
    Build SAME features used in model_app.py (3-hour version):
    ['bg_now', 'bg_mean_3h', 'bg_std_3h', 'carbs_sum_3h', 'insulin_sum_3h', 'steps_sum_3h', 'bg_trend_60m']

    Uses last 180 minutes of logs.
    Trend uses closest reading to 60 minutes ago.
    """
    df = df.sort_values("timestamp")

    window = df[df["timestamp"] >= (now_ts - pd.Timedelta(minutes=180))]
    if window.empty:
        return None

    # Current (most recent) bg (mmol/L)
    bg_now = float(window["bg"].iloc[-1])

    # Stats over last 3 hours (mmol/L)
    bg_mean_3h = float(window["bg"].mean())
    bg_std_3h_val = window["bg"].std()
    bg_std_3h = float(0.0 if pd.isna(bg_std_3h_val) else bg_std_3h_val)

    # Sums over last 3 hours
    carbs_sum_3h = float(window["carbs"].sum())
    insulin_sum_3h = float(window["insulin"].sum())
    steps_sum_3h = float(window["steps"].sum())

    # Trend over 60 minutes: now - ~60 mins ago
    target_time = now_ts - pd.Timedelta(minutes=60)
    idx = (window["timestamp"] - target_time).abs().idxmin()
    bg_60m = float(window.loc[idx, "bg"])
    bg_trend_60m = float(bg_now - bg_60m)

    X = pd.DataFrame([{
        "bg_now": bg_now,
        "bg_mean_3h": bg_mean_3h,
        "bg_std_3h": bg_std_3h,
        "carbs_sum_3h": carbs_sum_3h,
        "insulin_sum_3h": insulin_sum_3h,
        "steps_sum_3h": steps_sum_3h,
        "bg_trend_60m": bg_trend_60m,
    }])

    return X



# Chart helpers
# Display glucose in mg/dL (convert from stored mmol/L)
# pred is in mmol/L

def plot_bg_line(df_view: pd.DataFrame, now_ts: pd.Timestamp, pred_mmoll: float | None):
    fig, ax = plt.subplots()

    # Actual glucose line (mg/dL)
    ax.plot(
        df_view["timestamp"],
        df_view["bg"] * MGDL_PER_MMOLL,
        marker="o",
        linewidth=1.5,
        label="bg (mg/dL)",
    )

    # Shade last 3 hours window
    start_3h = now_ts - pd.Timedelta(hours=3)
    ax.axvspan(start_3h, now_ts, alpha=0.15, label="last 3 hours")

    # Predicted point at +1 hour + dotted connector from last actual to future
    if pred_mmoll is not None and not df_view.empty:
        pred_mgdl = pred_mmoll * MGDL_PER_MMOLL
        pred_time = now_ts + pd.Timedelta(hours=1)

        last_time = df_view["timestamp"].iloc[-1]
        last_bg_mgdl = float(df_view["bg"].iloc[-1]) * MGDL_PER_MMOLL

        # dotted connector
        ax.plot(
            [last_time, pred_time],
            [last_bg_mgdl, pred_mgdl],
            linestyle="--",
            linewidth=1.5,
            label="to predicted +1h",
        )

        # future prediction point
        ax.scatter(
            [pred_time],
            [pred_mgdl],
            s=80,
            marker="o",
            label=f"pred +1h: {pred_mgdl:.0f} mg/dL",
        )

    ax.set_title("Glucose over time")
    ax.set_xlabel("time")
    ax.set_ylabel("bg (mg/dL)")
    ax.legend()
    st.pyplot(fig)


def plot_bars(df_view: pd.DataFrame, col: str, title: str, ylab: str):
    fig, ax = plt.subplots()
    ax.bar(df_view["timestamp"], df_view[col])
    ax.set_title(title)
    ax.set_xlabel("time")
    ax.set_ylabel(ylab)
    st.pyplot(fig)



# Streamlit UI

st.set_page_config(page_title="Glucose Forecast Dashboard", layout="wide")

st.title("Glucose Forecast Dashboard")
st.subheader(
    "Log your glucose, carbs, insulin, and activity. The app uses your last 3 hours of history "
    "to forecast your glucose 1 hour ahead."
)
st.caption("Inputs and charts are shown in mg/dL.")

model = load_model()
log = prep_log(load_log())

# Sidebar controls
st.sidebar.header("Add a log entry")
ts = st.sidebar.datetime_input("Timestamp", value=datetime.now())

bg_mgdl = st.sidebar.number_input(
    "Glucose (mg/dL)",
    min_value=20.0,
    max_value=600.0,
    value=110.0,
    step=1.0,
)

carbs = st.sidebar.number_input(
    "Carbs since last entry (grams)",
    min_value=0.0,
    max_value=500.0,
    value=0.0,
    step=1.0,
)

insulin = st.sidebar.number_input(
    "Insulin since last entry (units)",
    min_value=0.0,
    max_value=100.0,
    value=0.0,
    step=0.5,
)

steps = st.sidebar.number_input(
    "Steps since last entry",
    min_value=0.0,
    max_value=50000.0,
    value=0.0,
    step=50.0,
)

if st.sidebar.button("Save entry"):
    # Convert mg/dL -> mmol/L for storage + model consistency
    bg_mmoll = mgdl_to_mmoll(float(bg_mgdl))

    new_row = pd.DataFrame([{
        "timestamp": pd.to_datetime(ts),
        "bg": float(bg_mmoll),  # stored as mmol/L
        "carbs": float(carbs),
        "insulin": float(insulin),
        "steps": float(steps),
    }])

    log = pd.concat([log, new_row], ignore_index=True)
    log = prep_log(log)
    save_log(log)

    st.sidebar.success("Saved!")
    st.rerun()

st.sidebar.divider()

if st.sidebar.button("🧹 Clear all logs"):
    save_log(pd.DataFrame(columns=["timestamp", "bg", "carbs", "insulin", "steps"]))
    st.sidebar.success("Cleared!")
    st.rerun()

if not log.empty:
    # Download log in mg/dL for user convenience
    log_dl = log.copy()
    log_dl["bg"] = log_dl["bg"].apply(mmoll_to_mgdl)

    csv_bytes = log_dl.to_csv(index=False).encode("utf-8")
    st.sidebar.download_button(
        "⬇️ Download log CSV",
        data=csv_bytes,
        file_name="user_log.csv",
        mime="text/csv",
    )

# Main layout
left, right = st.columns([2.2, 1])

with left:
    st.subheader("Your log (latest first)")

    if log.empty:
        st.info("No entries yet. Add entries from the sidebar.")
    else:
        # Display bg in mg/dL
        log_display = log.copy()
        log_display["bg"] = log_display["bg"].apply(mmoll_to_mgdl)

        st.dataframe(
            log_display.sort_values("timestamp", ascending=False).head(100),
            use_container_width=True,
        )

with right:
    st.subheader("Prediction (+1 hour)")

    pred_mmoll = None
    now_ts = pd.to_datetime(ts)

    if log.empty:
        st.warning("Add at least 1 entry to predict.")
    else:
        earliest = log["timestamp"].min()
        minutes_of_history = (now_ts - earliest).total_seconds() / 60.0
        if minutes_of_history < 60:
            st.info("Tip: Predictions get better once you have ~60+ minutes of logs.")

        X = build_features_from_log(log, now_ts)

        if X is None:
            st.warning("Not enough data in the last 3 hours.")
        else:
            pred_mmoll = float(model.predict(X)[0])
            pred_mgdl = mmoll_to_mgdl(pred_mmoll)

            # Predicted glucose on a new line (clean)
            st.write("Predicted glucose (+1 hour)")
            st.markdown(f"### {pred_mgdl:.0f} mg/dL")

            with st.expander("What the model used (3-hour features)"):
                st.dataframe(X, use_container_width=True)

st.divider()
st.subheader("📊 Charts")

if log.empty:
    st.info("Add entries to see charts.")
else:
    view = st.radio("Chart range", ["Last 3 hours", "Last 24 hours", "All"], horizontal=True)

    if view == "Last 3 hours":
        df_view = log[log["timestamp"] >= (now_ts - pd.Timedelta(hours=3))]
    elif view == "Last 24 hours":
        df_view = log[log["timestamp"] >= (now_ts - pd.Timedelta(hours=24))]
    else:
        df_view = log

    if df_view.empty:
        st.warning("No data in the selected time range.")
    else:
        plot_bg_line(df_view, now_ts, pred_mmoll)

        st.caption("These charts help you visually connect carbs/insulin/steps to glucose changes.")
        c1, c2, c3 = st.columns(3)
        with c1:
            plot_bars(df_view, "carbs", "Carbs over time", "grams")
        with c2:
            plot_bars(df_view, "insulin", "Insulin over time", "units")
        with c3:
            plot_bars(df_view, "steps", "Steps over time", "steps")