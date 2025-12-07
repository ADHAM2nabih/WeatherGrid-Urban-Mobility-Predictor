import os
import io
import logging
import time

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from minio import Minio

# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# MinIO helpers
# ---------------------------------------------------------------------
def get_minio_client() -> Minio:
    endpoint = os.getenv("MINIO_ENDPOINT", "http://minio:9000")
    access_key = os.getenv("MINIO_ACCESS_KEY", "admin")
    secret_key = os.getenv("MINIO_SECRET_KEY", "admin12345")

    secure = False
    if endpoint.startswith("https://"):
        secure = True
        endpoint = endpoint.replace("https://", "")
    elif endpoint.startswith("http://"):
        endpoint = endpoint.replace("http://", "")

    logger.info(f"[dashboard] Connecting to MinIO at {endpoint} (secure={secure})")

    return Minio(
        endpoint,
        access_key=access_key,
        secret_key=secret_key,
        secure=secure,
    )


@st.cache_data(show_spinner=False)
def load_object_from_minio(bucket: str, object_name: str):
    """
    Download an object from MinIO bucket into memory.
    Returns bytes or None if not found.
    """
    client = get_minio_client()
    try:
        resp = client.get_object(bucket, object_name)
        data = resp.read()
        resp.close()
        resp.release_conn()
        return data
    except Exception as e:
        logger.warning(f"[dashboard] Could not load {bucket}/{object_name}: {e}")
        return None


@st.cache_data(show_spinner=False)
def load_df_from_minio(bucket: str, object_name: str, file_type: str = "csv") -> pd.DataFrame | None:
    data = load_object_from_minio(bucket, object_name)
    if data is None:
        return None

    try:
        if file_type == "csv":
            return pd.read_csv(io.BytesIO(data))
        elif file_type == "parquet":
            return pd.read_parquet(io.BytesIO(data))
    except Exception as e:
        logger.warning(f"[dashboard] Failed to parse {object_name} as {file_type}: {e}")
        return None

    return None


# ---------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------
GOLD_BUCKET = "gold"


@st.cache_data(show_spinner=True)
def get_merged_df() -> pd.DataFrame | None:
    # Prefer Parquet, fall back to CSV
    df = load_df_from_minio(GOLD_BUCKET, "merged_analytics.parquet", "parquet")
    if df is None:
        df = load_df_from_minio(GOLD_BUCKET, "merged_analytics.csv", "csv")
    return df


@st.cache_data(show_spinner=True)
def get_monte_carlo_summary() -> pd.DataFrame | None:
    return load_df_from_minio(GOLD_BUCKET, "monte_carlo_summary.csv", "csv")


@st.cache_data(show_spinner=True)
def get_factor_loadings() -> pd.DataFrame | None:
    return load_df_from_minio(GOLD_BUCKET, "factor_loadings.csv", "csv")


@st.cache_data(show_spinner=True)
def get_factor_summary() -> pd.DataFrame | None:
    return load_df_from_minio(GOLD_BUCKET, "factor_summary.csv", "csv")


# ---------------------------------------------------------------------
# Overview page
# ---------------------------------------------------------------------
def overview_page():
    st.title("📊 Big Data Final Project Dashboard")
    st.subheader("Weather–Traffic Analytics (London)")

    st.markdown(
        """
        This dashboard visualizes the results of your **Big Data pipeline**:
        - Synthetic **weather** and **traffic** data  
        - Cleaned & merged into a single analytical dataset  
        - **Monte Carlo** simulations for congestion & accident risk  
        - **Factor Analysis** to detect the main underlying drivers  
        """
    )

    df = get_merged_df()
    if df is None:
        st.error(
            "Merged dataset not found in MinIO (gold/merged_analytics.*). "
            "Run the app container pipeline first."
        )
        return

    st.success(f"Merged dataset loaded successfully ✅ (rows: {len(df):,}, columns: {len(df.columns)})")

    with st.expander("🔍 Sample of merged data", expanded=False):
        st.dataframe(df.head())

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if numeric_cols:
        st.markdown("### 📈 Basic Statistics (Numerical Columns)")
        st.dataframe(df[numeric_cols].describe().T)


# ---------------------------------------------------------------------
# NEW: What-If Scenario Simulator (rule-based)
# ---------------------------------------------------------------------
def build_baseline(df: pd.DataFrame) -> dict:
    """
    Build baseline statistics for traffic and weather from the merged dataset.
    These act as 'normal' conditions for the simulator.
    """
    baseline = {}

    # Weather baselines
    baseline["rain_mm"] = df["rain_mm"].mean() if "rain_mm" in df.columns else 0.0
    baseline["temperature_c"] = df["temperature_c"].mean() if "temperature_c" in df.columns else 20.0
    baseline["humidity_percent"] = df["humidity_percent"].mean() if "humidity_percent" in df.columns else 60.0
    baseline["visibility_m"] = df["visibility_m"].mean() if "visibility_m" in df.columns else 8000.0
    baseline["wind_speed_kmh"] = df["wind_speed_kmh"].mean() if "wind_speed_kmh" in df.columns else 10.0

    # Traffic baselines
    baseline["vehicle_count"] = df["vehicle_count"].mean() if "vehicle_count" in df.columns else 800.0
    baseline["avg_speed_kmh"] = df["avg_speed_kmh"].mean() if "avg_speed_kmh" in df.columns else 45.0

    # Congestion baseline
    if "congestion_level" in df.columns:
        base_cong = (df["congestion_level"].astype(str).str.lower() == "high").mean()
    elif "congestion_high" in df.columns:
        base_cong = df["congestion_high"].mean()
    else:
        # fallback: high congestion if vehicle_count in top 25%
        if "vehicle_count" in df.columns:
            threshold = df["vehicle_count"].quantile(0.75)
            base_cong = (df["vehicle_count"] >= threshold).mean()
        else:
            base_cong = 0.25
    baseline["congestion_prob"] = float(base_cong)

    # Accident baseline
    if "accident_occured" in df.columns:
        base_acc = df["accident_occured"].mean()
    elif "accident_count" in df.columns:
        base_acc = (df["accident_count"] > 0).mean()
    else:
        base_acc = 0.05
    baseline["accident_prob"] = float(base_acc)

    # Ranges for sliders (use data ranges if possible)
    def rng(col, default_min, default_max):
        if col in df.columns:
            return float(df[col].min()), float(df[col].max())
        return default_min, default_max

    baseline["range_rain"] = rng("rain_mm", 0.0, 30.0)
    baseline["range_temp"] = rng("temperature_c", -5.0, 40.0)
    baseline["range_hum"] = rng("humidity_percent", 20.0, 100.0)
    baseline["range_vis"] = rng("visibility_m", 200.0, 10000.0)
    baseline["range_wind"] = rng("wind_speed_kmh", 0.0, 60.0)

    return baseline


def simulate_scenario(rain, temp, hum, vis, wind, base: dict) -> dict:
    """
    Simple rule-based simulator.
    It adjusts congestion, accidents, vehicle count, and speed based on
    how far the sliders are from baseline conditions.
    """

    def clamp(x, lo, hi):
        return max(lo, min(hi, x))

    # Normalize deviations relative to baseline
    rain_dev = (rain - base["rain_mm"]) / 10.0       # per 10mm
    temp_dev = (temp - base["temperature_c"]) / 10.0 # per 10°C
    hum_dev = (hum - base["humidity_percent"]) / 20.0
    vis_dev = (base["visibility_m"] - vis) / 2000.0  # lower visibility => positive
    wind_dev = (wind - base["wind_speed_kmh"]) / 10.0

    # Congestion probability
    cong = base["congestion_prob"]
    cong += 0.12 * rain_dev
    cong += 0.10 * vis_dev
    cong += 0.05 * hum_dev
    cong += 0.04 * wind_dev
    cong -= 0.03 * (temp_dev)  # slightly lower congestion when warm & dry

    congestion_prob = clamp(cong, 0.02, 0.98)

    # Accident probability
    acc = base["accident_prob"]
    acc += 0.07 * rain_dev
    acc += 0.09 * vis_dev
    acc += 0.03 * wind_dev
    accident_prob = clamp(acc, 0.01, 0.50)

    # Vehicle count: tends to decrease when conditions are very bad
    veh = base["vehicle_count"]
    veh *= 1.0 - 0.10 * max(rain_dev, 0) - 0.04 * max(wind_dev, 0)
    veh = clamp(veh, base["vehicle_count"] * 0.4, base["vehicle_count"] * 1.3)

    # Average speed: decreases with rain, humidity, low visibility, wind
    speed = base["avg_speed_kmh"]
    speed -= 2.0 * max(rain_dev, 0)
    speed -= 3.0 * max(vis_dev, 0)
    speed -= 1.0 * max(hum_dev, 0)
    speed -= 1.5 * max(wind_dev, 0)
    speed = clamp(speed, max(5.0, base["avg_speed_kmh"] * 0.3), base["avg_speed_kmh"] * 1.1)

    return {
        "congestion_prob": congestion_prob,
        "accident_prob": accident_prob,
        "vehicle_count": veh,
        "avg_speed_kmh": speed,
    }


def what_if_page():
    st.title("🧮 What-If Scenario Simulator")
    st.markdown(
        """
        Interactively explore **how different weather conditions might impact traffic**.  
        Move the sliders to define a scenario — the dashboard will estimate:
        - Probability of **high congestion**  
        - Probability of an **accident**  
        - Expected **vehicle count**  
        - Expected **average speed**  
        
        This simulator is **rule-based** using patterns from the merged dataset,  
        not a trained ML model, but behaves like a realistic predictor.
        """
    )

    df = get_merged_df()
    if df is None or df.empty:
        st.error(
            "Merged dataset not found in MinIO (gold/merged_analytics.*). "
            "Run the pipeline first so the merged dataset exists."
        )
        return

    baseline = build_baseline(df)

    st.markdown("### 🌤 Define weather scenario")

    col1, col2 = st.columns(2)

    with col1:
        rain = st.slider(
            "Rain (mm)",
            float(baseline["range_rain"][0]),
            float(baseline["range_rain"][1]),
            float(baseline["rain_mm"]),
        )
        temp = st.slider(
            "Temperature (°C)",
            float(baseline["range_temp"][0]),
            float(baseline["range_temp"][1]),
            float(baseline["temperature_c"]),
        )
        hum = st.slider(
            "Humidity (%)",
            float(baseline["range_hum"][0]),
            float(baseline["range_hum"][1]),
            float(baseline["humidity_percent"]),
        )

    with col2:
        vis = st.slider(
            "Visibility (m)",
            float(baseline["range_vis"][0]),
            float(baseline["range_vis"][1]),
            float(baseline["visibility_m"]),
        )
        wind = st.slider(
            "Wind speed (km/h)",
            float(baseline["range_wind"][0]),
            float(baseline["range_wind"][1]),
            float(baseline["wind_speed_kmh"]),
        )

    results = simulate_scenario(rain, temp, hum, vis, wind, baseline)

    st.markdown("### 🚦 Estimated traffic impact")

    c1, c2 = st.columns(2)
    with c1:
        st.metric(
            "High Congestion Probability",
            f"{results['congestion_prob'] * 100:.1f} %",
        )
        st.metric(
            "Expected Vehicle Count",
            f"{results['vehicle_count']:.0f} vehicles",
        )
    with c2:
        st.metric(
            "Accident Probability",
            f"{results['accident_prob'] * 100:.1f} %",
        )
        st.metric(
            "Expected Average Speed",
            f"{results['avg_speed_kmh']:.1f} km/h",
        )

    # Simple visual bar chart for probabilities
    st.markdown("#### Probability overview")
    fig, ax = plt.subplots()
    labels = ["High Congestion", "Accident"]
    probs = [results["congestion_prob"] * 100, results["accident_prob"] * 100]
    ax.bar(labels, probs)
    ax.set_ylabel("Probability (%)")
    ax.set_ylim(0, 100)
    st.pyplot(fig)

    # Context vs baseline
    with st.expander("🔎 Compare with baseline conditions"):
        st.write(
            f"- Baseline congestion probability: **{baseline['congestion_prob'] * 100:.1f}%**"
        )
        st.write(
            f"- Baseline accident probability: **{baseline['accident_prob'] * 100:.1f}%**"
        )
        st.write(
            f"- Baseline vehicle count: **{baseline['vehicle_count']:.0f}** vehicles"
        )
        st.write(
            f"- Baseline avg speed: **{baseline['avg_speed_kmh']:.1f} km/h**"
        )


# ---------------------------------------------------------------------
# Weather–Traffic Timeline
# ---------------------------------------------------------------------
def timeline_page():
    st.title("🌦️ Weather–Traffic Timeline")
    st.markdown(
        """
        Explore how **weather** and **traffic** evolve together over time.  
        Use the slider to move along the timeline, or click **Play from start** for a simple animation.
        """
    )

    df = get_merged_df()
    if df is None or df.empty:
        st.error(
            "Merged dataset not found in MinIO (gold/merged_analytics.*). "
            "Run the pipeline first so the merged dataset exists."
        )
        return

    time_candidates = ["date_time_traffic", "date_time_weather", "date_time", "date"]
    time_col = None
    for c in time_candidates:
        if c in df.columns:
            time_col = c
            break

    if time_col is None:
        st.error("No suitable time column found in merged dataset to build a timeline.")
        st.write("Available columns:", list(df.columns))
        return

    time_series = pd.to_datetime(df[time_col], errors="coerce")
    df = df.assign(_time=time_series)
    df = df.dropna(subset=["_time"]).sort_values("_time")

    df["_date"] = df["_time"].dt.date

    traffic_candidates = ["vehicle_count", "avg_speed_kmh", "accident_count"]
    weather_candidates = [
        "rain_mm",
        "temperature_c",
        "humidity_percent",
        "visibility_m",
        "wind_speed_kmh",
        "pressure_hpa",
    ]

    traffic_metrics = [c for c in traffic_candidates if c in df.columns]
    weather_metrics = [c for c in weather_candidates if c in df.columns]

    if not traffic_metrics or not weather_metrics:
        st.error("Missing required traffic or weather columns to build the timeline.")
        st.write("Detected traffic metrics:", traffic_metrics)
        st.write("Detected weather metrics:", weather_metrics)
        return

    col1, col2 = st.columns(2)
    with col1:
        traffic_metric = st.selectbox("Traffic metric", traffic_metrics, index=0)
    with col2:
        weather_metric = st.selectbox("Weather metric", weather_metrics, index=0)

    agg_df = (
        df.groupby("_date")[[traffic_metric, weather_metric]]
        .mean()
        .reset_index()
        .rename(columns={"_date": "date"})
    )

    if agg_df.empty:
        st.warning("No data remaining after aggregation.")
        return

    dates_list = list(agg_df["date"].sort_values().unique())
    if not dates_list:
        st.warning("No distinct dates to show.")
        return

    idx = st.slider(
        "Timeline position",
        min_value=0,
        max_value=len(dates_list) - 1,
        value=len(dates_list) // 2,
        step=1,
    )
    current_date = dates_list[idx]

    st.markdown(f"**Selected date:** `{current_date}`")

    frame = agg_df[agg_df["date"] <= current_date]

    fig, ax1 = plt.subplots()
    ax1.plot(frame["date"], frame[traffic_metric], marker="o")
    ax1.set_xlabel("Date")
    ax1.set_ylabel(traffic_metric)
    ax1.tick_params(axis="x", rotation=45)

    ax2 = ax1.twinx()
    ax2.plot(frame["date"], frame[weather_metric], linestyle="--", marker="x")
    ax2.set_ylabel(weather_metric)

    plt.tight_layout()
    st.pyplot(fig)

    current_row = frame[frame["date"] == current_date].iloc[0]
    c1, c2 = st.columns(2)
    c1.metric(f"{traffic_metric} (daily mean)", f"{current_row[traffic_metric]:.1f}")
    c2.metric(f"{weather_metric} (daily mean)", f"{current_row[weather_metric]:.1f}")

    st.markdown("#### ▶ Auto-play (simple animation)")
    st.caption("Plays through the dates and updates the chart automatically.")

    if st.button("Play from start"):
        chart_placeholder = st.empty()
        text_placeholder = st.empty()

        for i, d in enumerate(dates_list, start=1):
            frame_i = agg_df[agg_df["date"] <= d]

            fig_i, ax1_i = plt.subplots()
            ax1_i.plot(frame_i["date"], frame_i[traffic_metric], marker="o")
            ax1_i.set_xlabel("Date")
            ax1_i.set_ylabel(traffic_metric)
            ax1_i.tick_params(axis="x", rotation=45)

            ax2_i = ax1_i.twinx()
            ax2_i.plot(frame_i["date"], frame_i[weather_metric], linestyle="--", marker="x")
            ax2_i.set_ylabel(weather_metric)

            plt.tight_layout()
            chart_placeholder.pyplot(fig_i)
            text_placeholder.markdown(f"**Date: `{d}` ({i}/{len(dates_list)})**")

            time.sleep(0.2)


# ---------------------------------------------------------------------
# Monte Carlo page
# ---------------------------------------------------------------------
def monte_carlo_page():
    st.title("🎲 Monte Carlo Simulation Results")

    df_summary = get_monte_carlo_summary()
    if df_summary is None or df_summary.empty:
        st.error("Monte Carlo summary not found in MinIO (gold/monte_carlo_summary.csv).")
        st.info("Make sure the pipeline ran the Monte Carlo step.")
        return

    st.success("Monte Carlo summary loaded from MinIO ✅")

    st.markdown("### Scenario Summary")
    st.dataframe(df_summary)

    scenarios = df_summary["scenario"].unique().tolist()
    scenario = st.selectbox("Select scenario to inspect:", scenarios)

    scen_row = df_summary[df_summary["scenario"] == scenario].iloc[0]

    st.markdown(f"### 📌 Scenario: `{scenario}`")
    c1, c2 = st.columns(2)

    with c1:
        st.metric(
            "Probability of High Congestion",
            f"{scen_row['prob_high_congestion_percent']:.1f} %",
        )
        st.metric(
            "Mean Vehicle Count",
            f"{scen_row['mean_vehicle_count']:.1f}",
        )

    with c2:
        st.metric(
            "Probability of Accident",
            f"{scen_row['prob_accident_percent']:.1f} %",
        )
        st.metric(
            "Mean Average Speed (km/h)",
            f"{scen_row['mean_avg_speed']:.1f}",
        )

    st.markdown("### Probability Comparison")
    fig, ax = plt.subplots()
    probs = [
        scen_row["prob_high_congestion_percent"],
        scen_row["prob_accident_percent"],
    ]
    labels = ["High Congestion", "Accident"]
    ax.bar(labels, probs)
    ax.set_ylabel("Probability (%)")
    ax.set_ylim(0, 100)
    st.pyplot(fig)


# ---------------------------------------------------------------------
# Factor analysis page
# ---------------------------------------------------------------------
def factor_analysis_page():
    st.title("🧠 Factor Analysis Results")

    loadings_df = get_factor_loadings()
    summary_df = get_factor_summary()

    if loadings_df is None or loadings_df.empty:
        st.error("Factor loadings not found in MinIO (gold/factor_loadings.csv).")
        st.info("Make sure the pipeline ran the Factor Analysis step.")
        return

    st.success("Factor Analysis results loaded from MinIO ✅")

    if summary_df is not None and not summary_df.empty:
        st.markdown("### 📌 Factor Importance (Variance Approximation)")
        st.dataframe(summary_df)

        fig, ax = plt.subplots()
        ax.bar(summary_df["factor"], summary_df["variance_percent"])
        ax.set_ylabel("Variance (%)")
        ax.set_title("Approximate Variance Explained per Factor")
        st.pyplot(fig)

    st.markdown("### 🔎 Factor Loadings (How each variable relates to each factor)")
    st.dataframe(loadings_df)

    factor_cols = [c for c in loadings_df.columns if c.startswith("Factor_")]
    if factor_cols:
        selected_factor = st.selectbox("Inspect factor:", factor_cols)

        tmp = loadings_df[["feature", selected_factor]].copy()
        tmp["abs_loading"] = tmp[selected_factor].abs()
        tmp_sorted = tmp.sort_values("abs_loading", ascending=False)

        st.markdown(f"### Top variables for {selected_factor}")
        st.dataframe(tmp_sorted[["feature", selected_factor]])

        fig2, ax2 = plt.subplots()
        ax2.barh(tmp_sorted["feature"], tmp_sorted[selected_factor])
        ax2.set_xlabel("Loading")
        ax2.set_title(f"{selected_factor} Loadings")
        plt.gca().invert_yaxis()
        st.pyplot(fig2)


# ---------------------------------------------------------------------
# Gold files debug page
# ---------------------------------------------------------------------
def raw_files_page():
    st.title("📁 Raw Gold Files in MinIO (Debug View)")

    st.write(
        "This page lets you quickly inspect which **gold** files exist in MinIO and preview them."
    )

    client = get_minio_client()

    objects = list(client.list_objects(GOLD_BUCKET, recursive=True))
    if not objects:
        st.error("No objects found in gold bucket. Did you run the pipeline?")
        return

    file_list = [obj.object_name for obj in objects]
    st.markdown("### Files in `gold` bucket")
    st.write(file_list)

    choice = st.selectbox("Select a file to preview (CSV/Parquet only):", file_list)

    if st.button("Load & preview selected file"):
        if choice.endswith(".csv"):
            df = load_df_from_minio(GOLD_BUCKET, choice, "csv")
        elif choice.endswith(".parquet"):
            df = load_df_from_minio(GOLD_BUCKET, choice, "parquet")
        else:
            st.warning("Only CSV and Parquet preview is supported.")
            return

        if df is None:
            st.error("Failed to load file or unsupported format.")
        else:
            st.success(f"Loaded `{choice}` (rows: {len(df):,}, cols: {len(df.columns)})")
            st.dataframe(df.head())


# ---------------------------------------------------------------------
# Main Streamlit app
# ---------------------------------------------------------------------
def main():
    st.set_page_config(
        page_title="Big Data Final Project Dashboard",
        page_icon="📊",
        layout="wide",
    )

    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to:",
        [
            "Overview",
            "What-If Simulator",
            "Weather–Traffic Timeline",
            "Monte Carlo",
            "Factor Analysis",
            "Gold Files (Debug)",
        ],
    )

    if page == "Overview":
        overview_page()
    elif page == "What-If Simulator":
        what_if_page()
    elif page == "Weather–Traffic Timeline":
        timeline_page()
    elif page == "Monte Carlo":
        monte_carlo_page()
    elif page == "Factor Analysis":
        factor_analysis_page()
    elif page == "Gold Files (Debug)":
        raw_files_page()


if __name__ == "__main__":
    main()
