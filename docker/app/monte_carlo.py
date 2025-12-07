import os
import logging
from typing import Dict, Callable, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Reuse MinIO helpers from etl_pipeline to stay consistent
from etl_pipeline import (
    get_minio_client,
    ensure_bucket,
    upload_file_to_minio,
)

# ------------------------------------------------------------------------------
# Logging setup
# ------------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------------------
# Monte Carlo core logic
# ------------------------------------------------------------------------------

def load_merged_dataset() -> pd.DataFrame:
    """
    Load the merged analytical dataset created by the ETL step.
    Prefer Parquet; fall back to CSV.
    """
    parquet_path = "merged_analytics.parquet"
    csv_path = "merged_analytics.csv"

    if os.path.exists(parquet_path):
        logger.info(f"Loading merged dataset from {parquet_path}")
        df = pd.read_parquet(parquet_path)
    elif os.path.exists(csv_path):
        logger.info(f"Loading merged dataset from {csv_path}")
        df = pd.read_csv(csv_path)
    else:
        raise FileNotFoundError(
            "Merged dataset not found. Run ETL pipeline first to generate merged_analytics.*"
        )

    # Basic sanity check on key columns
    required_cols = [
        "city",
        "date_time",
        "vehicle_count",
        "avg_speed_kmh",
        "accident_count",
        "congestion_level",
        "rain_mm",
        "visibility_m",
        "wind_speed_kmh",
        "temperature_c",
        "humidity_percent",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        logger.warning(f"Merged dataset is missing columns: {missing}")

    return df


def define_scenarios() -> Dict[str, Callable[[pd.DataFrame], pd.Series]]:
    """
    Define Monte Carlo scenarios as filters on the merged dataset.

    Each scenario returns a boolean mask (Series[bool]) indicating
    which rows belong to that scenario.
    """
    scenarios = {}

    # 1) Heavy rain & low visibility
    def scenario_heavy_rain_low_vis(df: pd.DataFrame) -> pd.Series:
        return (df["rain_mm"] >= 10) & (df["visibility_m"] <= 2000)

    # 2) Foggy / very low visibility (regardless of rain)
    def scenario_fog_low_vis(df: pd.DataFrame) -> pd.Series:
        cond_vis = df["visibility_m"] <= 1000
        if "weather_condition" in df.columns:
            cond_weather = df["weather_condition"].str.contains(
                "Fog", case=False, na=False
            )
            return cond_vis | cond_weather
        return cond_vis

    # 3) Strong wind
    def scenario_strong_wind(df: pd.DataFrame) -> pd.Series:
        return df["wind_speed_kmh"] >= 40

    # 4) Hot & dry
    def scenario_hot_dry(df: pd.DataFrame) -> pd.Series:
        cond_temp = df["temperature_c"] >= 28
        cond_rain = df["rain_mm"] <= 1
        return cond_temp & cond_rain

    # 5) Normal / baseline conditions
    def scenario_normal(df: pd.DataFrame) -> pd.Series:
        return (
            (df["rain_mm"] < 5)
            & (df["visibility_m"] > 3000)
            & (df["wind_speed_kmh"] < 25)
        )

    scenarios["heavy_rain_low_visibility"] = scenario_heavy_rain_low_vis
    scenarios["fog_or_low_visibility"] = scenario_fog_low_vis
    scenarios["strong_wind"] = scenario_strong_wind
    scenarios["hot_and_dry"] = scenario_hot_dry
    scenarios["normal_conditions"] = scenario_normal

    return scenarios


def run_scenario_simulation(
    df: pd.DataFrame,
    scenario_name: str,
    mask: pd.Series,
    n_simulations: int = 5000,
) -> pd.DataFrame:
    """
    Run Monte Carlo simulation for a single scenario.

    Approach: empirical resampling
    - Filter df by mask (scenario subset)
    - Sample rows with replacement n_simulations times
    - For each sample, compute:
        - congestion_high: 1 if congestion_level == 'High'
        - accident_occured: 1 if accident_count > 0
        - vehicle_count
        - avg_speed_kmh

    Returns a DataFrame with one row per simulation.
    """
    scenario_df = df[mask].copy()

    if scenario_df.empty:
        logger.warning(f"Scenario '{scenario_name}' has no matching rows. Skipping.")
        return pd.DataFrame()

    logger.info(
        f"Running Monte Carlo for scenario '{scenario_name}' "
        f"with {len(scenario_df)} base rows and {n_simulations} simulations..."
    )

    # Standardize congestion_level to detect 'High'
    if "congestion_level" in scenario_df.columns:
        scenario_df["congestion_level_clean"] = (
            scenario_df["congestion_level"]
            .astype(str)
            .str.strip()
            .str.lower()
        )
    else:
        scenario_df["congestion_level_clean"] = ""

    # Ensure numeric accident_count
    scenario_df["accident_count"] = pd.to_numeric(
        scenario_df.get("accident_count", 0), errors="coerce"
    ).fillna(0)

    # We'll sample indices from scenario_df
    indices = scenario_df.index.to_list()
    if not indices:
        logger.warning(f"No indices available for scenario '{scenario_name}'.")
        return pd.DataFrame()

    samples_idx = np.random.choice(indices, size=n_simulations, replace=True)

    results = {
        "scenario": [],
        "run_id": [],
        "congestion_high": [],
        "accident_occured": [],
        "vehicle_count": [],
        "avg_speed_kmh": [],
    }

    for i, idx in enumerate(samples_idx, start=1):
        row = scenario_df.loc[idx]

        congestion_high = int(row["congestion_level_clean"] == "high")
        accident_occured = int(row["accident_count"] > 0)
        vehicle_count = row.get("vehicle_count", np.nan)
        avg_speed_kmh = row.get("avg_speed_kmh", np.nan)

        results["scenario"].append(scenario_name)
        results["run_id"].append(i)
        results["congestion_high"].append(congestion_high)
        results["accident_occured"].append(accident_occured)
        results["vehicle_count"].append(vehicle_count)
        results["avg_speed_kmh"].append(avg_speed_kmh)

    scenario_results_df = pd.DataFrame(results)
    return scenario_results_df


def summarize_results(sim_results: pd.DataFrame) -> pd.DataFrame:
    """
    Group by scenario and compute summary statistics:
    - probability of high congestion
    - probability of accident
    - mean vehicle count
    - mean avg speed
    """
    if sim_results.empty:
        logger.warning("No simulation results to summarize.")
        return pd.DataFrame()

    grouped = sim_results.groupby("scenario").agg(
        n_runs=("run_id", "count"),
        prob_high_congestion=("congestion_high", "mean"),
        prob_accident=("accident_occured", "mean"),
        mean_vehicle_count=("vehicle_count", "mean"),
        mean_avg_speed=("avg_speed_kmh", "mean"),
    )

    # Convert probabilities to percentages for readability
    grouped["prob_high_congestion_percent"] = grouped["prob_high_congestion"] * 100.0
    grouped["prob_accident_percent"] = grouped["prob_accident"] * 100.0

    return grouped.reset_index()


def save_plots_per_scenario(sim_results: pd.DataFrame, output_dir: str = "plots_monte_carlo"):
    """
    For each scenario, generate simple plots:
    - Histogram of vehicle_count
    - Histogram of avg_speed_kmh
    - Bar of congestion_high probability (0/1 distribution)

    Saves PNG files locally in output_dir.
    """
    if sim_results.empty:
        logger.warning("No simulation results. Skipping plot generation.")
        return

    os.makedirs(output_dir, exist_ok=True)

    scenarios = sim_results["scenario"].unique()
    for scen in scenarios:
        scen_df = sim_results[sim_results["scenario"] == scen]

        # Vehicle count histogram
        plt.figure()
        scen_df["vehicle_count"].dropna().hist(bins=30)
        plt.title(f"Vehicle Count Distribution - {scen}")
        plt.xlabel("Vehicle Count")
        plt.ylabel("Frequency")
        fname = os.path.join(output_dir, f"vehicle_count_{scen}.png")
        plt.savefig(fname, bbox_inches="tight")
        plt.close()

        # Average speed histogram
        plt.figure()
        scen_df["avg_speed_kmh"].dropna().hist(bins=30)
        plt.title(f"Average Speed Distribution - {scen}")
        plt.xlabel("Avg Speed (km/h)")
        plt.ylabel("Frequency")
        fname = os.path.join(output_dir, f"avg_speed_{scen}.png")
        plt.savefig(fname, bbox_inches="tight")
        plt.close()

        # Congestion high probability (bar of 0/1)
        plt.figure()
        scen_df["congestion_high"].value_counts(normalize=True).sort_index().plot(kind="bar")
        plt.title(f"High Congestion Probability - {scen}")
        plt.xlabel("congestion_high (0 = No, 1 = Yes)")
        plt.ylabel("Proportion")
        fname = os.path.join(output_dir, f"congestion_high_{scen}.png")
        plt.savefig(fname, bbox_inches="tight")
        plt.close()

        logger.info(f"Saved plots for scenario '{scen}' in {output_dir}/")


# ------------------------------------------------------------------------------
# Public entry point
# ------------------------------------------------------------------------------

def run_monte_carlo():
    """
    Main function:
    - Load merged dataset
    - Define scenarios
    - Run Monte Carlo simulations
    - Save detailed results and summaries
    - Upload them to MinIO gold bucket
    - Generate basic plots (saved locally)
    """
    logger.info("=== Monte Carlo simulation started ===")

    # 1) Load merged dataset
    df = load_merged_dataset()

    # 2) Define scenarios
    scenarios = define_scenarios()

    # 3) Run simulations for each scenario
    all_results: List[pd.DataFrame] = []
    for scen_name, scen_fn in scenarios.items():
        try:
            mask = scen_fn(df)
            scen_results = run_scenario_simulation(
                df,
                scenario_name=scen_name,
                mask=mask,
                n_simulations=5000,  # you can tune this number
            )
            if not scen_results.empty:
                all_results.append(scen_results)
        except Exception as e:
            logger.error(f"Error running scenario '{scen_name}': {e}")

    if not all_results:
        logger.warning("No scenario produced simulation results. Exiting Monte Carlo.")
        return

    sim_results = pd.concat(all_results, ignore_index=True)

    # 4) Summarize results
    summary_df = summarize_results(sim_results)

    # 5) Save to local files
    results_path = "monte_carlo_results.csv"
    summary_path = "monte_carlo_summary.csv"

    logger.info(f"Saving detailed simulation results -> {results_path}")
    sim_results.to_csv(results_path, index=False)

    logger.info(f"Saving summary results -> {summary_path}")
    summary_df.to_csv(summary_path, index=False)

    # 6) Generate plots per scenario
    save_plots_per_scenario(sim_results, output_dir="plots_monte_carlo")

    # 7) Upload outputs to MinIO gold bucket
    minio_client = get_minio_client()
    gold_bucket = "gold"
    ensure_bucket(minio_client, gold_bucket)

    upload_file_to_minio(
        minio_client,
        gold_bucket,
        "monte_carlo_results.csv",
        results_path,
        content_type="text/csv",
    )
    upload_file_to_minio(
        minio_client,
        gold_bucket,
        "monte_carlo_summary.csv",
        summary_path,
        content_type="text/csv",
    )

    # Optionally upload some plots (e.g., all PNGs)
    plots_dir = "plots_monte_carlo"
    if os.path.isdir(plots_dir):
        for fname in os.listdir(plots_dir):
            if fname.endswith(".png"):
                local_path = os.path.join(plots_dir, fname)
                object_name = f"monte_carlo_plots/{fname}"
                upload_file_to_minio(
                    minio_client,
                    gold_bucket,
                    object_name,
                    local_path,
                    content_type="image/png",
                )

    logger.info("✅ Monte Carlo simulation finished successfully.")


if __name__ == "__main__":
    run_monte_carlo()
