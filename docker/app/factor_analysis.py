import os
import logging
from typing import List

import numpy as np
import pandas as pd
from sklearn.decomposition import FactorAnalysis
from sklearn.preprocessing import StandardScaler

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
# Helpers
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

    return df


def select_numeric_features(df: pd.DataFrame) -> List[str]:
    """
    Select the numeric columns to be used in Factor Analysis.
    We focus on key weather + traffic variables.
    """
    candidates = [
        "temperature_c",
        "humidity_percent",
        "rain_mm",
        "wind_speed_kmh",
        "visibility_m",
        "pressure_hpa",
        "vehicle_count",
        "avg_speed_kmh",
        "accident_count",
    ]

    available = [c for c in candidates if c in df.columns]
    missing = [c for c in candidates if c not in df.columns]

    if missing:
        logger.warning(f"Some expected numeric columns are missing and will be skipped: {missing}")

    if len(available) < 2:
        logger.error(
            f"Not enough numeric features available for Factor Analysis. Found only: {available}"
        )

    logger.info(f"Using numeric features for Factor Analysis: {available}")
    return available


# ------------------------------------------------------------------------------
# Core Factor Analysis
# ------------------------------------------------------------------------------

def run_factor_analysis():
    """
    Main function:
    - Load merged dataset
    - Select numeric weather + traffic variables
    - Standardize them
    - Run Factor Analysis with 1–3 factors (depending on available features)
    - Save factor loadings, factor scores, and summary
    - Upload them to MinIO gold bucket
    """
    logger.info("=== Factor Analysis started ===")

    # 1) Load data
    df = load_merged_dataset()

    if df.empty:
        logger.warning("Merged dataset is empty. Skipping Factor Analysis.")
        return

    # 2) Select numeric features
    feature_cols = select_numeric_features(df)
    if len(feature_cols) < 2:
        logger.warning("Factor Analysis aborted: fewer than 2 numeric features available.")
        return

    work_df = df.copy()

    # 3) Handle missing values: convert to numeric & fill with column mean
    X = work_df[feature_cols].apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.mean())

    n_rows, n_features = X.shape
    logger.info(f"Prepared numeric matrix for FA with {n_rows} rows and {n_features} features.")

    if n_rows == 0:
        logger.warning("Numeric matrix has 0 rows. Skipping Factor Analysis.")
        return

    # 4) Standardize features (zero mean, unit variance)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.values)

    # 5) Decide number of factors (1–3, but not more than #features)
    n_factors = min(3, n_features)
    if n_factors < 1:
        logger.warning("Cannot run Factor Analysis: n_factors < 1.")
        return

    logger.info(f"Running Factor Analysis with n_factors = {n_factors}")

    fa = FactorAnalysis(n_components=n_factors, random_state=42)
    factor_scores = fa.fit_transform(X_scaled)  # shape: (n_samples, n_factors)

    # 6) Factor loadings (features x factors)
    loadings = fa.components_.T  # shape: (n_features, n_factors)
    factor_names = [f"Factor_{i+1}" for i in range(n_factors)]

    loadings_df = pd.DataFrame(
        loadings,
        index=feature_cols,
        columns=factor_names,
    ).reset_index().rename(columns={"index": "feature"})

    # 7) Factor scores per row (optional but useful)
    scores_df = pd.DataFrame(
        factor_scores,
        columns=factor_names,
    )

    context_cols = []
    if "city" in df.columns:
        context_cols.append("city")
    if "date_time" in df.columns:
        context_cols.append("date_time")
    if "date" in df.columns:
        context_cols.append("date")

    if context_cols:
        scores_df = pd.concat(
            [df[context_cols].reset_index(drop=True), scores_df.reset_index(drop=True)],
            axis=1,
        )

    # 8) Simple summary: variance of each factor (approximate importance)
    factor_variances = scores_df[factor_names].var().to_dict()
    summary_rows = []
    total_var = sum(factor_variances.values()) if factor_variances else 0.0

    for fname in factor_names:
        var = factor_variances.get(fname, 0.0)
        perc = (var / total_var * 100.0) if total_var > 0 else 0.0
        summary_rows.append(
            {
                "factor": fname,
                "variance": var,
                "variance_percent": perc,
            }
        )

    summary_df = pd.DataFrame(summary_rows)

    # 9) Save locally
    loadings_path = "factor_loadings.csv"
    scores_path = "factor_scores.csv"
    summary_path = "factor_summary.csv"

    logger.info(f"Saving factor loadings -> {loadings_path}")
    loadings_df.to_csv(loadings_path, index=False)

    logger.info(f"Saving factor scores -> {scores_path}")
    scores_df.to_csv(scores_path, index=False)

    logger.info(f"Saving factor summary -> {summary_path}")
    summary_df.to_csv(summary_path, index=False)

    # 10) Upload to MinIO gold bucket
    minio_client = get_minio_client()
    gold_bucket = "gold"
    ensure_bucket(minio_client, gold_bucket)

    upload_file_to_minio(
        minio_client,
        gold_bucket,
        "factor_loadings.csv",
        loadings_path,
        content_type="text/csv",
    )
    upload_file_to_minio(
        minio_client,
        gold_bucket,
        "factor_scores.csv",
        scores_path,
        content_type="text/csv",
    )
    upload_file_to_minio(
        minio_client,
        gold_bucket,
        "factor_summary.csv",
        summary_path,
        content_type="text/csv",
    )

    logger.info("✅ Factor Analysis finished successfully.")


if __name__ == "__main__":
    run_factor_analysis()
