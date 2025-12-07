import os
import io
import logging
from typing import Dict

import numpy as np
import pandas as pd
from minio import Minio
from minio.error import S3Error

from hdfs import InsecureClient

# ------------------------------------------------------------------------------
# Logging setup
# ------------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------------------
# MinIO helpers
# ------------------------------------------------------------------------------
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

    logger.info(f"Connecting to MinIO at {endpoint} (secure={secure})")
    return Minio(
        endpoint,
        access_key=access_key,
        secret_key=secret_key,
        secure=secure,
    )


def ensure_bucket(minio_client: Minio, bucket_name: str):
    found = minio_client.bucket_exists(bucket_name)
    if not found:
        logger.info(f"Creating bucket: {bucket_name}")
        minio_client.make_bucket(bucket_name)
    else:
        logger.info(f"Bucket already exists: {bucket_name}")


def upload_file_to_minio(
    minio_client: Minio,
    bucket: str,
    object_name: str,
    file_path: str,
    content_type: str = "application/octet-stream",
):
    """
    Upload a local file (file_path) to MinIO (bucket/object_name).
    """
    logger.info(f"Uploading {file_path} -> s3://{bucket}/{object_name}")
    file_size = os.path.getsize(file_path)
    with open(file_path, "rb") as f:
        minio_client.put_object(
            bucket_name=bucket,
            object_name=object_name,
            data=f,
            length=file_size,
            content_type=content_type,
        )


def upload_bytes_to_minio(
    minio_client: Minio,
    bucket: str,
    object_name: str,
    data: bytes,
    content_type: str = "application/octet-stream",
):
    logger.info(f"Uploading bytes -> s3://{bucket}/{object_name}")
    minio_client.put_object(
        bucket_name=bucket,
        object_name=object_name,
        data=io.BytesIO(data),
        length=len(data),
        content_type=content_type,
    )


# ------------------------------------------------------------------------------
# HDFS helpers
# ------------------------------------------------------------------------------
def get_hdfs_client() -> InsecureClient:
    """
    Connect using WebHDFS.

    By default we assume the NameNode web UI is at http://namenode:9870.
    You can override with env var HDFS_WEB_ENDPOINT.
    """
    web_endpoint = os.getenv("HDFS_WEB_ENDPOINT", "http://namenode:9870")
    user = os.getenv("HDFS_USER", "root")

    logger.info(f"Connecting to HDFS WebHDFS at {web_endpoint} as user={user}")
    client = InsecureClient(web_endpoint, user=user)
    logger.info(f"Instantiated {client}.")
    return client


def upload_to_hdfs(client: InsecureClient, local_path: str, hdfs_path: str):
    """
    Upload local file to HDFS path, overwrite if exists.
    """
    logger.info(f"Uploading to HDFS: {local_path} -> {hdfs_path}")
    client.makedirs(os.path.dirname(hdfs_path))
    logger.info(f"Uploading '{os.path.basename(local_path)}' to '{hdfs_path}'.")
    client.upload(hdfs_path, local_path, overwrite=True)
    logger.info(f"Listing '{hdfs_path}'.")
    client.list(hdfs_path)
    logger.info(f"Writing to '{hdfs_path}'.")


# ------------------------------------------------------------------------------
# Cleaning logic
# ------------------------------------------------------------------------------

def standardize_city(series: pd.Series) -> pd.Series:
    """
    Clean 'city' column:
    - lower, strip
    - if it looks like 'london' with small typo, standardize to 'London'
    """
    def clean_city(x):
        if pd.isna(x):
            return np.nan
        s = str(x).strip().lower()
        if any(tok in s for tok in ["lond", "lonodn", "ldnon"]):
            return "London"
        if s == "":
            return np.nan
        # fallback: capitalize first letter
        return s.capitalize()

    return series.apply(clean_city)


def parse_datetime_column(series: pd.Series) -> pd.Series:
    """
    Convert messy date strings (multiple formats, invalid values) to datetime.
    Invalid values -> NaT.
    """
    # infer_datetime_format is now default, so we don't pass the deprecated arg
    return pd.to_datetime(series, errors="coerce", dayfirst=True)


def clean_numeric_column(
    series: pd.Series,
    valid_min: float,
    valid_max: float,
    fill_strategy: str = "median",
) -> pd.Series:
    """
    - Convert to numeric
    - Invalid strings -> NaN
    - Values outside [valid_min, valid_max] -> NaN
    - Fill NaN with median/mean or leave as NaN
    """
    s = pd.to_numeric(series, errors="coerce")
    s = s.where((s >= valid_min) & (s <= valid_max), np.nan)

    if fill_strategy == "median":
        fill_value = s.median()
        s = s.fillna(fill_value)
    elif fill_strategy == "mean":
        fill_value = s.mean()
        s = s.fillna(fill_value)
    else:
        # keep NaNs
        pass

    return s


def standardize_categorical(series: pd.Series, mapping: Dict[str, str]) -> pd.Series:
    """
    Standardize categorical values using a mapping like:
        {"low": "Low", "LOW": "Low", ...}
    Unmapped values remain as they are (except NaN).
    """
    def normalize(x):
        if pd.isna(x):
            return np.nan
        s = str(x).strip()
        key = s.lower()
        return mapping.get(key, s)  # fallback to original

    return series.apply(normalize)


def clean_weather_df(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Cleaning weather dataset...")

    df = df.copy()

    expected_cols = [
        "weather_id",
        "date_time",
        "city",
        "temperature_c",
        "humidity_percent",
        "rain_mm",
        "wind_speed_kmh",
        "visibility_m",
        "pressure_hpa",
        "weather_condition",
        "season",
    ]
    for col in expected_cols:
        if col not in df.columns:
            logger.warning(f"Weather: missing column '{col}', creating empty.")
            df[col] = np.nan

    # Date/time
    df["date_time"] = parse_datetime_column(df["date_time"])

    # City
    df["city"] = standardize_city(df["city"])

    # Numeric columns
    df["temperature_c"] = clean_numeric_column(df["temperature_c"], -30, 60)
    df["humidity_percent"] = clean_numeric_column(df["humidity_percent"], 0, 100)
    df["rain_mm"] = clean_numeric_column(df["rain_mm"], 0, 500)
    df["wind_speed_kmh"] = clean_numeric_column(df["wind_speed_kmh"], 0, 200)
    df["visibility_m"] = clean_numeric_column(df["visibility_m"], 0, 100000)
    df["pressure_hpa"] = clean_numeric_column(df["pressure_hpa"], 800, 1200)

    # Categorical: weather_condition
    weather_map = {
        "clear": "Clear",
        "clr": "Clear",
        "cloudy": "Cloudy",
        "clody": "Cloudy",
        "rain": "Rain",
        "rainy": "Rain",
        "raiin": "Rain",
        "storm": "Storm",
        "fog": "Fog",
        "snow": "Snow",
        "sn0w": "Snow",
    }
    df["weather_condition"] = standardize_categorical(df["weather_condition"], weather_map)

    # Categorical: season
    season_map = {
        "winter": "Winter",
        "summr": "Summer",
        "summer": "Summer",
        "autum": "Autumn",
        "autumn": "Autumn",
        "sprng": "Spring",
        "spring": "Spring",
        "fall": "Autumn",
    }
    df["season"] = standardize_categorical(df["season"], season_map)

    # Drop rows without date_time or city because we can't join them later
    before = len(df)
    df = df.dropna(subset=["date_time", "city"])
    after = len(df)
    logger.info(f"Weather: dropped {before - after} rows with missing city/date_time")

    return df


def clean_traffic_df(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Cleaning traffic dataset...")

    df = df.copy()

    expected_cols = [
        "traffic_id",
        "date_time",
        "city",
        "area",
        "vehicle_count",
        "avg_speed_kmh",
        "congestion_level",
        "accident_count",
        "road_condition",
        "is_holiday",
        "event_type",
    ]
    for col in expected_cols:
        if col not in df.columns:
            logger.warning(f"Traffic: missing column '{col}', creating empty.")
            df[col] = np.nan

    # Date/time
    df["date_time"] = parse_datetime_column(df["date_time"])

    # City
    df["city"] = standardize_city(df["city"])

    # Numeric columns
    df["vehicle_count"] = clean_numeric_column(df["vehicle_count"], 0, 20000)
    df["avg_speed_kmh"] = clean_numeric_column(df["avg_speed_kmh"], 0, 300)
    df["accident_count"] = clean_numeric_column(df["accident_count"], 0, 100)

    # Categorical: congestion_level
    congestion_map = {
        "low": "Low",
        "l": "Low",
        "1": "Low",

        "medium": "Medium",
        "med": "Medium",
        "2": "Medium",

        "high": "High",
        "hi": "High",
        "h": "High",
        "3": "High",
    }
    df["congestion_level"] = standardize_categorical(df["congestion_level"], congestion_map)

    # Categorical: road_condition
    road_map = {
        "dry": "Dry",
        "dryy": "Dry",
        "wet": "Wet",
        "snowy": "Snowy",
        "snoy": "Snowy",
        "damaged": "Damaged",
        "dmged": "Damaged",
    }
    df["road_condition"] = standardize_categorical(df["road_condition"], road_map)

    # is_holiday: normalize to Yes/No
    def normalize_holiday(x):
        if pd.isna(x):
            return "No"  # assume non-holiday if unknown
        s = str(x).strip().lower()
        if s in ["yes", "y", "1", "true", "t"]:
            return "Yes"
        if s in ["no", "n", "0", "false", "f"]:
            return "No"
        return "No"

    df["is_holiday"] = df["is_holiday"].apply(normalize_holiday)

    # Drop rows without date_time or city
    before = len(df)
    df = df.dropna(subset=["date_time", "city"])
    after = len(df)
    logger.info(f"Traffic: dropped {before - after} rows with missing city/date_time")

    return df


# ------------------------------------------------------------------------------
# Main ETL
# ------------------------------------------------------------------------------

def run_etl():
    """
    End-to-end ETL pipeline:
    1) Read local raw CSV files
    2) Upload to MinIO Bronze
    3) Clean & write Parquet to MinIO Silver
    4) Sync cleaned Parquet files to HDFS
    5) Create merged analytical dataset and upload to MinIO Gold
    """
    logger.info("=== ETL pipeline started ===")

    # ------------------ Step 0: Config & clients ------------------
    minio_client = get_minio_client()
    bronze_bucket = "bronze"
    silver_bucket = "silver"
    gold_bucket = "gold"

    for b in [bronze_bucket, silver_bucket, gold_bucket]:
        ensure_bucket(minio_client, b)

    # HDFS client (may fail if HDFS not configured; we catch it)
    hdfs_client = None
    try:
        hdfs_client = get_hdfs_client()
    except Exception as e:
        logger.warning(f"Could not connect to HDFS yet: {e}")

    # ------------------ Step 1: Read local raw CSVs ------------------
    weather_raw_path = "weather_raw.csv"
    traffic_raw_path = "traffic_raw.csv"

    if not os.path.exists(weather_raw_path) or not os.path.exists(traffic_raw_path):
        raise FileNotFoundError(
            "Raw CSV files not found. Please run generate_datasets.py first."
        )

    logger.info("Reading raw weather CSV...")
    weather_raw_df = pd.read_csv(weather_raw_path)

    logger.info("Reading raw traffic CSV...")
    traffic_raw_df = pd.read_csv(traffic_raw_path)

    # ------------------ Step 2: Upload raw to MinIO Bronze ------------------
    upload_file_to_minio(
        minio_client,
        bronze_bucket,
        "weather_raw.csv",
        weather_raw_path,
        content_type="text/csv",
    )
    upload_file_to_minio(
        minio_client,
        bronze_bucket,
        "traffic_raw.csv",
        traffic_raw_path,
        content_type="text/csv",
    )

    # ------------------ Step 3: Clean datasets -> Parquet (Silver) ------------------
    weather_clean_df = clean_weather_df(weather_raw_df)
    traffic_clean_df = clean_traffic_df(traffic_raw_df)

    # Save locally as Parquet
    weather_clean_path = "weather_clean.parquet"
    traffic_clean_path = "traffic_clean.parquet"

    logger.info(f"Writing cleaned weather parquet -> {weather_clean_path}")
    weather_clean_df.to_parquet(weather_clean_path, index=False)

    logger.info(f"Writing cleaned traffic parquet -> {traffic_clean_path}")
    traffic_clean_df.to_parquet(traffic_clean_path, index=False)

    # Upload to MinIO silver
    upload_file_to_minio(
        minio_client,
        silver_bucket,
        "weather_clean.parquet",
        weather_clean_path,
        content_type="application/octet-stream",
    )
    upload_file_to_minio(
        minio_client,
        silver_bucket,
        "traffic_clean.parquet",
        traffic_clean_path,
        content_type="application/octet-stream",
    )

    # ------------------ Step 4: Sync cleaned to HDFS ------------------
    if hdfs_client is not None:
        try:
            upload_to_hdfs(
                hdfs_client,
                weather_clean_path,
                "/data/weather/clean/weather_clean.parquet",
            )
            upload_to_hdfs(
                hdfs_client,
                traffic_clean_path,
                "/data/traffic/clean/traffic_clean.parquet",
            )
        except Exception as e:
            logger.warning(f"Failed to upload cleaned data to HDFS: {e}")
    else:
        logger.warning("Skipping HDFS sync because HDFS client is not available.")

    # ------------------ Step 5: Create merged analytical dataset (Gold) ------------------
    logger.info("Merging cleaned weather + traffic datasets...")

    # Ensure date_time is datetime in both
    weather_clean_df["date_time"] = pd.to_datetime(weather_clean_df["date_time"])
    traffic_clean_df["date_time"] = pd.to_datetime(traffic_clean_df["date_time"])

    # NEW: join on city + DATE (not full timestamp) to get more matches
    weather_clean_df["date"] = weather_clean_df["date_time"].dt.date
    traffic_clean_df["date"] = traffic_clean_df["date_time"].dt.date

    merged_df = pd.merge(
        traffic_clean_df,
        weather_clean_df,
        on=["city", "date"],
        how="inner",
        suffixes=("_traffic", "_weather"),
    )

    logger.info(f"Merged dataset rows: {len(merged_df)}")

    merged_path_parquet = "merged_analytics.parquet"
    merged_path_csv = "merged_analytics.csv"

    logger.info(f"Writing merged analytics parquet -> {merged_path_parquet}")
    merged_df.to_parquet(merged_path_parquet, index=False)

    logger.info(f"Writing merged analytics CSV -> {merged_path_csv}")
    merged_df.to_csv(merged_path_csv, index=False)

    # Upload merged dataset to MinIO gold
    upload_file_to_minio(
        minio_client,
        gold_bucket,
        "merged_analytics.parquet",
        merged_path_parquet,
        content_type="application/octet-stream",
    )
    upload_file_to_minio(
        minio_client,
        gold_bucket,
        "merged_analytics.csv",
        merged_path_csv,
        content_type="text/csv",
    )

    logger.info("✅ ETL pipeline finished successfully.")


if __name__ == "__main__":
    run_etl()
