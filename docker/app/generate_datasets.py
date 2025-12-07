"""
This module:
- Creates two messy datasets (weather, traffic)
- Saves them as CSV so ETL can upload them to MinIO (Bronze layer)
"""

import random
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


# -----------------------------
# CONFIG
# -----------------------------
N_WEATHER_ROWS = 5000
N_TRAFFIC_ROWS = 10000

random.seed(42)
np.random.seed(42)


# -----------------------------
# HELPER FUNCTIONS
# -----------------------------

def random_datetime(start, end):
    """Return a random datetime between start and end."""
    delta = end - start
    rand_sec = random.randint(0, int(delta.total_seconds()))
    return start + timedelta(seconds=rand_sec)


def format_datetime_messy(dt):
    """
    Return the same datetime in different random formats
    to simulate messy date formats.
    """
    formats = [
        "%Y-%m-%d %H:%M:%S",   # 2024-01-15 13:45:00
        "%d-%m-%Y %H:%M",      # 15-01-2024 13:45
        "%m/%d/%Y %H:%M",      # 01/15/2024 13:45
        "%Y/%m/%d",            # 2024/01/15
        "%d %b %Y %H:%M",      # 15 Jan 2024 13:45
        "%Y-%m-%d"             # 2024-01-15
    ]
    fmt = random.choice(formats)
    return dt.strftime(fmt)


def inject_datetime_noise(dt_str):
    """
    Occasionally replace a valid datetime string with invalid or missing values.
    """
    r = random.random()
    if r < 0.03:   # 3% invalid string
        return "not-a-date"
    elif r < 0.06:  # 3% empty
        return ""
    else:
        return dt_str


def random_city_with_noise():
    """
    Mostly 'London' but with casing variations and typos, plus some missing.
    """
    variants = ["London", "london", "LONDON", "Lonodn", "Ldnon"]
    r = random.random()
    if r < 0.05:
        return None  # missing
    return random.choice(variants)


def random_weather_condition():
    base = ["Clear", "Cloudy", "Rain", "Storm", "Fog", "Snow"]
    typos = ["clr", "RAIN", "raiin", "clody", "STORM", "sn0w"]
    if random.random() < 0.15:
        return random.choice(typos)
    return random.choice(base)


def random_season():
    base = ["Winter", "Spring", "Summer", "Autumn"]
    typos = ["winter", "summr", "Autum", "sprng", "FALL"]
    if random.random() < 0.10:
        return random.choice(typos)
    return random.choice(base)


def random_road_condition():
    base = ["Dry", "Wet", "Snowy", "Damaged"]
    typos = ["dryy", "WET", "snoy", "dmged"]
    if random.random() < 0.15:
        return random.choice(typos)
    return random.choice(base)


def random_congestion_level():
    base = ["Low", "Medium", "High"]
    noise = ["low", "LOW", "Med", "H", "3", "hi", "HIGH"]
    if random.random() < 0.20:
        return random.choice(noise)
    return random.choice(base)


def random_is_holiday():
    """
    Random holiday flag with mixed encodings.
    """
    encodings = ["Yes", "No", "Y", "N", "1", "0", "TRUE", "FALSE"]
    if random.random() < 0.10:
        return None
    return random.choice(encodings)


def add_numeric_noise(base_value, min_val=None, max_val=None,
                      null_prob=0.05, outlier_prob=0.05, outlier_range=None):
    """
    Generate a numeric value with:
    - some probability of being NaN
    - some probability of being an out-of-range outlier
    """
    r = random.random()
    if r < null_prob:
        return np.nan

    if r < null_prob + outlier_prob and outlier_range is not None:
        # generate outlier
        return random.uniform(*outlier_range)

    # normal noise around base_value
    value = base_value + np.random.normal(
        0,
        (max_val - min_val) * 0.02 if (min_val is not None and max_val is not None) else 1
    )

    if min_val is not None and max_val is not None:
        # clip into valid range for "typical" values
        value = max(min_val, min(max_val, value))

    return value


def main():
    """
    Entry point used by run_all.py.

    - Builds shared datetime pool
    - Generates weather_raw.csv
    - Generates traffic_raw.csv
    """
    # -----------------------------
    # 1) GENERATE BASE DATETIMES
    # -----------------------------
    start_dt = datetime(2024, 1, 1)
    end_dt = datetime(2024, 12, 31, 23, 59, 59)

    # Create a pool of datetimes to share between weather & traffic so joins are possible
    pool_size = 3000
    datetime_pool = [random_datetime(start_dt, end_dt) for _ in range(pool_size)]

    weather_datetimes = [random.choice(datetime_pool) for _ in range(N_WEATHER_ROWS)]
    traffic_datetimes = [random.choice(datetime_pool) for _ in range(N_TRAFFIC_ROWS)]

    # -----------------------------
    # 2) GENERATE WEATHER DATASET
    # -----------------------------
    weather_records = []

    for i in range(N_WEATHER_ROWS):
        dt = weather_datetimes[i]
        # start with clean formatted datetime
        dt_str = format_datetime_messy(dt)
        dt_str = inject_datetime_noise(dt_str)

        # base realistic values
        base_temp = random.uniform(-3, 30)         # °C
        base_hum = random.uniform(30, 90)          # %
        base_rain = max(0, np.random.exponential(2))  # mm
        base_wind = random.uniform(0, 40)          # km/h
        base_vis = random.uniform(500, 10000)      # meters
        base_press = random.uniform(980, 1030)     # hPa

        record = {
            "weather_id": i + 1,
            "date_time": dt_str,
            "city": random_city_with_noise(),
            "temperature_c": add_numeric_noise(
                base_temp, min_val=-10, max_val=45,
                null_prob=0.05, outlier_prob=0.03, outlier_range=(-30, 60)
            ),
            "humidity_percent": add_numeric_noise(
                base_hum, min_val=0, max_val=100,
                null_prob=0.05, outlier_prob=0.03, outlier_range=(-20, 150)
            ),
            "rain_mm": add_numeric_noise(
                base_rain, min_val=0, max_val=50,
                null_prob=0.05, outlier_prob=0.03, outlier_range=(-5, 200)
            ),
            "wind_speed_kmh": add_numeric_noise(
                base_wind, min_val=0, max_val=120,
                null_prob=0.05, outlier_prob=0.03, outlier_range=(-20, 200)
            ),
            "visibility_m": add_numeric_noise(
                base_vis, min_val=50, max_val=20000,
                null_prob=0.05, outlier_prob=0.03, outlier_range=(-100, 100000)
            ),
            "pressure_hpa": add_numeric_noise(
                base_press, min_val=900, max_val=1100,
                null_prob=0.05, outlier_prob=0.03, outlier_range=(500, 1300)
            ),
            "weather_condition": random_weather_condition(),
            "season": random_season()
        }

        weather_records.append(record)

    weather_df = pd.DataFrame(weather_records)

    # -----------------------------
    # 3) GENERATE TRAFFIC DATASET
    # -----------------------------
    traffic_records = []

    areas = ["Central", "North", "South", "East", "West", "Downtown", "Suburb-A", "Suburb-B"]
    event_types = ["None", "Roadwork", "SportsEvent", "Concert", "Parade"]

    for i in range(N_TRAFFIC_ROWS):
        dt = traffic_datetimes[i]
        dt_str = format_datetime_messy(dt)
        dt_str = inject_datetime_noise(dt_str)

        base_veh_count = np.random.poisson(300)  # vehicles
        base_speed = random.uniform(10, 80)      # km/h
        base_accidents = np.random.binomial(2, 0.05)

        record = {
            "traffic_id": i + 1,
            "date_time": dt_str,
            "city": random_city_with_noise(),
            "area": random.choice(areas),
            "vehicle_count": add_numeric_noise(
                base_veh_count, min_val=0, max_val=5000,
                null_prob=0.05, outlier_prob=0.03, outlier_range=(-100, 20000)
            ),
            "avg_speed_kmh": add_numeric_noise(
                base_speed, min_val=0, max_val=130,
                null_prob=0.05, outlier_prob=0.03, outlier_range=(-50, 300)
            ),
            "congestion_level": random_congestion_level(),
            "accident_count": add_numeric_noise(
                base_accidents, min_val=0, max_val=10,
                null_prob=0.05, outlier_prob=0.03, outlier_range=(-5, 50)
            ),
            "road_condition": random_road_condition(),
            "is_holiday": random_is_holiday(),
            "event_type": random.choice(event_types)
        }

        traffic_records.append(record)

    traffic_df = pd.DataFrame(traffic_records)

    # -----------------------------
    # 4) SAVE TO CSV
    # -----------------------------
    weather_df.to_csv("weather_raw.csv", index=False)
    traffic_df.to_csv("traffic_raw.csv", index=False)

    print("Generated files:")
    print(f" - weather_raw.csv  (rows: {len(weather_df)})")
    print(f" - traffic_raw.csv  (rows: {len(traffic_df)})")


if __name__ == "__main__":
    # Allows running `python generate_datasets.py` manually
    main()
