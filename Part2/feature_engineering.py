"""Task 2 - feature engineering on the cleaned traffic data.

Adds time, weather and scaling features plus a data-driven congestion category,
then writes ``data/processed/traffic_features.csv``.

Feature groups
--------------
time        hour, day_of_week (0=Monday), day_name, month, is_weekend,
            hour_sin / hour_cos (cyclical encoding of the hour so that 23:00 and
            00:00 sit next to each other), day_of_week_sin / day_of_week_cos.
weather     temp_c (Kelvin -> Celsius), is_raining, is_snowing, is_adverse_weather,
            is_holiday, one-hot weather_main (wm_*) and one-hot top-7
            weather_description labels (wd_*).
scaled      temp_c_minmax, traffic_volume_minmax, rain_1h_log_minmax
            (all min-max scaled to [0, 1] with NumPy).
target      congestion (Low / Moderate / High / Severe) and congestion_rank (0-3).

Congestion category logic
-------------------------
The four bands are the quartiles of ``traffic_volume`` computed on the whole
cleaned dataset (40,575 hourly records), so the label means "relative to normal
traffic on this interstate" rather than an arbitrary fixed cut-off::

    traffic_volume <= Q1 (25th percentile)          -> "Low"
    Q1 <  traffic_volume <= Q2 (50th percentile)    -> "Moderate"
    Q2 <  traffic_volume <= Q3 (75th percentile)    -> "High"
    traffic_volume >  Q3                            -> "Severe"

Each band therefore holds roughly a quarter of the hours. The exact thresholds
are logged at DEBUG and written into the README / report.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CLEAN_CSV = BASE_DIR / "data" / "processed" / "traffic_clean.csv"
DEFAULT_FEATURES_CSV = BASE_DIR / "data" / "processed" / "traffic_features.csv"

CONGESTION_LABELS = ["Low", "Moderate", "High", "Severe"]
ADVERSE_WEATHER = {"Rain", "Snow", "Thunderstorm", "Drizzle", "Fog", "Mist", "Squall", "Smoke", "Haze"}
TOP_DESCRIPTIONS = 7


class FeatureEngineeringError(RuntimeError):
    """The cleaned dataset is missing or unreadable."""


def load_clean_data(path: str | Path = DEFAULT_CLEAN_CSV) -> pd.DataFrame:
    """Read the cleaned CSV produced by ``pipeline.py``."""
    path = Path(path)
    logger.info("Loading cleaned data from %s", path)
    try:
        df = pd.read_csv(path, parse_dates=["date_time"], encoding="utf-8-sig")
    except FileNotFoundError as exc:
        raise FeatureEngineeringError(
            f"Cleaned dataset not found: {path}. Run pipeline.py first."
        ) from exc
    except pd.errors.EmptyDataError as exc:
        raise FeatureEngineeringError(f"Cleaned dataset is empty: {path}") from exc
    except pd.errors.ParserError as exc:
        raise FeatureEngineeringError(f"Cleaned dataset is not parseable: {path}") from exc

    if df.empty:
        raise FeatureEngineeringError(f"Cleaned dataset has no rows: {path}")
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calendar features plus cyclical encodings of hour and weekday."""
    df = df.copy()
    df["hour"] = df["date_time"].dt.hour
    df["day_of_week"] = df["date_time"].dt.dayofweek  # Monday=0 ... Sunday=6
    df["day_name"] = df["date_time"].dt.day_name()
    df["month"] = df["date_time"].dt.month
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    # sin/cos keep the circular distance: hour 23 is one hour from hour 0.
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["day_of_week_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["day_of_week_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

    logger.info("Time features added: hour, day_of_week, day_name, month, is_weekend, hour_sin/cos, day_of_week_sin/cos")
    logger.debug(
        "Weekend share: %d of %d hours (%.1f%%)",
        int(df["is_weekend"].sum()),
        len(df),
        100 * df["is_weekend"].mean(),
    )
    return df


def add_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    """Weather indicators, one-hot encodings and the Celsius conversion."""
    df = df.copy()

    df["temp_c"] = df["temp"] - 273.15
    df["is_raining"] = (df["rain_1h"] > 0).astype(int)
    df["is_snowing"] = (df["snow_1h"] > 0).astype(int)
    df["is_adverse_weather"] = df["weather_main"].isin(ADVERSE_WEATHER).astype(int)
    df["is_holiday"] = (df["holiday"] != "No Holiday").astype(int)

    # One-hot for the main weather category (11 labels).
    main_dummies = pd.get_dummies(df["weather_main"], prefix="wm", dtype=int)
    main_dummies.columns = [c.lower().replace(" ", "_") for c in main_dummies.columns]

    # weather_description has ~37 labels; keep the 7 most frequent and group the rest.
    top_descriptions = df["weather_description"].value_counts().head(TOP_DESCRIPTIONS).index
    grouped = df["weather_description"].where(df["weather_description"].isin(top_descriptions), "other")
    desc_dummies = pd.get_dummies(grouped, prefix="wd", dtype=int)
    desc_dummies.columns = [c.lower().replace(" ", "_") for c in desc_dummies.columns]

    df = pd.concat([df, main_dummies, desc_dummies], axis=1)

    logger.info(
        "Weather features added: temp_c, is_raining, is_snowing, is_adverse_weather, is_holiday, "
        "%d weather_main dummies, %d grouped weather_description dummies",
        main_dummies.shape[1],
        desc_dummies.shape[1],
    )
    logger.debug("One-hot weather_main columns: %s", ", ".join(main_dummies.columns))
    logger.debug("Rainy hours: %d, snowy hours: %d, holiday hours: %d",
                 int(df["is_raining"].sum()), int(df["is_snowing"].sum()), int(df["is_holiday"].sum()))
    return df


def add_scaled_features(df: pd.DataFrame) -> pd.DataFrame:
    """Min-max scale two continuous variables (plus a log-scaled rain variable)."""
    df = df.copy()

    def min_max(series: pd.Series) -> pd.Series:
        low, high = float(np.min(series)), float(np.max(series))
        if high - low == 0:  # constant column: avoid dividing by zero
            return pd.Series(np.zeros(len(series)), index=series.index)
        return (series - low) / (high - low)

    df["temp_c_minmax"] = min_max(df["temp_c"])
    df["traffic_volume_minmax"] = min_max(df["traffic_volume"])
    # rain is extremely skewed (median 0.0, max 28.7 mm after cleaning), so log1p first
    df["rain_1h_log_minmax"] = min_max(np.log1p(df["rain_1h"]))

    logger.info(
        "Scaled features added: temp_c_minmax (%.1f..%.1f C), traffic_volume_minmax (%d..%d veh/h), "
        "rain_1h_log_minmax (log1p of %d..%.2f mm)",
        df["temp_c"].min(),
        df["temp_c"].max(),
        int(df["traffic_volume"].min()),
        int(df["traffic_volume"].max()),
        int(df["rain_1h"].min()),
        df["rain_1h"].max(),
    )
    return df


def add_congestion_category(df: pd.DataFrame) -> pd.DataFrame:
    """Label each hour by traffic-volume quartile (see the module docstring).

    Thresholds are derived from the data with ``numpy.percentile`` rather than
    hard-coded, so the bands stay balanced if the input file changes.
    """
    df = df.copy()
    q1, q2, q3 = (float(np.percentile(df["traffic_volume"], p)) for p in (25, 50, 75))
    logger.debug(
        "Traffic-volume quartile thresholds (veh/h): Q1=%.1f Q2=%.1f Q3=%.1f", q1, q2, q3
    )

    conditions = [
        df["traffic_volume"] <= q1,
        (df["traffic_volume"] > q1) & (df["traffic_volume"] <= q2),
        (df["traffic_volume"] > q2) & (df["traffic_volume"] <= q3),
        df["traffic_volume"] > q3,
    ]
    df["congestion"] = np.select(conditions, CONGESTION_LABELS, default="Unknown")
    df["congestion_rank"] = df["congestion"].map({label: i for i, label in enumerate(CONGESTION_LABELS)}).astype(int)

    counts = df["congestion"].value_counts().reindex(CONGESTION_LABELS).fillna(0).astype(int)
    logger.info(
        "Congestion categories created from quartiles Q1=%.0f, Q2=%.0f, Q3=%.0f veh/h: %s",
        q1,
        q2,
        q3,
        ", ".join(f"{label}={counts[label]}" for label in CONGESTION_LABELS),
    )
    return df


def build_features(clean_csv: str | Path = DEFAULT_CLEAN_CSV,
                   features_csv: str | Path = DEFAULT_FEATURES_CSV) -> pd.DataFrame:
    """Run all feature steps and save the features CSV."""
    features_csv = Path(features_csv)

    df = load_clean_data(clean_csv)
    logger.info("Shape before feature engineering: %d rows x %d columns", df.shape[0], df.shape[1])

    df = add_time_features(df)
    df = add_weather_features(df)
    df = add_scaled_features(df)
    df = add_congestion_category(df)

    logger.info("Shape after feature engineering: %d rows x %d columns", df.shape[0], df.shape[1])

    features_csv.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_csv(features_csv, index=False)
    except OSError as exc:
        raise FeatureEngineeringError(f"Could not write the features CSV to {features_csv}") from exc

    logger.info("Saved engineered features to %s (%d rows x %d columns)",
                features_csv, df.shape[0], df.shape[1])
    return df


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task 2: build features from the cleaned traffic data.")
    parser.add_argument("--input", default=str(DEFAULT_CLEAN_CSV), help="cleaned CSV from pipeline.py")
    parser.add_argument("--output", default=str(DEFAULT_FEATURES_CSV), help="features CSV to write")
    parser.add_argument("--debug", action="store_true", help="enable DEBUG logging (logs/pipeline_debug.log)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    from logging_setup import configure_logging

    configure_logging(debug=args.debug)
    logger.info("Starting feature_engineering.py (debug=%s)", args.debug)

    try:
        build_features(clean_csv=args.input, features_csv=args.output)
    except FeatureEngineeringError as exc:
        logger.error("Feature engineering could not complete: %s", exc, exc_info=True)
        return 1
    except Exception as exc:
        logger.error("Unexpected failure in feature_engineering.py: %s", exc, exc_info=True)
        return 1

    logger.info("feature_engineering.py finished successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main())


