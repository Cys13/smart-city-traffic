"""Task 1 - load, validate and clean the raw smart-city traffic CSV.

The raw file is an hourly extract of the *Metro Interstate Traffic Volume*
dataset (Interstate 94, Minneapolis-St Paul, Oct 2012 - Sep 2018).

Pipeline order (schema validation always comes first, before any processing):

1. ``load_raw_data``            read the CSV with explicit exception handling.
2. ``validate_schema``          every expected column must exist, and the
                                numeric columns must be convertible to numbers.
3. ``standardise_categories``   trim/lowercase ``weather_description``, map
                                ``weather_main`` to canonical spelling, turn the
                                literal string ``"None"`` in ``holiday`` into a
                                real category ("no holiday that day").
4. ``parse_datetime``           parse ``date_time``; rows whose timestamp cannot
                                be parsed are dropped (and counted).
5. ``remove_duplicates``        exact duplicate rows, then one row per hour
                                (the source contains several weather
                                observations for the same hour).
6. ``handle_missing_values``    numeric gaps filled with the *monthly* median,
                                categorical gaps filled with ``"Unknown"``.
7. ``impute_impossible_values`` physically impossible readings (0 Kelvin,
                                9831 mm of rain in one hour, counts under
                                10 vehicles/hour ...) replaced with the
                                *monthly* median.

Every step logs its own INFO/WARNING line: nothing is dropped or modified
silently, and the log always states how many rows were affected and why.

Run directly with::

    python pipeline.py            # normal run, INFO level
    python pipeline.py --debug    # DEBUG detail into logs/pipeline_debug.log
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
DEFAULT_RAW_CSV = BASE_DIR / "data" / "raw" / "metro_interstate_traffic_volume.csv"
DEFAULT_PROCESSED_CSV = BASE_DIR / "data" / "processed" / "traffic_clean.csv"

# --------------------------------------------------------------------------- #
# Expected schema, derived from inspecting the real file (48,204 rows, 9 columns)
# --------------------------------------------------------------------------- #
REQUIRED_COLUMNS = [
    "holiday",
    "temp",
    "rain_1h",
    "snow_1h",
    "clouds_all",
    "weather_main",
    "weather_description",
    "date_time",
    "traffic_volume",
]
NUMERIC_COLUMNS = ["temp", "rain_1h", "snow_1h", "clouds_all", "traffic_volume"]
CATEGORICAL_COLUMNS = ["holiday", "weather_main", "weather_description"]

# Physically plausible ranges, used to catch sensor errors rather than extremes.
#   temp            Kelvin  -> 200 K (-73 C) is a Minnesota record low, 333 K (60 C) an upper bound
#   rain_1h / snow_1h  mm    -> 200 mm of rain or 100 mm of snow in one hour is beyond any observed event
#   clouds_all      %       -> 0-100 by definition
#   traffic_volume  veh/h   -> the interstate never drops below ~10 vehicles/hour when the
#                              sensor works, so 0-9 counts are treated as a sensor dropout
PLAUSIBLE_RANGES: dict[str, tuple[float, float]] = {
    "temp": (200.0, 333.0),
    "rain_1h": (0.0, 200.0),
    "snow_1h": (0.0, 100.0),
    "clouds_all": (0.0, 100.0),
    "traffic_volume": (10.0, 10000.0),
}

# The raw file writes the literal string "None" for days that are not holidays.
# It is renamed to "No Holiday" because pandas treats "None" as a missing value
# on every later read_csv, which would silently turn the category into NaN.
NO_HOLIDAY_LABEL = "No Holiday"
UNKNOWN_LABEL = "Unknown"

# Canonical spellings for weather_main (the raw file is already close, but the
# mapping makes the rule explicit and survives future exports).
WEATHER_MAIN_CANONICAL = {
    "clouds": "Clouds",
    "clear": "Clear",
    "mist": "Mist",
    "rain": "Rain",
    "snow": "Snow",
    "drizzle": "Drizzle",
    "haze": "Haze",
    "thunderstorm": "Thunderstorm",
    "fog": "Fog",
    "smoke": "Smoke",
    "squall": "Squall",
}

# Spelling variants found in weather_description that must collapse to one label.
WEATHER_DESCRIPTION_FIXES = {
    "squalls": "squall",
    "sky is clear": "sky is clear",
}


class RawDataError(RuntimeError):
    """The raw CSV could not be read at all."""


class SchemaError(ValueError):
    """The raw CSV does not contain the columns this pipeline expects."""


# --------------------------------------------------------------------------- #
# Step 1 - load
# --------------------------------------------------------------------------- #
def load_raw_data(path: str | Path = DEFAULT_RAW_CSV) -> pd.DataFrame:
    """Read the raw CSV, converting pandas I/O errors into ``RawDataError``.

    The file is read with ``utf-8-sig`` because the export carries a UTF-8 BOM
    (without it the first column would be named ``"\\ufeffholiday"``).
    """
    path = Path(path)
    logger.info("Loading raw data from %s", path)

    try:
        df = pd.read_csv(path, encoding="utf-8-sig")
    except FileNotFoundError as exc:
        raise RawDataError(f"Raw data file not found: {path}") from exc
    except pd.errors.EmptyDataError as exc:
        raise RawDataError(f"Raw data file is empty (no header, no rows): {path}") from exc
    except pd.errors.ParserError as exc:
        raise RawDataError(f"Raw data file is not parseable as CSV: {path} ({exc})") from exc
    except UnicodeDecodeError as exc:
        raise RawDataError(f"Raw data file is not valid UTF-8: {path}") from exc
    except PermissionError as exc:
        raise RawDataError(f"Raw data file is not readable (permission denied): {path}") from exc

    if df.empty:
        raise RawDataError(f"Raw data file contains a header but no data rows: {path}")

    logger.info("Loaded raw data: %d rows x %d columns", df.shape[0], df.shape[1])
    logger.debug("Raw columns and dtypes: %s", {c: str(t) for c, t in df.dtypes.items()})
    return df


# --------------------------------------------------------------------------- #
# Step 2 - schema validation (must run before anything else touches the data)
# --------------------------------------------------------------------------- #
def validate_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Check that every required column exists, then coerce numeric columns.

    Raises
    ------
    SchemaError
        If one or more required columns are missing. The message lists them so
        the log states exactly what is wrong with the file.
    """
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise SchemaError(
            "Schema validation failed: missing required column(s): "
            + ", ".join(missing)
            + f". Found columns: {', '.join(map(str, df.columns))}"
        )

    unexpected = [col for col in df.columns if col not in REQUIRED_COLUMNS]
    if unexpected:
        logger.warning(
            "Schema validation: %d unexpected column(s) present and kept: %s",
            len(unexpected),
            ", ".join(unexpected),
        )

    # Numeric columns must actually hold numbers; non-numeric text becomes NaN
    # and is handled later by the monthly-median imputation.
    df = df.copy()
    for col in NUMERIC_COLUMNS:
        converted = pd.to_numeric(df[col], errors="coerce")
        broken = int(converted.isna().sum() - df[col].isna().sum())
        if broken > 0:
            logger.warning(
                "Schema validation: %d row(s) in '%s' were non-numeric text and are now missing",
                broken,
                col,
            )
        df[col] = converted

    logger.info(
        "Schema validation passed: all %d required columns present (%s)",
        len(REQUIRED_COLUMNS),
        ", ".join(REQUIRED_COLUMNS),
    )
    return df


# --------------------------------------------------------------------------- #
# Step 3 - standardise categorical values
# --------------------------------------------------------------------------- #
def standardise_categories(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse inconsistent spellings into one canonical label per category."""
    df = df.copy()

    # --- weather_main ------------------------------------------------------ #
    original_main = df["weather_main"].astype("string")
    normalised_main = original_main.str.strip().str.lower()
    canonical_main = normalised_main.map(WEATHER_MAIN_CANONICAL)
    fallback = normalised_main.str.replace("_", " ", regex=False).str.title()
    df["weather_main"] = canonical_main.fillna(fallback)
    changed_main = int((df["weather_main"] != original_main).fillna(False).sum())
    if changed_main:
        logger.warning(
            "Standardised weather_main: %d row(s) re-spelled (trim / case / synonym mapping)",
            changed_main,
        )
    else:
        logger.info("Standardised weather_main: no rows needed re-spelling")

    # --- weather_description ----------------------------------------------- #
    original_desc = df["weather_description"].astype("string")
    desc = (
        original_desc.str.strip()
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
        .replace(WEATHER_DESCRIPTION_FIXES)
    )
    df["weather_description"] = desc
    changed_desc = int((df["weather_description"] != original_desc).fillna(False).sum())
    if changed_desc:
        logger.warning(
            "Standardised weather_description: %d row(s) re-spelled to lower case "
            "(e.g. 'Sky is Clear' -> 'sky is clear', 'SQUALLS' -> 'squall')",
            changed_desc,
        )

    # --- holiday ------------------------------------------------------------ #
    # pandas reads the literal string "None" as NaN, so a missing holiday means
    # "this day is not a holiday" rather than "unknown".
    original_holiday = df["holiday"].astype("string")
    holiday = original_holiday.fillna(NO_HOLIDAY_LABEL).str.strip()
    holiday = holiday.replace({"": NO_HOLIDAY_LABEL, "none": NO_HOLIDAY_LABEL})
    df["holiday"] = holiday
    no_holiday_rows = int((df["holiday"] == NO_HOLIDAY_LABEL).sum())
    holiday_rows = int(len(df) - no_holiday_rows)
    logger.info(
        "Standardised holiday: %d row(s) marked '%s' (not a holiday), %d row(s) on a named holiday (%d distinct)",
        no_holiday_rows,
        NO_HOLIDAY_LABEL,
        holiday_rows,
        int(df.loc[df["holiday"] != NO_HOLIDAY_LABEL, "holiday"].nunique()),
    )

    # --- categorical gaps --------------------------------------------------- #
    for col in ("weather_main", "weather_description"):
        n_missing = int(df[col].isna().sum())
        if n_missing:
            df[col] = df[col].fillna(UNKNOWN_LABEL)
            logger.warning(
                "Filled %d missing '%s' value(s) with '%s'",
                n_missing,
                col,
                UNKNOWN_LABEL,
            )

    return df


# --------------------------------------------------------------------------- #
# Step 4 - parse the timestamp
# --------------------------------------------------------------------------- #
def parse_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """Parse ``date_time`` and drop rows whose timestamp cannot be understood."""
    df = df.copy()
    parsed = pd.to_datetime(df["date_time"], errors="coerce")

    bad_mask = parsed.isna()
    n_bad = int(bad_mask.sum())
    if n_bad:
        examples = df.loc[bad_mask, "date_time"].astype(str).head(3).tolist()
        logger.warning(
            "Dropped %d row(s) with an unparseable date_time (examples: %s)",
            n_bad,
            ", ".join(repr(x) for x in examples),
        )
        df = df.loc[~bad_mask].copy()
        parsed = parsed.loc[~bad_mask]

    df["date_time"] = parsed
    df = df.sort_values("date_time", kind="stable").reset_index(drop=True)

    out_of_range = df["date_time"].dt.year.lt(2010) | df["date_time"].dt.year.gt(2030)
    n_out_of_range = int(out_of_range.sum())
    if n_out_of_range:
        logger.warning(
            "Dropped %d row(s) whose timestamp falls outside the expected 2010-2030 window",
            n_out_of_range,
        )
        df = df.loc[~out_of_range].reset_index(drop=True)

    logger.info(
        "Parsed date_time: %d valid timestamp(s), range %s to %s",
        len(df),
        df["date_time"].min(),
        df["date_time"].max(),
    )
    logger.debug(
        "Timestamp grid: %d unique hours between %s and %s (%d missing hours in the series)",
        df["date_time"].nunique(),
        df["date_time"].min(),
        df["date_time"].max(),
        len(pd.date_range(df["date_time"].min(), df["date_time"].max(), freq="h"))
        - df["date_time"].nunique(),
    )
    return df


# --------------------------------------------------------------------------- #
# Step 5 - duplicates
# --------------------------------------------------------------------------- #
def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove exact duplicate rows, then keep one row per hourly timestamp.

    Inspection of the raw file showed 5,445 hours carrying two to six rows: the
    traffic and temperature values are identical, only the weather observation
    differs. Keeping the first observation gives a clean one-row-per-hour series
    that the feature engineering and the CLI can index by timestamp.
    """
    df = df.copy()

    n_exact = int(df.duplicated().sum())
    if n_exact:
        df = df.drop_duplicates(keep="first").reset_index(drop=True)
        logger.warning(
            "Removed %d exact duplicate row(s) (every column identical, keeping the first)",
            n_exact,
        )
    else:
        logger.info("Duplicate check: no exact duplicate rows")

    n_same_hour = int(df.duplicated(subset=["date_time"], keep="first").sum())
    if n_same_hour:
        n_affected_hours = int(df.loc[df.duplicated(subset=["date_time"], keep=False), "date_time"].nunique())
        df = df.drop_duplicates(subset=["date_time"], keep="first").reset_index(drop=True)
        logger.warning(
            "Removed %d duplicate row(s) sharing an hourly timestamp across %d hour(s) "
            "(kept the first weather observation; traffic and temperature were identical)",
            n_same_hour,
            n_affected_hours,
        )
    else:
        logger.info("Duplicate check: no repeated timestamps")

    logger.info("Duplicate handling complete: %d row(s) remaining", len(df))
    return df


# --------------------------------------------------------------------------- #
# Monthly-median imputation helper (explicit loop over months)
# --------------------------------------------------------------------------- #
def _impute_monthly_median(
    df: pd.DataFrame,
    column: str,
    bad_mask: pd.Series,
    reason: str,
) -> tuple[pd.DataFrame, int]:
    """Replace ``bad_mask`` values of ``column`` with that month's median.

    The loop is deliberately explicit: for every calendar month present in the
    data a *conditional* check decides whether anything needs imputing, and the
    replacement value is computed from that month's own valid observations. A
    monthly median follows the seasonal cycle (a January temperature is nothing
    like a July temperature), which a single global median would blur.
    """
    total = 0
    bad_mask = bad_mask.fillna(False)

    for month in sorted(df.loc[bad_mask, "month"].unique()):
        in_month = df["month"] == month
        rows = in_month & bad_mask
        n_rows = int(rows.sum())
        if n_rows == 0:  # conditional: nothing to do for this month
            continue

        reference = df.loc[in_month & ~bad_mask, column].dropna()
        source = f"month {int(month):02d} median"
        if reference.empty:
            reference = df.loc[~bad_mask, column].dropna()
            source = "whole-dataset median (month had no valid value)"

        if reference.empty:
            logger.error(
                "Cannot impute '%s' in month %02d: no valid reference value anywhere",
                column,
                int(month),
            )
            continue

        fill_value = float(np.median(reference))
        df.loc[rows, column] = fill_value
        total += n_rows
        logger.warning(
            "Imputed %d %s value(s) in month %02d with the %s (%.2f) - reason: %s",
            n_rows,
            column,
            int(month),
            source,
            fill_value,
            reason,
        )
        logger.debug(
            "Imputation detail: column=%s month=%02d rows=%d source=%s reference_rows=%d",
            column,
            int(month),
            n_rows,
            source,
            len(reference),
        )

    return df, total


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Fill numeric gaps with the monthly median; categoricals were handled earlier."""
    df = df.copy()
    logger.info("Checking missing values: %s", df[NUMERIC_COLUMNS].isna().sum().to_dict())

    for col in NUMERIC_COLUMNS:
        n_missing = int(df[col].isna().sum())
        if n_missing == 0:
            logger.info("Missing values in '%s': none", col)
            continue
        logger.warning(
            "Handling %d missing value(s) in '%s' with the monthly median",
            n_missing,
            col,
        )
        df, _ = _impute_monthly_median(df, col, df[col].isna(), "missing value")

    remaining = int(df[NUMERIC_COLUMNS].isna().sum().sum())
    if remaining:
        logger.error("Missing-value handling left %d numeric gap(s)", remaining)
    return df


# --------------------------------------------------------------------------- #
# Step 7 - impossible values
# --------------------------------------------------------------------------- #
def impute_impossible_values(df: pd.DataFrame) -> pd.DataFrame:
    """Detect physically impossible readings and replace them with the monthly median."""
    df = df.copy()

    for col, (low, high) in PLAUSIBLE_RANGES.items():
        mask = (df[col] < low) | (df[col] > high)
        n_bad = int(mask.sum())
        if n_bad == 0:
            logger.info(
                "Impossible-value check for '%s': none outside [%s, %s]",
                col,
                low,
                high,
            )
            continue
        logger.warning(
            "Detected %d impossible '%s' value(s) outside the plausible range [%s, %s] - "
            "replacing with the monthly median",
            n_bad,
            col,
            low,
            high,
        )
        logger.debug(
            "Impossible values in '%s': min=%s max=%s",
            col,
            df.loc[mask, col].min(),
            df.loc[mask, col].max(),
        )
        df, _ = _impute_monthly_median(
            df, col, mask, f"value outside plausible range [{low}, {high}]"
        )

    return df


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run_pipeline(
    raw_csv: str | Path = DEFAULT_RAW_CSV,
    processed_csv: str | Path = DEFAULT_PROCESSED_CSV,
) -> pd.DataFrame:
    """Run every cleaning stage in order and write the cleaned CSV."""
    raw_csv = Path(raw_csv)
    processed_csv = Path(processed_csv)

    df = load_raw_data(raw_csv)
    rows_loaded = len(df)

    df = validate_schema(df)          # schema first, before any processing
    df = standardise_categories(df)
    df = parse_datetime(df)
    df = remove_duplicates(df)

    df["month"] = df["date_time"].dt.month
    df = handle_missing_values(df)
    df = impute_impossible_values(df)
    df = df.drop(columns=["month"])

    df = df.sort_values("date_time", kind="stable").reset_index(drop=True)

    processed_csv.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_csv(processed_csv, index=False)
    except OSError as exc:
        raise RawDataError(f"Could not write the cleaned CSV to {processed_csv}") from exc

    logger.info(
        "Saved cleaned data: %s (%d rows x %d columns)",
        processed_csv,
        df.shape[0],
        df.shape[1],
    )
    logger.info(
        "Cleaning summary: %d row(s) loaded, %d row(s) kept, %d row(s) dropped in total (%.2f%%)",
        rows_loaded,
        len(df),
        rows_loaded - len(df),
        100 * (rows_loaded - len(df)) / rows_loaded,
    )
    return df


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Task 1: load, validate and clean the raw traffic CSV."
    )
    parser.add_argument("--raw", default=str(DEFAULT_RAW_CSV), help="path to the raw CSV")
    parser.add_argument(
        "--output",
        default=str(DEFAULT_PROCESSED_CSV),
        help="path of the cleaned CSV to write",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="enable DEBUG level logging (written to logs/pipeline_debug.log)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    # Imported here only so that the module can be imported without side effects.
    from logging_setup import configure_logging

    configure_logging(debug=args.debug)
    logger.info("Starting pipeline.py (debug=%s)", args.debug)

    try:
        run_pipeline(raw_csv=args.raw, processed_csv=args.output)
    except (RawDataError, SchemaError) as exc:
        logger.error("Pipeline could not complete: %s", exc, exc_info=True)
        return 1
    except Exception as exc:  # top-level guard: log, never let a traceback escape
        logger.error("Unexpected failure in pipeline.py: %s", exc, exc_info=True)
        return 1

    logger.info("pipeline.py finished successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main())



