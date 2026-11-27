"""Task 4 - mini command line app for the smart-city traffic dataset.

Five read-only commands query the engineered dataset:

    at        traffic at one specific date and time
    peak      the busiest hours of the day
    compare   weekday vs weekend traffic
    recommend quietest hours / best 3-hour travel window for a day
    weather   weather and traffic summary for one calendar day

Every invocation is logged at INFO with its arguments, and bad input (for
example a malformed date) is logged at ERROR and exits with ``sys.exit(1)``
instead of printing a traceback. ``print()`` is used only for the answer.

Examples
--------
    python traffic_app.py at --datetime "2018-06-15 08:00"
    python traffic_app.py peak --top 5 --scope weekday
    python traffic_app.py compare
    python traffic_app.py recommend --day Wednesday
    python traffic_app.py weather --date 2018-06-15
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_FEATURES_CSV = BASE_DIR / "data" / "processed" / "traffic_features.csv"

DATETIME_FORMAT = "%Y-%m-%d %H:%M"
DATE_FORMAT = "%Y-%m-%d"
DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


class AppError(RuntimeError):
    """A user-facing error: message is logged and the app exits with status 1."""


def load_data(path: str | Path = DEFAULT_FEATURES_CSV) -> pd.DataFrame:
    """Load the features CSV used by every command."""
    path = Path(path)
    logger.info("Loading dataset from %s", path)
    try:
        df = pd.read_csv(path, parse_dates=["date_time"], encoding="utf-8-sig")
    except FileNotFoundError as exc:
        raise AppError(
            f"Dataset not found: {path}. Run 'python run_all.py' (or pipeline.py then "
            "feature_engineering.py) first."
        ) from exc
    except pd.errors.EmptyDataError as exc:
        raise AppError(f"Dataset is empty: {path}") from exc
    except pd.errors.ParserError as exc:
        raise AppError(f"Dataset is not parseable: {path}") from exc

    if df.empty:
        raise AppError(f"Dataset has no rows: {path}")
    logger.info("Dataset ready: %d rows, %s to %s", len(df), df["date_time"].min(), df["date_time"].max())
    return df


# --------------------------------------------------------------------------- #
# Input parsing helpers (raise AppError with a readable message)
# --------------------------------------------------------------------------- #
def parse_datetime_arg(value: str) -> datetime:
    try:
        return datetime.strptime(value.strip(), DATETIME_FORMAT)
    except ValueError as exc:
        raise AppError(
            f"Invalid --datetime '{value}': expected 'YYYY-MM-DD HH:MM', e.g. '2018-06-15 08:00'"
        ) from exc


def parse_date_arg(value: str) -> datetime:
    try:
        return datetime.strptime(value.strip(), DATE_FORMAT)
    except ValueError as exc:
        raise AppError(
            f"Invalid --date '{value}': expected 'YYYY-MM-DD', e.g. '2018-06-15'"
        ) from exc


def parse_day_arg(value: str) -> str:
    for day in DAY_NAMES:
        if day.lower() == value.strip().lower():
            return day
    raise AppError(f"Invalid --day '{value}': choose one of {', '.join(DAY_NAMES)}")


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #
def cmd_at(df: pd.DataFrame, args: argparse.Namespace) -> None:
    """Traffic volume at one specific hour."""
    target = parse_datetime_arg(args.datetime)
    matches = df[df["date_time"] == pd.Timestamp(target)]

    note = ""
    if matches.empty:
        # Fall back to the nearest recorded hour within a 3-hour window.
        deltas = (df["date_time"] - pd.Timestamp(target)).abs()
        if deltas.min() > pd.Timedelta(hours=3):
            raise AppError(
                f"No traffic record within 3 hours of {target:%Y-%m-%d %H:%M}. "
                f"Data covers {df['date_time'].min():%Y-%m-%d %H:%M} to {df['date_time'].max():%Y-%m-%d %H:%M}."
            )
        matches = df.loc[[deltas.idxmin()]]
        note = " (nearest recorded hour; the exact timestamp is missing from the series)"

    row = matches.iloc[0]
    print(f"Traffic at {row['date_time']:%Y-%m-%d %H:%M}{note}")
    print(f"  volume      : {int(row['traffic_volume']):,} vehicles/hour")
    print(f"  congestion  : {row['congestion']} (rank {int(row['congestion_rank'])}/3)")
    print(f"  weather     : {row['weather_main']} - {row['weather_description']}")
    print(f"  temperature : {row['temp_c']:.1f} C, rain {row['rain_1h']:.2f} mm, snow {row['snow_1h']:.2f} mm")
    print(f"  calendar    : {row['day_name']}, {'weekend' if row['is_weekend'] else 'weekday'}"
          f"{', holiday (' + str(row['holiday']) + ')' if row['is_holiday'] else ''}")


def cmd_peak(df: pd.DataFrame, args: argparse.Namespace) -> None:
    """The busiest hours of the day, by mean traffic volume."""
    if args.scope == "weekday":
        subset = df[df["is_weekend"] == 0]
    elif args.scope == "weekend":
        subset = df[df["is_weekend"] == 1]
    else:
        subset = df

    if subset.empty:
        raise AppError(f"No records available for scope '{args.scope}'")

    profile = subset.groupby("hour")["traffic_volume"].agg(["mean", "count"]).sort_values("mean", ascending=False)
    top = profile.head(args.top)

    print(f"Busiest hours of the day ({args.scope} hours, mean traffic volume)")
    print(f"  {'hour':>6}  {'mean veh/h':>12}  {'hours':>6}")
    for hour, row in top.iterrows():
        print(f"  {hour:>4d}:00  {row['mean']:>12,.0f}  {int(row['count']):>6,}")

    busiest = top.index[0]
    print(f"\nPeak hour: {busiest:02d}:00-{busiest:02d}:59 with {top['mean'].iloc[0]:,.0f} vehicles/hour on average.")
    print(f"Quietest hour: {profile.index[-1]:02d}:00 with {profile['mean'].iloc[-1]:,.0f} vehicles/hour.")


def cmd_compare(df: pd.DataFrame, args: argparse.Namespace) -> None:
    """Weekday vs weekend comparison."""
    weekday = df[df["is_weekend"] == 0]
    weekend = df[df["is_weekend"] == 1]
    if weekday.empty or weekend.empty:
        raise AppError("Both weekday and weekend records are needed for this comparison.")

    wd_mean, we_mean = weekday["traffic_volume"].mean(), weekend["traffic_volume"].mean()
    wd_median, we_median = weekday["traffic_volume"].median(), weekend["traffic_volume"].median()
    severe_share = lambda part: 100 * (part["congestion"] == "Severe").mean()  # noqa: E731

    print("Weekday vs weekend traffic")
    print(f"  {'':<12}{'weekday':>12}{'weekend':>12}{'difference':>14}")
    print(f"  {'hours':<12}{len(weekday):>12,}{len(weekend):>12,}{len(weekday) - len(weekend):>14,}")
    print(f"  {'mean veh/h':<12}{wd_mean:>12,.0f}{we_mean:>12,.0f}{we_mean - wd_mean:>14,.0f}")
    print(f"  {'median veh/h':<12}{wd_median:>12,.0f}{we_median:>12,.0f}{we_median - wd_median:>14,.0f}")
    print(f"  {'peak hour':<12}{weekday.groupby('hour')['traffic_volume'].mean().idxmax():>9d}:00"
          f"{weekend.groupby('hour')['traffic_volume'].mean().idxmax():>9d}:00")
    print(f"  {'severe share':<12}{severe_share(weekday):>11.1f}%{severe_share(weekend):>11.1f}%")

    change = 100 * (we_mean - wd_mean) / wd_mean
    print(f"\nWeekend traffic is {abs(change):.1f}% {'lower' if change < 0 else 'higher'} than weekday traffic on average.")
    print("Weekday demand has two commuter peaks; weekend demand is flatter and peaks around midday.")


def cmd_recommend(df: pd.DataFrame, args: argparse.Namespace) -> None:
    """Quietest hours and the best 3-hour window for a day or weekday."""
    if args.date:
        date = parse_date_arg(args.date)
        day_name = DAY_NAMES[date.weekday()]
        scope_label = f"{date:%Y-%m-%d} ({day_name})"
    elif args.day:
        day_name = parse_day_arg(args.day)
        scope_label = day_name
    else:
        raise AppError("Provide --date YYYY-MM-DD or --day <weekday name>")

    profile = (
        df[df["day_name"] == day_name]
        .groupby("hour")["traffic_volume"]
        .agg(["mean", "count"])
        .reindex(range(24))
        .dropna()
    )
    if profile.empty:
        raise AppError(f"No records for {scope_label} in the dataset")

    quietest = profile.sort_values("mean").head(args.top)
    print(f"Quietest hours for {scope_label} (mean over {int(profile['count'].sum()):,} recorded hours)")
    print(f"  {'hour':>6}  {'mean veh/h':>12}")
    for hour, row in quietest.iterrows():
        print(f"  {hour:>4d}:00  {row['mean']:>12,.0f}")

    # Best 3-hour window: lowest rolling mean, computed on the 24-hour profile.
    full = profile["mean"].reindex(range(24)).interpolate(limit_direction="both")
    window = full.rolling(window=3).mean().dropna()
    start = int(window.idxmin()) - 2
    print(
        f"\nRecommended 3-hour window: {start:02d}:00-{start + 3:02d}:00 "
        f"(mean {window.min():,.0f} vehicles/hour, about "
        f"{100 * (window.min() / profile['mean'].max()):.0f}% of the day's peak hour)."
    )
    print(f"Busiest hour that day: {int(profile['mean'].idxmax()):02d}:00 ({profile['mean'].max():,.0f} vehicles/hour).")


def cmd_weather(df: pd.DataFrame, args: argparse.Namespace) -> None:
    """Weather and traffic summary for one calendar day."""
    date = parse_date_arg(args.date)
    day = df[df["date_time"].dt.date == date.date()]
    if day.empty:
        raise AppError(
            f"No records for {date:%Y-%m-%d}. Data covers {df['date_time'].min():%Y-%m-%d} to {df['date_time'].max():%Y-%m-%d}."
        )

    dominant = day["weather_main"].value_counts()
    temp_col = day["temp_c"]
    print(f"Weather and traffic on {date:%Y-%m-%d} ({day['day_name'].iloc[0]}, {len(day)} recorded hours)")
    print(f"  temperature : min {temp_col.min():.1f} C, mean {temp_col.mean():.1f} C, max {temp_col.max():.1f} C")
    print(f"  rain        : {day['rain_1h'].sum():.2f} mm over {int(day['is_raining'].sum())} hour(s)")
    print(f"  snow        : {day['snow_1h'].sum():.2f} mm over {int(day['is_snowing'].sum())} hour(s)")
    print(f"  weather     : {', '.join(f'{k} ({v}h)' for k, v in dominant.items())}")
    print(f"  traffic     : mean {day['traffic_volume'].mean():,.0f}, max {int(day['traffic_volume'].max()):,} vehicles/hour")
    print(f"  congestion  : {', '.join(f'{k}={v}' for k, v in day['congestion'].value_counts().items())}")


# --------------------------------------------------------------------------- #
# CLI wiring
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="traffic_app.py",
        description="Mini CLI for the smart-city traffic dataset (Task 4).",
    )
    parser.add_argument("--data", default=str(DEFAULT_FEATURES_CSV), help="path to traffic_features.csv")
    parser.add_argument("--debug", action="store_true", help="enable DEBUG logging (logs/pipeline_debug.log)")
    subparsers = parser.add_subparsers(dest="command", required=True, metavar="command")

    at = subparsers.add_parser("at", help="traffic at one date and time")
    at.add_argument("--datetime", required=True, help="'YYYY-MM-DD HH:MM'")
    at.set_defaults(func=cmd_at)

    peak = subparsers.add_parser("peak", help="busiest hours of the day")
    peak.add_argument("--top", type=int, default=5, help="how many hours to list (default 5)")
    peak.add_argument("--scope", choices=["all", "weekday", "weekend"], default="all", help="which hours to rank")
    peak.set_defaults(func=cmd_peak)

    compare = subparsers.add_parser("compare", help="weekday vs weekend comparison")
    compare.set_defaults(func=cmd_compare)

    recommend = subparsers.add_parser("recommend", help="quietest hours / best travel window")
    group = recommend.add_mutually_exclusive_group(required=True)
    group.add_argument("--date", help="'YYYY-MM-DD'")
    group.add_argument("--day", help="weekday name, e.g. Wednesday")
    recommend.add_argument("--top", type=int, default=4, help="how many quiet hours to list (default 4)")
    recommend.set_defaults(func=cmd_recommend)

    weather = subparsers.add_parser("weather", help="weather and traffic for one day")
    weather.add_argument("--date", required=True, help="'YYYY-MM-DD'")
    weather.set_defaults(func=cmd_weather)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    from logging_setup import configure_logging

    configure_logging(debug=args.debug)

    command = args.command
    arguments = {k: v for k, v in vars(args).items() if k not in ("func", "debug", "command")}
    logger.info("Command invoked: %s | arguments: %s", command, arguments)

    try:
        if hasattr(args, "top") and args.top <= 0:
            raise AppError("--top must be a positive integer")
        df = load_data(args.data)
        args.func(df, args)
    except AppError as exc:
        logger.error("Command '%s' failed: %s", command, exc)
        return 1
    except Exception as exc:  # top-level guard: no traceback for the user
        logger.error("Unexpected failure in command '%s': %s", command, exc, exc_info=True)
        return 1

    logger.info("Command '%s' completed", command)
    return 0


if __name__ == "__main__":
    sys.exit(main())


