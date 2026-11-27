"""Task 3 - Matplotlib figures that show what the data actually does.

Five figures are written to ``figures/`` (PNG, 160 dpi):

1. ``fig1_traffic_by_hour.png``       mean hourly traffic, weekday vs weekend
2. ``fig2_weekday_vs_weekend.png``    mean traffic per day of week
3. ``fig3_traffic_by_weather.png``    mean traffic per weather category, with hour counts
4. ``fig4_temperature_vs_traffic.png`` temperature vs traffic, scatter + binned mean
5. ``fig5_traffic_distribution.png``  traffic-volume distribution with the quartile bands

Each figure logs the numbers it is drawn from at INFO, so the interpretation in
the README/report can be checked against the log.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless rendering: no window is opened
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_FEATURES_CSV = BASE_DIR / "data" / "processed" / "traffic_features.csv"
DEFAULT_FIGURES_DIR = BASE_DIR / "figures"

WEEKDAY_COLOUR = "#1f4e79"
WEEKEND_COLOUR = "#d9822b"
ACCENT_COLOUR = "#2e7d32"
DAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


class VisualizationError(RuntimeError):
    """The features file is missing or a figure could not be written."""


def load_features(path: str | Path = DEFAULT_FEATURES_CSV) -> pd.DataFrame:
    """Read the engineered features CSV."""
    path = Path(path)
    logger.info("Loading features for plotting from %s", path)
    try:
        df = pd.read_csv(path, parse_dates=["date_time"], encoding="utf-8-sig")
    except FileNotFoundError as exc:
        raise VisualizationError(
            f"Features dataset not found: {path}. Run run_all.py or feature_engineering.py first."
        ) from exc
    except pd.errors.EmptyDataError as exc:
        raise VisualizationError(f"Features dataset is empty: {path}") from exc
    except pd.errors.ParserError as exc:
        raise VisualizationError(f"Features dataset is not parseable: {path}") from exc
    return df


def _save(fig: plt.Figure, path: Path) -> Path:
    """Save a figure and log where it went."""
    try:
        fig.savefig(path, dpi=160, bbox_inches="tight")
    except OSError as exc:
        raise VisualizationError(f"Could not write figure {path}") from exc
    finally:
        plt.close(fig)
    logger.info("Saved figure: %s", path)
    return path


def plot_traffic_by_hour(df: pd.DataFrame, out_dir: Path) -> Path:
    """Figure 1 - the daily double peak, split into weekday and weekend."""
    profile = (
        df.pivot_table(index="hour", columns="is_weekend", values="traffic_volume", aggfunc="mean")
        .reindex(columns=[0, 1])
        .rename(columns={0: "Weekday", 1: "Weekend"})
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(profile.index, profile["Weekday"], marker="o", color=WEEKDAY_COLOUR, label="Weekday (Mon-Fri)")
    ax.plot(profile.index, profile["Weekend"], marker="s", color=WEEKEND_COLOUR, label="Weekend (Sat-Sun)")

    peak_hour = int(profile["Weekday"].idxmax())
    peak_value = float(profile["Weekday"].max())
    ax.annotate(
        f"weekday peak {peak_hour:02d}:00\n{peak_value:,.0f} veh/h",
        xy=(peak_hour, peak_value),
        xytext=(peak_hour - 7.5, peak_value * 0.90),
        arrowprops={"arrowstyle": "->", "color": "grey"},
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 2},
    )
    ax.set_title("Mean traffic volume by hour of day")
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Mean traffic volume (vehicles/hour)")
    ax.set_xticks(range(0, 24, 2))
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)

    logger.info(
        "Figure 1 data: weekday peak %02d:00 at %.0f veh/h, weekday trough %02d:00 at %.0f veh/h; "
        "weekend peak %02d:00 at %.0f veh/h",
        peak_hour,
        peak_value,
        int(profile["Weekday"].idxmin()),
        float(profile["Weekday"].min()),
        int(profile["Weekend"].idxmax()),
        float(profile["Weekend"].max()),
    )
    return _save(fig, out_dir / "fig1_traffic_by_hour.png")


def plot_weekday_vs_weekend(df: pd.DataFrame, out_dir: Path) -> Path:
    """Figure 2 - mean traffic per weekday, weekend bars highlighted."""
    means = df.groupby("day_name")["traffic_volume"].mean().reindex(DAY_ORDER)
    colours = [WEEKEND_COLOUR if day in ("Saturday", "Sunday") else WEEKDAY_COLOUR for day in DAY_ORDER]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(DAY_ORDER, means.values, color=colours)
    ax.bar_label(bars, labels=[f"{v:,.0f}" for v in means.values], padding=3, fontsize=8)

    weekday_mean = df.loc[df["is_weekend"] == 0, "traffic_volume"].mean()
    weekend_mean = df.loc[df["is_weekend"] == 1, "traffic_volume"].mean()
    ax.axhline(weekday_mean, color=WEEKDAY_COLOUR, linestyle="--", linewidth=1, alpha=0.6)
    ax.axhline(weekend_mean, color=WEEKEND_COLOUR, linestyle="--", linewidth=1, alpha=0.6)
    ax.set_title("Mean traffic volume by day of week")
    ax.set_xlabel("Day of week")
    ax.set_ylabel("Mean traffic volume (vehicles/hour)")
    ax.set_ylim(0, means.max() * 1.15)
    ax.grid(axis="y", alpha=0.3)

    logger.info(
        "Figure 2 data: weekday mean %.0f veh/h vs weekend mean %.0f veh/h (%.1f%% lower), "
        "busiest day %s (%.0f), quietest day %s (%.0f)",
        weekday_mean,
        weekend_mean,
        100 * (weekday_mean - weekend_mean) / weekday_mean,
        means.idxmax(),
        means.max(),
        means.idxmin(),
        means.min(),
    )
    return _save(fig, out_dir / "fig2_weekday_vs_weekend.png")


def plot_traffic_by_weather(df: pd.DataFrame, out_dir: Path) -> Path:
    """Figure 3 - mean traffic per weather category (categories with enough hours)."""
    stats = (
        df.groupby("weather_main")
        .agg(mean_traffic=("traffic_volume", "mean"), hours=("traffic_volume", "size"))
        .sort_values("mean_traffic")
    )
    # Categories seen in fewer than 20 hours (e.g. one single Squall hour) are too
    # thin to average, so they are left out of the chart and named in the log.
    thin = stats[stats["hours"] < 20]
    stats = stats[stats["hours"] >= 20]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(stats.index, stats["mean_traffic"], color=ACCENT_COLOUR, alpha=0.85)
    ax.bar_label(
        bars,
        labels=[f"{m:,.0f}  ({h:,} h)" for m, h in zip(stats["mean_traffic"], stats["hours"])],
        padding=4,
        fontsize=8,
    )
    ax.set_title("Mean traffic volume by weather category")
    ax.set_xlabel("Mean traffic volume (vehicles/hour)")
    ax.set_ylabel("weather_main")
    ax.set_xlim(0, stats["mean_traffic"].max() * 1.3)
    ax.grid(axis="x", alpha=0.3)

    logger.info(
        "Figure 3 data: highest mean under '%s' (%.0f veh/h, %d h), lowest under '%s' (%.0f veh/h, %d h); "
        "excluded as too rare: %s",
        stats["mean_traffic"].idxmax(),
        stats["mean_traffic"].max(),
        int(stats.loc[stats["mean_traffic"].idxmax(), "hours"]),
        stats["mean_traffic"].idxmin(),
        stats["mean_traffic"].min(),
        int(stats.loc[stats["mean_traffic"].idxmin(), "hours"]),
        ", ".join(f"{idx} ({int(row.hours)} h)" for idx, row in thin.iterrows()) or "none",
    )
    return _save(fig, out_dir / "fig3_traffic_by_weather.png")


def plot_temperature_vs_traffic(df: pd.DataFrame, out_dir: Path) -> Path:
    """Figure 4 - temperature vs traffic scatter with a binned mean line."""
    sample = df.sample(n=min(6000, len(df)), random_state=42)
    correlation = float(np.corrcoef(df["temp_c"], df["traffic_volume"])[0, 1])

    bins = np.arange(np.floor(df["temp_c"].min() / 5) * 5, np.ceil(df["temp_c"].max() / 5) * 5 + 5, 5)
    binned = df.groupby(pd.cut(df["temp_c"], bins), observed=True).agg(
        temp_mid=("temp_c", "mean"), mean_traffic=("traffic_volume", "mean")
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(sample["temp_c"], sample["traffic_volume"], s=6, alpha=0.15, color=WEEKDAY_COLOUR, label="hourly records (sample)")
    ax.plot(binned["temp_mid"], binned["mean_traffic"], color=WEEKEND_COLOUR, linewidth=2.2, marker="o", label="mean per 5 C bin")
    ax.set_title(f"Temperature vs traffic volume (Pearson r = {correlation:.2f})")
    ax.set_xlabel("Temperature (degrees C)")
    ax.set_ylabel("Traffic volume (vehicles/hour)")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, loc="upper left")

    logger.info(
        "Figure 4 data: r(temp_c, traffic_volume) = %.3f, %d points sampled; hottest 5 C bin mean %.0f veh/h",
        correlation,
        len(sample),
        float(binned["mean_traffic"].iloc[-1]),
    )
    return _save(fig, out_dir / "fig4_temperature_vs_traffic.png")


def plot_traffic_distribution(df: pd.DataFrame, out_dir: Path) -> Path:
    """Figure 5 - distribution of hourly traffic with the congestion quartiles."""
    q1, q2, q3 = (float(np.percentile(df["traffic_volume"], p)) for p in (25, 50, 75))

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(df["traffic_volume"], bins=60, color=WEEKDAY_COLOUR, alpha=0.8, edgecolor="white", linewidth=0.4)
    top = ax.get_ylim()[1]
    for value, label in ((q1, "Q1"), (q2, "Q2"), (q3, "Q3")):
        ax.axvline(value, color=WEEKEND_COLOUR, linestyle="--", linewidth=1.4)
        ax.text(
            value - 60,
            top * 0.97,
            label,
            color=WEEKEND_COLOUR,
            fontsize=9,
            rotation=90,
            va="top",
            ha="right",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 1},
        )

    band_y = top * 0.72
    for x, label in ((q1 / 2, "Low"), ((q1 + q2) / 2, "Moderate"), ((q2 + q3) / 2, "High"), (q3 + (df["traffic_volume"].max() - q3) / 2, "Severe")):
        ax.text(x, band_y, label, ha="center", fontsize=9, color="dimgrey")

    ax.set_title(
        "Distribution of hourly traffic volume with congestion bands\n"
        f"Q1 = {q1:,.0f}, Q2 (median) = {q2:,.0f}, Q3 = {q3:,.0f} veh/h",
        fontsize=11,
    )
    ax.set_xlabel("Traffic volume (vehicles/hour)")
    ax.set_ylabel("Number of hours")
    ax.grid(axis="y", alpha=0.3)

    logger.info(
        "Figure 5 data: min %d, Q1 %.0f, median %.0f, Q3 %.0f, max %d veh/h; mean %.0f",
        int(df["traffic_volume"].min()),
        q1,
        q2,
        q3,
        int(df["traffic_volume"].max()),
        df["traffic_volume"].mean(),
    )
    return _save(fig, out_dir / "fig5_traffic_distribution.png")


def create_all_figures(
    features_csv: str | Path = DEFAULT_FEATURES_CSV,
    figures_dir: str | Path = DEFAULT_FIGURES_DIR,
) -> list[Path]:
    """Build every figure and return the list of saved paths."""
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    df = load_features(features_csv)
    logger.info("Plotting %d rows x %d columns", df.shape[0], df.shape[1])

    paths = [
        plot_traffic_by_hour(df, figures_dir),
        plot_weekday_vs_weekend(df, figures_dir),
        plot_traffic_by_weather(df, figures_dir),
        plot_temperature_vs_traffic(df, figures_dir),
        plot_traffic_distribution(df, figures_dir),
    ]
    logger.info("%d figure(s) written to %s", len(paths), figures_dir)
    return paths


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task 3: build the Matplotlib figures.")
    parser.add_argument("--input", default=str(DEFAULT_FEATURES_CSV), help="features CSV from feature_engineering.py")
    parser.add_argument("--figures-dir", default=str(DEFAULT_FIGURES_DIR), help="folder for the PNG files")
    parser.add_argument("--debug", action="store_true", help="enable DEBUG logging (logs/pipeline_debug.log)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    from logging_setup import configure_logging

    configure_logging(debug=args.debug)
    logger.info("Starting visualizations.py (debug=%s)", args.debug)

    try:
        create_all_figures(features_csv=args.input, figures_dir=args.figures_dir)
    except VisualizationError as exc:
        logger.error("Visualization step could not complete: %s", exc, exc_info=True)
        return 1
    except Exception as exc:
        logger.error("Unexpected failure in visualizations.py: %s", exc, exc_info=True)
        return 1

    logger.info("visualizations.py finished successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main())



