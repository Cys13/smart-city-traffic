# Smart City Traffic Analytics - Part 2 (Python)

Hourly traffic-volume analytics for Interstate 94 (Minneapolis-St Paul, October 2012 - September 2018).
This part of the capstone covers the Python data pipeline: loading and validating the raw CSV,
cleaning it, engineering features, plotting the patterns and querying the result from a small CLI.

* Raw data: 48,204 rows x 9 columns
* Cleaned data: 40,575 hourly rows (one row per hour)
* Engineered features: 40,575 rows x 47 columns
* Figures: 5 PNG files in `figures/`
* Report: 'report.docx`

---

## 1. Project structure

```
capstone_part2/
├── pipeline.py              # Task 1: load, validate schema, clean, save data/processed/traffic_clean.csv
├── feature_engineering.py   # Task 2: time/weather/scaled features + congestion category
├── visualizations.py        # Task 3: five Matplotlib figures in figures/
├── traffic_app.py           # Task 4: mini CLI (at, peak, compare, recommend, weather)
├── run_all.py               # entry point: clean -> features -> figures in one command
├── logging_setup.py         # single configure_logging() used by the entry points only
├── requirements.txt
├── README.md
├── REPORT.md
├── report.docx
├── data/
│   ├── raw/
│   │   └── metro_interstate_traffic_volume.csv     # raw export, never modified
│   └── processed/
│       ├── traffic_clean.csv                       # output of pipeline.py
│       └── traffic_features.csv                    # output of feature_engineering.py
├── figures/
│   ├── fig1_traffic_by_hour.png
│   ├── fig2_weekday_vs_weekend.png
│   ├── fig3_traffic_by_weather.png
│   ├── fig4_temperature_vs_traffic.png
│   └── fig5_traffic_distribution.png
└── logs/
    ├── pipeline.log            # sample log of a normal (INFO) run
    └── pipeline_debug.log      # sample log of a --debug run
```

---

## 2. Setup

Requires Python 3.10+ (developed on 3.11).

```bash
cd capstone_part2
python -m venv .venv
source .venv/Scripts/activate      # Git Bash on Windows
# or: source .venv/bin/activate    # Linux / macOS
pip install -r requirements.txt
```

`requirements.txt` pins `pandas`, `numpy`, `matplotlib` and `python-docx` (the last one is only
needed if you regenerate `report.docx`).

---

## 3. How to run

Everything in one go:

```bash
python run_all.py            # INFO/WARNING into logs/pipeline.log
python run_all.py --debug    # DEBUG detail into logs/pipeline_debug.log
```

Stage by stage:

```bash
python pipeline.py                       # raw CSV -> data/processed/traffic_clean.csv
python feature_engineering.py            # cleaned CSV -> data/processed/traffic_features.csv
python visualizations.py                 # features CSV -> figures/*.png
```

Each stage accepts `--debug`, and `pipeline.py` also accepts `--raw` / `--output`:

```bash
python pipeline.py --raw data/raw/metro_interstate_traffic_volume.csv --output data/processed/traffic_clean.csv
python pipeline.py --debug
```

### Mini CLI (`traffic_app.py`)

| Command | What it answers | Example |
|---|---|---|
| `at` | traffic and conditions at one timestamp | `python traffic_app.py at --datetime "2018-06-15 08:00"` |
| `peak` | busiest hours of the day (all / weekday / weekend) | `python traffic_app.py peak --top 5 --scope weekday` |
| `compare` | weekday vs weekend traffic | `python traffic_app.py compare` |
| `recommend` | quietest hours and best 3-hour window for a date or weekday | `python traffic_app.py recommend --day Wednesday` / `--date 2018-06-16` |
| `weather` | weather + traffic summary for one day | `python traffic_app.py weather --date 2018-06-15` |

Global options: `--data <path to traffic_features.csv>`, `--debug`.
Only the answer goes to standard output; diagnostics go through the logger. A malformed date
(e.g. `--datetime "2018-13-45 99:00"`) is logged as an ERROR and the app exits with status 1,
without a traceback.

---

## 4. What the data inspection found

The raw export is the *Metro Interstate Traffic Volume* dataset, but nothing was assumed: the file
was profiled first and every rule below comes from what it actually contains.

| Check | Finding | How it is handled |
|---|---|---|
| Columns / types | 9 columns: `holiday`, `temp`, `rain_1h`, `snow_1h`, `clouds_all`, `weather_main`, `weather_description`, `date_time`, `traffic_volume`. `holiday` and the two weather columns are text, `temp`/`rain_1h`/`snow_1h` float, `clouds_all`/`traffic_volume` integer | Expected schema declared in `REQUIRED_COLUMNS`; validated before anything else |
| Encoding | UTF-8 with a BOM | read with `encoding="utf-8-sig"` |
| Missing values | Only `holiday`: 48,143 rows hold the literal string `None`; no numeric gaps at all | `None` means "not a holiday", renamed to `No Holiday` (pandas would otherwise read it as NaN on the next load) |
| Duplicate rows | 17 fully identical rows, plus 7,612 rows sharing an hourly timestamp with another row (5,430 hours have 2-6 rows) | exact duplicates dropped first, then one row per hour kept; the duplicates carry identical traffic and temperature, only the weather observation differs |
| Repeated sensor hours | After de-duplication the series has 40,575 unique hours out of 52,551 hours between the first and last timestamp | the 11,976 missing hours are left as gaps (they are absent observations, not zeros) |
| Category spellings | `weather_description`: `Sky is Clear` (1,726 rows) and `sky is clear`; `SQUALLS` (4 rows) and `squall` | lower-cased, trimmed, whitespace collapsed, variant mapping applied: 1,730 rows changed |
| Other categories | `weather_main` (11 labels) and `holiday` (11 distinct holidays) already consistent | canonical mapping applied defensively |
| Dates | all 48,204 timestamps parse, format `YYYY-MM-DD HH:MM:SS`, range 2012-10-02 09:00 to 2018-09-30 23:00 | parsed with `errors="coerce"`; unparseable rows would be dropped and counted |
| Impossible values | 10 rows at exactly 0.00 K (-273 C, Jan/Feb 2014); 1 row with 9,831.3 mm of rain in one hour; 33 rows with 0-9 vehicles/hour (whole hours of 2016-07-23 among them) | replaced with the **monthly median** |
| Range after cleaning | temp -29.8 to 36.9 C, rain 0 to 55.63 mm, traffic 10 to 7,280 veh/h | consistent with a Minnesota interstate |

---

## 5. Cleaning rules (pipeline.py) and their counts

| Step | Rule | Rows affected |
|---|---|---|
| Load | `try/except` for `FileNotFoundError`, `EmptyDataError`, `ParserError`, `UnicodeDecodeError`, `PermissionError` | 48,204 x 9 loaded |
| Schema | all 9 required columns must exist; numeric columns coerced with `pd.to_numeric` | 0 problems |
| Categories | trim/lower-case, variant mapping, `None` -> `No Holiday` | 1,730 description rows |
| Dates | parse, sort, drop unparseable / out-of-window rows | 0 dropped |
| Duplicates | exact duplicates, then one row per timestamp | 17 + 7,612 = 7,629 dropped (15.83%) |
| Missing values | numeric gaps -> monthly median (no gaps existed); categorical gaps -> `Unknown` | 0 |
| Impossible values | outside the plausible ranges below -> monthly median | 44 values (10 temp, 1 rain, 33 traffic) |

Plausible ranges used for the impossible-value check:

| Column | Range | Why |
|---|---|---|
| `temp` | 200-333 K | -73 C to 60 C, beyond any observed Minnesota hour |
| `rain_1h` | 0-200 mm | 200 mm in one hour is far beyond any recorded event |
| `snow_1h` | 0-100 mm | same |
| `clouds_all` | 0-100 % | definition |
| `traffic_volume` | 10-10,000 veh/h | the interstate never falls below ~10 vehicles/hour when the sensor works; above 10,000 is beyond capacity |

Imputation is a **loop over calendar months**: for each month the code checks whether any value in
that month is invalid and, if so, replaces it with that month's own median (a January temperature is
not comparable to a July one). Actual values used in this run:

* temperature: 4 January values -> 265.48 K, 6 February values -> 265.89 K
* rain: 1 July value (9,831.3 mm) -> 0.00 mm (the July median)
* traffic: 25 July values -> 3,347 veh/h, 1 August value -> 3,726 veh/h, 7 September values -> 3,539 veh/h

---

## 6. Feature engineering (feature_engineering.py)

| Group | Columns |
|---|---|
| Calendar | `hour`, `day_of_week` (0 = Monday), `day_name`, `month`, `is_weekend` |
| Cyclical | `hour_sin`, `hour_cos`, `day_of_week_sin`, `day_of_week_cos` |
| Weather | `temp_c`, `is_raining`, `is_snowing`, `is_adverse_weather`, `is_holiday` |
| Categorical encoding | 11 `wm_*` one-hot columns for `weather_main`; 8 `wd_*` columns for the 7 most frequent `weather_description` labels plus `wd_other` |
| Scaled | `temp_c_minmax`, `traffic_volume_minmax`, `rain_1h_log_minmax` (min-max to [0, 1], rain log-transformed first because it is extremely skewed) |
| Target | `congestion` (Low / Moderate / High / Severe), `congestion_rank` (0-3) |

Shape: 40,575 x 9 before, 40,575 x 47 after.

### Congestion category logic

The bands are the **quartiles of `traffic_volume`** on the cleaned dataset, so each label means
"relative to a normal hour on this interstate" and each band holds roughly a quarter of the hours:

| Band | Rule | Threshold (veh/h) | Hours |
|---|---|---|---|
| Low | `volume <= Q1` | <= 1,256 | 10,145 |
| Moderate | `Q1 < volume <= Q2` | 1,256 - 3,429 | 10,145 |
| High | `Q2 < volume <= Q3` | 3,429 - 4,952 | 10,143 |
| Severe | `volume > Q3` | > 4,952 | 10,142 |

Thresholds are computed with `numpy.percentile(..., [25, 50, 75])` at run time (logged at DEBUG),
not hard-coded, so the labels stay balanced if the input file changes.

---

## 7. Figures and what they show

### Figure 1 - `fig1_traffic_by_hour.png` (mean traffic by hour, weekday vs weekend)

Weekday traffic has two sharp commuter peaks: 06:00-07:00 rises to 6,061 veh/h at 07:00, dips to
4,401 veh/h at 10:00, then climbs to the day's maximum of **6,241 veh/h at 16:00**, and collapses to
309 veh/h by 02:00. Weekend traffic is a single broad hump that peaks at **4,410 veh/h at 12:00**
and never reaches the weekday peaks. Between 19:00 and 21:00 the two curves converge (3,332 vs
3,258 veh/h at 19:00), which is why evening travel is equally busy on both.

### Figure 2 - `fig2_weekday_vs_weekend.png` (mean traffic per day of week)

Friday is the busiest day (**3,675 veh/h**) and Sunday the quietest (**2,429 veh/h**), a 34% gap.
The weekday mean is 3,558 veh/h against 2,631 veh/h at the weekend, i.e. **weekend traffic is 26.1%
lower**. The drop is not caused by fewer hours of daylight: the weekend curve in Figure 1 simply
lacks the two commuter peaks.

### Figure 3 - `fig3_traffic_by_weather.png` (mean traffic per weather category)

Differences between weather categories are much smaller than the time-of-day differences:
Clouds 3,619 veh/h (15,123 h), Haze 3,602 (766 h), Rain 3,361 (4,769 h), Drizzle 3,160 (515 h),
Clear 3,058 (13,366 h), Thunderstorm 3,008 (453 h), Snow 3,002 (2,297 h), Mist 2,858 (3,069 h),
Fog 2,724 (201 h). "Clear" is *below* "Clouds" because clear hours are concentrated at night and
at weekends. Fog and snow are the only categories associated with clearly lower volumes. `Smoke`
(15 h) and `Squall` (1 h) are too rare to average and are excluded from the chart.

### Figure 4 - `fig4_temperature_vs_traffic.png` (temperature vs traffic)

The linear relationship is weak: **Pearson r = 0.14**. The binned mean nevertheless rises steadily,
from about 1,900 veh/h in the -30 to -25 C bin to **5,074 veh/h in the hottest 5 C bin**, because
warm hours are also daytime and summer hours. Temperature is therefore not a good single predictor
of traffic - hour of day and weekday carry nearly all the signal.

### Figure 5 - `fig5_traffic_distribution.png` (traffic distribution and congestion bands)

The distribution is clearly bimodal: a large spike at 300-600 veh/h (night hours) and a broad mass
between 2,500 and 6,000 veh/h (daytime). Mean 3,293, median 3,429, Q1 1,256, Q3 4,952, maximum
7,280 veh/h. Because of the night spike, a quarter of all hours still fall below 1,256 veh/h, so the
"Low" band is mostly night traffic rather than light daytime traffic.

---

## 8. Key findings

1. **Time of day dominates.** Weekday demand swings from 309 veh/h at 02:00 to 6,241 veh/h at 16:00,
   a factor of 20. Weather categories differ by at most ~30%.
2. **The weekend is structurally different, not just quieter.** Weekday hours average 3,558 veh/h
   and weekend hours 2,631 (-26.1%); 33.8% of weekday hours are "Severe" against 3.1% at weekends.
   The weekday curves peak twice (07:00 and 16:00), the weekend curve once (12:00).
3. **Holidays are the quietest days of the year.** 53 holiday hours average 864 veh/h, 74% below the
   3,297 veh/h of normal hours.
4. **Rain barely changes volume** (3,334 veh/h in rainy hours vs 3,293 overall, 5.1% of hours are
   rainy), while snow and fog coincide with the lowest averages (2,900 and 2,724 veh/h).
5. **The dataset is incomplete but usable.** 11,976 of the 52,551 hours in the period are missing,
   and one whole day (2016-07-23) had a traffic sensor dropout that reported 0-8 veh/h.

---

## 9. Logging configuration

* `logging_setup.py` holds a single `configure_logging(debug)` function. It is called **only** from
  the `main()` of an entry-point script (`run_all.py`, `pipeline.py`, `feature_engineering.py`,
  `visualizations.py`, `traffic_app.py`), never at import time.
* Every module uses `logger = logging.getLogger(__name__)`, so each line names its module
  (`pipeline`, `feature_engineering`, `visualizations`, `traffic_app`).
* Two handlers: console (stdout) and a file. Format:
  `2026-11-26 13:54:59 | WARNING  | pipeline | Removed 17 exact duplicate row(s) ...`
  (timestamp, level, module, message).
* Files: `logs/pipeline.log` for a normal run, `logs/pipeline_debug.log` for a `--debug` run.
  Each run starts a fresh file.
* Levels:
  * **DEBUG** - internal detail: raw dtypes, timestamp grid, per-month imputation reference counts,
    quartile thresholds. Only with `--debug`.
  * **INFO** - a step finished (rows loaded, shape before/after, figure saved, command invoked).
  * **WARNING** - data dropped, modified or imputed, always with the count and the reason.
  * **ERROR** - a step could not finish; logged with `exc_info=True` and the process exits with
    status 1 instead of raising an unhandled traceback.
* No `print()` is used for status anywhere; `traffic_app.py` uses `print()` only for the answer.

---

## 10. Assumptions and limitations

* The raw file was taken as the authoritative source; its `holiday = None` string was interpreted as
  "no holiday", not as missing data.
* Duplicate timestamps were resolved by keeping the first weather observation. Traffic and
  temperature are identical inside each duplicated hour, so no measurement is lost, but the choice of
  weather label is arbitrary for those 5,430 hours.
* "Plausible ranges" are engineering choices, documented in `pipeline.py`; a different lower bound
  for `traffic_volume` (for example 1 instead of 10) would keep the 33 sensor-dropout hours instead
  of imputing them.
* Missing hours are not interpolated: the CLI and the figures use only observed hours.
* The congestion bands are relative to this dataset, not to a traffic-engineering level-of-service
  standard.
* Part 1 findings were not available while writing this, so the interpretation is based only on this
  dataset; align the wording with Part 1 before submitting.

