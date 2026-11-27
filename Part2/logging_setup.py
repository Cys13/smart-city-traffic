"""Central logging configuration for the smart-city traffic pipeline.

Design rule used throughout this project:
    * every module creates ``logger = logging.getLogger(__name__)``;
    * only the *entry-point* scripts (``run_all.py``, ``pipeline.py``,
      ``feature_engineering.py``, ``visualizations.py``, ``traffic_app.py``)
      call :func:`configure_logging` from inside ``main()``;
    * therefore importing a module never attaches a handler to the root logger
      and never changes global logging behaviour.

Two handlers are attached to the root logger:

    console  -> stdout, human readable progress for the person running the script
    file     -> logs/pipeline.log (normal run) or logs/pipeline_debug.log (--debug)

Log levels in this project
--------------------------
DEBUG    internal detail useful while developing: quartile thresholds, per-month
         medians used for imputation, intermediate shapes.
INFO     a step completed successfully (rows loaded, file written, command run).
WARNING  data was dropped, modified or imputed -- always with the row count and
         the reason, so nothing changes silently.
ERROR    a step could not be completed; the script exits with ``sys.exit(1)``
         after ``logger.error(..., exc_info=True)`` instead of raising a traceback.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
LOG_DIR = BASE_DIR / "logs"

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)-22s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Loggers from third-party libraries are noisy at DEBUG level and are not part
# of the pipeline, so they are pinned to WARNING even in a --debug run.
NOISY_LOGGERS = ("matplotlib", "PIL", "urllib3")


def configure_logging(debug: bool = False, log_file: str | Path | None = None) -> Path:
    """Attach a console + file handler to the root logger.

    Parameters
    ----------
    debug:
        When True the level is DEBUG and the default log file becomes
        ``logs/pipeline_debug.log`` so the normal-run log stays clean.
    log_file:
        Explicit log file path; overrides the default.

    Returns
    -------
    Path of the log file that receives the records.
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    if log_file is None:
        log_file = LOG_DIR / ("pipeline_debug.log" if debug else "pipeline.log")
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    level = logging.DEBUG if debug else logging.INFO
    formatter = logging.Formatter(fmt=LOG_FORMAT, datefmt=DATE_FORMAT)

    root = logging.getLogger()
    # Re-configuring (e.g. run_all.py then a stage's main()) must not duplicate lines.
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()
    root.setLevel(level)

    console = logging.StreamHandler(stream=sys.stdout)
    console.setLevel(level)
    console.setFormatter(formatter)
    root.addHandler(console)

    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    for name in NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)

    logger.debug("Logging configured: level=%s, file=%s", logging.getLevelName(level), log_path)
    return log_path

