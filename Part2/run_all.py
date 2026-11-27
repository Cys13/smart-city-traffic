"""End-to-end entry point: clean -> features -> figures in one command.

    python run_all.py             # normal run, INFO/WARNING into logs/pipeline.log
    python run_all.py --debug     # DEBUG detail into logs/pipeline_debug.log

This is the only place besides each script's own ``main()`` where logging
handlers are configured. Each stage keeps its own module logger
(``pipeline``, ``feature_engineering``, ``visualizations``), so the log trail
shows which step produced each line.
"""

from __future__ import annotations

import argparse
import logging
import sys

import feature_engineering
import pipeline
import visualizations
from logging_setup import configure_logging

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the whole smart-city traffic pipeline end to end.")
    parser.add_argument("--raw", default=str(pipeline.DEFAULT_RAW_CSV), help="raw CSV input")
    parser.add_argument("--debug", action="store_true", help="enable DEBUG logging (logs/pipeline_debug.log)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    log_path = configure_logging(debug=args.debug)
    logger.info("=== Smart-city traffic pipeline: full run started (debug=%s) ===", args.debug)
    logger.info("Log file for this run: %s", log_path)

    try:
        pipeline.run_pipeline(raw_csv=args.raw)
        feature_engineering.build_features()
        visualizations.create_all_figures()
    except (
        pipeline.RawDataError,
        pipeline.SchemaError,
        feature_engineering.FeatureEngineeringError,
        visualizations.VisualizationError,
    ) as exc:
        logger.error("Pipeline run aborted: %s", exc, exc_info=True)
        return 1
    except Exception as exc:  # top-level guard: log and exit, never raise a bare traceback
        logger.error("Unexpected failure during the full run: %s", exc, exc_info=True)
        return 1

    logger.info("=== Full run finished successfully: data/processed/ and figures/ are up to date ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())

