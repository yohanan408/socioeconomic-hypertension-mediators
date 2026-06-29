"""
logger.py
=========
Centralised logging configuration for the mediation pipeline.

Sets up a root logger with two handlers:
  1. **StreamHandler** – writes ``INFO``+ messages to the terminal.
  2. **FileHandler** – writes ``DEBUG``+ messages to ``pipeline.log``.

Format
------
``%(asctime)s - %(levelname)s - %(message)s``

Usage
-----
.. code-block:: python

   from src.logger import logger
   logger.info("Data loading complete – shape: %s", df.shape)
"""

import logging
import sys


_LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def _setup_logger() -> logging.Logger:
    """Build and return the pre-configured root logger."""
    log = logging.getLogger("mediation_pipeline")
    log.setLevel(logging.DEBUG)

    # ── avoid duplicate handlers when module is reloaded ────────────
    if log.handlers:
        return log

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    # Console handler – INFO and above ───────────────────────────────
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)

    # File handler – DEBUG and above, append mode ────────────────────
    file_h = logging.FileHandler("pipeline.log", mode="a", encoding="utf-8")
    file_h.setLevel(logging.DEBUG)
    file_h.setFormatter(formatter)

    log.addHandler(console)
    log.addHandler(file_h)

    return log


logger = _setup_logger()
