"""Centralised logging configuration for the project.

Imports this module early to suppress TensorFlow deprecation warnings
consistently across all modules (predict.py, app.py, train_model.py).
"""

import logging
import os
import sys
from pathlib import Path

# ── Suppress TF 2.15 internal deprecation warnings (centralised) ─────────
# These MUST be set before importing tensorflow in any module.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

# Suppress TF Python-level logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)

LOG_FILE = Path(__file__).parent / "logs" / "app.log"
LOG_FILE.parent.mkdir(exist_ok=True)

_FMT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE = "%Y-%m-%d %H:%M:%S"


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger. Call once per module.

    Usage::
        from logger import get_logger
        log = get_logger(__name__)
        log.info("Model loaded successfully")
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(logging.Formatter(_FMT, _DATE))
    logger.addHandler(ch)

    # File handler
    fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
    fh.setFormatter(logging.Formatter(_FMT, _DATE))
    logger.addHandler(fh)

    return logger
