"""
Logging utility for EduSolve - Fixed for Windows Unicode support
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
import io

# Global logger instance to prevent re-initialization
_logger_instance = None


def setup_logger(name: str, log_file: str = None, level=logging.INFO):
    """
    Setup logger with UTF-8 encoding support for Windows
    Returns singleton logger instance
    """

    global _logger_instance

    # If logger already exists, return it (SINGLETON PATTERN)
    if _logger_instance is not None:
        return _logger_instance

    # Create new logger
    logger = logging.getLogger(name)

    # Clear any existing handlers (important for Streamlit reruns)
    if logger.handlers:
        logger.handlers.clear()

    logger.setLevel(level)

    # Prevent propagation to root logger (stops duplication)
    logger.propagate = False

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler with UTF-8 encoding
    # Use sys.stdout with UTF-8 encoding wrapper for Windows compatibility
    try:
        # Reconfigure stdout to use UTF-8 (Python 3.7+)
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")

        # Create stream with UTF-8 encoding
        console_stream = io.TextIOWrapper(
            sys.stdout.buffer,
            encoding="utf-8",
            errors="replace",  # Replace unencodable characters instead of crashing
            line_buffering=True,
        )
    except Exception:
        # Fallback to regular stdout if reconfiguration fails
        console_stream = sys.stdout

    console_handler = logging.StreamHandler(console_stream)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(level)
    logger.addHandler(console_handler)

    # File handler with UTF-8 encoding
    if log_file:
        log_path = Path("logs")
        log_path.mkdir(exist_ok=True)

        # Add date to log filename
        log_filename = f"{datetime.now().strftime('%Y-%m-%d')}_{log_file}"

        # Use UTF-8 encoding for file handler
        file_handler = logging.FileHandler(
            log_path / log_filename,
            mode="a",
            encoding="utf-8",  # Ensure file is written with UTF-8
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(level)
        logger.addHandler(file_handler)

    # Store as singleton
    _logger_instance = logger

    return logger


def get_logger():
    """Get existing logger instance or create new one"""
    global _logger_instance

    if _logger_instance is None:
        _logger_instance = setup_logger("edusolve", "edusolve.log")

    return _logger_instance


# Global logger instance (singleton)
logger = get_logger()
