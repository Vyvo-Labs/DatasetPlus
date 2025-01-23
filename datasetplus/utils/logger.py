import logging
import os
from typing import Optional

import colorlog


def setup_logger(
    name: str = "datasetplus",
    level: int = logging.INFO,
    log_file: Optional[str] = None,
) -> logging.Logger:
    """Set up a logger with colored output.

    Args:
        name: Name of the logger
        level: Logging level
        log_file: Optional path to a log file

    Returns:
        logging.Logger: Configured logger instance
    """
    formatter = colorlog.ColoredFormatter(
        "%(log_color)s%(levelname)-8s%(reset)s %(blue)s%(message)s",
        log_colors={
            "DEBUG": "cyan",
            "INFO": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "red,bg_white",
        },
    )

    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        if log_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(logging.Formatter("%(levelname)-8s %(message)s"))
            logger.addHandler(file_handler)

    return logger


def get_logger(name: str = "datasetplus") -> logging.Logger:
    """Get the logger instance.

    Args:
        name: Name of the logger

    Returns:
        logging.Logger: Logger instance
    """
    return logging.getLogger(name)
