"""
Centralized logging configuration for the application.

This module provides a standardized way to get a logger instance, ensuring
that all logs produced by the application (including those from Prefect tasks)
are consistent in format and output.
"""

import logging
import sys
from typing import Optional

# Define a standard logging format
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Configures and retrieves a logger instance.

    This function ensures that all loggers use a consistent format and
    stream to stdout. This is crucial for visibility within Prefect logs
    and Docker container logs.

    Args:
        name: The name for the logger, typically __name__.
        level: The logging level to set (e.g., logging.INFO, logging.DEBUG).

    Returns:
        A configured logging.Logger instance.
    """
    # Get the logger instance
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Check if the logger already has handlers configured
    # This prevents duplicate log messages if get_logger is called multiple times
    if not logger.hasHandlers():
        # Create a stream handler to output to stdout
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(level)

        # Create a formatter
        formatter = logging.Formatter(LOG_FORMAT)

        # Set the formatter for the handler
        stream_handler.setFormatter(formatter)

        # Add the handler to the logger
        logger.addHandler(stream_handler)
        
        # Prevent logs from propagating to the root logger
        logger.propagate = False

    return logger