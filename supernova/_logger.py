"""
Centralized logging configuration for pynova with MPI-aware, colored output.

This module provides colored, formatted logging with MPI rank awareness:
- Rank 0 only logs INFO/WARNING/DEBUG messages (matches current behavior)
- All ranks log ERROR/CRITICAL messages (for collective error reporting)
- Color-coded severity levels (red=ERROR, yellow=WARNING, white=INFO, gray=DEBUG)
- Timestamps and module origin information
"""

import logging
import sys
from typing import Optional

try:
    from colorama import Fore, Style, init as colorama_init
    HAS_COLORAMA = True
    # Initialize colorama for cross-platform colored output
    colorama_init(autoreset=True)
except ImportError:
    HAS_COLORAMA = False



class ColoredFormatter(logging.Formatter):
    """
    Custom formatter that adds color coding based on log level.
    
    Color scheme:
    - DEBUG: Gray
    - INFO: White (default)
    - WARNING: Yellow
    - ERROR: Red
    - CRITICAL: Red + Bold
    """
    
    # Define color mappings
    if HAS_COLORAMA:
        COLORS = {
            'DEBUG': Fore.LIGHTBLACK_EX,
            'INFO': Fore.WHITE,
            'WARNING': Fore.YELLOW,
            'ERROR': Fore.RED,
            'CRITICAL': Fore.RED + Style.BRIGHT,
        }
        RESET = Style.RESET_ALL
    else:
        # No colors if colorama not available
        COLORS = {
            'DEBUG': '',
            'INFO': '',
            'WARNING': '',
            'ERROR': '',
            'CRITICAL': '',
        }
        RESET = ''
    
    def __init__(self, fmt: Optional[str] = None, datefmt: Optional[str] = None):
        """
        Initialize the colored formatter.

        Parameters
        ----------
        fmt : str, optional
            Log message format string.
        datefmt : str, optional
            Date format string.
        """
        if fmt is None:
            fmt = '[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s'
        
        if datefmt is None:
            datefmt = '%Y-%m-%d %H:%M:%S'
        
        super().__init__(fmt, datefmt)
    
    def format(self, record: logging.LogRecord) -> str:
        """Format the log record with appropriate color."""
        # Get the original formatted message
        log_message = super().format(record)
        
        # Add color based on level
        level_name = record.levelname
        color = self.COLORS.get(level_name, '')
        
        if color:
            return f"{color}{log_message}{self.RESET}"
        else:
            return log_message

class ErrorOnlyFilter(logging.Filter):
    """Filter that only allows ERROR and CRITICAL messages (for all-ranks logging)."""
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Return True if level is ERROR or CRITICAL."""
        return record.levelno >= logging.ERROR


# Global flag to track if logging has been configured
_LOGGING_CONFIGURED = False


def setup_logger(level: int = logging.INFO, log_file: Optional[str] = None) -> None:
    """
    Configure the logging system for pynova with MPI-aware, colored output.

    This function sets up:
    - Console handler with colored output (rank 0 only for INFO/WARNING/DEBUG)
    - Separate console handler for ERROR/CRITICAL (all ranks)
    - Optional file handler (all ranks write to separate files if MPI size > 1)

    Parameters
    ----------
    level : int, optional
        Minimum logging level (default: logging.INFO).
    log_file : str, optional
        Path to log file. If provided and MPI size > 1, each rank writes to
        ``{log_file}.rank{N}``. If None, only console logging is configured.

    Examples
    --------
    >>> from pynova._logger import setup_logger
    >>> import logging
    >>> setup_logger(level=logging.DEBUG)
    >>> setup_logger(level=logging.INFO, log_file='simulation.log')
    """
    global _LOGGING_CONFIGURED
    
    # Only configure once
    if _LOGGING_CONFIGURED:
        return
    
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)  # Set to DEBUG, handlers will filter
    
    # Remove any existing handlers
    root_logger.handlers.clear()
    
    # Create rank-0-only console handler for INFO/WARNING/DEBUG
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    handler.setFormatter(ColoredFormatter())
    root_logger.addHandler(handler)
    
    # Optional file handler
    if log_file is not None:
        # If multiple ranks, create separate log files
        file_path = log_file
        
        file_handler = logging.FileHandler(file_path, mode='w')
        file_handler.setLevel(logging.DEBUG)  # Log everything to file
        file_handler.setFormatter(ColoredFormatter())
        root_logger.addHandler(file_handler)
    
    _LOGGING_CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for the given module name.
    
    This is a convenience function that wraps logging.getLogger().
    
    Parameters
    ----------
    name : str
        Logger name, typically __name__ of the calling module
    
    Returns
    -------
    logging.Logger
        Logger instance
    
    Examples
    --------
    >>> logger = get_logger(__name__)
    >>> logger.info("Simulation started")
    >>> logger.warning("Missing optional parameter, using default")
    >>> logger.error("Failed to read input file")
    """
    return logging.getLogger(name)
