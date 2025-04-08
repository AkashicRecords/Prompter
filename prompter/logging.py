"""
Logging module for Prompter.

This module provides centralized logging functionality for the Prompter package.
"""

import logging
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

# Default log level
DEFAULT_LOG_LEVEL = logging.INFO

# Default log format
DEFAULT_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Default log file
DEFAULT_LOG_FILE = Path.home() / ".prompter" / "logs" / "prompter.log"

class PrompterLogger:
    """Centralized logger for Prompter."""
    
    _instance: Optional["PrompterLogger"] = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(PrompterLogger, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if PrompterLogger._initialized:
            return
        
        self.logger = logging.getLogger("prompter")
        self.logger.setLevel(DEFAULT_LOG_LEVEL)
        
        # Create log directory if it doesn't exist
        log_dir = DEFAULT_LOG_FILE.parent
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Add file handler
        file_handler = logging.FileHandler(DEFAULT_LOG_FILE)
        file_handler.setLevel(DEFAULT_LOG_LEVEL)
        file_formatter = logging.Formatter(DEFAULT_LOG_FORMAT)
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)
        
        # Add console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(DEFAULT_LOG_LEVEL)
        console_formatter = logging.Formatter(DEFAULT_LOG_FORMAT)
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        PrompterLogger._initialized = True
    
    def debug(self, message: str, *args, **kwargs):
        """Log a debug message."""
        self.logger.debug(message, *args, **kwargs)
    
    def info(self, message: str, *args, **kwargs):
        """Log an info message."""
        self.logger.info(message, *args, **kwargs)
    
    def warning(self, message: str, *args, **kwargs):
        """Log a warning message."""
        self.logger.warning(message, *args, **kwargs)
    
    def error(self, message: str, *args, **kwargs):
        """Log an error message."""
        self.logger.error(message, *args, **kwargs)
    
    def critical(self, message: str, *args, **kwargs):
        """Log a critical message."""
        self.logger.critical(message, *args, **kwargs)
    
    def set_level(self, level: int):
        """Set the log level."""
        self.logger.setLevel(level)
        for handler in self.logger.handlers:
            handler.setLevel(level)

# Create a singleton instance
logger = PrompterLogger() 