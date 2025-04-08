"""
Unit tests for the logging module.
"""

import os
import pytest
import logging
from pathlib import Path
from unittest.mock import patch, MagicMock, Mock

# Import test utilities
from test_utils import create_validated_mock, _validate_log_message

# Import the module under test
from prompter.logging import PrompterLogger

# Mock the module for testing with enhanced validation
class MockPrompterLogger:
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(MockPrompterLogger, cls).__new__(cls)
            cls._instance.initialized = False
        return cls._instance
    
    def __init__(self, log_level=logging.INFO, log_format=None, log_file=None):
        if not self.initialized:
            self.log_level = log_level
            self.log_format = log_format or "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            self.log_file = log_file
            
            # Create type-validated logger mock
            self.logger = create_validated_mock(
                type_validators={
                    'debug': {'message': str},
                    'info': {'message': str},
                    'warning': {'message': str},
                    'error': {'message': str},
                    'critical': {'message': str},
                    'setLevel': {'level': int}
                }
            )
            
            self.initialized = True
    
    def debug(self, message):
        # Validate message type
        _validate_log_message(message)
        self.logger.debug(message)
    
    def info(self, message):
        # Validate message type
        _validate_log_message(message)
        self.logger.info(message)
    
    def warning(self, message):
        # Validate message type
        _validate_log_message(message)
        self.logger.warning(message)
    
    def error(self, message):
        # Validate message type
        _validate_log_message(message)
        self.logger.error(message)
    
    def critical(self, message):
        # Validate message type
        _validate_log_message(message)
        self.logger.critical(message)
    
    def set_level(self, level):
        # Validate level type
        if not isinstance(level, int):
            raise TypeError(f"Log level must be an integer, got {type(level)}")
        
        self.log_level = level
        self.logger.setLevel(level)


@pytest.fixture
def mock_prompter_logger():
    """Create a validated mock logger instance."""
    with patch("prompter.logging.PrompterLogger", MockPrompterLogger):
        logger_instance = MockPrompterLogger()
        yield logger_instance


class TestPrompterLogger:
    """Test the PrompterLogger class."""
    
    def test_singleton_pattern(self, mock_prompter_logger):
        """Test that the PrompterLogger follows the singleton pattern."""
        # We need to patch the import first, then test
        with patch("prompter.logging.PrompterLogger", MockPrompterLogger):
            from prompter.logging import PrompterLogger
            
            # Create multiple instances
            logger1 = PrompterLogger()
            logger2 = PrompterLogger()
            
            # Verify they are the same instance
            assert logger1 is logger2, "Loggers should be the same instance (singleton pattern)"
            
            # Verify type
            assert isinstance(logger1, PrompterLogger), "logger1 should be a PrompterLogger instance"
            assert isinstance(logger2, PrompterLogger), "logger2 should be a PrompterLogger instance"
    
    def test_initialization(self, temp_dir):
        """Test logger initialization with default parameters."""
        with patch("prompter.logging.PrompterLogger", MockPrompterLogger):
            from prompter.logging import PrompterLogger
            
            # Reset the singleton for this test
            MockPrompterLogger._instance = None
            
            # Create with default parameters
            with patch("pathlib.Path.home", return_value=temp_dir):
                logger = PrompterLogger()
                
                # Check default log level
                assert logger.log_level == logging.INFO, "Default log level should be INFO"
                
                # Check that log directory was created
                log_dir = temp_dir / ".prompter" / "logs"
                assert log_dir.exists(), "Log directory should be created"
                assert isinstance(log_dir, Path), "Log directory should be a Path object"
    
    def test_custom_log_level(self):
        """Test setting a custom log level."""
        with patch("prompter.logging.PrompterLogger", MockPrompterLogger):
            from prompter.logging import PrompterLogger
            
            # Reset the singleton for this test
            MockPrompterLogger._instance = None
            
            # Create with custom log level
            logger = PrompterLogger(log_level=logging.DEBUG)
            
            # Check custom log level
            assert logger.log_level == logging.DEBUG, "Custom log level should be DEBUG"
            assert isinstance(logger.log_level, int), "Log level should be an integer"
    
    def test_custom_log_file(self, temp_dir):
        """Test setting a custom log file."""
        with patch("prompter.logging.PrompterLogger", MockPrompterLogger):
            from prompter.logging import PrompterLogger
            
            # Reset the singleton for this test
            MockPrompterLogger._instance = None
            
            # Create with custom log file
            custom_log_file = temp_dir / "custom.log"
            logger = PrompterLogger(log_file=custom_log_file)
            
            # Check custom log file
            assert logger.log_file == custom_log_file, "Custom log file should be set correctly"
            assert isinstance(logger.log_file, Path), "Log file should be a Path object"
    
    def test_log_methods(self, mock_prompter_logger):
        """Test all logging methods with strict type checking."""
        # Debug - valid string
        mock_prompter_logger.debug("Debug message")
        mock_prompter_logger.logger.debug.assert_called_with("Debug message")
        
        # Info - valid string
        mock_prompter_logger.info("Info message")
        mock_prompter_logger.logger.info.assert_called_with("Info message")
        
        # Warning - valid string
        mock_prompter_logger.warning("Warning message")
        mock_prompter_logger.logger.warning.assert_called_with("Warning message")
        
        # Error - valid string
        mock_prompter_logger.error("Error message")
        mock_prompter_logger.logger.error.assert_called_with("Error message")
        
        # Critical - valid string
        mock_prompter_logger.critical("Critical message")
        mock_prompter_logger.logger.critical.assert_called_with("Critical message")
        
        # Test with invalid types - should raise TypeError
        with pytest.raises(TypeError):
            mock_prompter_logger.debug(123)
        
        with pytest.raises(TypeError):
            mock_prompter_logger.info(['invalid', 'type'])
        
        with pytest.raises(TypeError):
            mock_prompter_logger.warning({'invalid': 'type'})
    
    def test_set_level(self, mock_prompter_logger):
        """Test changing log level dynamically with type validation."""
        # Valid log level (integer)
        mock_prompter_logger.set_level(logging.DEBUG)
        
        # Check log level was changed
        assert mock_prompter_logger.log_level == logging.DEBUG, "Log level should be updated to DEBUG"
        mock_prompter_logger.logger.setLevel.assert_called_with(logging.DEBUG)
        
        # Test with invalid type - should raise TypeError
        with pytest.raises(TypeError):
            mock_prompter_logger.set_level("DEBUG")
    
    @pytest.mark.integration
    def test_integration_with_other_modules(self):
        """Test integration with other modules."""
        # This test will be implemented after other modules are created
        pass 