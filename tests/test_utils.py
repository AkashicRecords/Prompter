"""
Test utilities and common fixtures for Prompter tests.
"""

import os
import pytest
import tempfile
from pathlib import Path
import json
import asyncio
import ssl
import torch
from unittest.mock import MagicMock, patch, Mock, call
from cryptography.fernet import Fernet
from typing import Dict, Any, List, Optional, Union, Type

# Path fixtures
@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)

@pytest.fixture
def temp_file():
    """Create a temporary file for tests."""
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        file_path = Path(temp_file.name)
        yield file_path
        if file_path.exists():
            os.unlink(file_path)

# SSL fixtures
@pytest.fixture
def ssl_context():
    """Create a self-signed SSL context for testing."""
    context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE
    return context

@pytest.fixture
def ssl_cert_key(temp_dir):
    """Generate self-signed certificate and key for testing."""
    cert_path = temp_dir / "test.crt"
    key_path = temp_dir / "test.key"
    
    # This would normally create real certs, but for tests we'll simulate
    cert_path.write_text("TEST CERTIFICATE")
    key_path.write_text("TEST KEY")
    
    return cert_path, key_path

# Encryption fixtures
@pytest.fixture
def encryption_key():
    """Generate a Fernet encryption key for testing."""
    return Fernet.generate_key()

@pytest.fixture
def fernet(encryption_key):
    """Create a Fernet encryption instance for testing."""
    return Fernet(encryption_key)

# Async test helpers
@pytest.fixture
def event_loop():
    """Create an event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

# Enhanced mock fixtures with validation
@pytest.fixture
def mock_logger():
    """Create a mock logger for testing with strict type checking."""
    logger = Mock(spec=True)
    
    # Configure type checking for debug
    debug_mock = Mock()
    debug_mock.side_effect = lambda msg: _validate_log_message(msg)
    logger.debug = debug_mock
    
    # Configure type checking for info
    info_mock = Mock()
    info_mock.side_effect = lambda msg: _validate_log_message(msg)
    logger.info = info_mock
    
    # Configure type checking for warning
    warning_mock = Mock()
    warning_mock.side_effect = lambda msg: _validate_log_message(msg)
    logger.warning = warning_mock
    
    # Configure type checking for error
    error_mock = Mock()
    error_mock.side_effect = lambda msg: _validate_log_message(msg)
    logger.error = error_mock
    
    # Configure type checking for critical
    critical_mock = Mock()
    critical_mock.side_effect = lambda msg: _validate_log_message(msg)
    logger.critical = critical_mock
    
    return logger

def _validate_log_message(message):
    """Validate log message is a string."""
    if not isinstance(message, str):
        raise TypeError(f"Log message must be a string, got {type(message)}")
    return None

# Type-validated mock creator
def create_validated_mock(spec: Optional[Type] = None, 
                          return_value: Any = None, 
                          side_effect: Any = None,
                          type_validators: Dict[str, Dict[str, Type]] = None):
    """
    Create a mock with type validation for method arguments.
    
    Args:
        spec: The specification object for the mock
        return_value: The return value for the mock
        side_effect: The side effect for the mock
        type_validators: Dictionary mapping method names to argument validators
            Example: {'method_name': {'arg1': str, 'arg2': int}}
    
    Returns:
        A mock object with type validation
    """
    mock = Mock(spec=spec)
    
    if return_value is not None:
        mock.return_value = return_value
        
    if side_effect is not None:
        mock.side_effect = side_effect
    
    if type_validators:
        for method_name, arg_types in type_validators.items():
            original_method = getattr(mock, method_name, None)
            if original_method is not None:
                wrapped_method = Mock()
                
                def create_side_effect(method_name, arg_types):
                    def side_effect(*args, **kwargs):
                        # Skip self argument if it exists
                        start_idx = 1 if len(args) > 0 and args[0] == mock else 0
                        
                        # Validate positional args
                        for i, (arg_name, arg_type) in enumerate(arg_types.items()):
                            if i + start_idx < len(args):
                                if not isinstance(args[i + start_idx], arg_type):
                                    raise TypeError(
                                        f"Argument '{arg_name}' of method '{method_name}' "
                                        f"must be of type {arg_type.__name__}, "
                                        f"got {type(args[i + start_idx]).__name__}"
                                    )
                        
                        # Validate keyword args
                        for arg_name, arg_type in arg_types.items():
                            if arg_name in kwargs:
                                if not isinstance(kwargs[arg_name], arg_type):
                                    raise TypeError(
                                        f"Argument '{arg_name}' of method '{method_name}' "
                                        f"must be of type {arg_type.__name__}, "
                                        f"got {type(kwargs[arg_name]).__name__}"
                                    )
                        
                        return original_method(*args, **kwargs)
                    return side_effect
                
                wrapped_method.side_effect = create_side_effect(method_name, arg_types)
                setattr(mock, method_name, wrapped_method)
    
    return mock

# Pattern fixtures
@pytest.fixture
def sample_pattern_dict():
    """Create a sample pattern dictionary for testing."""
    return {
        "id": "test-pattern-1",
        "name": "Test Pattern",
        "description": "A test pattern for unit tests",
        "template": "This is a {placeholder} template",
        "variables": ["placeholder"],
        "score": 0.75,
        "tags": ["test", "example"],
        "created_at": "2023-04-07T12:00:00Z",
        "updated_at": "2023-04-07T12:00:00Z"
    }

# Enhanced validation for pattern dictionaries
def validate_pattern_dict(pattern_dict: Dict[str, Any]) -> bool:
    """
    Validate a pattern dictionary has the correct structure and types.
    
    Args:
        pattern_dict: Dictionary representing a pattern
        
    Returns:
        True if valid, raises exception otherwise
    """
    required_keys = ["id", "template"]
    for key in required_keys:
        if key not in pattern_dict:
            raise KeyError(f"Pattern dictionary missing required key: {key}")
    
    # Validate types
    if not isinstance(pattern_dict["id"], str):
        raise TypeError(f"Pattern 'id' must be a string, got {type(pattern_dict['id'])}")
    
    if not isinstance(pattern_dict["template"], str):
        raise TypeError(f"Pattern 'template' must be a string, got {type(pattern_dict['template'])}")
    
    if "variables" in pattern_dict and not isinstance(pattern_dict["variables"], list):
        raise TypeError(f"Pattern 'variables' must be a list, got {type(pattern_dict['variables'])}")
    
    if "score" in pattern_dict and not isinstance(pattern_dict["score"], (int, float)):
        raise TypeError(f"Pattern 'score' must be a number, got {type(pattern_dict['score'])}")
    
    if "tags" in pattern_dict and not isinstance(pattern_dict["tags"], list):
        raise TypeError(f"Pattern 'tags' must be a list, got {type(pattern_dict['tags'])}")
    
    return True

# Test data fixtures
@pytest.fixture
def test_report_data():
    """Create sample test report data."""
    report_data = {
        "timestamp": "2023-04-07T12:00:00Z",
        "tests": [
            {
                "name": "test_example",
                "status": "passed",
                "duration": 0.01,
                "timestamp": "2023-04-07T12:00:00Z"
            }
        ],
        "summary": {
            "total": 1,
            "passed": 1,
            "failed": 0,
            "skipped": 0,
            "errors": 0,
            "duration": 0.01
        }
    }
    
    # Validate the report data
    validate_test_report(report_data)
    
    return report_data

def validate_test_report(report: Dict[str, Any]) -> bool:
    """
    Validate a test report has the correct structure and types.
    
    Args:
        report: Dictionary representing a test report
        
    Returns:
        True if valid, raises exception otherwise
    """
    # Check required keys
    required_keys = ["timestamp", "tests", "summary"]
    for key in required_keys:
        if key not in report:
            raise KeyError(f"Test report missing required key: {key}")
    
    # Validate timestamp
    if not isinstance(report["timestamp"], str):
        raise TypeError(f"Report 'timestamp' must be a string, got {type(report['timestamp'])}")
    
    # Validate tests
    if not isinstance(report["tests"], list):
        raise TypeError(f"Report 'tests' must be a list, got {type(report['tests'])}")
    
    for test in report["tests"]:
        if not isinstance(test, dict):
            raise TypeError(f"Test must be a dictionary, got {type(test)}")
        
        test_required_keys = ["name", "status", "duration"]
        for key in test_required_keys:
            if key not in test:
                raise KeyError(f"Test missing required key: {key}")
        
        if not isinstance(test["name"], str):
            raise TypeError(f"Test 'name' must be a string, got {type(test['name'])}")
        
        if not isinstance(test["status"], str):
            raise TypeError(f"Test 'status' must be a string, got {type(test['status'])}")
        
        if not isinstance(test["duration"], (int, float)):
            raise TypeError(f"Test 'duration' must be a number, got {type(test['duration'])}")
    
    # Validate summary
    if not isinstance(report["summary"], dict):
        raise TypeError(f"Report 'summary' must be a dictionary, got {type(report['summary'])}")
    
    summary_required_keys = ["total", "passed", "failed", "skipped", "errors", "duration"]
    for key in summary_required_keys:
        if key not in report["summary"]:
            raise KeyError(f"Summary missing required key: {key}")
        
        if not isinstance(report["summary"][key], (int, float)):
            raise TypeError(f"Summary '{key}' must be a number, got {type(report['summary'][key])}")
    
    return True

# Check if CUDA is available for PyTorch tests
@pytest.fixture
def device():
    """Get the appropriate device (CPU or CUDA) for testing."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Mock tensor validation
def validate_tensor(tensor, expected_shape=None, expected_dtype=None, expected_device=None):
    """
    Validate a PyTorch tensor has expected properties.
    
    Args:
        tensor: The tensor to validate
        expected_shape: Expected shape of the tensor
        expected_dtype: Expected data type of the tensor
        expected_device: Expected device of the tensor
        
    Returns:
        True if valid, raises exception otherwise
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(tensor)}")
    
    if expected_shape is not None and tensor.shape != expected_shape:
        raise ValueError(f"Expected tensor shape {expected_shape}, got {tensor.shape}")
    
    if expected_dtype is not None and tensor.dtype != expected_dtype:
        raise ValueError(f"Expected tensor dtype {expected_dtype}, got {tensor.dtype}")
    
    if expected_device is not None:
        if isinstance(expected_device, str):
            expected_device = torch.device(expected_device)
        
        if tensor.device.type != expected_device.type:
            raise ValueError(f"Expected tensor device {expected_device}, got {tensor.device}")
    
    return True 