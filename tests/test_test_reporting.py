"""
Unit tests for the test reporting module.
"""

import json
import pytest
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import patch, MagicMock

# Import will be available after implementation
with pytest.raises(ImportError):
    from prompter.test_reporting import TestReporter, PrompterTestReporter

# Use our test utilities
from test_utils import temp_dir, temp_file, mock_logger


class TestTestReporter:
    """Test the TestReporter class."""
    
    @pytest.fixture
    def reporter(self, temp_dir):
        """Create a test reporter instance."""
        with patch("prompter.test_reporting.TestReporter", create=True) as MockTestReporter:
            from prompter.test_reporting import TestReporter
            reporter = TestReporter(output_dir=temp_dir)
            yield reporter
    
    def test_initialization(self, reporter, temp_dir):
        """Test reporter initialization."""
        # Check output directory
        assert reporter.output_dir == temp_dir
        
        # Check initial report structure
        assert "timestamp" in reporter.current_report
        assert "tests" in reporter.current_report
        assert "summary" in reporter.current_report
        
        # Check summary structure
        summary = reporter.current_report["summary"]
        assert summary["total"] == 0
        assert summary["passed"] == 0
        assert summary["failed"] == 0
        assert summary["skipped"] == 0
        assert summary["errors"] == 0
        assert summary["duration"] == 0.0
    
    def test_add_test_result_passed(self, reporter):
        """Test adding a passed test result."""
        # Add a passed test
        reporter.add_test_result(
            name="test_example",
            status="passed",
            duration=0.01
        )
        
        # Check test was added
        assert len(reporter.current_report["tests"]) == 1
        test = reporter.current_report["tests"][0]
        assert test["name"] == "test_example"
        assert test["status"] == "passed"
        assert test["duration"] == 0.01
        
        # Check summary was updated
        summary = reporter.current_report["summary"]
        assert summary["total"] == 1
        assert summary["passed"] == 1
        assert summary["duration"] == 0.01
    
    def test_add_test_result_failed(self, reporter):
        """Test adding a failed test result."""
        # Add a failed test
        reporter.add_test_result(
            name="test_example_fail",
            status="failed",
            duration=0.02,
            error_message="AssertionError",
            traceback="File test_example.py, line 10"
        )
        
        # Check test was added
        assert len(reporter.current_report["tests"]) == 1
        test = reporter.current_report["tests"][0]
        assert test["name"] == "test_example_fail"
        assert test["status"] == "failed"
        assert test["duration"] == 0.02
        assert test["error_message"] == "AssertionError"
        assert test["traceback"] == "File test_example.py, line 10"
        
        # Check summary was updated
        summary = reporter.current_report["summary"]
        assert summary["total"] == 1
        assert summary["failed"] == 1
        assert summary["duration"] == 0.02
    
    def test_add_test_result_skipped(self, reporter):
        """Test adding a skipped test result."""
        # Add a skipped test
        reporter.add_test_result(
            name="test_example_skip",
            status="skipped",
            duration=0.0,
            error_message="Skipped: feature not implemented"
        )
        
        # Check test was added
        assert len(reporter.current_report["tests"]) == 1
        test = reporter.current_report["tests"][0]
        assert test["name"] == "test_example_skip"
        assert test["status"] == "skipped"
        
        # Check summary was updated
        summary = reporter.current_report["summary"]
        assert summary["total"] == 1
        assert summary["skipped"] == 1
    
    def test_add_test_result_error(self, reporter):
        """Test adding a test with an error."""
        # Add a test with error
        reporter.add_test_result(
            name="test_example_error",
            status="error",
            duration=0.03,
            error_message="ImportError: No module named 'nonexistent'",
            traceback="File test_example.py, line 5"
        )
        
        # Check test was added
        assert len(reporter.current_report["tests"]) == 1
        test = reporter.current_report["tests"][0]
        assert test["name"] == "test_example_error"
        assert test["status"] == "error"
        
        # Check summary was updated
        summary = reporter.current_report["summary"]
        assert summary["total"] == 1
        assert summary["errors"] == 1
    
    def test_multiple_tests(self, reporter):
        """Test adding multiple test results."""
        # Add multiple tests
        reporter.add_test_result(name="test1", status="passed", duration=0.01)
        reporter.add_test_result(name="test2", status="failed", duration=0.02)
        reporter.add_test_result(name="test3", status="skipped", duration=0.0)
        
        # Check all tests were added
        assert len(reporter.current_report["tests"]) == 3
        
        # Check summary was updated
        summary = reporter.current_report["summary"]
        assert summary["total"] == 3
        assert summary["passed"] == 1
        assert summary["failed"] == 1
        assert summary["skipped"] == 1
        assert summary["duration"] == 0.03
    
    def test_save_report(self, reporter, temp_dir):
        """Test saving the report to a file."""
        # Add a test
        reporter.add_test_result(name="test_example", status="passed", duration=0.01)
        
        # Save the report
        filename = "test_report.json"
        file_path = reporter.save_report(filename)
        
        # Check file was created
        assert file_path == temp_dir / filename
        assert file_path.exists()
        
        # Check file contents
        with open(file_path) as f:
            report_data = json.load(f)
            assert "tests" in report_data
            assert "summary" in report_data
            assert len(report_data["tests"]) == 1
    
    def test_save_report_default_filename(self, reporter):
        """Test saving the report with default filename."""
        # Save the report with default filename
        with patch("datetime.now") as mock_now:
            mock_now.return_value.strftime.return_value = "20230407_120000"
            file_path = reporter.save_report()
        
        # Check filename
        assert file_path.name == "test_report_20230407_120000.json"
    
    def test_get_summary(self, reporter):
        """Test getting a summary of the report."""
        # Add some tests
        reporter.add_test_result(name="test1", status="passed", duration=0.01)
        reporter.add_test_result(name="test2", status="failed", duration=0.02)
        
        # Get summary
        summary = reporter.get_summary()
        
        # Check summary
        assert summary["total"] == 2
        assert summary["passed"] == 1
        assert summary["failed"] == 1
        assert summary["duration"] == 0.03


class TestPrompterTestReporter:
    """Test the PrompterTestReporter class."""
    
    @pytest.fixture
    def pytest_plugin(self):
        """Create a pytest plugin instance."""
        with patch("prompter.test_reporting.PrompterTestReporter", create=True) as MockPlugin:
            from prompter.test_reporting import PrompterTestReporter
            
            # Mock the reporter
            reporter = MagicMock()
            plugin = PrompterTestReporter()
            plugin.reporter = reporter
            
            yield plugin
    
    def test_initialization(self):
        """Test plugin initialization."""
        with patch("prompter.test_reporting.PrompterTestReporter", create=True):
            with patch("prompter.test_reporting.TestReporter", create=True) as MockReporter:
                from prompter.test_reporting import PrompterTestReporter
                
                plugin = PrompterTestReporter()
                
                # Check reporter was created
                MockReporter.assert_called_once()
                assert plugin.start_time is None
    
    def test_pytest_sessionstart(self, pytest_plugin):
        """Test session start hook."""
        # Call the hook
        pytest_plugin.pytest_sessionstart(None)
        
        # Check start time was set
        assert pytest_plugin.start_time is not None
    
    def test_pytest_runtest_logreport_passed(self, pytest_plugin):
        """Test processing a passed test report."""
        # Create a mock report
        report = MagicMock()
        report.when = "call"
        report.outcome = "passed"
        report.nodeid = "test_example.py::test_function"
        report.duration = 0.01
        
        # Call the hook
        pytest_plugin.pytest_runtest_logreport(report)
        
        # Check the reporter was called
        pytest_plugin.reporter.add_test_result.assert_called_once_with(
            name="test_example.py::test_function",
            status="passed",
            duration=0.01,
            error_message=None,
            traceback=None
        )
    
    def test_pytest_runtest_logreport_failed(self, pytest_plugin):
        """Test processing a failed test report."""
        # Create a mock report
        report = MagicMock()
        report.when = "call"
        report.outcome = "failed"
        report.nodeid = "test_example.py::test_function"
        report.duration = 0.01
        report.longrepr = "AssertionError"
        report.longrepr.reprtraceback = "Traceback info"
        
        # Call the hook
        pytest_plugin.pytest_runtest_logreport(report)
        
        # Check the reporter was called
        pytest_plugin.reporter.add_test_result.assert_called_once_with(
            name="test_example.py::test_function",
            status="failed",
            duration=0.01,
            error_message="AssertionError",
            traceback="Traceback info"
        )
    
    def test_pytest_runtest_logreport_skipped(self, pytest_plugin):
        """Test processing a skipped test report."""
        # Create a mock report
        report = MagicMock()
        report.when = "call"
        report.outcome = "skipped"
        report.nodeid = "test_example.py::test_function"
        report.duration = 0.0
        report.longrepr = "Skipped: feature not implemented"
        
        # Call the hook
        pytest_plugin.pytest_runtest_logreport(report)
        
        # Check the reporter was called
        pytest_plugin.reporter.add_test_result.assert_called_once_with(
            name="test_example.py::test_function",
            status="skipped",
            duration=0.0,
            error_message="Skipped: feature not implemented",
            traceback=None
        )
    
    def test_pytest_runtest_logreport_setup(self, pytest_plugin):
        """Test processing a setup phase report."""
        # Create a mock report for setup phase
        report = MagicMock()
        report.when = "setup"
        
        # Call the hook
        pytest_plugin.pytest_runtest_logreport(report)
        
        # Check the reporter was not called for setup phase
        pytest_plugin.reporter.add_test_result.assert_not_called()
    
    def test_pytest_sessionfinish(self, pytest_plugin):
        """Test session finish hook."""
        # Set start time
        pytest_plugin.start_time = datetime.now()
        
        # Call the hook
        result = pytest_plugin.pytest_sessionfinish(None, 0)
        
        # Check summary was retrieved
        pytest_plugin.reporter.get_summary.assert_called_once()
        
        # Check report was saved
        pytest_plugin.reporter.save_report.assert_called_once()
        
        # Check return value
        assert result == 0


@pytest.mark.integration
class TestReportingIntegration:
    """Integration tests for the test reporting module."""
    
    def test_integration_with_pytest(self):
        """Test integration with pytest execution."""
        # This test will be implemented after the module is created
        pass 