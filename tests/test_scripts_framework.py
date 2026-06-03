#!/usr/bin/env python3
"""
EFD3D Scripts Testing Framework
Common utilities and test patterns for testing CLI scripts

Created: 2026-03-08
Provides:
- CLI argument testing utilities
- Function-level testing patterns
- Integration/output testing patterns
- Test configuration for scripts
"""

from collections.abc import Callable
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any
from unittest.mock import MagicMock

import pytest

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============================================================================
# Common Test Utilities
# ============================================================================


class ScriptTestHelper:
    """Helper class for script testing"""

    def __init__(self, script_path: str):
        """
        Initialize script test helper

        Args:
            script_path: Relative or absolute path to the script file
        """
        self.script_path = Path(script_path)
        if not self.script_path.is_absolute():
            self.script_path = Path(os.getcwd()) / self.script_path

    def run_script(self, args: list[str], capture_output: bool = True) -> subprocess.CompletedProcess:
        """
        Run script with given arguments

        Args:
            args: List of command line arguments
            capture_output: Whether to capture stdout/stderr

        Returns:
            CompletedProcess object
        """
        cmd = [sys.executable, str(self.script_path), *args]
        return subprocess.run(cmd, capture_output=capture_output, text=True, cwd=os.getcwd(), check=False)

    def create_temp_config(self, config_dict: dict[str, Any]) -> str:
        """
        Create a temporary config file from dict

        Args:
            config_dict: Configuration dictionary

        Returns:
            Path to temporary config file
        """
        fd, path = tempfile.mkstemp(suffix=".json", text=True)
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(config_dict, f, indent=2)
            return path
        except:
            os.close(fd)
            raise

    def create_temp_file(self, content: str, suffix: str = ".txt") -> str:
        """
        Create a temporary file with content

        Args:
            content: File content
            suffix: File suffix

        Returns:
            Path to temporary file
        """
        fd, path = tempfile.mkstemp(suffix=suffix, text=True)
        try:
            with os.fdopen(fd, "w") as f:
                f.write(content)
            return path
        except:
            os.close(fd)
            raise

    def parse_json_output(self, output: str) -> dict[str, Any]:
        """
        Parse JSON output from script

        Args:
            output: JSON string output

        Returns:
            Parsed dictionary
        """
        try:
            return json.loads(output)
        except json.JSONDecodeError as err:
            msg = f"Failed to parse JSON output: {err}\nOutput: {output}"
            raise ValueError(msg) from err


class ConfigFixture:
    """Fixture for creating test configurations"""

    @staticmethod
    def minimal_config() -> dict[str, Any]:
        """Create minimal valid configuration"""
        return {
            "model": {
                "input_dim": 6,
                "output_dim": 5,
                "hidden_layers": [64, 64],
                "activation": "relu",
            },
            "training": {"epochs": 100, "batch_size": 32},
            "physics": {"theta0": 120.0, "epsilon_r": 12.0, "sigma": 0.015},
        }

    @staticmethod
    def training_config() -> dict[str, Any]:
        """Create training configuration"""
        return {
            "model": {
                "input_dim": 6,
                "output_dim": 5,
                "hidden_layers": [128, 128, 128],
                "activation": "gelu",
                "use_batch_norm": True,
            },
            "training": {
                "epochs": 1000,
                "batch_size": 64,
                "learning_rate": 0.001,
                "stage1_epochs": 100,
                "stage2_epochs": 500,
            },
            "physics": {
                "theta0": 120.0,
                "epsilon_r": 12.0,
                "sigma": 0.015,
                "tau": 0.005,
                "zeta": 0.8,
            },
            "data": {"train_size": 1000, "val_size": 200, "test_size": 200},
        }

    @staticmethod
    def analysis_config() -> dict[str, Any]:
        """Create analysis configuration"""
        return {
            "analysis": {
                "type": "training",
                "log_dir": "logs/",
                "output_dir": "results/analysis/",
                "metrics": ["loss", "residuals", "volume_error"],
            },
            "visualization": {"save_plots": True, "plot_format": "png", "dpi": 300},
        }


# ============================================================================
# CLI Argument Testing Patterns
# ============================================================================


class CLITestCase:
    """Base class for CLI argument testing"""

    def assert_exit_success(self, result: subprocess.CompletedProcess):
        """Assert script exited with success (return code 0)"""
        assert (
            result.returncode == 0
        ), f"Script failed with return code {result.returncode}\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}"

    def assert_exit_failure(self, result: subprocess.CompletedProcess):
        """Assert script exited with failure (non-zero return code)"""
        assert (
            result.returncode != 0
        ), f"Script succeeded when it should have failed\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}"

    def assert_output_contains(self, result: subprocess.CompletedProcess, text: str):
        """Assert output contains specific text"""
        assert (
            text in result.stdout or text in result.stderr
        ), f"Expected output to contain '{text}'\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}"

    def assert_json_output(self, result: subprocess.CompletedProcess) -> dict[str, Any]:
        """Assert output is valid JSON and return parsed data"""
        try:
            return json.loads(result.stdout)
        except json.JSONDecodeError as err:
            msg = f"Expected JSON output, got:\n{result.stdout}\nSTDERR: {result.stderr}"
            raise AssertionError(msg) from err

    def assert_file_exists(self, filepath: str):
        """Assert file exists"""
        assert os.path.exists(filepath), f"Expected file to exist: {filepath}"

    def assert_file_not_exists(self, filepath: str):
        """Assert file does not exist"""
        assert not os.path.exists(filepath), f"Expected file NOT to exist: {filepath}"


@pytest.fixture
def script_helper():
    """Fixture providing ScriptTestHelper for test scripts"""

    def _helper(script_name: str):
        script_path = Path(os.getcwd()) / script_name
        return ScriptTestHelper(script_path)

    return _helper


@pytest.fixture
def temp_config():
    """Fixture providing temporary config file"""

    def _create_config(config_dict: dict[str, Any]):
        helper = ScriptTestHelper("dummy.py")
        return helper.create_temp_config(config_dict)

    return _create_config


# ============================================================================
# Function-Level Testing Patterns
# ============================================================================


class FunctionTestCase:
    """Base class for function-level testing"""

    def assert_function_signature(self, func: Callable, expected_params: list[str]):
        """
        Assert function has expected parameters

        Args:
            func: Function to test
            expected_params: List of expected parameter names
        """
        import inspect

        sig = inspect.signature(func)
        actual_params = list(sig.parameters.keys())
        assert actual_params == expected_params, f"Expected parameters {expected_params}, got {actual_params}"

    def assert_return_type(self, func: Callable, *args, expected_type: type):
        """
        Assert function returns expected type

        Args:
            func: Function to test
            *args: Arguments to pass to function
            expected_type: Expected return type
        """
        result = func(*args)
        assert isinstance(result, expected_type), f"Expected return type {expected_type}, got {type(result)}"
        return result

    def assert_raises(self, func: Callable, *args, exception: type):
        """
        Assert function raises specific exception

        Args:
            func: Function to test
            *args: Arguments to pass to function
            exception: Expected exception type
        """
        with pytest.raises(exception):
            func(*args)

    def assert_output_shape(self, result: Any, expected_shape: tuple[int, ...]):
        """
        Assert output has expected shape (for numpy/torch arrays)

        Args:
            result: Output array
            expected_shape: Expected shape tuple
        """
        if hasattr(result, "shape"):
            assert result.shape == expected_shape, f"Expected shape {expected_shape}, got {result.shape}"
        else:
            msg = "Result has no shape attribute"
            raise TypeError(msg)


class MockTestCase:
    """Helper for mocking and patching"""

    @staticmethod
    def mock_config_parser():
        """Create mock config parser"""
        mock = MagicMock()
        mock.load = MagicMock(return_value=ConfigFixture.minimal_config())
        mock.validate = MagicMock(return_value=[])
        return mock

    @staticmethod
    def mock_model():
        """Create mock PINN model"""
        mock = MagicMock()
        mock.eval = MagicMock()
        mock.train = MagicMock()
        mock.to = MagicMock(return_value=mock)
        mock.state_dict = MagicMock(return_value={})
        mock.load_state_dict = MagicMock()
        return mock

    @staticmethod
    def mock_data_generator():
        """Create mock data generator"""
        mock = MagicMock()
        mock.generate_train_data = MagicMock(return_value=(None, None))
        mock.generate_test_data = MagicMock(return_value=(None, None))
        mock.get_validation_data = MagicMock(return_value=(None, None))
        return mock


# ============================================================================
# Integration/Output Testing Patterns
# ============================================================================


class IntegrationTestCase:
    """Base class for integration and output testing"""

    def setup_test_environment(self, temp_dir: str):
        """
        Setup test environment with required directories

        Args:
            temp_dir: Temporary directory path
        """
        os.makedirs(os.path.join(temp_dir, "logs"), exist_ok=True)
        os.makedirs(os.path.join(temp_dir, "results"), exist_ok=True)
        os.makedirs(os.path.join(temp_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(temp_dir, "data"), exist_ok=True)

    def assert_directory_structure(self, base_path: str, expected_dirs: list[str]):
        """
        Assert expected directory structure exists

        Args:
            base_path: Base directory path
            expected_dirs: List of expected directory names
        """
        for dir_name in expected_dirs:
            dir_path = os.path.join(base_path, dir_name)
            assert os.path.isdir(dir_path), f"Expected directory {dir_path} does not exist"

    def assert_output_files(self, expected_files: list[str]):
        """
        Assert expected output files exist

        Args:
            expected_files: List of expected file paths
        """
        for filepath in expected_files:
            assert os.path.exists(filepath), f"Expected output file {filepath} does not exist"

    def assert_plot_saved(self, filepath: str):
        """
        Assert plot file was saved correctly

        Args:
            filepath: Path to plot file
        """
        assert os.path.exists(filepath), f"Plot file not found: {filepath}"
        assert os.path.getsize(filepath) > 0, f"Plot file is empty: {filepath}"

    def assert_log_contains(self, log_path: str, expected_text: str):
        """
        Assert log file contains expected text

        Args:
            log_path: Path to log file
            expected_text: Text to search for
        """
        with open(log_path) as f:
            content = f.read()
        assert expected_text in content, f"Expected text '{expected_text}' not found in log {log_path}"


@pytest.fixture
def integration_env(tmp_path):
    """Fixture providing integration test environment"""
    test_case = IntegrationTestCase()
    test_case.setup_test_environment(str(tmp_path))
    return test_case, tmp_path


# ============================================================================
# Test Configuration
# ============================================================================


class ScriptsTestConfig:
    """Configuration for scripts testing"""

    # Test data paths
    TEST_DATA_DIR = "tests/data/scripts"
    TEST_CONFIGS_DIR = "tests/data/configs"

    # Common test parameters
    DEFAULT_BATCH_SIZE = 32
    DEFAULT_EPOCHS = 10  # Small number for testing
    DEFAULT_LEARNING_RATE = 0.001

    # Test timeout (seconds)
    SCRIPT_TIMEOUT = 60

    # Output directories for tests
    TEST_OUTPUT_DIR = "tests/output"
    TEST_LOGS_DIR = "tests/logs"

    @classmethod
    def setup_directories(cls):
        """Setup test directories"""
        for dir_path in [
            cls.TEST_DATA_DIR,
            cls.TEST_CONFIGS_DIR,
            cls.TEST_OUTPUT_DIR,
            cls.TEST_LOGS_DIR,
        ]:
            os.makedirs(dir_path, exist_ok=True)

    @classmethod
    def cleanup_test_outputs(cls):
        """Clean up test outputs"""
        import shutil

        for dir_path in [cls.TEST_OUTPUT_DIR, cls.TEST_LOGS_DIR]:
            if os.path.exists(dir_path):
                shutil.rmtree(dir_path)
                os.makedirs(dir_path, exist_ok=True)


# ============================================================================
# Performance Testing Utilities
# ============================================================================


class PerformanceTestCase:
    """Base class for performance testing"""

    def measure_execution_time(self, func: Callable, *args, **kwargs) -> tuple[float, Any]:
        """
        Measure function execution time

        Args:
            func: Function to measure
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Tuple of (execution_time_seconds, result)
        """
        import time

        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        return end_time - start_time, result

    def assert_execution_time_under(self, func: Callable, max_seconds: float, *args, **kwargs):
        """
        Assert function completes within time limit

        Args:
            func: Function to test
            max_seconds: Maximum allowed seconds
            *args: Positional arguments
            **kwargs: Keyword arguments
        """
        elapsed, _ = self.measure_execution_time(func, *args, **kwargs)
        assert elapsed < max_seconds, f"Function took {elapsed:.2f}s, expected < {max_seconds}s"


# ============================================================================
# Pytest Configuration and Hooks
# ============================================================================


def pytest_configure(config):
    """Pytest configuration hook"""
    config.addinivalue_line("markers", "scripts: mark test as script integration test")
    config.addinivalue_line("markers", "slow: mark test as slow-running")
    config.addinivalue_line("markers", "cli: mark test as CLI argument test")


def pytest_collection_modifyitems(config, items):
    """Modify test collection"""
    for item in items:
        # Add markers based on test class
        if "CLITestCase" in item.parent.__class__.__name__:
            item.add_marker(pytest.mark.cli)
        if "IntegrationTestCase" in item.parent.__class__.__name__:
            item.add_marker(pytest.mark.scripts)


# ============================================================================
# Example Test Cases (Documentation)
# ============================================================================


class ExampleCLITests(CLITestCase):
    """Example CLI test cases showing usage patterns"""

    @pytest.fixture
    def example_helper(self, script_helper):
        return script_helper("scripts/example_script.py")

    def test_basic_cli(self, example_helper):
        """Example: Basic CLI test"""
        result = example_helper.run_script([])
        self.assert_exit_success(result)

    def test_cli_with_args(self, example_helper):
        """Example: CLI with arguments"""
        result = example_helper.run_script(["--config", "test.json", "--verbose"])
        self.assert_exit_success(result)

    def test_cli_error_handling(self, example_helper):
        """Example: CLI error handling"""
        result = example_helper.run_script(["--invalid-arg"])
        self.assert_exit_failure(result)


class ExampleFunctionTests(FunctionTestCase):
    """Example function-level test cases"""

    def test_function_signature(self):
        """Example: Test function signature"""

        def example_func(x: int, y: int) -> int:
            return x + y

        self.assert_function_signature(example_func, ["x", "y"])

    def test_return_type(self):
        """Example: Test return type"""

        def example_func(x: int) -> int:
            return x * 2

        result = self.assert_return_type(example_func, 5, expected_type=int)
        assert result == 10

    def test_exception_raised(self):
        """Example: Test exception handling"""

        def example_func(x: int):
            if x < 0:
                msg = "x must be non-negative"
                raise ValueError(msg)
            return x

        self.assert_raises(example_func, -1, exception=ValueError)


class ExampleIntegrationTests(IntegrationTestCase):
    """Example integration test cases"""

    def test_end_to_end_workflow(self, integration_env):
        """Example: End-to-end workflow test"""
        _test_case, temp_dir = integration_env

        # Setup
        self.setup_test_environment(str(temp_dir))

        # Execute workflow (example)
        # ... perform actions ...

        # Verify
        self.assert_directory_structure(str(temp_dir), ["logs", "results"])


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
