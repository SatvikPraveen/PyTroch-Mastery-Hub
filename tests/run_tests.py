# tests/run_tests.py
"""
Test runner script for PyTorch Mastery Hub
"""

import pytest
import sys
import os
from pathlib import Path

# Add src to Python path
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))


def run_all_tests():
    """Run all tests."""
    return pytest.main([
        "tests/",
        "-v",
        "--tb=short",
        "--color=yes"
    ])


def run_unit_tests():
    """Run only unit tests (fast tests)."""
    return pytest.main([
        "tests/",
        "-v",
        "-m", "not slow and not gpu",
        "--tb=short",
        "--color=yes"
    ])


def run_integration_tests():
    """Run integration tests."""
    return pytest.main([
        "tests/test_integration.py",
        "-v",
        "--tb=short",
        "--color=yes"
    ])


def run_specific_module(module_name):
    """Run tests for a specific module."""
    test_path = f"tests/test_{module_name}/"
    if not os.path.exists(test_path):
        test_path = f"tests/test_{module_name}.py"
    
    return pytest.main([
        test_path,
        "-v",
        "--tb=short",
        "--color=yes"
    ])


def run_coverage():
    """Run tests with coverage report."""
    return pytest.main([
        "tests/",
        "--cov=src",
        "--cov-report=html",
        "--cov-report=term",
        "-v"
    ])


if __name__ == "__main__":
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "unit":
            exit_code = run_unit_tests()
        elif command == "integration":
            exit_code = run_integration_tests()
        elif command == "coverage":
            exit_code = run_coverage()
        elif command.startswith("module:"):
            module = command.split(":")[1]
            exit_code = run_specific_module(module)
        else:
            print(f"Unknown command: {command}")
            print("Available commands: unit, integration, coverage, module:<name>")
            sys.exit(1)
    else:
        exit_code = run_all_tests()
    
    sys.exit(exit_code)