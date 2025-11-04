#!/usr/bin/env python3
"""
Test runner script for Phishing Website Checker project.
"""
import sys
import subprocess
import os
from pathlib import Path

def run_tests():
    """Run all tests with pytest."""
    print("🧪 Running Phishing Website Checker Tests")
    print("=" * 50)
    
    # Change to project directory
    project_dir = Path(__file__).parent
    os.chdir(project_dir)
    
    # Run pytest with coverage
    cmd = [
        sys.executable, "-m", "pytest",
        "tests/",
        "-v",
        "--tb=short",
        "--cov=.",
        "--cov-report=html",
        "--cov-report=term-missing"
    ]
    
    try:
        result = subprocess.run(cmd, check=True)
        print("\n✅ All tests passed!")
        print("\n📊 Coverage report generated in htmlcov/index.html")
        return 0
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Tests failed with exit code {e.returncode}")
        return e.returncode

def run_specific_test(test_file):
    """Run a specific test file."""
    print(f"🧪 Running {test_file}")
    print("=" * 50)
    
    cmd = [sys.executable, "-m", "pytest", f"tests/{test_file}", "-v"]
    
    try:
        result = subprocess.run(cmd, check=True)
        print(f"\n✅ {test_file} passed!")
        return 0
    except subprocess.CalledProcessError as e:
        print(f"\n❌ {test_file} failed with exit code {e.returncode}")
        return e.returncode

def run_unit_tests_only():
    """Run only unit tests."""
    print("🧪 Running Unit Tests Only")
    print("=" * 50)
    
    cmd = [
        sys.executable, "-m", "pytest",
        "tests/test_feature_extraction.py",
        "tests/test_url_processing.py",
        "-v"
    ]
    
    try:
        result = subprocess.run(cmd, check=True)
        print("\n✅ Unit tests passed!")
        return 0
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Unit tests failed with exit code {e.returncode}")
        return e.returncode

def run_integration_tests_only():
    """Run only integration tests."""
    print("🧪 Running Integration Tests Only")
    print("=" * 50)
    
    cmd = [
        sys.executable, "-m", "pytest",
        "tests/test_flask_routes.py",
        "tests/test_model_integration.py",
        "-v"
    ]
    
    try:
        result = subprocess.run(cmd, check=True)
        print("\n✅ Integration tests passed!")
        return 0
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Integration tests failed with exit code {e.returncode}")
        return e.returncode

def main():
    """Main function to handle command line arguments."""
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "unit":
            return run_unit_tests_only()
        elif command == "integration":
            return run_integration_tests_only()
        elif command.endswith(".py"):
            return run_specific_test(command)
        else:
            print(f"Unknown command: {command}")
            print("Usage: python run_tests.py [unit|integration|test_file.py]")
            return 1
    else:
        return run_tests()

if __name__ == "__main__":
    sys.exit(main())
