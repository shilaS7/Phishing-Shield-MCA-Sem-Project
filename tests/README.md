# Phishing Website Checker - Test Suite

This directory contains comprehensive unit tests for the Phishing Website Checker project.

## Test Structure

```
tests/
├── __init__.py                 # Test package initialization
├── conftest.py                 # Pytest configuration and fixtures
├── test_feature_extraction.py  # Tests for feature extraction functionality
├── test_url_processing.py      # Tests for URL processing functions
├── test_flask_routes.py        # Tests for Flask routes and API endpoints
├── test_model_integration.py   # Tests for model loading and prediction
└── README.md                   # This file
```

## Test Categories

### 1. Feature Extraction Tests (`test_feature_extraction.py`)
- Tests for all 30 feature extraction methods
- URL analysis functions (IP detection, length, short URLs, etc.)
- SSL/Security feature detection
- Domain analysis features
- Content analysis features
- Error handling and edge cases

### 2. URL Processing Tests (`test_url_processing.py`)
- URL validation and conversion functions
- Short link detection
- CSV file operations
- Edge cases and error handling

### 3. Flask Routes Tests (`test_flask_routes.py`)
- Home page and result routes
- API endpoints (`/api/analysis/`, `/detailed_report/`)
- Form submission handling
- Error handling and edge cases
- Recommendation generation

### 4. Model Integration Tests (`test_model_integration.py`)
- Model loading and initialization
- Prediction functionality
- Confidence score calculation
- Feature extraction integration
- Error handling and edge cases

## Running Tests

### Prerequisites
Install test dependencies:
```bash
pip install -r requirements.txt
```

### Run All Tests
```bash
# Using pytest directly
pytest

# Using the test runner script
python run_tests.py
```

### Run Specific Test Categories
```bash
# Unit tests only
python run_tests.py unit

# Integration tests only
python run_tests.py integration

# Specific test file
python run_tests.py test_feature_extraction.py
```

### Run with Coverage
```bash
pytest --cov=. --cov-report=html
```

## Test Configuration

The tests are configured using:
- `pytest.ini` - Pytest configuration
- `conftest.py` - Shared fixtures and test setup
- `run_tests.py` - Custom test runner script

## Fixtures

The test suite includes several useful fixtures:

- `client` - Flask test client
- `sample_urls` - Collection of test URLs (legitimate, phishing, short, etc.)
- `mock_whois_response` - Mock whois data
- `mock_requests_response` - Mock HTTP responses

## Mocking Strategy

Tests use extensive mocking to:
- Avoid external API calls (whois, requests)
- Isolate units under test
- Ensure consistent test results
- Speed up test execution

## Coverage

The test suite aims for comprehensive coverage of:
- All public methods and functions
- Error handling paths
- Edge cases and boundary conditions
- Integration points between components

## Adding New Tests

When adding new tests:

1. Follow the existing naming convention (`test_*.py`)
2. Use descriptive test method names
3. Include both positive and negative test cases
4. Mock external dependencies
5. Add appropriate assertions
6. Update this README if adding new test categories

## Test Data

Test data is generated programmatically to avoid:
- Large test data files
- Sensitive information in tests
- Maintenance overhead

## Continuous Integration

These tests are designed to run in CI/CD environments:
- No external dependencies
- Deterministic results
- Fast execution
- Clear pass/fail indicators
