"""
Pytest configuration and fixtures for Phishing Website Checker tests.
"""
import pytest
import sys
import os
from unittest.mock import Mock, patch

# Add the parent directory to the path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app import app
from feature import FeatureExtraction
from convert import convertion, shortlink

@pytest.fixture
def client():
    """Create a test client for the Flask application."""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

@pytest.fixture
def sample_urls():
    """Sample URLs for testing."""
    return {
        'legitimate': 'https://www.google.com',
        'phishing': 'http://fake-bank-login.com',
        'short_url': 'https://bit.ly/example',
        'ip_url': 'http://192.168.1.1',
        'long_url': 'https://www.example.com/very/long/path/with/many/segments/and/parameters?param1=value1&param2=value2&param3=value3',
        'invalid_url': 'not-a-valid-url'
    }

@pytest.fixture
def mock_whois_response():
    """Mock whois response for testing."""
    mock_response = Mock()
    mock_response.creation_date = [Mock(year=2020, month=1)]
    mock_response.expiration_date = [Mock(year=2025, month=1)]
    return mock_response

@pytest.fixture
def mock_requests_response():
    """Mock requests response for testing."""
    mock_response = Mock()
    mock_response.text = '<html><head><title>Test</title></head><body>Test content</body></html>'
    mock_response.history = []
    return mock_response
