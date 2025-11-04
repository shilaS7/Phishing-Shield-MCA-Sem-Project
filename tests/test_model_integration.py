"""
Unit tests for model integration and prediction functionality.
"""
import pytest
import numpy as np
from unittest.mock import patch, Mock, mock_open
import pickle


class TestModelIntegration:
    """Test cases for model loading and prediction functionality."""
    
    def test_model_loading_success(self):
        """Test successful model loading."""
        # Mock the pickle file and model
        mock_model = Mock()
        mock_model.predict.return_value = [1]
        mock_model.predict_proba.return_value = [[0.1, 0.9]]
        
        with patch("builtins.open", mock_open()) as mock_file:
            with patch("pickle.load", return_value=mock_model):
                # Import and test model loading
                from app import gbc
                assert gbc is not None
    
    def test_model_prediction_safe(self):
        """Test model prediction for safe URL."""
        # Mock feature extraction
        with patch('app.FeatureExtraction') as mock_fe:
            mock_fe.return_value.getFeaturesList.return_value = [1] * 30
            
            # Mock model
            with patch('app.gbc') as mock_gbc:
                mock_gbc.predict.return_value = [1]  # Safe
                mock_gbc.predict_proba.return_value = [[0.1, 0.9]]  # 90% safe
                
                from app import app
                with app.test_client() as client:
                    response = client.post('/result', data={'name': 'https://www.google.com'})
                    assert response.status_code == 200
                    assert b'Safe' in response.data or b'Continue' in response.data
    
    def test_model_prediction_unsafe(self):
        """Test model prediction for unsafe URL."""
        # Mock feature extraction
        with patch('app.FeatureExtraction') as mock_fe:
            mock_fe.return_value.getFeaturesList.return_value = [-1] * 30
            
            # Mock model
            with patch('app.gbc') as mock_gbc:
                mock_gbc.predict.return_value = [-1]  # Unsafe
                mock_gbc.predict_proba.return_value = [[0.9, 0.1]]  # 90% unsafe
                
                from app import app
                with app.test_client() as client:
                    response = client.post('/result', data={'name': 'https://fake-bank.com'})
                    assert response.status_code == 200
                    assert b'Not Safe' in response.data or b'Still want to Continue' in response.data
    
    def test_model_prediction_confidence_scores(self):
        """Test model prediction with confidence scores."""
        # Mock feature extraction
        with patch('app.FeatureExtraction') as mock_fe:
            mock_fe.return_value.getFeaturesList.return_value = [1] * 30
            
            # Mock model with specific confidence scores
            with patch('app.gbc') as mock_gbc:
                mock_gbc.predict.return_value = [1]  # Safe
                mock_gbc.predict_proba.return_value = [[0.2, 0.8]]  # 80% safe, 20% unsafe
                
                from app import app
                with app.test_client() as client:
                    response = client.post('/result', data={'name': 'https://www.google.com'})
                    assert response.status_code == 200
                    # Check that confidence scores are included in response
                    assert b'80.0' in response.data or b'80' in response.data
    
    def test_model_prediction_edge_case_single_class(self):
        """Test model prediction when predict_proba returns single class."""
        # Mock feature extraction
        with patch('app.FeatureExtraction') as mock_fe:
            mock_fe.return_value.getFeaturesList.return_value = [1] * 30
            
            # Mock model with single class probability
            with patch('app.gbc') as mock_gbc:
                mock_gbc.predict.return_value = [1]  # Safe
                mock_gbc.predict_proba.return_value = [[0.9]]  # Single class
                
                from app import app
                with app.test_client() as client:
                    response = client.post('/result', data={'name': 'https://www.google.com'})
                    assert response.status_code == 200
                    # Should handle single class gracefully
                    assert b'Safe' in response.data or b'Continue' in response.data
    
    def test_model_prediction_empty_features(self):
        """Test model prediction with empty features."""
        # Mock feature extraction returning empty list
        with patch('app.FeatureExtraction') as mock_fe:
            mock_fe.return_value.getFeaturesList.return_value = []
            
            # Mock model
            with patch('app.gbc') as mock_gbc:
                mock_gbc.predict.return_value = [1]
                mock_gbc.predict_proba.return_value = [[0.1, 0.9]]
                
                from app import app
                with app.test_client() as client:
                    # Should raise ValueError for empty features
                    with pytest.raises(ValueError):
                        client.post('/result', data={'name': 'https://www.google.com'})
    
    def test_model_prediction_invalid_features(self):
        """Test model prediction with invalid features."""
        # Mock feature extraction returning invalid features
        with patch('app.FeatureExtraction') as mock_fe:
            mock_fe.return_value.getFeaturesList.return_value = [None] * 30
            
            # Mock model
            with patch('app.gbc') as mock_gbc:
                mock_gbc.predict.return_value = [1]
                mock_gbc.predict_proba.return_value = [[0.1, 0.9]]
                
                from app import app
                with app.test_client() as client:
                    response = client.post('/result', data={'name': 'https://www.google.com'})
                    # Should handle invalid features gracefully
                    assert response.status_code == 200
    
    def test_model_prediction_exception_handling(self):
        """Test model prediction with exception handling."""
        # Mock feature extraction
        with patch('app.FeatureExtraction') as mock_fe:
            mock_fe.return_value.getFeaturesList.return_value = [1] * 30
            
            # Mock model that raises exception
            with patch('app.gbc') as mock_gbc:
                mock_gbc.predict.side_effect = Exception("Model prediction failed")
                
                from app import app
                with app.test_client() as client:
                    # Should raise exception when model fails
                    with pytest.raises(Exception):
                        client.post('/result', data={'name': 'https://www.google.com'})
    
    def test_feature_extraction_integration(self):
        """Test feature extraction integration with model."""
        # Test with real FeatureExtraction class but mocked external dependencies
        with patch('feature.requests.get') as mock_get:
            mock_response = Mock()
            mock_response.text = '<html><head><title>Test</title></head><body>Test content</body></html>'
            mock_response.history = []
            mock_get.return_value = mock_response
            
            with patch('feature.whois') as mock_whois:
                mock_whois.return_value = Mock()
                
                with patch('feature.urlparse') as mock_parse:
                    mock_parse.return_value.netloc = "www.google.com"
                    mock_parse.return_value.scheme = "https"
                    
                    from feature import FeatureExtraction
                    fe = FeatureExtraction("https://www.google.com")
                    features = fe.getFeaturesList()
                    
                    assert len(features) == 30
                    assert all(isinstance(f, int) for f in features)
                    assert all(f in [-1, 0, 1] for f in features)
    
    def test_model_prediction_with_different_url_types(self):
        """Test model prediction with different types of URLs."""
        test_urls = [
            "https://www.google.com",
            "http://example.com",
            "https://subdomain.example.com/path",
            "https://www.example.com/path?param=value",
            "https://www.example.com/path#fragment"
        ]
        
        for url in test_urls:
            with patch('app.FeatureExtraction') as mock_fe:
                mock_fe.return_value.getFeaturesList.return_value = [1] * 30
                
                with patch('app.gbc') as mock_gbc:
                    mock_gbc.predict.return_value = [1]
                    mock_gbc.predict_proba.return_value = [[0.1, 0.9]]
                    
                    from app import app
                    with app.test_client() as client:
                        response = client.post('/result', data={'name': url})
                        assert response.status_code == 200
    
    def test_model_prediction_confidence_calculation(self):
        """Test confidence score calculation logic."""
        # Test the confidence calculation logic directly
        from app import app
        
        # Mock feature extraction
        with patch('app.FeatureExtraction') as mock_fe:
            mock_fe.return_value.getFeaturesList.return_value = [1] * 30
            
            # Test with different confidence scenarios
            test_cases = [
                ([0.1, 0.9], 90.0, 10.0),  # 90% safe, 10% unsafe
                ([0.5, 0.5], 50.0, 50.0),  # 50% safe, 50% unsafe
                ([0.9, 0.1], 10.0, 90.0),  # 10% safe, 90% unsafe
            ]
            
            for proba, expected_safe, expected_unsafe in test_cases:
                with patch('app.gbc') as mock_gbc:
                    mock_gbc.predict.return_value = [1]
                    mock_gbc.predict_proba.return_value = [proba]
                    
                    with app.test_client() as client:
                        response = client.post('/result', data={'name': 'https://www.google.com'})
                        assert response.status_code == 200
                        # Check that confidence scores are in the response
                        response_text = response.data.decode('utf-8')
                        assert str(expected_safe) in response_text or str(int(expected_safe)) in response_text
