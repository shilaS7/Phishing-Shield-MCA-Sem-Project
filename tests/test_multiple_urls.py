"""
Unit tests for multiple URL processing functionality.
"""
import pytest
from unittest.mock import patch, Mock
from app import app


class TestMultipleURLProcessing:
    """Test cases for multiple URL processing functionality."""
    
    def test_multiple_urls_processing_success(self, client):
        """Test processing multiple URLs successfully."""
        with patch('app.FeatureExtraction') as mock_fe:
            mock_fe.return_value.getFeaturesList.return_value = [1] * 30
            
            with patch('app.gbc') as mock_gbc:
                mock_gbc.predict.return_value = [1]
                mock_gbc.predict_proba.return_value = [[0.1, 0.9]]
                
                urls_text = "https://www.google.com\nhttps://www.github.com\nhttps://www.example.com"
                response = client.post('/result', data={'urls': urls_text})
                
                assert response.status_code == 200
                assert b'multiple_results' in response.data or b'Scan Results' in response.data
    
    def test_multiple_urls_processing_with_errors(self, client):
        """Test processing multiple URLs with some errors."""
        with patch('app.FeatureExtraction') as mock_fe:
            # First URL succeeds, second fails
            def side_effect(url):
                if 'google.com' in url:
                    mock = Mock()
                    mock.getFeaturesList.return_value = [1] * 30
                    return mock
                else:
                    raise Exception("Connection failed")
            
            mock_fe.side_effect = side_effect
            
            with patch('app.gbc') as mock_gbc:
                mock_gbc.predict.return_value = [1]
                mock_gbc.predict_proba.return_value = [[0.1, 0.9]]
                
                urls_text = "https://www.google.com\nhttps://invalid-url.com"
                response = client.post('/result', data={'urls': urls_text})
                
                assert response.status_code == 200
                assert b'Error' in response.data
    
    def test_multiple_urls_empty_input(self, client):
        """Test processing with empty URL input."""
        response = client.post('/result', data={'urls': ''})
        assert response.status_code == 200
        assert b'Please enter at least one URL' in response.data
    
    def test_multiple_urls_whitespace_only(self, client):
        """Test processing with only whitespace."""
        response = client.post('/result', data={'urls': '   \n  \n  '})
        assert response.status_code == 200
        assert b'Please enter at least one URL' in response.data
    
    def test_multiple_urls_limit_exceeded(self, client):
        """Test processing with more than 10 URLs."""
        with patch('app.FeatureExtraction') as mock_fe:
            mock_fe.return_value.getFeaturesList.return_value = [1] * 30
            
            with patch('app.gbc') as mock_gbc:
                mock_gbc.predict.return_value = [1]
                mock_gbc.predict_proba.return_value = [[0.1, 0.9]]
                
                # Create 15 URLs
                urls = [f"https://www.example{i}.com" for i in range(15)]
                urls_text = '\n'.join(urls)
                
                response = client.post('/result', data={'urls': urls_text})
                
                assert response.status_code == 200
                assert b'Limited to first 10 URLs' in response.data
    
    def test_multiple_urls_single_url(self, client):
        """Test processing single URL in multiple URL mode."""
        with patch('app.FeatureExtraction') as mock_fe:
            mock_fe.return_value.getFeaturesList.return_value = [1] * 30
            
            with patch('app.gbc') as mock_gbc:
                mock_gbc.predict.return_value = [1]
                mock_gbc.predict_proba.return_value = [[0.1, 0.9]]
                
                response = client.post('/result', data={'urls': 'https://www.google.com'})
                
                assert response.status_code == 200
                assert b'multiple_results' in response.data or b'Scan Results' in response.data
    
    def test_multiple_urls_mixed_valid_invalid(self, client):
        """Test processing mix of valid and invalid URLs."""
        with patch('app.FeatureExtraction') as mock_fe:
            def side_effect(url):
                if 'google.com' in url:
                    mock = Mock()
                    mock.getFeaturesList.return_value = [1] * 30
                    return mock
                else:
                    raise Exception("Invalid URL")
            
            mock_fe.side_effect = side_effect
            
            with patch('app.gbc') as mock_gbc:
                mock_gbc.predict.return_value = [1]
                mock_gbc.predict_proba.return_value = [[0.1, 0.9]]
                
                urls_text = "https://www.google.com\ninvalid-url\nhttps://www.github.com"
                response = client.post('/result', data={'urls': urls_text})
                
                assert response.status_code == 200
                # Should have both successful and error results
                assert b'Error' in response.data
    
    def test_multiple_urls_duplicate_urls(self, client):
        """Test processing duplicate URLs."""
        with patch('app.FeatureExtraction') as mock_fe:
            mock_fe.return_value.getFeaturesList.return_value = [1] * 30
            
            with patch('app.gbc') as mock_gbc:
                mock_gbc.predict.return_value = [1]
                mock_gbc.predict_proba.return_value = [[0.1, 0.9]]
                
                urls_text = "https://www.google.com\nhttps://www.google.com\nhttps://www.google.com"
                response = client.post('/result', data={'urls': urls_text})
                
                assert response.status_code == 200
                # Should process all URLs even if duplicates
                assert b'multiple_results' in response.data or b'Scan Results' in response.data
    
    def test_multiple_urls_with_newlines_and_spaces(self, client):
        """Test processing URLs with various whitespace."""
        with patch('app.FeatureExtraction') as mock_fe:
            mock_fe.return_value.getFeaturesList.return_value = [1] * 30
            
            with patch('app.gbc') as mock_gbc:
                mock_gbc.predict.return_value = [1]
                mock_gbc.predict_proba.return_value = [[0.1, 0.9]]
                
                urls_text = "  https://www.google.com  \n  \n  https://www.github.com  \n  "
                response = client.post('/result', data={'urls': urls_text})
                
                assert response.status_code == 200
                assert b'multiple_results' in response.data or b'Scan Results' in response.data
    
    def test_multiple_urls_analysis_storage(self, client):
        """Test that analysis results are properly stored."""
        # Clear existing analysis results
        from app import analysis_results
        analysis_results.clear()
        
        with patch('app.FeatureExtraction') as mock_fe:
            mock_fe.return_value.getFeaturesList.return_value = [1] * 30
            
            with patch('app.gbc') as mock_gbc:
                mock_gbc.predict.return_value = [1]
                mock_gbc.predict_proba.return_value = [[0.1, 0.9]]
                
                urls_text = "https://www.google.com\nhttps://www.github.com"
                response = client.post('/result', data={'urls': urls_text})
                
                assert response.status_code == 200
                
                # Check that analysis results are stored
                assert len(analysis_results) == 2
                
                # Check that each result has required fields
                for analysis_id, analysis_data in analysis_results.items():
                    assert 'id' in analysis_data
                    assert 'url' in analysis_data
                    assert 'prediction' in analysis_data
                    assert 'confidence_safe' in analysis_data
                    assert 'confidence_phishing' in analysis_data
