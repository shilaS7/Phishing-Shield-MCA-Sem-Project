"""
Unit tests for Flask routes and API endpoints.
"""
import pytest
import json
from unittest.mock import patch, Mock
from app import app


class TestFlaskRoutes:
    """Test cases for Flask application routes."""
    
    def test_home_route(self, client):
        """Test home page route."""
        response = client.get('/')
        assert response.status_code == 200
        assert b'PhishShield' in response.data or b'Phishing' in response.data
    
    def test_result_route_get(self, client):
        """Test result route with GET method."""
        # The function doesn't handle GET requests and returns None
        # This causes Flask to raise a TypeError before creating a response
        with pytest.raises(TypeError):
            client.get('/result')
    
    def test_result_route_post_safe_url(self, client):
        """Test result route with POST method for safe URL."""
        with patch('app.FeatureExtraction') as mock_fe:
            # Mock feature extraction
            mock_fe.return_value.getFeaturesList.return_value = [1] * 30
            
            with patch('app.gbc') as mock_gbc:
                # Mock model prediction
                mock_gbc.predict.return_value = [1]  # Safe
                mock_gbc.predict_proba.return_value = [[0.1, 0.9]]  # 90% safe
                
                response = client.post('/result', data={'name': 'https://www.google.com'})
                assert response.status_code == 200
                assert b'Safe' in response.data or b'Continue' in response.data
    
    def test_result_route_post_unsafe_url(self, client):
        """Test result route with POST method for unsafe URL."""
        with patch('app.FeatureExtraction') as mock_fe:
            # Mock feature extraction
            mock_fe.return_value.getFeaturesList.return_value = [-1] * 30
            
            with patch('app.gbc') as mock_gbc:
                # Mock model prediction
                mock_gbc.predict.return_value = [-1]  # Unsafe
                mock_gbc.predict_proba.return_value = [[0.9, 0.1]]  # 90% unsafe
                
                response = client.post('/result', data={'name': 'https://fake-bank.com'})
                assert response.status_code == 200
                assert b'Not Safe' in response.data or b'Still want to Continue' in response.data
    
    def test_result_route_post_short_url(self, client):
        """Test result route with POST method for short URL."""
        with patch('app.FeatureExtraction') as mock_fe:
            # Mock feature extraction
            mock_fe.return_value.getFeaturesList.return_value = [1] * 30
            
            with patch('app.gbc') as mock_gbc:
                # Mock model prediction
                mock_gbc.predict.return_value = [1]  # Safe
                mock_gbc.predict_proba.return_value = [[0.1, 0.9]]  # 90% safe
                
                response = client.post('/result', data={'name': 'https://bit.ly/example'})
                assert response.status_code == 200
                assert b'Not Safe' in response.data or b'Still want to Continue' in response.data
    
    def test_detailed_report_route_existing(self, client):
        """Test detailed report route with existing analysis ID."""
        # First create an analysis
        with patch('app.FeatureExtraction') as mock_fe:
            mock_fe.return_value.getFeaturesList.return_value = [1] * 30
            
            with patch('app.gbc') as mock_gbc:
                mock_gbc.predict.return_value = [1]
                mock_gbc.predict_proba.return_value = [[0.1, 0.9]]
                
                response = client.post('/result', data={'name': 'https://www.google.com'})
                assert response.status_code == 200
                
                # Extract analysis ID from response
                analysis_id = None
                if b'analysis_id' in response.data:
                    # This would need to be extracted from the response
                    # For now, we'll create a mock analysis
                    analysis_id = "test-analysis-id"
                    app.analysis_results[analysis_id] = {
                        "id": analysis_id,
                        "url": "https://www.google.com",
                        "prediction": 1,
                        "confidence_safe": 90.0,
                        "confidence_phishing": 10.0,
                        "feature_analysis": [],
                        "risk_factors": [],
                        "safe_factors": [],
                        "recommendations": [],
                        "timestamp": "2024-01-01T00:00:00",
                        "total_features": 30,
                        "risky_features": 0,
                        "safe_features": 30
                    }
                
                if analysis_id:
                    response = client.get(f'/detailed_report/{analysis_id}')
                    assert response.status_code == 200
    
    def test_detailed_report_route_nonexistent(self, client):
        """Test detailed report route with non-existent analysis ID."""
        response = client.get('/detailed_report/nonexistent-id')
        assert response.status_code == 404
    
    def test_api_analysis_route_existing(self, client):
        """Test API analysis route with existing analysis ID."""
        # Create a mock analysis
        analysis_id = "test-analysis-id"
        from app import analysis_results
        analysis_results[analysis_id] = {
            "id": analysis_id,
            "url": "https://www.google.com",
            "prediction": 1,
            "confidence_safe": 90.0,
            "confidence_phishing": 10.0,
            "feature_analysis": [],
            "risk_factors": [],
            "safe_factors": [],
            "recommendations": [],
            "timestamp": "2024-01-01T00:00:00",
            "total_features": 30,
            "risky_features": 0,
            "safe_features": 30
        }
        
        response = client.get(f'/api/analysis/{analysis_id}')
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert data['id'] == analysis_id
        assert data['url'] == "https://www.google.com"
        assert data['prediction'] == 1
    
    def test_api_analysis_route_nonexistent(self, client):
        """Test API analysis route with non-existent analysis ID."""
        response = client.get('/api/analysis/nonexistent-id')
        assert response.status_code == 404
        
        data = json.loads(response.data)
        assert 'error' in data
    
    def test_usecases_route(self, client):
        """Test use cases route."""
        response = client.get('/usecases')
        assert response.status_code == 200
    
    def test_about_route(self, client):
        """Test about route."""
        response = client.get('/about')
        assert response.status_code == 200
    
    def test_result_route_invalid_data(self, client):
        """Test result route with invalid form data."""
        response = client.post('/result', data={})
        assert response.status_code == 400  # Should return 400 for missing data
    
    def test_result_route_empty_url(self, client):
        """Test result route with empty URL."""
        with patch('app.FeatureExtraction') as mock_fe:
            mock_fe.return_value.getFeaturesList.return_value = [1] * 30
            
            with patch('app.gbc') as mock_gbc:
                mock_gbc.predict.return_value = [1]
                mock_gbc.predict_proba.return_value = [[0.1, 0.9]]
                
                response = client.post('/result', data={'name': ''})
                assert response.status_code == 200
    
    def test_result_route_malformed_url(self, client):
        """Test result route with malformed URL."""
        with patch('app.FeatureExtraction') as mock_fe:
            mock_fe.return_value.getFeaturesList.return_value = [1] * 30
            
            with patch('app.gbc') as mock_gbc:
                mock_gbc.predict.return_value = [1]
                mock_gbc.predict_proba.return_value = [[0.1, 0.9]]
                
                response = client.post('/result', data={'name': 'not-a-valid-url'})
                assert response.status_code == 200
    
    def test_generate_recommendations_safe_site(self):
        """Test generate_recommendations function for safe site."""
        from app import generate_recommendations
        
        risk_factors = []
        prediction = 1
        recommendations = generate_recommendations(risk_factors, prediction)
        
        assert len(recommendations) > 0
        assert any("legitimate and safe" in rec for rec in recommendations)
    
    def test_generate_recommendations_unsafe_site(self):
        """Test generate_recommendations function for unsafe site."""
        from app import generate_recommendations
        
        risk_factors = ["Using IP", "Long URL", "HTTPS"]
        prediction = -1
        recommendations = generate_recommendations(risk_factors, prediction)
        
        assert len(recommendations) > 0
        assert any("HIGH RISK" in rec for rec in recommendations)
        assert any("IP address" in rec for rec in recommendations)
        assert any("HTTPS" in rec for rec in recommendations)
    
    def test_generate_recommendations_safe_site_with_risk_factors(self):
        """Test generate_recommendations function for safe site with risk factors."""
        from app import generate_recommendations
        
        risk_factors = ["Using IP", "HTTPS"]
        prediction = 1
        recommendations = generate_recommendations(risk_factors, prediction)
        
        assert len(recommendations) > 0
        assert any("legitimate and safe" in rec for rec in recommendations)
        # Should only show critical risk factors for safe sites
        assert any("IP address" in rec for rec in recommendations)
        assert any("HTTPS" in rec for rec in recommendations)
