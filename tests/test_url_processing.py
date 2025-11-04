"""
Unit tests for URL processing functionality.
"""
import pytest
from convert import convertion, shortlink, find_url_in_csv


class TestURLProcessing:
    """Test cases for URL processing functions."""
    
    def test_shortlink_legitimate_url(self):
        """Test shortlink function with legitimate URL."""
        url = "https://www.google.com"
        result = shortlink(url)
        assert result == 1  # Should return 1 for legitimate URL
    
    def test_shortlink_bitly(self):
        """Test shortlink function with bit.ly URL."""
        url = "https://bit.ly/example"
        result = shortlink(url)
        assert result == -1  # Should return -1 for shortened URL
    
    def test_shortlink_tinyurl(self):
        """Test shortlink function with tinyurl."""
        url = "https://tinyurl.com/example"
        result = shortlink(url)
        assert result == -1  # Should return -1 for shortened URL
    
    def test_shortlink_googl(self):
        """Test shortlink function with goo.gl URL."""
        url = "https://goo.gl/example"
        result = shortlink(url)
        assert result == -1  # Should return -1 for shortened URL
    
    def test_shortlink_tco(self):
        """Test shortlink function with t.co URL."""
        url = "https://t.co/example"
        result = shortlink(url)
        assert result == -1  # Should return -1 for shortened URL
    
    def test_convertion_safe_url(self):
        """Test convertion function with safe URL."""
        url = "https://www.google.com"
        prediction = 1  # Safe
        result = convertion(url, prediction)
        expected = [url, "Safe", "Continue", "1"]
        assert result == expected
    
    def test_convertion_unsafe_url(self):
        """Test convertion function with unsafe URL."""
        url = "https://www.google.com"
        prediction = -1  # Unsafe
        result = convertion(url, prediction)
        expected = [url, "Not Safe", "Still want to Continue"]
        assert result == expected
    
    def test_convertion_short_url(self):
        """Test convertion function with short URL."""
        url = "https://bit.ly/example"
        prediction = 1  # Safe (but short URL should override)
        result = convertion(url, prediction)
        expected = [url, "Not Safe", "Still want to Continue"]
        assert result == expected
    
    def test_find_url_in_csv_existing_url(self, tmp_path):
        """Test find_url_in_csv with existing URL."""
        # Create a temporary CSV file
        csv_file = tmp_path / "test_urls.csv"
        csv_file.write_text("https://www.google.com\nhttps://www.example.com\n")
        
        result = find_url_in_csv(str(csv_file), "https://www.google.com")
        assert result == "https://www.google.com"
    
    def test_find_url_in_csv_nonexistent_url(self, tmp_path):
        """Test find_url_in_csv with non-existent URL."""
        # Create a temporary CSV file
        csv_file = tmp_path / "test_urls.csv"
        csv_file.write_text("https://www.google.com\nhttps://www.example.com\n")
        
        result = find_url_in_csv(str(csv_file), "https://www.nonexistent.com")
        assert result is None
    
    def test_find_url_in_csv_empty_file(self, tmp_path):
        """Test find_url_in_csv with empty file."""
        # Create an empty CSV file
        csv_file = tmp_path / "empty.csv"
        csv_file.write_text("")
        
        result = find_url_in_csv(str(csv_file), "https://www.google.com")
        assert result is None
    
    def test_find_url_in_csv_nonexistent_file(self):
        """Test find_url_in_csv with non-existent file."""
        with pytest.raises(FileNotFoundError):
            find_url_in_csv("nonexistent.csv", "https://www.google.com")
    
    def test_shortlink_various_shorteners(self):
        """Test shortlink function with various URL shorteners."""
        short_urls = [
            "https://bit.ly/example",
            "https://goo.gl/example", 
            "https://tinyurl.com/example",
            "https://t.co/example",
            "https://ow.ly/example",
            "https://is.gd/example",
            "https://short.to/example",
            "https://j.mp/example"
        ]
        
        for url in short_urls:
            result = shortlink(url)
            assert result == -1, f"Failed for URL: {url}"
    
    def test_shortlink_legitimate_variations(self):
        """Test shortlink function with legitimate URL variations."""
        legitimate_urls = [
            "https://www.google.com",
            "http://example.com",
            "https://subdomain.example.com",
            "https://www.example.com/path/to/page",
            "https://www.example.com/path?param=value",
            "https://www.example.com/path#fragment"
        ]
        
        for url in legitimate_urls:
            result = shortlink(url)
            assert result == 1, f"Failed for URL: {url}"
    
    def test_convertion_edge_cases(self):
        """Test convertion function with edge cases."""
        # Test with prediction = 0 (neutral)
        url = "https://www.google.com"
        prediction = 0
        result = convertion(url, prediction)
        expected = [url, "Not Safe", "Still want to Continue"]
        assert result == expected
        
        # Test with empty URL
        url = ""
        prediction = 1
        result = convertion(url, prediction)
        expected = [url, "Safe", "Continue", "1"]
        assert result == expected
