# PhishShield - Technical Documentation

## Table of Contents
1. [Project Overview](#project-overview)
2. [System Architecture](#system-architecture)
3. [Technology Stack](#technology-stack)
4. [Feature Extraction Engine](#feature-extraction-engine)
5. [Machine Learning Model](#machine-learning-model)
6. [API Documentation](#api-documentation)
7. [Frontend Interface](#frontend-interface)
8. [Installation and Setup](#installation-and-setup)
9. [File Structure](#file-structure)
10. [Security Considerations](#security-considerations)
11. [Performance Metrics](#performance-metrics)
12. [Future Enhancements](#future-enhancements)

---

## Project Overview

**PhishShield** is a machine learning-powered web application designed to detect and prevent phishing attacks by analyzing website URLs and their characteristics. The system extracts 30 distinct features from URLs and web content to classify websites as legitimate or phishing attempts.

### Key Features
- Real-time URL analysis and classification
- Comprehensive feature extraction (30 security features)
- Detailed security reports with confidence scores
- Interactive web interface with Bootstrap UI
- RESTful API for programmatic access
- Educational content about phishing attacks

### Project Metrics
- **30 Security Features** analyzed per URL
- **Machine Learning Model**: Gradient Boosting Classifier
- **Web Framework**: Flask (Python)
- **Frontend**: Bootstrap 4 with custom CSS
- **Deployment Ready**: Heroku-compatible with Procfile

---

## System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │   Flask App     │    │   ML Model      │
│   (HTML/CSS/JS) │───▶│   (app.py)      │───▶│   (newmodel.pkl)│
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │ Feature         │
                       │ Extraction      │
                       │ (feature.py)    │
                       └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │ URL Processing  │
                       │ (convert.py)    │
                       └─────────────────┘
```

### Core Components

1. **Web Interface** (`templates/`)
   - User input form for URL submission
   - Results display with confidence scores
   - Detailed analysis reports
   - Educational content about phishing

2. **Flask Application** (`app.py`)
   - Route handling and request processing
   - Model inference coordination
   - Session management and result storage

3. **Feature Extraction Engine** (`feature.py`)
   - 30 distinct security feature analyzers
   - Web scraping and content analysis
   - Domain and network-level checks

4. **URL Processing** (`convert.py`)
   - URL validation and preprocessing
   - Short link detection
   - Result formatting

---

## Technology Stack

### Backend Technologies
```python
# Core Framework
Flask==3.0.3                    # Web framework
gunicorn==20.1.0               # WSGI server for production

# Machine Learning & Data Processing
scikit_learn==1.3.1           # ML algorithms
numpy==1.26.4                 # Numerical computing
pandas==2.2.2                 # Data manipulation

# Web Scraping & Analysis
beautifulsoup4==4.12.3        # HTML/XML parsing
Requests==2.31.0              # HTTP library
urllib3==2.2.1                # URL handling

# External APIs
whois==0.9.27                 # Domain information
googlesearch_python==1.2.3    # Google search integration

# Utilities
python_dateutil==2.8.2       # Date/time parsing
```

### Frontend Technologies
- **HTML5** with semantic markup
- **CSS3** with custom styles and animations
- **Bootstrap 4** for responsive design
- **JavaScript** for client-side interactions
- **Chart.js** for data visualization
- **Font Awesome** for icons

---

## Feature Extraction Engine

The `FeatureExtraction` class implements 30 distinct security features to analyze URLs:

### URL-Based Features (7 features)
1. **UsingIP**: Detects if URL uses IP address instead of domain
2. **LongURL**: Analyzes URL length (suspicious if >75 characters)
3. **ShortURL**: Detects URL shortening services
4. **Symbol@**: Checks for '@' symbol in URL
5. **Redirecting//**: Identifies suspicious redirects
6. **PrefixSuffix**: Detects hyphens in domain names
7. **SubDomains**: Counts number of subdomains

### SSL/Security Features (3 features)
8. **HTTPS**: Verifies SSL certificate usage
9. **HTTPSDomainURL**: Checks for HTTPS in domain
10. **NonStdPort**: Detects non-standard ports

### Domain Analysis Features (4 features)
11. **DomainRegLen**: Domain registration length
12. **AgeofDomain**: Domain age analysis
13. **DNSRecording**: DNS record verification
14. **AbnormalURL**: URL structure analysis

### Content Analysis Features (8 features)
15. **Favicon**: Favicon source analysis
16. **RequestURL**: External resource analysis
17. **AnchorURL**: Anchor tag analysis
18. **LinksInScriptTags**: Script source analysis
19. **ServerFormHandler**: Form action analysis
20. **InfoEmail**: Email information detection
21. **StatusBarCust**: Status bar customization
22. **DisableRightClick**: Right-click disabling

### Advanced Analysis Features (8 features)
23. **UsingPopupWindow**: Popup window detection
24. **IframeRedirection**: Iframe analysis
25. **WebsiteForwarding**: Redirect chain analysis
26. **WebsiteTraffic**: Alexa rank analysis
27. **PageRank**: Google PageRank analysis
28. **GoogleIndex**: Google indexing status
29. **LinksPointingToPage**: Backlink analysis
30. **StatsReport**: Statistical reporting

### Feature Values
- **1**: Safe/Legitimate indicator
- **0**: Neutral/Suspicious indicator
- **-1**: Risky/Phishing indicator

---

## Machine Learning Model

### Model Architecture
- **Algorithm**: Gradient Boosting Classifier (GBC)
- **Features**: 30-dimensional feature vector
- **Output**: Binary classification (0=Phishing, 1=Legitimate)
- **Model File**: `newmodel.pkl` (serialized scikit-learn model)

### Prediction Process
```python
# Feature extraction
obj = FeatureExtraction(url)
features = obj.getFeaturesList()
x = np.array(features).reshape(1, 30)

# Model prediction
y_pred = gbc.predict(x)[0]           # Binary classification
y_proba = gbc.predict_proba(x)[0]    # Confidence scores
```

### Confidence Scoring
- **Safe Confidence**: `y_proba[1] * 100` (probability of being legitimate)
- **Phishing Confidence**: `y_proba[0] * 100` (probability of being phishing)

---

## API Documentation

### Core Endpoints

#### 1. Home Page
```http
GET /
```
Returns the main application interface.

#### 2. URL Analysis
```http
POST /result
Content-Type: application/x-www-form-urlencoded

name=https://example.com
```

**Response Format**:
```python
# Template variables
name = [url, status, action, confidence, analysis_id]
analysis_id = "uuid-string"
```

#### 3. Detailed Analysis Report
```http
GET /detailed_report/<analysis_id>
```
Returns comprehensive analysis with feature breakdown.

#### 4. JSON API
```http
GET /api/analysis/<analysis_id>
```

**Response Example**:
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "url": "https://example.com",
  "prediction": 1,
  "confidence_safe": 92.3,
  "confidence_phishing": 7.7,
  "feature_analysis": [
    {
      "name": "Using IP",
      "value": 1,
      "status": "Safe"
    }
  ],
  "risk_factors": [],
  "safe_factors": ["HTTPS", "Domain Reg Length"],
  "recommendations": [
    "✅ This website appears to be legitimate and safe to visit."
  ],
  "timestamp": "2024-01-15T10:30:00",
  "total_features": 30,
  "risky_features": 2,
  "safe_features": 25
}
```

#### 5. Use Cases Page
```http
GET /usecases
```
Returns educational content about phishing attack scenarios.

---

## Frontend Interface

### Main Components

#### 1. Landing Page (`index.html`)
- **URL Input Form**: Accepts HTTP/HTTPS URLs
- **Results Display**: Shows classification with confidence
- **Progress Bar**: Visual confidence indicator
- **Action Buttons**: Continue/Warning options
- **Educational Content**: FAQ about phishing

#### 2. Detailed Report (`detailed_report.html`)
- **Analysis Overview**: URL, timestamp, overall assessment
- **Confidence Meter**: Circular progress indicator
- **Feature Breakdown**: 30 features with status indicators
- **Risk Factors**: Highlighted security concerns
- **Recommendations**: Actionable security advice
- **Download Options**: JSON export and print functionality

#### 3. Use Cases (`usecases.html`)
- **Real-world Examples**: Historical phishing attacks
- **Case Studies**: BenefitMall, Sony Pictures, etc.
- **Educational Content**: Attack vectors and prevention

### UI/UX Features
- **Responsive Design**: Bootstrap 4 grid system
- **Accessibility**: ARIA labels and semantic HTML
- **Performance**: Optimized assets and lazy loading
- **Animations**: AOS (Animate On Scroll) library
- **Charts**: Chart.js for data visualization

---

## Installation and Setup

### Prerequisites
- Python 3.7+
- pip (Python package manager)

### Local Development Setup

1. **Download Project**
```bash
# Download and extract the project files to your local machine
cd Phishing-Website-Checker
```

2. **Create Virtual Environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install Dependencies**
```bash
pip install -r requirements.txt
```

4. **Run Application**
```bash
python app.py
```

5. **Access Application**
```
http://127.0.0.1:5000/
```

### Production Deployment

#### Heroku Deployment
```bash
# Login to Heroku
heroku login

# Create app
heroku create your-app-name

# Deploy
# Upload your project files to your hosting platform
```

#### Configuration Files
- **Procfile**: `web: gunicorn app:app`
- **requirements.txt**: Production dependencies
- **Runtime**: Python 3.9+ recommended

---

## File Structure

```
Phishing-Website-Checker/
├── app.py                          # Main Flask application
├── feature.py                      # Feature extraction engine
├── convert.py                      # URL processing utilities
├── newmodel.pkl                    # Trained ML model
├── requirements.txt                # Python dependencies
├── Procfile                        # Heroku deployment config
├── README.md                       # Project documentation
├── Phishingproject.ipynb          # Model training notebook
├── DataFiles/                      # Training datasets
│   ├── legitimateurls.csv
│   ├── phishing.csv
│   └── phishurls.csv
├── templates/                      # HTML templates
│   ├── index.html                 # Main interface
│   ├── detailed_report.html       # Analysis report
│   ├── usecases.html             # Educational content
│   └── error.html                # Error handling
├── static/                        # Static assets
│   ├── style.css                 # Custom styles
│   ├── assets/                   # Images and resources
│   │   ├── img/
│   │   ├── js/
│   │   └── vendor/               # Third-party libraries
│   └── assets/vendor/
│       ├── bootstrap/            # Bootstrap framework
│       ├── jquery/               # jQuery library
│       ├── aos/                  # Animation library
│       └── ...                   # Other vendor libraries
└── screenshots/                   # Application screenshots
    ├── screenshot1.png
    └── screenshot2.png
```

---

## Security Considerations

### Input Validation
- **URL Sanitization**: Proper URL parsing and validation
- **XSS Prevention**: Template escaping and CSP headers
- **CSRF Protection**: Flask-WTF token validation (recommended)

### Data Privacy
- **In-Memory Storage**: Analysis results stored temporarily
- **No Persistent Data**: URLs not logged permanently
- **Session Management**: Secure session handling

### API Security
- **Rate Limiting**: Implement request throttling (recommended)
- **Authentication**: Add API key validation for production
- **HTTPS Only**: Force secure connections in production

### Model Security
- **Model Integrity**: Verify pickle file authenticity
- **Feature Validation**: Sanitize extracted features
- **Error Handling**: Graceful failure for invalid inputs

---

## Performance Metrics

### Analysis Performance
- **Feature Extraction**: ~2-5 seconds per URL
- **Model Inference**: <100ms
- **Memory Usage**: ~50MB base application
- **Concurrent Users**: 10-20 (single instance)

### Accuracy Metrics
- **True Positive Rate**: High legitimate detection
- **False Positive Rate**: Low false alarms
- **Feature Coverage**: 30 comprehensive security indicators

### Optimization Opportunities
- **Caching**: Implement Redis for repeated analyses
- **Async Processing**: Use Celery for background tasks
- **Model Updates**: Regular retraining with new data
- **CDN**: Static asset delivery optimization

---

## Future Enhancements

### Technical Improvements
1. **Database Integration**
   - PostgreSQL for analysis history
   - User account management
   - Analytics dashboard

2. **Advanced ML Features**
   - Deep learning models (LSTM/CNN)
   - Ensemble methods
   - Real-time model updates

3. **API Enhancements**
   - GraphQL endpoint
   - Batch processing capabilities
   - Webhook notifications

4. **Browser Integration**
   - Chrome/Firefox extensions
   - Real-time URL checking
   - Automatic warning system

### Security Enhancements
1. **Enhanced Detection**
   - Behavioral analysis
   - Computer vision for visual phishing
   - Natural language processing

2. **Threat Intelligence**
   - Integration with threat feeds
   - Collaborative filtering
   - Zero-day detection

### User Experience
1. **Mobile Application**
   - React Native app
   - Offline capability
   - Push notifications

2. **Educational Platform**
   - Interactive training modules
   - Phishing simulation
   - Certification programs

---

## Conclusion

PhishShield represents a comprehensive solution for phishing detection, combining machine learning, web development, and cybersecurity best practices. The modular architecture allows for easy extension and maintenance, while the educational components help users understand and prevent phishing attacks.

For questions, contributions, or support, please refer to the project repository or contact the development team.

---

**Document Version**: 1.0  
**Last Updated**: September 2025
**Authors**: Shila Shrestha 
**License**: [Specify License]