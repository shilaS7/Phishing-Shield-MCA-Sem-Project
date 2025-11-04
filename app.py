#importing required libraries

from flask import Flask, request, render_template, jsonify, session
import numpy as np
import pandas as pd
from sklearn import metrics
import warnings
import pickle
from convert import convertion
import uuid
import datetime
import json
warnings.filterwarnings('ignore')
from feature import FeatureExtraction

file = open("newmodel.pkl","rb")
gbc = pickle.load(file)
file.close()

# Feature names for detailed analysis
FEATURE_NAMES = [
    "Using IP", "Long URL", "Short URL", "Symbol@", "Redirecting//", 
    "Prefix Suffix", "Sub Domains", "HTTPS", "Domain Reg Length", "Favicon",
    "Non-Std Port", "HTTPS Domain URL", "Request URL", "Anchor URL", 
    "Links in Script Tags", "Server Form Handler", "Info Email", "Abnormal URL",
    "Website Forwarding", "Status Bar Cust", "Disable Right Click", "Using Popup Window",
    "Iframe Redirection", "Age of Domain", "DNS Recording", "Website Traffic",
    "Page Rank", "Google Index", "Links Pointing to Page", "Stats Report"
]

# In-memory storage for analysis results (use database in production)
analysis_results = {}

app = Flask(__name__)
app.secret_key = 'your-secret-key-change-in-production'  # Change this in production
@app.route("/")
def home():
    return render_template("index.html")
@app.route('/result',methods=['POST','GET'])
def predict():
    if request.method == "POST":
        # Check if it's single URL or multiple URLs
        if 'urls' in request.form:
            # Multiple URLs processing
            urls_text = request.form["urls"]
            urls = [url.strip() for url in urls_text.split('\n') if url.strip()]
            
            if not urls:
                return render_template("index.html", error="Please enter at least one URL")
            
            # Limit to 10 URLs for performance
            if len(urls) > 10:
                urls = urls[:10]
                warning = f"Limited to first 10 URLs. {len(urls)} URLs will be processed."
            else:
                warning = None
            
            results = []
            for i, url in enumerate(urls):
                try:
                    # Extract features
                    obj = FeatureExtraction(url)
                    features = obj.getFeaturesList()
                    x = np.array(features).reshape(1,30)
                    
                    # Get prediction and probabilities
                    y_pred = gbc.predict(x)[0]
                    y_proba = gbc.predict_proba(x)[0]
                    
                    # Calculate confidence scores
                    if len(y_proba) >= 2:
                        confidence_safe = float(y_proba[1] * 100)
                        confidence_phishing = float(y_proba[0] * 100)
                    else:
                        confidence_safe = 80.0 if y_pred == 1 else 20.0
                        confidence_phishing = 20.0 if y_pred == 1 else 80.0
                    
                    # Generate unique analysis ID
                    analysis_id = str(uuid.uuid4())
                    
                    # Create result data
                    result = {
                        "url": url,
                        "prediction": int(y_pred),
                        "confidence_safe": confidence_safe,
                        "confidence_phishing": confidence_phishing,
                        "analysis_id": analysis_id,
                        "status": "Safe" if y_pred == 1 else "Not Safe"
                    }
                    
                    # Store detailed analysis
                    analysis_data = {
                        "id": analysis_id,
                        "url": url,
                        "prediction": int(y_pred),
                        "confidence_safe": confidence_safe,
                        "confidence_phishing": confidence_phishing,
                        "feature_analysis": [],
                        "risk_factors": [],
                        "safe_factors": [],
                        "recommendations": [],
                        "timestamp": datetime.datetime.now().isoformat(),
                        "total_features": len(features),
                        "risky_features": 0,
                        "safe_features": 0
                    }
                    
                    # Store in memory
                    analysis_results[analysis_id] = analysis_data
                    results.append(result)
                    
                except Exception as e:
                    # Handle errors for individual URLs
                    result = {
                        "url": url,
                        "prediction": -1,
                        "confidence_safe": 0,
                        "confidence_phishing": 0,
                        "analysis_id": None,
                        "status": "Error",
                        "error": str(e)
                    }
                    results.append(result)
            
            return render_template("index.html", multiple_results=results, warning=warning)
        
        else:
            # Single URL processing (existing code)
            url = request.form["name"]
            
            # Extract features
            obj = FeatureExtraction(url)
            features = obj.getFeaturesList()
            x = np.array(features).reshape(1,30)
            
            # Get prediction and probabilities
            y_pred = gbc.predict(x)[0]
            y_proba = gbc.predict_proba(x)[0]
        
        # Calculate confidence scores
        if len(y_proba) >= 2:
            confidence_safe = float(y_proba[1] * 100)      # Probability of being safe
            confidence_phishing = float(y_proba[0] * 100)  # Probability of being phishing
        else:
            # Fallback if predict_proba doesn't work as expected
            confidence_safe = 80.0 if y_pred == 1 else 20.0
            confidence_phishing = 20.0 if y_pred == 1 else 80.0
        
        # Generate unique analysis ID
        analysis_id = str(uuid.uuid4())
        
        # Create detailed analysis data
        feature_analysis = []
        risk_factors = []
        safe_factors = []
        
        for i, (feature_name, feature_value) in enumerate(zip(FEATURE_NAMES, features)):
            feature_info = {
                "name": feature_name,
                "value": feature_value,
                "status": "Safe" if feature_value == 1 else ("Suspicious" if feature_value == 0 else "Risky")
            }
            feature_analysis.append(feature_info)
            
            # Categorize risk factors
            if feature_value == -1:
                risk_factors.append(feature_name)
            elif feature_value == 1:
                safe_factors.append(feature_name)
        
        # Smart risk factor filtering based on prediction
        if y_pred == 1:  # Safe site
            if len(risk_factors) > 0:
                # For safe sites, only show critical risk factors to avoid overwhelming users
                # This prevents showing minor technical details that don't affect security
                critical_risk_factors = []
                for factor in risk_factors:
                    if factor in ["Using IP", "HTTPS", "Age of Domain", "Abnormal URL", "Symbol@", "Redirecting//"]:
                        critical_risk_factors.append(factor)
                risk_factors = critical_risk_factors
        else:  # Unsafe site (prediction == -1)
            # For unsafe sites, show ALL risk factors to help users understand the full scope of threats
            # This helps users make informed decisions about why the site is considered unsafe
            pass  # Keep all risk factors as they are
        
        # Generate recommendations
        recommendations = generate_recommendations(risk_factors, y_pred)
        
        # Store detailed analysis
        analysis_data = {
            "id": analysis_id,
            "url": url,
            "prediction": int(y_pred),
            "confidence_safe": confidence_safe,
            "confidence_phishing": confidence_phishing,
            "feature_analysis": feature_analysis,
            "risk_factors": risk_factors,
            "safe_factors": safe_factors,
            "recommendations": recommendations,
            "timestamp": datetime.datetime.now().isoformat(),
            "total_features": len(features),
            "risky_features": len(risk_factors),
            "safe_features": len(safe_factors)
        }
        
        # Store in memory (use database in production)
        analysis_results[analysis_id] = analysis_data
        
        # Get basic result for main page
        name = convertion(url, int(y_pred))
        
        # Add analysis ID and confidence to the result
        if len(name) >= 2:
            name.append(f"{confidence_safe:.1f}")  # Add confidence
            name.append(analysis_id)  # Add analysis ID for detailed view
        
        return render_template("index.html", name=name, analysis_id=analysis_id)

def generate_recommendations(risk_factors, prediction):
    """Generate security recommendations based on detected risk factors"""
    recommendations = []
    
    if prediction == -1:  # Phishing detected
        recommendations.append("⚠️ HIGH RISK: This website is likely a phishing site. Do not enter personal information.")
        recommendations.append("🚫 Avoid clicking links or downloading files from this site.")
        recommendations.append("🔍 Verify the website URL carefully - look for misspellings or suspicious domains.")
        recommendations.append("📱 Report this website to your security team or relevant authorities.")
        
        # Add specific risk factor recommendations for unsafe sites
        if "Using IP" in risk_factors:
            recommendations.append("🌐 The website uses an IP address instead of a domain name, which is suspicious.")
        if "Long URL" in risk_factors:
            recommendations.append("📏 The URL is unusually long, which may indicate URL manipulation.")
        if "Short URL" in risk_factors:
            recommendations.append("🔗 This appears to be a shortened URL. Be cautious as it may redirect to malicious sites.")
        if "Symbol@" in risk_factors:
            recommendations.append("@ The URL contains '@' symbol, which can be used to deceive users about the real destination.")
        if "HTTPS" in risk_factors:
            recommendations.append("🔓 The website doesn't use HTTPS, making it less secure for data transmission.")
        if "Age of Domain" in risk_factors:
            recommendations.append("🆕 This is a newly registered domain, which increases phishing risk.")
        if "Sub Domains" in risk_factors:
            recommendations.append("📂 The URL has suspicious subdomain structure.")
        if "Brand Impersonation" in risk_factors:
            recommendations.append("🎭 This domain appears to impersonate a well-known brand. This is a common phishing technique.")
    
    elif prediction == 1:  # Safe website
        recommendations.append("✅ This website appears to be legitimate and safe to visit.")
        recommendations.append("🔒 However, always exercise caution when entering personal information online.")
        recommendations.append("🛡️ Make sure you're on the official website by checking the URL carefully.")
        
        # For safe sites, only show recommendations for critical risk factors
        if "Using IP" in risk_factors:
            recommendations.append("🌐 Note: This site uses an IP address, which is unusual but not necessarily unsafe.")
        if "HTTPS" in risk_factors:
            recommendations.append("🔓 Note: This site doesn't use HTTPS, consider using a VPN for additional security.")
        if "Age of Domain" in risk_factors:
            recommendations.append("🆕 Note: This is a newly registered domain, but appears legitimate based on other factors.")
    
    return recommendations

@app.route('/detailed_report/<analysis_id>')
def detailed_report(analysis_id):
    """Display detailed analysis report for a specific URL analysis"""
    if analysis_id not in analysis_results:
        return render_template('error.html', error="Analysis not found"), 404
    
    analysis_data = analysis_results[analysis_id]
    return render_template('detailed_report.html', analysis=analysis_data)

@app.route('/api/analysis/<analysis_id>')
def api_analysis(analysis_id):
    """API endpoint to get analysis data in JSON format"""
    if analysis_id not in analysis_results:
        return jsonify({"error": "Analysis not found"}), 404
    
    return jsonify(analysis_results[analysis_id])

@app.route('/usecases', methods=['GET', 'POST'])
def usecases():
    return render_template('usecases.html')

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == "__main__":
    app.run(debug=True)
