# PhishShield - Use Case Specification

## Use Case Diagram Overview

The PhishShield system involves multiple actors interacting with various functionalities. The system serves as a comprehensive phishing detection and education platform.

## Actors

### Primary Actors
1. **End User** - General public seeking to verify URL safety
2. **Security Analyst** - IT professionals analyzing threats
3. **Developer/Administrator** - System maintenance and configuration

### Secondary Actors (External Systems)
4. **WHOIS Service** - Domain registration information
5. **Google APIs** - Search indexing and PageRank data
6. **Traffic Analytics Services** - Website traffic statistics
7. **DNS Servers** - Domain name resolution

---

## Use Case Specifications

### UC-01: Analyze URL for Phishing

**Primary Actor:** End User, Security Analyst  
**Goal:** Determine if a given URL is legitimate or a phishing attempt  
**Scope:** PhishShield System  
**Level:** User Goal  

**Preconditions:**
- User has access to PhishShield web interface
- URL is properly formatted (http/https)
- Internet connection is available

**Postconditions:**
- URL analysis is completed
- Classification result is generated
- Confidence score is calculated
- Analysis ID is created for detailed report access

**Main Success Scenario:**
1. User navigates to PhishShield homepage
2. User enters URL in the input field
3. User clicks "Scan URL" button
4. System validates URL format
5. System extracts 30 security features from URL
6. System calls machine learning model for prediction
7. System calculates confidence scores
8. System generates unique analysis ID
9. System displays results with safety classification
10. System provides option to view detailed report

**Extensions:**
- 4a. Invalid URL format
  - 4a1. System displays error message
  - 4a2. User returns to step 2
- 5a. URL is inaccessible
  - 5a1. System uses available features for analysis
  - 5a2. System notes accessibility issues in report
- 6a. Model prediction fails
  - 6a1. System uses fallback classification logic
  - 6a2. System logs error for administrator review

---

### UC-02: View Analysis Results

**Primary Actor:** End User, Security Analyst  
**Goal:** Review the classification results of URL analysis  
**Scope:** PhishShield System  
**Level:** User Goal  

**Preconditions:**
- URL analysis has been completed (UC-01)
- Analysis results are available

**Postconditions:**
- User understands the safety classification
- User can make informed decision about URL safety

**Main Success Scenario:**
1. System displays analysis results on same page
2. System shows URL being analyzed
3. System displays safety classification (Safe/Not Safe)
4. System shows confidence percentage with progress bar
5. System provides action buttons (Continue/Still want to Continue)
6. System offers link to detailed analysis report
7. User reviews the information
8. User decides on next action

**Extensions:**
- 3a. Result is "Not Safe"
  - 3a1. System displays warning styling (red/danger)
  - 3a2. System shows "Still want to Continue" button
- 3b. Result is "Safe"
  - 3b1. System displays success styling (green)
  - 3b2. System shows "Continue" button

---

### UC-03: View Detailed Security Report

**Primary Actor:** End User, Security Analyst  
**Goal:** Access comprehensive analysis breakdown with feature details  
**Scope:** PhishShield System  
**Level:** User Goal  

**Preconditions:**
- Analysis has been completed with valid analysis ID
- User clicks on "View Detailed Report" link

**Postconditions:**
- User has access to comprehensive security analysis
- User understands specific risk factors and safe features

**Main Success Scenario:**
1. User clicks "View Detailed Report" link
2. System retrieves analysis data using analysis ID
3. System displays detailed report page with:
   - URL analysis overview
   - Confidence meter visualization
   - Feature analysis summary (Safe/Neutral/Risk counts)
   - Feature breakdown chart (pie chart)
   - Individual feature status for all 30 features
   - Detected risk factors (if any)
   - Security recommendations
   - Action buttons (back, download, print)
4. User reviews detailed information
5. User may download or print report

**Extensions:**
- 2a. Analysis ID not found
  - 2a1. System displays error page
  - 2a2. System provides link back to homepage
- 3a. No risk factors detected
  - 3a1. System displays positive security message
  - 3a2. System still shows general security recommendations

---

### UC-04: Download Analysis Report

**Primary Actor:** End User, Security Analyst  
**Goal:** Export analysis results in JSON format for external use  
**Scope:** PhishShield System  
**Level:** User Goal  

**Preconditions:**
- Detailed report is displayed (UC-03)
- Browser supports file downloads

**Postconditions:**
- JSON file is downloaded to user's device
- File contains complete analysis data

**Main Success Scenario:**
1. User clicks "Download Report (JSON)" button
2. System serializes analysis data to JSON format
3. System creates download link with unique filename
4. Browser initiates file download
5. File is saved to user's default download location

---

### UC-05: Learn About Phishing

**Primary Actor:** End User  
**Goal:** Gain knowledge about phishing attacks and prevention  
**Scope:** PhishShield System  
**Level:** User Goal  

**Preconditions:**
- User is on PhishShield homepage

**Postconditions:**
- User has increased awareness of phishing threats
- User knows prevention strategies

**Main Success Scenario:**
1. User scrolls down to FAQ section on homepage
2. User clicks on various phishing-related questions
3. System expands content showing:
   - Definition of phishing
   - Types of phishing attacks
   - Why phishing awareness matters
   - Prevention strategies for individuals
   - Prevention strategies for companies
4. User reads educational content
5. User gains understanding of phishing threats

---

### UC-06: View Use Case Scenarios

**Primary Actor:** End User, Security Analyst  
**Goal:** Learn from real-world phishing attack examples  
**Scope:** PhishShield System  
**Level:** User Goal  

**Preconditions:**
- User clicks on "Usecases" in navigation menu

**Postconditions:**
- User understands real-world impact of phishing attacks
- User can relate to practical scenarios

**Main Success Scenario:**
1. User navigates to Use Cases page
2. System displays historical phishing attack cases:
   - John Podesta's Email (2016 Election)
   - BenefitMall breach (2018)
   - Sony Pictures attack (2014)
   - Methodist Hospitals breach (2019)
   - University of Wisconsin-Parkside (2019)
3. User reviews case studies with details about:
   - Attack vectors used
   - Data compromised
   - Financial impact
   - Lessons learned
4. User gains practical understanding of phishing risks

---

### UC-07: Access RESTful API

**Primary Actor:** Developer, Security Analyst  
**Goal:** Programmatically access PhishShield analysis capabilities  
**Scope:** PhishShield System  
**Level:** User Goal  

**Preconditions:**
- Developer has API endpoint knowledge
- Valid analysis ID exists (for data retrieval)

**Postconditions:**
- JSON data is returned with analysis results
- Developer can integrate PhishShield into external systems

**Main Success Scenario:**
1. Developer makes HTTP GET request to `/api/analysis/<analysis_id>`
2. System validates analysis ID
3. System retrieves analysis data from memory store
4. System serializes data to JSON format
5. System returns JSON response with analysis details
6. Developer receives structured data for integration

**Extensions:**
- 2a. Invalid analysis ID
  - 2a1. System returns 404 error with error message
  - 2a2. Developer handles error appropriately

---

### UC-08: Report Phishing Websites

**Primary Actor:** End User  
**Goal:** Report suspected phishing sites to appropriate authorities  
**Scope:** External Integration  
**Level:** User Goal  

**Preconditions:**
- User has identified a potential phishing website
- User is on PhishShield interface

**Postconditions:**
- Phishing report is submitted to external service
- User contributes to internet safety

**Main Success Scenario:**
1. User clicks on "Help" menu
2. User selects "Report Phishing Cases"
3. User chooses reporting option:
   - Google Safe Browsing
   - Google Support
4. System redirects to external reporting service
5. User completes report on external platform

---

### UC-09: Manage System Administration

**Primary Actor:** Developer/Administrator  
**Goal:** Maintain and configure PhishShield system  
**Scope:** PhishShield System  
**Level:** User Goal  

**Preconditions:**
- Administrator has system access
- System is deployed and running

**Postconditions:**
- System configuration is updated
- System performance is optimized

**Main Success Scenario:**
1. Administrator accesses server environment
2. Administrator performs maintenance tasks:
   - Monitor application logs
   - Update dependencies
   - Configure security settings
   - Manage memory storage
   - Update ML model if needed
3. Administrator verifies system functionality
4. Administrator documents changes

---

### UC-10: Handle Feature Extraction

**Primary Actor:** System (Internal Process)  
**Goal:** Extract 30 security features from submitted URL  
**Scope:** PhishShield System  
**Level:** Subfunction  

**Preconditions:**
- Valid URL is provided for analysis
- External services are accessible

**Postconditions:**
- 30-dimensional feature vector is created
- Features are ready for ML model input

**Main Success Scenario:**
1. System initializes FeatureExtraction class with URL
2. System attempts to fetch web content
3. System parses URL structure
4. System queries WHOIS database
5. System analyzes web content (HTML, CSS, JavaScript)
6. System checks external services (Google, traffic data)
7. System calculates each of 30 features:
   - URL-based features (7)
   - SSL/Security features (3)
   - Domain analysis features (4)
   - Content analysis features (8)
   - Advanced analysis features (8)
8. System returns feature vector for ML prediction

**Extensions:**
- 2a. Web content inaccessible
  - 2a1. System uses URL-based features only
  - 2a2. System marks content features as neutral
- 4a. WHOIS query fails
  - 4a1. System marks domain features as risky
  - 4a2. System continues with other features
- 6a. External service unavailable
  - 6a1. System uses cached data if available
  - 6a2. System marks affected features as neutral

---

## Use Case Relationships

### Include Relationships
- **View Analysis Results** includes **View Detailed Security Report**
- **View Detailed Security Report** includes **Download Analysis Report**
- **Analyze URL** includes **Handle Feature Extraction**

### Extend Relationships
- **Handle Analysis Errors** extends **Analyze URL**
- **Handle API Rate Limiting** extends **Access RESTful API**
- **Validate URL Format** extends **Analyze URL**

### Generalization Relationships
- **End User** and **Security Analyst** are specializations of **System User**
- **Download Report** and **Print Report** are specializations of **Export Data**

---

## Non-Functional Requirements

### Performance Requirements
- URL analysis should complete within 10 seconds
- System should handle 10+ concurrent users
- API responses should be under 500ms (excluding analysis time)

### Security Requirements
- All data transmission must use HTTPS
- No persistent storage of analyzed URLs
- Input validation for all user inputs
- Protection against XSS and injection attacks

### Usability Requirements
- Interface should be responsive (mobile-friendly)
- Analysis results should be clearly understandable
- Educational content should be accessible to non-technical users

### Reliability Requirements
- System uptime of 99%+ for production deployment
- Graceful degradation when external services are unavailable
- Error handling for all edge cases

---

## Conclusion

This use case specification provides a comprehensive view of the PhishShield system's functionality, covering all major interactions between actors and the system. The use cases are designed to address both security analysis needs and educational objectives, making PhishShield a complete solution for phishing detection and awareness.