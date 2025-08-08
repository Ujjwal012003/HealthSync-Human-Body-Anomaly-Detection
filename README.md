# HealthSync AI - Human Body Anomaly Detection

A comprehensive web-based platform for medical anomaly detection that helps diagnose bone fractures and tuberculosis using machine learning.

## Project Overview

HealthSync AI is an advanced medical diagnostic tool that applies AI and machine learning to help healthcare professionals and patients detect health anomalies early. The system currently focuses on two primary use cases:

1. **Bone Fracture Detection**: Analysis of X-ray images to detect fractures
2. **Tuberculosis Detection**: Analysis of chest X-rays to identify signs of tuberculosis

## Who Can Use This System

- **Doctors**: Access to patient histories, ability to generate detailed reports, and collaborative features
- **Patients**: Self-service preliminary diagnosis, personal health tracking, and secure communication with healthcare providers
- **Healthcare Institutions**: Deployment options for institutional use with features for managing multiple practitioners

## Key Features

- **AI-Powered Diagnostics**: Using XGBoost models trained on medical image data
- **Dual-Interface System**: Separate dashboards for doctors and patients
- **Secure Data Handling**: HTTPS support with SSL certificates for protecting sensitive medical data
- **Comprehensive Reporting**: Detailed diagnostic reports with confidence metrics
- **User Management**: Account creation, profile management, and role-based access control
- **Patient History Tracking**: Chronological record of diagnoses and treatments

## Technologies Used

### Backend
- **Flask**: Python web framework for creating the application server
- **SQLAlchemy**: ORM for database operations and management
- **XGBoost**: Machine learning library for anomaly detection models
- **OpenCV & scikit-image**: Image processing libraries for feature extraction
- **scikit-learn**: Machine learning utilities for model training and evaluation

### Frontend
- **HTML/CSS/JavaScript**: Core web technologies for the user interface
- **Tailwind CSS**: Utility-first CSS framework for responsive design
- **jQuery**: JavaScript library for DOM manipulation
- **Chart.js**: JavaScript charting library for data visualization
- **jsPDF**: Client-side PDF generation

### Security & Deployment
- **PyOpenSSL**: For generating SSL certificates
- **Werkzeug**: WSGI web application library with security features
- **Flask-Migrate**: Database migration management
- **Gunicorn**: WSGI HTTP server for production deployment

## Setup Instructions

### Prerequisites

* Python 3.7+ installed
* pip (Python package manager)
* Git (optional, for cloning the repository)

### Installation Steps

1. **Clone the repository** (or download and extract the ZIP file):  
```  
git clone https://github.com/Ujjwal012003/HealthSync-Human-Body-Anomaly-Detection.git  
cd HealthSync-Human-Body-Anomaly-Detection  
```
2. **Navigate to the main application directory**:  
```  
cd "Human Body Anomaly Detection/CancerDetectionWeb"  
```
3. **Create and activate a virtual environment**:  
```  
python -m venv .venv  
# On Windows  
.\.venv\Scripts\activate  
# On macOS/Linux  
source .venv/bin/activate  
```
4. **Install dependencies**:  
```  
pip install -r requirements.txt  
```
5. **Run the setup script** (optional but recommended):  
```  
python setup.py  
```
6. **Initialize the database** (if needed):  
```  
python init_db.py  
```
7. **Run the application**:  
```  
python run.py  
```
8. **Access the application**: Open your browser and navigate to <https://localhost:5000>

### Model Files Setup

The trained XGBoost models are large binary files that need to be downloaded separately:

#### Option 1: Download from GitHub Releases (Recommended)
1. Go to [Releases](https://github.com/Ujjwal012003/HealthSync-Human-Body-Anomaly-Detection/releases/latest)
2. Download the model files directly:
   - [Fracture_XGBoost](https://github.com/Ujjwal012003/HealthSync-Human-Body-Anomaly-Detection/releases/download/v1.0.0/Fracture_XGBoost)
   - [TB_XGBoost](https://github.com/Ujjwal012003/HealthSync-Human-Body-Anomaly-Detection/releases/download/v1.0.0/TB_XGBoost)
3. Place them in the `CancerDetectionWeb/` directory

#### Option 2: Train Your Own Models
If you prefer to train your own models:
1. Use the provided training scripts in the `data/` directory
2. Follow the instructions in the training notebooks
3. Place the trained models in the `CancerDetectionWeb/` directory

#### Option 3: Run Without Models
The application will run without model files, but prediction functionality will be limited to basic image upload and display.

## Usage Guide

### For Patients

1. **Create an account**: Sign up with your email and personal details
2. **Complete your profile**: Add relevant medical history
3. **Upload medical images**: Submit X-ray images for analysis
4. **View results**: Receive preliminary diagnostic results
5. **Share with doctor**: Send results to your healthcare provider
6. **Track history**: Monitor your diagnostic history over time

### For Doctors

1. **Create a doctor account**: Register with professional credentials
2. **View patient records**: Access patient histories and previous diagnoses
3. **Analyze images**: Review uploaded images and AI predictions
4. **Generate reports**: Create detailed diagnostic reports
5. **Communicate**: Send findings to patients securely

## Project Structure

- **CancerDetectionWeb/**: Main application directory
  - **app.py**: Main Flask application
  - **run.py**: Application startup script
  - **templates/**: HTML templates for the web interface
  - **static/**: CSS, JavaScript, and images
  - **uploads/**: Temporary storage for uploaded images
  - **data/**: Training data and model evaluation
  - **models/**: Trained machine learning models
  - **migrations/**: Database migration files
  - **certificates/**: SSL certificates for HTTPS

## Model Information

### Fracture Detection Model
- **Algorithm**: XGBoost (Gradient Boosted Decision Trees)
- **Features**: Mean intensity, variance, edge density
- **Input**: X-ray images (processed for feature extraction)
- **Output**: Binary classification (fracture present/absent) with confidence score

### Tuberculosis Detection Model
- **Algorithm**: XGBoost
- **Features**: Mean intensity, variance
- **Input**: Chest X-ray images (processed for feature extraction)
- **Output**: Binary classification (TB signs present/absent) with confidence score

## Security Considerations

This application handles sensitive medical data and implements several security measures:
- HTTPS with SSL encryption for data in transit
- Secure password hashing
- Role-based access control
- Data privacy compliance features

## Contributing

Contributions to improve the system are welcome. Please follow these steps:
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request

## License

[Insert License Information Here]

## Contact

[Insert Contact Information Here]

# Add to your app.py
from flask_cors import CORS

# Enable CORS
CORS(app, origins=[os.environ.get('ALLOWED_ORIGINS')])

mkdir healthsync-frontend
cd healthsync-frontend 

git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/healthsync-frontend.git
git push -u origin main 