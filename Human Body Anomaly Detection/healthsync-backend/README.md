# HealthSync AI Backend

This is the backend for the HealthSync AI medical diagnostic application, handling bone fracture and tuberculosis detection using machine learning.

## Technologies Used
- Flask (Python web framework)
- SQLAlchemy (Database ORM)
- XGBoost (Machine learning for predictions)
- OpenCV and scikit-image (Image processing)

## Setup Instructions

### Local Development
1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Initialize the database:
   ```
   python init_db.py
   ```

3. Run the application:
   ```
   python run.py
   ```

### Deployment
This application is configured for deployment on Render.com using the provided Procfile.

## API Endpoints
- `/predict` - Submit medical images for diagnosis
- `/api/analytics/summary` - Get analytics data
- `/api/patients/{id}/scans` - Get patient scan history
- Many more endpoints available for the full application functionality 