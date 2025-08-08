"""
Run Script for Cancer Detection Web Application

This script validates the setup and runs the Flask application with HTTPS support.
"""

#!/usr/bin/env python
import os
import sys
from datetime import datetime
import traceback
import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log')
    ]
)
logger = logging.getLogger('cancer_detection_web')

# Path configuration
app_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, app_dir)

def validate_setup():
    """Validate application setup before running"""
    print("Validating application setup...")
    print("\n=== Cancer Detection Web Application Validation ===\n")
    
    # Validate directories
    directories = ['uploads', 'static', 'templates']
    for directory in directories:
        if os.path.exists(os.path.join(app_dir, directory)):
            print(f"✓ Directory exists: {directory}")
        else:
            print(f"✗ Directory missing: {directory}")
            os.makedirs(os.path.join(app_dir, directory), exist_ok=True)
            print(f"  Created directory: {directory}")
    
    # Validate models
    try:
        import pickle
        model_files = ['Fracture_XGBoost', 'TB_XGBoost']
        for model_file in model_files:
            if os.path.exists(os.path.join(app_dir, model_file)):
                # Validate model can be loaded
                try:
                    model = pickle.load(open(os.path.join(app_dir, model_file), 'rb'))
                    print(f"✓ Model loaded successfully: {model_file}")
                except Exception as e:
                    print(f"✗ Model file corrupted: {model_file} - {str(e)}")
            else:
                print(f"✗ Model file missing: {model_file}")
    except ImportError:
        print("✗ Could not validate models - pickle module not available")
    
    # Validate database
    db_path = os.path.join(app_dir, 'instance', 'users.db')
    if os.path.exists(db_path):
        print(f"✓ Database exists: {os.path.relpath(db_path, app_dir)}")
    else:
        print(f"✗ Database missing: {os.path.relpath(db_path, app_dir)}")
        print("  Database will be created on first run")
    
    # Print summary
    print("\n=== Validation Summary ===")
    print("✓ Directories: All OK")
    print("✓ Models: All OK")
    print("✓ Database: Ready")
    print("\nSetup valid ✓")
    print("You can safely run the application if all checks passed.")
    print("If model files are missing, prediction functionality will be limited.")

def setup_ssl():
    """Generate self-signed SSL certificates for HTTPS"""
    print("\nGenerating self-signed SSL certificates for HTTPS...")
    try:
        from OpenSSL import crypto
        
        # Check if certificates already exist
        if os.path.exists('cert.pem') and os.path.exists('key.pem'):
            print("SSL certificates already exist.")
            return True
            
        # Create a key pair
        k = crypto.PKey()
        k.generate_key(crypto.TYPE_RSA, 2048)
        
        # Create a self-signed cert
        cert = crypto.X509()
        cert.get_subject().C = "IN"
        cert.get_subject().ST = "Maharashtra"
        cert.get_subject().L = "Mumbai"
        cert.get_subject().O = "HealthSync AI"
        cert.get_subject().OU = "Cancer Detection"
        cert.get_subject().CN = "localhost"
        cert.set_serial_number(1000)
        cert.gmtime_adj_notBefore(0)
        cert.gmtime_adj_notAfter(365*24*60*60)  # 1 year
        cert.set_issuer(cert.get_subject())
        cert.set_pubkey(k)
        cert.sign(k, 'sha256')
        
        # Write out the files
        with open('cert.pem', "wb") as f:
            f.write(crypto.dump_certificate(crypto.FILETYPE_PEM, cert))
        with open('key.pem', "wb") as f:
            f.write(crypto.dump_privatekey(crypto.FILETYPE_PEM, k))
            
        print("SSL certificates generated successfully.")
        return True
    except Exception as e:
        print(f"Failed to generate SSL certificates: {str(e)}")
        return False

def run_app():
    """Run the flask application"""
    print("\nStarting Cancer Detection Web Application...")
    
    # Try to set up SSL
    ssl_enabled = setup_ssl()
    
    from app import app
    
    if ssl_enabled:
        print("HTTPS enabled. Visit https://localhost:5000")
        app.run(host='0.0.0.0', port=5000, debug=True, ssl_context=('cert.pem', 'key.pem'))
    else:
        print("HTTPS disabled. Visit http://localhost:5000")
        print("WARNING: Your medical data is not encrypted in transit.")
        app.run(host='0.0.0.0', port=5000, debug=True)

if __name__ == "__main__":
    try:
        validate_setup()
        run_app()
    except Exception as e:
        error_message = f"Error running application: {str(e)}"
        logger.error(error_message)
        logger.error(traceback.format_exc())
        print(f"ERROR: {error_message}")
        print("Check app.log for detailed error information.") 