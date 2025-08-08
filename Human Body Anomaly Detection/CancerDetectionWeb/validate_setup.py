"""
Validation Script for Cancer Detection Web Application

This script checks if all required components are properly set up:
1. Required directories (uploads, instance)
2. Model files (Fracture_XGBoost, TB_XGBoost)
3. Database existence

Usage: python validate_setup.py
"""

import os
import sys
import pickle

def check_directory(directory):
    """Check if directory exists and create it if not"""
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
            print(f"✓ Created directory: {directory}")
        except Exception as e:
            print(f"✗ Failed to create directory {directory}: {str(e)}")
            return False
    else:
        print(f"✓ Directory exists: {directory}")
    return True

def check_model_file(model_path):
    """Check if model file exists and can be loaded"""
    if not os.path.exists(model_path):
        print(f"✗ Model file not found: {model_path}")
        return False
    
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
            print(f"✓ Model loaded successfully: {model_path}")
            return True
    except Exception as e:
        print(f"✗ Failed to load model {model_path}: {str(e)}")
        return False

def check_database():
    """Check if SQLite database exists or can be created"""
    db_dir = "instance"
    db_file = os.path.join(db_dir, "users.db")
    
    if not os.path.exists(db_dir):
        try:
            os.makedirs(db_dir)
            print(f"✓ Created database directory: {db_dir}")
        except Exception as e:
            print(f"✗ Failed to create database directory: {str(e)}")
            return False
    
    if os.path.exists(db_file):
        print(f"✓ Database exists: {db_file}")
    else:
        print(f"! Database file does not exist: {db_file}")
        print("  It will be created when you run the application")
    
    return True

def validate_setup():
    """Run all validation checks"""
    print("\n=== Cancer Detection Web Application Validation ===\n")
    
    # Check required directories
    directories_valid = True
    directories_valid &= check_directory("uploads")
    directories_valid &= check_directory("static")
    directories_valid &= check_directory("templates")
    
    # Check model files
    models_valid = True
    models_valid &= check_model_file("Fracture_XGBoost")
    models_valid &= check_model_file("TB_XGBoost")
    
    # Check database
    database_valid = check_database()
    
    # Overall validation result
    print("\n=== Validation Summary ===")
    print(f"✓ Directories: {'All OK' if directories_valid else 'Issues Found'}")
    print(f"✓ Models: {'All OK' if models_valid else 'Issues Found'}")
    print(f"✓ Database: {'Ready' if database_valid else 'Issues Found'}")
    
    if not models_valid:
        print("\nMissing model files. You need to train the models first.")
        print("Run the following scripts:")
        print("  - python train_fracture.py")
        print("  - python train_tb_xgboost.py")
    
    print("\nSetup " + ("valid ✓" if (directories_valid and database_valid) else "invalid ✗"))
    print("You can safely run the application if all checks passed.")
    print("If model files are missing, prediction functionality will be limited.\n")

if __name__ == "__main__":
    # Change to the directory where this script is located
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    validate_setup() 