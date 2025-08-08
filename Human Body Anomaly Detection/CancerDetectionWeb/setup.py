#!/usr/bin/env python3
"""
Setup script for HealthSync AI - Human Body Anomaly Detection
"""

import os
import sys
import subprocess
import shutil

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 7):
        print("❌ Python 3.7 or higher is required")
        print(f"Current version: {sys.version}")
        return False
    print(f"✅ Python version: {sys.version}")
    return True

def create_directories():
    """Create necessary directories"""
    directories = ['uploads', 'static', 'templates', 'instance']
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"✅ Created directory: {directory}")
        else:
            print(f"✅ Directory exists: {directory}")

def install_dependencies():
    """Install required packages"""
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def check_model_files():
    """Check if model files exist"""
    model_files = ['Fracture_XGBoost', 'TB_XGBoost']
    missing_models = []
    
    for model in model_files:
        if os.path.exists(model):
            print(f"✅ Model file exists: {model}")
        else:
            print(f"⚠️  Model file missing: {model}")
            missing_models.append(model)
    
    if missing_models:
        print("\n📝 Note: Missing model files will limit prediction functionality.")
        print("You can train models using the provided training scripts.")
    
    return len(missing_models) == 0

def main():
    """Main setup function"""
    print("🚀 Setting up HealthSync AI - Human Body Anomaly Detection")
    print("=" * 60)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create directories
    print("\n📁 Creating directories...")
    create_directories()
    
    # Install dependencies
    print("\n📦 Installing dependencies...")
    if not install_dependencies():
        sys.exit(1)
    
    # Check model files
    print("\n🤖 Checking model files...")
    check_model_files()
    
    print("\n✅ Setup complete!")
    print("\n📋 Next steps:")
    print("1. Run: python run.py")
    print("2. Open: https://localhost:5000")
    print("3. Create an account and start using the application")

if __name__ == "__main__":
    main() 