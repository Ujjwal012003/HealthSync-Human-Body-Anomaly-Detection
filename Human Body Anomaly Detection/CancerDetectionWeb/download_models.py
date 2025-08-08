#!/usr/bin/env python3
"""
Download script for HealthSync AI model files
"""

import os
import sys
import requests
import zipfile
from pathlib import Path

def download_file(url, filename):
    """Download a file from URL"""
    try:
        print(f"📥 Downloading {filename}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"✅ Downloaded {filename}")
        return True
    except Exception as e:
        print(f"❌ Failed to download {filename}: {e}")
        return False

def extract_zip(zip_path, extract_to):
    """Extract zip file"""
    try:
        print(f"📦 Extracting {zip_path}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print(f"✅ Extracted to {extract_to}")
        return True
    except Exception as e:
        print(f"❌ Failed to extract {zip_path}: {e}")
        return False

def main():
    """Main download function"""
    print("🤖 HealthSync AI Model Downloader")
    print("=" * 40)
    
    # Check if models already exist
    model_files = ['Fracture_XGBoost', 'TB_XGBoost']
    existing_models = [f for f in model_files if os.path.exists(f)]
    
    if existing_models:
        print(f"✅ Found existing models: {', '.join(existing_models)}")
        response = input("Do you want to re-download? (y/N): ")
        if response.lower() != 'y':
            print("Skipping download.")
            return
    
    print("\n📋 Available download options:")
    print("1. GitHub Releases (recommended)")
    print("2. Direct download links")
    print("3. Manual download instructions")
    
    choice = input("\nSelect option (1-3): ")
    
    if choice == "1":
        print("\n📥 Downloading from GitHub Releases...")
        # GitHub releases URL (you'll need to update this with actual release URL)
        release_url = "https://github.com/Ujjwal012003/HealthSync-Human-Body-Anomaly-Detection/releases/latest"
        print(f"Please visit: {release_url}")
        print("Download the latest release and extract model files to this directory.")
        
    elif choice == "2":
        print("\n📥 Direct download links:")
        # GitHub release download URLs
        urls = {
            "Fracture_XGBoost": "https://github.com/Ujjwal012003/HealthSync-Human-Body-Anomaly-Detection/releases/download/v1.0.0/Fracture_XGBoost",
            "TB_XGBoost": "https://github.com/Ujjwal012003/HealthSync-Human-Body-Anomaly-Detection/releases/download/v1.0.0/TB_XGBoost"
        }
        
        for model_name, url in urls.items():
            download_file(url, model_name)
                
    elif choice == "3":
        print("\n📋 Manual download instructions:")
        print("1. Visit the project repository")
        print("2. Go to Releases section")
        print("3. Download the latest release")
        print("4. Extract model files to this directory")
        print("5. Ensure files are named: Fracture_XGBoost and TB_XGBoost")
    
    # Verify downloaded files
    print("\n🔍 Verifying model files...")
    for model in model_files:
        if os.path.exists(model):
            size = os.path.getsize(model) / (1024 * 1024)  # Size in MB
            print(f"✅ {model} ({size:.1f} MB)")
        else:
            print(f"❌ {model} not found")
    
    print("\n✅ Download process complete!")
    print("You can now run: python run.py")

if __name__ == "__main__":
    main() 